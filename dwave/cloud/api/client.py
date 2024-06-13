# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from collections import deque, namedtuple
from typing import Optional, Union

import requests
import urllib3
from packaging.specifiers import SpecifierSet
from werkzeug.http import parse_options_header, dump_options_header

from dwave.cloud.api import constants, exceptions
from dwave.cloud.config import load_config, validate_config_v1
from dwave.cloud.config.models import ClientConfig
from dwave.cloud.utils.exception import is_caused_by
from dwave.cloud.utils.http import PretimedHTTPAdapter, BaseUrlSession, default_user_agent

__all__ = ['DWaveAPIClient', 'SolverAPIClient', 'MetadataAPIClient', 'LeapAPIClient']

logger = logging.getLogger(__name__)


class LoggingSession(BaseUrlSession):
    """:class:`.BaseUrlSession` extended to unify timeout exceptions and to log
    all requests (and responses).

    In addition to request logging, a history of responses, including exceptions
    (up to `history_size`), is kept in :attr:`.history`.
    """

    def __init__(self, history_size: int = 0, **kwargs):
        if not isinstance(history_size, int) or history_size < 0:
            raise ValueError("non-negative integer value required for 'history_size'")

        self.history = deque([], maxlen=history_size)

        super().__init__(**kwargs)

    def _request_unified(self, method: str, *args, **kwargs):
        # timeout exceptions unified with regular request exceptions
        try:
            return super().request(method, *args, **kwargs)
        except Exception as exc:
            if is_caused_by(exc, (requests.exceptions.Timeout,
                                  urllib3.exceptions.TimeoutError)):
                raise exceptions.RequestTimeout(
                    request=getattr(exc, 'request', None),
                    response=getattr(exc, 'response', None)) from exc
            else:
                raise

    RequestRecord = namedtuple('RequestRecord',
                               ('request', 'response', 'exception'))

    def send(self, request: requests.PreparedRequest, **kwargs) -> requests.Response:
        callee = type(self).__name__
        if logger.getEffectiveLevel() >= logging.DEBUG:
            logger.debug(f"[{callee!s}] send(method={request.method!r}, url={request.url!r}, ...)")
        else:   # pragma: no cover
            logger.trace(f"[{callee!s}] send(method={request.method!r}, url={request.url!r},"
                         f" headers={request.headers!r}, body={request.body!r})")
        return super().send(request, **kwargs)

    def request(self, method: str, *args, **kwargs):
        callee = type(self).__name__
        if logger.getEffectiveLevel() >= logging.DEBUG:
            logger.debug("[%s] request(%r, %r, *%r, ...)",
                         callee, method, self.base_url, args)
        else:   # pragma: no cover
            logger.trace("[%s] request(%r, %r, *%r, **%r)",
                         callee, method, self.base_url, args, kwargs)

        try:
            response = self._request_unified(method, *args, **kwargs)

            rec = LoggingSession.RequestRecord(
                request=response.request, response=response, exception=None)
            self.history.append(rec)

        except Exception as exc:
            logger.debug("[%s] request failed with %r", callee, exc)

            req = getattr(exc, 'request', None)
            if req is not None:
                logger.trace("[%s] failing request=%r", callee,
                             dict(method=req.method, url=req.url,
                                  headers=req.headers, body=req.body))

            res = getattr(exc, 'response', None)
            if res is not None:
                logger.trace("[%s] failing response=%r", callee,
                             dict(status_code=res.status_code,
                                  headers=res.headers, text=res.text))

            rec = LoggingSession.RequestRecord(
                request=req, response=res, exception=exc)
            self.history.append(rec)

            raise

        if logger.getEffectiveLevel() >= logging.DEBUG:
            logger.debug("[%s] request succeeded with status_code=%r",
                         callee, response.status_code)
        else:   # pragma: no cover
            logger.trace("[%s] request(...) = (code=%r, body=%r, headers=%r)",
                         callee, response.status_code, response.text, response.headers)

        return response


class VersionedAPISession(LoggingSession):
    """A `requests.Session` subclass (technically, further specialized
    :class:`.LoggingSession`) that enforces conformance of API response
    version with supported version range(s).

    Response format version is requested via `Accept` header field, and
    format/version of the response is checked via `Content-Type`.

    Args:
        strict_mode:
            In strict mode (the default), response validation fails unless an
            appropriate `Content-Type` is returned. Set to ``False`` to disable it.

    """

    _strict_mode: bool = True
    _media_type: Optional[str] = None
    _ask_version: Optional[str] = None
    _accept_version: Optional[str] = None
    _media_type_params: Optional[dict] = None

    def __init__(self, strict_mode: bool = True, **kwargs):
        self._strict_mode = strict_mode
        super().__init__(**kwargs)

    def set_accept(self,
                   media_type: Optional[str] = None,
                   ask_version: Optional[str] = None,
                   accept_version: Optional[str] = None,
                   media_type_params: Optional[dict] = None,
                   strict_mode: Optional[bool] = None) -> None:
        """Set the request and expected response media type, going forward."""

        self._media_type = media_type
        self._ask_version = ask_version
        self._accept_version = accept_version
        self._media_type_params = media_type_params

        if strict_mode is not None:
            self._strict_mode = strict_mode

    def unset_accept(self) -> None:
        """Stop response validation by removing expectations of media type."""
        self._media_type = None

    def _validate_response_content_type(self, response: requests.Response) -> None:
        """Validate response's `Content-Type` matches the expected media type
        and version range.
        """
        if self._media_type is None or not response.ok:
            return

        content_type = response.headers.get('Content-Type')
        if not content_type:
            if self._strict_mode:
                raise exceptions.ResourceBadResponseError(
                    f'Media type not present in the response while '
                    f'expecting {self._media_type!r}')
            else:
                return

        media_type, params = parse_options_header(content_type)

        if media_type != self._media_type:
            raise exceptions.ResourceBadResponseError(
                f'Received media type {media_type!r} while '
                f'expecting {self._media_type!r}')

        if self._accept_version is not None:
            # todo: move parsing level up?
            ss = SpecifierSet(self._accept_version, prereleases=True)

            version = params.pop('version', None)
            if version is None:
                raise exceptions.ResourceBadResponseError(
                    'API response format version undefined in the response.')
            else:
                if not ss.contains(version):
                    raise exceptions.ResourceBadResponseError(
                        f'API response format version {version!r} not compliant '
                        f'with supported version {self._accept_version!r}. '
                        'Try upgrading "dwave-cloud-client".')

    def request(self, *args, headers: Optional[dict] = None, **kwargs) -> requests.Response:
        # (1) communicate lower bound on version handled in the outgoing request
        #     (via media type in `Accept` header field)
        if self._media_type is not None:
            params = {} if self._media_type_params is None else self._media_type_params.copy()
            if self._ask_version:
                params['version'] = self._ask_version

            headers = {} if headers is None else headers.copy()
            headers['Accept'] = dump_options_header(self._media_type, params)

        response = super().request(*args, headers=headers, **kwargs)

        # (2) validate format/version supported in the incoming response
        #     (validate `Content-Type` if `media_type` and/or `accept_version` set)
        if self._media_type is not None:
            self._validate_response_content_type(response)

        return response


class DWaveAPIClient:
    """Low-level client for D-Wave APIs. A thin wrapper around
    `requests.Session` that handles API specifics such as authentication,
    response and error parsing, retrying, etc.

    Note:
        To make sure the session is closed, call :meth:`.close`, or use the
        context manager form (as show in the example below).

    Note:
        Since :class:`requests.Session` is **not thread-safe**, neither is
        :class:`DWaveAPIClient`. It's best to create (and dispose) a new client
        on demand, in each thread.

    Example:
        with DWaveAPIClient(endpoint='...', timeout=(5, 600)) as client:
            client.session.get('...')
    """

    DEFAULTS = {
        'endpoint': None,
        'token': None,
        'cert': None,

        # (connect, read) timeout in sec
        'timeout': (60, 120),

        # urllib3.Retry options
        'retry': dict(total=10, backoff_factor=0.01, backoff_max=60),

        # optional additional headers
        'headers': None,

        # ssl verify
        'verify': True,

        # proxy urls, see :attr:`requests.Session.proxies`
        'proxies': None,

        # number of most recent request records to keep in :attr:`.session.history`
        'history_size': 0,

        # response version strict mode validation, see :class:`VersionedAPISession`
        'version_strict_mode': True
    }

    # client instance config, populated on init from kwargs overridding DEFAULTS
    config = None

    def __init__(self, **config):
        self.config = {}
        for opt, default in self.DEFAULTS.items():
            val = config.get(opt)
            if val is None:
                val = default
            self.config[opt] = val

        logger.debug(f"{type(self).__name__} initialized with config={self.config!r}")

        self.session = self._create_session(self.config)

    @classmethod
    def from_config_model(cls, config: ClientConfig, **options):
        """Create class instance based on a
        :class:`~dwave.cloud.config.models.ClientConfig` config.
        """
        logger.trace(f"{cls.__name__}.from_config_model(config={config!r}, **{options!r})")

        if config.headers:
            headers = config.headers.copy()
        else:
            headers = {}
        if config.connection_close:
            headers.update({'Connection': 'close'})

        opts = dict(
            endpoint=config.endpoint,
            token=config.token,
            cert=config.cert,
            headers=headers,
            timeout=config.request_timeout,
            retry=config.request_retry.model_dump(),
            proxies=dict(http=config.proxy, https=config.proxy),
            verify=not config.permissive_ssl,
        )

        # add class-specific options not existing in ClientConfig
        opts.update(**options)

        return cls(**opts)

    @classmethod
    def from_config_file(cls, config_file: Optional[str] = None,
                         profile: Optional[str] = None, **kwargs):
        """Create class instance based on config file / environment / kwarg options
        (same format as :class:`~dwave.cloud.client.base.Client`).
        """
        logger.trace(f"{cls.__name__}.from_config_file("
                     f"config_file={config_file!r}, profile={profile!r}, **{kwargs!r})")

        options = load_config(config_file=config_file, profile=profile, **kwargs)
        config = validate_config_v1(options)
        return cls.from_config_model(config)

    @classmethod
    def from_config(cls, config: Optional[Union[ClientConfig, str]] = None, **kwargs):
        """Create class instance based on client config or config file.

        Args:
            config:
                Client config model or path to configuration file. Based on `config`
                type, dispatches object creation to either :meth:`.from_config_model`
                or :meth:`.from_config_file`. If omitted, attempts to load
                configuration from file.

            **kwargs:
                Arguments passed to the dispatched method. See ``config`` above.
        """
        logger.trace(f"{cls.__name__}.from_config(config={config!r}, **{kwargs!r}")

        if isinstance(config, ClientConfig):
            return cls.from_config_model(config, **kwargs)

        kwargs['config_file'] = config
        return cls.from_config_file(**kwargs)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _retry_config(backoff_max=None, **kwargs):
        """Create http idempotent urllib3.Retry config."""

        retry = urllib3.Retry(**kwargs)

        # note: prior to `urllib3==2`, backoff_max had to be set manually on object
        if backoff_max is not None:
            # handle `urllib3>=1.21.1,<1.27` AND `urllib3>=1.21.1,<3`
            retry.BACKOFF_MAX = retry.backoff_max = backoff_max

        return retry

    @classmethod
    def _set_session_auth(cls, session, config):
        if config['token']:
            session.headers.update({'X-Auth-Token': config['token']})
        if config['cert']:
            session.cert = config['cert']

    @classmethod
    def _create_session(cls, config):
        # allow endpoint path to not end with /
        # (handle incorrect user input when merging paths, see rfc3986, sec 5.2.3)
        endpoint = config['endpoint']
        if not endpoint:
            raise ValueError("API endpoint undefined")
        if not endpoint.endswith('/'):
            endpoint += '/'

        # configure request timeout and retries
        session = VersionedAPISession(
            base_url=endpoint,
            history_size=config['history_size'],
            strict_mode=config['version_strict_mode'])
        timeout = config['timeout']
        retry = config['retry']
        session.mount('http://', PretimedHTTPAdapter(
            timeout=timeout, max_retries=cls._retry_config(**retry)))
        session.mount('https://', PretimedHTTPAdapter(
            timeout=timeout, max_retries=cls._retry_config(**retry)))

        # configure headers
        session.headers.update({'User-Agent': default_user_agent()})
        if config['headers']:
            session.headers.update(config['headers'])

        # auth
        cls._set_session_auth(session, config)

        if config['proxies']:
            session.proxies = config['proxies']
        if config['verify'] is not None:
            session.verify = config['verify']

        # raise all response errors as exceptions automatically
        session.hooks['response'].append(cls._raise_for_status)

        # debug log
        logger.debug(f"{cls.__name__} session created using config={config!r}")

        return session

    @staticmethod
    def _raise_for_status(response, **kwargs):
        """Raises :class:`~dwave.cloud.api.exceptions.RequestError`, if one
        occurred, with message populated from SAPI error response.

        See:
            :meth:`requests.Response.raise_for_status`.

        Raises:
            :class:`dwave.cloud.api.exceptions.RequestError` subclass
        """
        # NOTE: the expected behavior is for SAPI to return JSON error on
        # failure. However, that is currently not the case. We need to work
        # around this until it's fixed.

        # no error -> content type verified by `VersionedAPISession ` @accepts
        # error -> body can be json or plain text error message
        if not response.ok:
            try:
                msg = response.json()
                error_msg = msg['error_msg']
                error_code = msg['error_code']
            except:
                # TODO: better message when error body blank
                error_msg = response.text or response.reason
                error_code = response.status_code

            kw = dict(error_msg=error_msg,
                      error_code=error_code,
                      response=response)

            # map known SAPI error codes to exceptions
            exception_map = {
                400: exceptions.ResourceBadRequestError,
                401: exceptions.ResourceAuthenticationError,
                403: exceptions.ResourceAccessForbiddenError,
                404: exceptions.ResourceNotFoundError,
                409: exceptions.ResourceConflictError,
                429: exceptions.ResourceLimitsExceededError,
            }
            if error_code in exception_map:
                raise exception_map[error_code](**kw)
            elif 500 <= error_code < 600:
                raise exceptions.InternalServerError(**kw)
            else:
                raise exceptions.RequestError(**kw)


class SolverAPIClient(DWaveAPIClient):
    """Client for D-Wave's Solver API."""

    def __init__(self, **config):
        # TODO: make endpoint region-sensitive
        self.DEFAULTS = super().DEFAULTS.copy()
        self.DEFAULTS.update(endpoint=constants.DEFAULT_SOLVER_API_ENDPOINT)

        super().__init__(**config)


class MetadataAPIClient(DWaveAPIClient):
    """Client for D-Wave's Metadata API."""

    def __init__(self, **config):
        # TODO: make endpoint region-sensitive
        self.DEFAULTS = super().DEFAULTS.copy()
        self.DEFAULTS.update(endpoint=constants.DEFAULT_METADATA_API_ENDPOINT)

        super().__init__(**config)

    @classmethod
    def from_config_model(cls, config: ClientConfig, **kwargs):
        kwargs.setdefault('endpoint', config.metadata_api_endpoint)
        return super().from_config_model(config, **kwargs)


class LeapAPIClient(DWaveAPIClient):
    """Client for D-Wave's Leap API."""

    @classmethod
    def _set_session_auth(cls, session, config):
        if config['token']:
            session.headers.update({'Authorization': f"Bearer {config['token']}"})

    def __init__(self, **config):
        # TODO: make endpoint region-sensitive
        self.DEFAULTS = super().DEFAULTS.copy()
        self.DEFAULTS.update(endpoint=constants.DEFAULT_LEAP_API_ENDPOINT)

        super().__init__(**config)

    @classmethod
    def from_config_model(cls, config: ClientConfig, **kwargs):
        kwargs.setdefault('endpoint', config.leap_api_endpoint)
        return super().from_config_model(config, **kwargs)

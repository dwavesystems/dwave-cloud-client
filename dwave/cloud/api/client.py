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

import logging
from collections import deque, namedtuple
from typing import Optional

import requests
import urllib3
from packaging.specifiers import SpecifierSet
from werkzeug.http import parse_options_header, dump_options_header

from dwave.cloud.api import constants, exceptions
from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.utils import (
    TimeoutingHTTPAdapter, BaseUrlSession, user_agent, is_caused_by)

__all__ = ['DWaveAPIClient', 'SolverAPIClient', 'MetadataAPIClient']

logger = logging.getLogger(__name__)


class LazyUserAgentClassProperty:
    # roughly equivalent to ``classmethod(property(cached_user_agent))``, but it
    # doesn't require chained decorators support available in py39+
    _user_agent = None

    def __get__(self, obj, objtype=None):
        # Note: The only tags that might change are platform tags, as returned
        # by `dwave.common.platform.tags` entry points, and `platform.platform()`
        # (like linux kernel version). Assuming OS/machine won't change during
        # client's lifespan, and typical platform tags defined via entry points
        # depend on process environment variables (which rarely change), it's
        # pretty safe to cache the user-agent per-class (or even globally).
        if self._user_agent is None:
            self._user_agent = user_agent(__packagename__, __version__)
        return self._user_agent


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

    def request(self, method: str, *args, **kwargs):
        callee = type(self).__name__
        logger.trace("[%s] request(%r, *%r, **%r)",
                     callee, method, args, kwargs)

        try:
            response = self._request_unified(method, *args, **kwargs)

            rec = LoggingSession.RequestRecord(
                request=response.request, response=response, exception=None)
            self.history.append(rec)

        except Exception as exc:
            logger.debug("[%s] request failed with %r", callee, exc)

            req = getattr(exc, 'request', None)
            if req:
                logger.trace("[%s] failing request=%r", callee,
                             dict(method=req.method, url=req.url,
                                  headers=req.headers, body=req.body))

            res = getattr(exc, 'response', None)
            if res:
                logger.trace("[%s] failing response=%r", callee,
                             dict(status_code=res.status_code,
                                  headers=res.headers, text=res.text))

            rec = LoggingSession.RequestRecord(
                request=req, response=res, exception=exc)
            self.history.append(rec)

            raise

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

    # User-Agent string used in API requests, as returned by
    # :meth:`~dwave.cloud.utils.user_agent`, computed on first access and
    # cached for the lifespan of the class.
    # TODO: consider exposing "user_agent" config parameter
    user_agent = LazyUserAgentClassProperty()

    def __init__(self, **config):
        self.config = {}
        for opt, default in self.DEFAULTS.items():
            self.config[opt] = config.get(opt, default)

        self.session = self._create_session(self.config)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _retry_config(backoff_max=None, **kwargs):
        """Create http idempotent urllib3.Retry config."""

        if not kwargs:
            return None

        retry = urllib3.Retry(**kwargs)

        # note: `Retry.BACKOFF_MAX` can't be set on construction
        if backoff_max is not None:
            retry.BACKOFF_MAX = backoff_max

        return retry

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
        session.mount('http://',
            TimeoutingHTTPAdapter(
                timeout=timeout, max_retries=cls._retry_config(**retry)))
        session.mount('https://',
            TimeoutingHTTPAdapter(
                timeout=timeout, max_retries=cls._retry_config(**retry)))

        # configure headers
        session.headers.update({'User-Agent': cls.user_agent})
        if config['headers']:
            session.headers.update(config['headers'])

        # auth
        if config['token']:
            session.headers.update({'X-Auth-Token': config['token']})
        if config['cert']:
            session.cert = config['cert']

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

        # no error -> body is json
        # error -> body can be json or plain text error message
        if response.ok:
            try:
                response.json()
            except:
                raise exceptions.ResourceBadResponseError("JSON response expected")

        else:
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
        config.setdefault('endpoint', constants.DEFAULT_SOLVER_API_ENDPOINT)
        super().__init__(**config)

    @classmethod
    def from_client_config(cls, client):
        """Create SAPI client instance configured from a
        :class:`~dwave.cloud.client.base.Client' instance.
        """

        headers = client.headers.copy()
        if client.connection_close:
            headers.update({'Connection': 'close'})

        opts = dict(
            endpoint=client.endpoint,
            token=client.token,
            cert=client.client_cert,
            timeout=client.request_timeout,
            proxies=dict(
                http=client.proxy,
                https=client.proxy,
            ),
            retry=dict(
                total=client.http_retry_total,
                connect=client.http_retry_connect,
                read=client.http_retry_read,
                redirect=client.http_retry_redirect,
                status=client.http_retry_status,
                raise_on_redirect=True,
                raise_on_status=True,
                respect_retry_after_header=True,
                backoff_factor=client.http_retry_backoff_factor,
                backoff_max=client.http_retry_backoff_max,
            ),
            headers=client.headers,
            verify=not client.permissive_ssl,
        )

        return cls(**opts)


class MetadataAPIClient(DWaveAPIClient):
    """Client for D-Wave's Metadata API."""

    def __init__(self, **config):
        config.setdefault('endpoint', constants.DEFAULT_METADATA_API_ENDPOINT)
        super().__init__(**config)

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

import requests
import urllib3

from dwave.cloud.api import constants, exceptions
from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.utils import (
    TimeoutingHTTPAdapter, BaseUrlSession, user_agent, is_caused_by)

__all__ = ['DWaveAPIClient', 'SolverAPIClient']

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
    """

    def request(self, method, *args, **kwargs):
        callee = type(self).__name__
        logger.trace("[%s] request(%r, *%r, **%r)",
                     callee, method, args, kwargs)

        # unify timeout exceptions
        try:
            response = super().request(method, *args, **kwargs)

        except Exception as exc:
            logger.trace("[%s] request failed with %r", callee, exc)

            if is_caused_by(exc, (requests.exceptions.Timeout,
                                  urllib3.exceptions.TimeoutError)):
                raise exceptions.RequestTimeout from exc
            else:
                raise

        logger.trace("[%s] request(...) = (code=%r, body=%r)",
                     callee, response.status_code, response.text)

        return response


class DWaveAPIClient:
    """Low-level D-Wave API client, as a thin wrapper around `requests.Session`,
    that handles API specifics like authentication, response and error parsing,
    retrying, etc.
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
        session = LoggingSession(base_url=endpoint)
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
        logger.debug("create_session from config={!r}".format(config))

        return session

    @staticmethod
    def _raise_for_status(response, **kwargs):
        """Raises :class:`~dwave.cloud.exceptions.SAPIRequestError`, if one
        occurred, with message populated from SAPI error response.

        See:
            :meth:`requests.Response.raise_for_status`.

        Raises:
            :class:`dwave.cloud.exceptions.SAPIRequestError` subclass
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

    def __init__(self, **config):
        config.setdefault('endpoint', constants.SOLVER_API_ENDPOINT)
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

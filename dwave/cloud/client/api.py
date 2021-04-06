import logging
from functools import lru_cache

import requests
import urllib3

from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.utils import TimeoutingHTTPAdapter, BaseUrlSession, user_agent

__all__ = ['SAPI']

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


class SAPI:
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

    # User-Agent string used in SAPI requests, as returned by
    # :meth:`~dwave.cloud.utils.user_agent`, computed on first access and
    # cached for the lifespan of the class.
    # TODO: consider exposing "user_agent" config parameter
    user_agent = LazyUserAgentClassProperty()

    def __init__(self, **kwargs):
        # populate .config with defaults overridden with supplied kwargs
        self.config = {}
        for opt, default in self.DEFAULTS.items():
            self.config[opt] = kwargs.get(opt, default)

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

    @staticmethod
    def _retry(backoff_max=None, **kwargs):
        """Create http idempotent urllib3.Retry config."""

        retry = urllib3.Retry(**kwargs)

        # note: `Retry.BACKOFF_MAX` can't be set on construction
        if backoff_max is not None:
            retry.BACKOFF_MAX = backoff_max

        return retry

    def create_session(self):
        # allow endpoint path to not end with /
        # (handle incorrect user input when merging paths, see rfc3986, sec 5.2.3)
        endpoint = self.config['endpoint']
        if not endpoint.endswith('/'):
            endpoint += '/'

        # configure request timeout and retries
        session = BaseUrlSession(base_url=endpoint)
        timeout = self.config['timeout']
        retry = self.config['retry']
        session.mount('http://',
            TimeoutingHTTPAdapter(
                timeout=timeout, max_retries=self._retry(**retry)))
        session.mount('https://',
            TimeoutingHTTPAdapter(
                timeout=timeout, max_retries=self._retry(**retry)))

        # configure headers
        session.headers.update({'User-Agent': self.user_agent})
        if self.config['headers']:
            session.headers.update(self.config['headers'])

        # auth
        if self.config['token']:
            session.headers.update({'X-Auth-Token': self.config['token']})
        if self.config['cert']:
            session.cert = self.config['cert']

        if self.config['proxies']:
            session.proxies = self.config['proxies']

        # debug log
        logger.debug("create_session from config={!r}".format(self.config))

        return session

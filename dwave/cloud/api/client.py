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

import io
import hashlib
import logging
import numbers
import os
import warnings
import zlib
from collections import deque, namedtuple, abc
from collections.abc import Iterable
from typing import IO, Optional, TypedDict, TYPE_CHECKING, Union

import http_sf
import orjson
import requests
import urllib3
import werkzeug
from packaging.specifiers import SpecifierSet

import dwave.cloud.config
from dwave.cloud.api import exceptions
from dwave.cloud.api.models import DeprecationMessage
from dwave.cloud.utils.exception import is_caused_by
from dwave.cloud.utils.http import PretimedHTTPAdapter, BaseUrlSessionMixin, default_user_agent
from dwave.cloud.utils.time import epochnow

if TYPE_CHECKING:
    from dwave.cloud.config.models import ClientConfig

__all__ = ['DWaveAPIClient', 'SolverAPIClient', 'MetadataAPIClient', 'LeapAPIClient']

logger = logging.getLogger(__name__)


class LoggingSessionMixin:
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
            logger.debug("[%s] request(%r, *%r, ...)",
                         callee, method, args)
        else:   # pragma: no cover
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
            logger.trace("[%s] request(...) = (code=%r, headers=%r, body=%r)",
                         callee, response.status_code, response.headers, response.text)

        return response


class LoggingSession(LoggingSessionMixin, BaseUrlSessionMixin, requests.Session):
    def __init__(self, *args, **kwargs):
        # note: this deprecation warning is raised by all subclasses, up to and
        # including `CachingSession`. The message adapts accordingly.
        cls = type(self).__name__
        warnings.warn(
            f"`{cls}` is deprecated in favor of `{cls}Mixin` since "
            f"dwave-cloud-client 0.13.6, and will be removed in 0.15.0.",
            DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class PayloadCompressingSessionMixin:
    """A :class:`requests.Session` subclass (technically, a further specialized
    :class:`.LoggingSession`) that adds support for payload compression on the
    fly.

    Args:
        compress (bool, default=False):
            Preemptively compress HTTP request body. When compression is enabled,
            the payload is always compressed in 32KiB chunks and streamed to
            minimize memory overhead.

    .. note::
        With compression enabled, the request body is transferred using chunked
        encoding (``Transfer-Encoding: chunked``) and content is compressed using
        the `DEFLATE <https://www.rfc-editor.org/rfc/rfc1951.txt>`_ compression
        algorithm (``Content-Encoding: deflate``).

    .. versionadded:: 0.13.3
        Preemptive payload compression support added to :class:`.DWaveAPIClient`.

    """

    _compress: bool = False

    def __init__(self, compress: bool = False, **kwargs):
        self._compress = compress
        super().__init__(**kwargs)

    def set_payload_compress(self, compress: bool = True) -> bool:
        """Modify payload compression on upload for current session.

        Args:
            compress (bool, default=True):
                Enable request body compression with DEFLATE on upload requests.
                Make sure the server supports compression on inbound requests,
                as most servers don't.

        Returns:
            bool:
                Previous compression setting.
        """
        previous = self._compress
        self._compress = compress
        return previous

    @staticmethod
    def _iter_compressed(data: Union[bytes, IO, Iterable[bytes]],
                         chunk_size: Optional[int] = None):
        if chunk_size is None:
            # match zlib window size
            chunk_size = 2**15

        if isinstance(data, bytes):
            data = io.BytesIO(data)

        if hasattr(data, 'read'):
            def get_chunks(f, chunk_size):
                while chunk := f.read(chunk_size):
                    yield chunk
            chunks = get_chunks(data, chunk_size)

        else:
            # assume data is iterable
            chunks = data

        zbuf = zlib.compressobj()

        for chunk in chunks:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            if c := zbuf.compress(chunk):
                yield c

        yield zbuf.flush()

    def request(self, *args, **kwargs):
        compress = kwargs.pop('compress', self._compress)

        # set compression for current request (assumes Session is not thread-safe)
        old = self._compress
        try:
            self._compress = compress
            return super().request(*args, **kwargs)
        finally:
            self._compress = old

    def send(self, request: requests.PreparedRequest, **kwargs) -> requests.Response:
        # Note: compress body on `send()` (rather than on `request()`) as by now
        # json, form, and multipart data are all converted to bytes. We only need
        # to handle bytes, files and iterators at this point.
        if self._compress and request.body:
            request.body = self._iter_compressed(request.body)
            # update the prepared request for chunked upload
            request.headers['Content-Encoding'] = 'deflate'
            request.headers['Transfer-Encoding'] = 'chunked'
            request.headers.pop('Content-Length', None)

        return super().send(request, **kwargs)


class PayloadCompressingSession(PayloadCompressingSessionMixin, LoggingSession):
    pass


class VersionedAPISessionMixin:
    """A :class:`requests.Session` subclass (technically, further specialized
    :class:`.PayloadCompressingSession`) that enforces conformance of API response
    version with supported version range(s).

    Response format version is requested via ``Accept`` header field, and
    format/version of the response is checked via ``Content-Type``.

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
        if self._media_type is None or not 200 <= response.status_code < 300:
            return

        content_type = response.headers.get('Content-Type')
        if not content_type:
            if self._strict_mode:
                raise exceptions.ResourceBadResponseError(
                    f'Media type not present in the response while '
                    f'expecting {self._media_type!r}')
            else:
                return

        media_type, params = werkzeug.http.parse_options_header(content_type)

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
            headers['Accept'] = werkzeug.http.dump_options_header(self._media_type, params)

        response = super().request(*args, headers=headers, **kwargs)

        # (2) validate format/version supported in the incoming response
        #     (validate `Content-Type` if `media_type` and/or `accept_version` set)
        if self._media_type is not None:
            self._validate_response_content_type(response)

        return response


def _create_default_cache_store(**kwargs) -> abc.Mapping:
    # TODO: de-dup vs `dwave.cloud.utils.decorators.cached`?

    # defer to break circular imports
    from dwave.cloud.config import get_cache_dir

    directory = kwargs.pop('directory', os.path.join(get_cache_dir(), 'api'))

    # defer import and construction until needed
    import diskcache
    return diskcache.Cache(directory=directory)


class VersionedAPISession(VersionedAPISessionMixin, PayloadCompressingSession):
    pass


class CachingSessionMixin:
    """A :class:`requests.Session` subclass (technically, further specialized
    :class:`.VersionedAPISession`) that caches responses and uses conditional
    requests for smart cache updates.

    Args:
        cache (dict):
            Cache configuration in a dict with keys:

                * ``enabled`` (bool):
                    Enable request caching.

                * ``maxage`` (float):
                    Default response maxage, in case server response
                    doesn't specify ``Cache-Control``.

                * ``store`` (mapping, callable):
                    Define cache storage (with a Mapping interface). Default to
                    ``diskcache.Cache`` with pickle serialization.

    :meth:`CachingSession.request` accepts the following keyword arguments,
    in addition to the base class method:

        * ``maxage_`` (float):
            Suggested maxage for heuristic caching, overriding the default value
            from cache config.

        * ``no_cache_`` (bool):
            Skip cache.

        * ``refresh_`` (bool):
            Force cache update, skipping time-based or etag-based validation.
    """

    class CacheConfig(TypedDict, total=False):
        enabled: bool
        maxage: float
        store: Union[abc.Mapping, abc.Callable[[], abc.Mapping]]

    # default cache config
    _default_cache_config = CacheConfig(enabled=True,
                                        maxage=0,
                                        store=_create_default_cache_store)

    def __init__(self, cache: Union[CacheConfig, bool, None] = None, **kwargs):
        if cache is None:
            cache = self._default_cache_config

        if not isinstance(cache, dict):
            enabled = bool(cache)
            cache = self._default_cache_config
            cache.update(enabled=enabled)

        self.configure_cache(cache)

        super().__init__(**kwargs)

    def configure_cache(self, cache: CacheConfig) -> None:
        config = {opt: cache.get(opt, default)
                  for opt, default in self._default_cache_config.items()}

        self._cache_enabled = enabled = config['enabled']
        if not enabled:
            logger.debug("[%s] cache disabled.", type(self).__name__)
            return

        _maxage = config.get('maxage')
        if not isinstance(_maxage, numbers.Real) or _maxage < 0:
            raise ValueError("A non-negative real value required for 'maxage'.")
        self._maxage = _maxage

        self._store = config['store']
        if callable(self._store):
            self._store = self._store()

        logger.debug("[%s] configured cache: (enabled=%r, maxage=%r, store=%r)",
                     type(self).__name__, self._cache_enabled, self._maxage, self._store)

        if hasattr(self._store, 'directory'):
            logger.debug("cache.store.directory=%r", self._store.directory)

    def _parse_cache_control(
            self,
            cache_control: Optional[str] = None,
            ) -> tuple[bool, Optional[int]]:
        # parse Cache-Control header field (if present) and derive a suitable max-age value
        # returns: (cache?, maxage)

        # note:
        # - we always revalidate after expiry, so we can ignore `must-revalidate`
        # - and we never transform the response, so safe to ignore `no-transform`
        # - our cache is private (for a single user), so we can ignore `private`,
        #   `public` and `s-maxage` directives

        cc = werkzeug.http.parse_cache_control_header(
                cache_control, cls=werkzeug.datastructures.ResponseCacheControl)

        if cc.no_cache:
            # cache, but always validate
            return (True, 0)

        if cc.no_store:
            # skip cache
            return (False, 0)

        # cache for up to max_age
        return (True, cc.max_age)

    def _update_cache(
            self,
            response: requests.Response,
            key_meta: str,
            key_data: Optional[str] = None,
            only_meta: bool = False
            ) -> bool:
        # update cache from response according to cache-control
        # returns: updated?

        # etag and cache-control are present in 304 if they're in 200 response
        # see: https://datatracker.ietf.org/doc/html/rfc9110#name-304-not-modified
        etag = response.headers.get('ETag')
        content_type = response.headers.get('Content-Type')
        cache_control = response.headers.get('Cache-Control')

        use_cache, maxage = self._parse_cache_control(cache_control)
        if use_cache:
            meta = self._store.get(key_meta, {})
            meta.update(created=epochnow(), maxage=maxage, etag=etag)
            if content_type is not None:
                meta.update(content_type=content_type)
            self._store[key_meta] = meta
            logger.debug("cache_write (meta): %r = %r", key_meta, meta)

            if not only_meta:
                content = self._store[key_data] = response.content
                logger.debug("cache_write (data): %r <- (%d bytes)",
                             key_data, len(content))

            logger.debug("response cached for maxage=%r", maxage)
            return True

        else:
            if key_meta in self._store:
                del self._store[key_meta]
            if key_data in self._store:
                del self._store[key_data]

            logger.debug("response caching skipped at server's request")
            return False

    def request(self, method, url, *,
                params: Optional[dict] = None,
                headers: Optional[dict] = None,
                **kwargs
                ) -> requests.Response:

        # read CachingSession-specific params
        refresh = kwargs.pop('refresh_', False)
        no_cache = kwargs.pop('no_cache_', False)
        default_maxage = kwargs.pop('maxage_', None)

        if not self._cache_enabled or no_cache or method.lower() != 'get':
            # completely bypass cache lookup and validation
            return super().request(method, url, params=params, headers=headers, **kwargs)

        if default_maxage is None:
            default_maxage = self._maxage
        elif not isinstance(default_maxage, numbers.Real) or default_maxage < 0:
           warnings.warn(
               f"non-negative real value required for 'maxage_'; ignoring {default_maxage}",
               UserWarning, stacklevel=2)
           default_maxage = self._maxage

        key = (method, url,
               self.params, params,
               self.headers, headers)

        # note: metadata split from data for faster metadata-only updates
        key_data = hashlib.sha256(repr(key).encode('utf8')).hexdigest()
        key_meta = f"{key_data}:meta"

        def make_response(meta, content):
            res = requests.Response()
            res.status_code = 200
            res.raw = io.BytesIO(content)
            res.headers['content-type'] = meta.get('content_type')
            return res

        meta = self._store.get(key_meta)
        content = self._store.get(key_data)

        if not refresh and meta and content:
            logger.debug("cache_read (meta): %r = %r", key_meta, meta)
            logger.debug("cache_read (data): %r -> (%d bytes)", key_data, len(content))

            # respect max-age from response cache-control
            maxage = meta.get('maxage')
            if maxage is None:
                # but if cache-control was absent, client gets do decide
                maxage = default_maxage

            if epochnow() - meta['created'] < maxage:
                logger.debug(f'cache hit within maxage ({maxage} seconds)')
                return make_response(meta, content)

            # validate with conditional request if possible
            if etag := meta.get('etag'):
                headers = {} if headers is None else headers.copy()
                headers['If-None-Match'] = etag

            res = super().request(method, url, params=params, headers=headers, **kwargs)

            if res.status_code == requests.codes['not_modified']:
                # see 4.3.4. in rfc9111 (http caching)
                # we know SAPI only uses weak validators
                logger.debug('resource "not modified", using cached value')
                self._update_cache(res, key_meta=key_meta, only_meta=True)
                return make_response(meta, content)

            else:
                logger.debug('resource modified, updating cache')
                self._update_cache(res, key_meta=key_meta, key_data=key_data)
                return res

        else:
            logger.debug('cache miss, updating cache')
            res = super().request(method, url, params=params, headers=headers, **kwargs)
            self._update_cache(res, key_meta=key_meta, key_data=key_data)
            return res


class CachingSession(CachingSessionMixin, VersionedAPISession):
    pass


class DeprecationAwareSessionMixin:
    """A :class:`requests.Session` mixin that interprets Leap deprecation
    messages in the API response headers, storing them, logging and raising
    warnings on the fly.

    Args:
        on_deprecation:
            Configuration typed dictionary describing behavior when a
            deprecation message is received. The following keys are supported
            (all optional and true by default):

                * ``log`` (bool):
                    Enable deprecation message logging (using ``WARNING`` log level).

                * ``warn`` (bool):
                    Raise each deprecation message using :exc:`DeprecationWarning`,
                    or :class:`~dwave.cloud.api.exceptions.ResourceDeprecationWarning`
                    warning category class.

                * ``store`` (bool):
                    Enable deprecation message storing in the session attribute,
                    :attr:`.deprecations`, as well as in the request response.

     .. note::
        Low-level API deprecations are raised using (usually silenced)
        :exc:`DeprecationWarning` warning category. User application-level
        deprecations of, for example, a solver, solver feature or solver
        parameter are raised using a more prominent (not silenced by default)
        :class:`~dwave.cloud.api.exceptions.ResourceDeprecationWarning` user
        warning category.

    .. versionadded:: 0.14.0
        Support for Leap deprecation messages added to :class:`.DWaveAPIClient`.

    """

    deprecations: list[DeprecationMessage] = None
    """Deprecation messages received in the last response, if storing enabled."""

    class OnDeprecationConfig(TypedDict, total=False):
        log: bool
        warn: bool
        store: bool

    DEFAULT_ON_DEPRECATION_CONFIG = OnDeprecationConfig(log=True, warn=True, store=True)

    def __init__(self, on_deprecation: Optional[OnDeprecationConfig] = None, **kwargs):
        super().__init__(**kwargs)

        self.set_on_deprecation(**self.DEFAULT_ON_DEPRECATION_CONFIG)
        if on_deprecation is not None:
            self.set_on_deprecation(**on_deprecation)

    def set_on_deprecation(self,
                           *,
                           log: Optional[bool] = None,
                           warn: Optional[bool] = None,
                           store: Optional[bool] = None,
                           ) -> None:
        """Configure the on-deprecation behavior, as described in
        :class:`.DeprecationAwareSessionMixin`.
        """
        if log is not None:
            self._log_deprecations = log
        if warn is not None:
            self._raise_deprecations = warn
        if store is not None:
            self._store_deprecations = store

    def _parse_deprecation_messages(self, response: requests.Response) -> list[DeprecationMessage]:
        """Parse deprecation notes returned in ``X-Deprecation`` header, encoded
        using "Structured Field Values for HTTP" (RFC 9651).

        Note: malformed headers are ignored.

        Note: this format of deprecation messages is Leap-specific.
        """

        x_dep = response.headers.get('X-Deprecation')
        if not x_dep:
            return []

        try:
            sfv = http_sf.parse(x_dep.encode(), tltype='list')
        except Exception as exc:
            logger.debug("Deprecation header %r parsing failed with %r.", x_dep, exc)
            return []

        try:
            deps = [DeprecationMessage(id=str(v[0]), **v[1]) for v in sfv]
        except Exception as exc:
            logger.debug("Deprecation message %r validation failed with %r.", sfv, exc)
            return []

        return deps

    def request(self, *args, **kwargs) -> requests.Response:
        response = super().request(*args, **kwargs)

        deps = self._parse_deprecation_messages(response)

        if self._store_deprecations:
            self.deprecations = deps
            setattr(response, 'deprecations', deps)

        if self._log_deprecations:
            for dep in deps:
                logger.warning("API Deprecation Warning: %r", dep)

        if self._raise_deprecations:
            for dep in deps:
                text = f"{dep.message} Sunset date: {dep.sunset.date().isoformat()}."
                if dep.link:
                    text = f"{text} For details, see: {dep.link!r}."
                if dep.is_app_level_deprecation:
                    # user warning: user should change the solver, parameter used, etc.
                    warnings.warn(text, exceptions.ResourceDeprecationWarning, stacklevel=3)
                else:
                    # developer warning
                    warnings.warn(text, DeprecationWarning, stacklevel=3)

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
        >>> with DWaveAPIClient(endpoint='...', timeout=(5, 600)) as client:    # doctest: +SKIP
        ...     client.session.get('...')

    """

    class DefaultSession(VersionedAPISessionMixin,
                         DeprecationAwareSessionMixin,
                         PayloadCompressingSessionMixin,
                         BaseUrlSessionMixin,
                         CachingSessionMixin,
                         LoggingSessionMixin,
                         requests.Session):
        """Default session class is based on `requests.Session` extended with
        base url, logging, compressing, API versioning and caching mixins.
        """

    DEFAULTS = {
        'session_class': DefaultSession,

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

        # preemptive payload compression on upload requests, see :class:`PayloadCompressingSession`
        'compress': False,

        # response version strict mode validation, see :class:`VersionedAPISession`
        'version_strict_mode': True,

        # enable conditional requests and response caching, see :class:`CachingSession`
        'cache': dict(enabled=False),       # type: CachingSession.CacheConfig

        # api deprecation message handling, see :class:`DeprecationAwareSessionMixin`
        'on_deprecation': dict(log=True, warn=True, store=True),
    }

    # client instance config, populated on init from kwargs overriding DEFAULTS
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

        options = dwave.cloud.config.load_config(config_file=config_file, profile=profile, **kwargs)
        config = dwave.cloud.config.validate_config_v1(options)
        return cls.from_config_model(config)

    @classmethod
    def from_config(cls, config: Union[ClientConfig, str, None] = None, **kwargs):
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

        if isinstance(config, dwave.cloud.config.ClientConfig):
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

        session_class = config['session_class']

        # build kwargs from mixins used
        # note: we could automate param extraction with a metaclass, but let's kiss
        session_kwargs = {}
        if issubclass(session_class, BaseUrlSessionMixin):
            session_kwargs.update(base_url=endpoint)
        if issubclass(session_class, LoggingSessionMixin):
            session_kwargs.update(history_size=config['history_size'])
        if issubclass(session_class, CachingSessionMixin):
            session_kwargs.update(cache=config['cache'])
        if issubclass(session_class, PayloadCompressingSessionMixin):
            session_kwargs.update(compress=config['compress'])
        if issubclass(session_class, VersionedAPISessionMixin):
            session_kwargs.update(strict_mode=config['version_strict_mode'])
        if issubclass(session_class, DeprecationAwareSessionMixin):
            session_kwargs.update(on_deprecation=config['on_deprecation'])

        session = session_class(**session_kwargs)

        # configure request timeout and retries
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
                msg = orjson.loads(response.content)
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
        self.DEFAULTS.update(endpoint=dwave.cloud.config.constants.DEFAULT_SOLVER_API_ENDPOINT)

        # add qpu data compression option
        self.DEFAULTS.update(compress_qpu_problem_data=False)

        super().__init__(**config)

    @classmethod
    def from_config_model(cls, config: ClientConfig, **kwargs):
        kwargs.setdefault('compress_qpu_problem_data', config.compress_qpu_problem_data)
        return super().from_config_model(config, **kwargs)


class MetadataAPIClient(DWaveAPIClient):
    """Client for D-Wave's Metadata API."""

    def __init__(self, **config):
        # TODO: make endpoint region-sensitive
        self.DEFAULTS = super().DEFAULTS.copy()
        self.DEFAULTS.update(endpoint=dwave.cloud.config.constants.DEFAULT_METADATA_API_ENDPOINT)

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
        self.DEFAULTS.update(endpoint=dwave.cloud.config.constants.DEFAULT_LEAP_API_ENDPOINT)

        super().__init__(**config)

    @classmethod
    def from_config_model(cls, config: ClientConfig, **kwargs):
        kwargs.setdefault('endpoint', config.leap_api_endpoint)
        return super().from_config_model(config, **kwargs)

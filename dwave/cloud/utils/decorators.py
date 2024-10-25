# Copyright 2024 D-Wave Inc.
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

"""Various useful decorators (grouped here for lack of a better namespace)
for private and Ocean-internal use.

.. versionchanged:: 0.12.0
   These functions previously lived under ``dwave.cloud.utils``.
"""

import logging
import math
import numbers
import time
import warnings

from collections import abc
from functools import wraps
from secrets import token_hex
from typing import Any, Optional, Union

from dwave.cloud.utils.time import epochnow

__all__ = ['aliasdict', 'cached', 'deprecated', 'retried']

logger = logging.getLogger(__name__)


class aliasdict(dict):
    """A dict subclass with support for item aliasing -- when you want to allow
    explicit access to some keys, but not to store them in the dict.

    :class:`aliasdict` can be used as a stand-in replacement for :class:`dict`.
    If no aliases are added, behavior is identical to :class:`dict`.

    Alias items added can be explicitly accessed, but they are not visible
    otherwise via the dict interface. Aliases shadow original keys, and their
    values can be computed on access only.

    Aliases are added with :meth:`.alias`, and they are stored in the
    :attr:`.aliases` class instance dictionary.

    Example:
        >>> from operator import itemgetter
        >>> from dwave.cloud.utils.decorators import aliasdict

        >>> d = aliasdict(a=1, b=2)
        >>> d.alias(c=itemgetter('a'))
        >>> d
        {'a': 1, 'b': 2}
        >>> 'c' in d
        True
        >>> d['c']
        1

    """
    __slots__ = ('aliases', )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # keep alias keys and reference values separate from the base dict
        self.aliases = {}

    def alias(self, *args, **kwargs):
        """Update aliases dictionary with the key/value pairs from ``other``,
        overwriting existing keys.

        Args:
            other (dict/Iterable[(key,value)]):
                Either another dictionary object or an iterable of key/value
                pairs (as tuples or other iterables of length two). If keyword
                arguments are specified, the dictionary is then updated with
                those key/value pairs ``d.alias(red=1, blue=2)``.

        Note:
            Alias key will become available via item getter, but it will not
            be listed in the container.

            Alias value can be a concrete value for the alias key, or it can be
            a callable that is evaluated on the aliasdict instance, on each
            access.

        """
        self.aliases.update(*args, **kwargs)

    def _alias_value(self, key):
        value = self.aliases[key]
        if callable(value):
            value = value(self)
        return value

    def __getitem__(self, key):
        if key in self.aliases:
            return self._alias_value(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if key in self.aliases:
            return self.aliases.__setitem__(key, value)
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        if key in self.aliases:
            return self.aliases.__delitem__(key)
        return super().__delitem__(key)

    def __contains__(self, key):
        if key in self.aliases:
            return True
        return super().__contains__(key)

    def copy(self):
        new = type(self)(self)
        new.alias(self.aliases)
        return new


class cached:
    """Caching decorator with max-age/expiry, forced refresh, and
    per-arguments-combo keys.

    Args:
        maxage:
            Default cache max-age. Overridden with cached function's ``maxage_``
            argument.
        store:
            Data store.
        key:
            Name of cached function's argument to be used as a cache key.
        bucket:
            Cache bucket prefix. By default, ``@cached`` instances use isolated
            buckets.

    The decorated function accepts two additional keyword arguments:
        refresh_ (bool):
            Force cache miss.
        maxage_ (float):
            Value's maximum allowed age for a cache hit.

    Examples:
        Cache for 5 minutes::

            @cached(maxage=300)
            def get_solvers(**features):
                return requests.get(...)

        Populate the cache on the first hit for a specific arguments combination::

            get_solvers(name='asd', count=5)

        Cache hit (note a different ordering of arguments)::

            get_solvers(count=5, name='asd')

        Not in cache::

            get_solvers(count=10, name='asd')

        But cache is refreshed, even on a hit, if ``refresh_=True``::

            get_solvers(count=5, name='asd', refresh_=True)

        By default, cache indefinitely::

            @cached()
            def f(x):
                return x**2

        Specify per-call value max-age::

            f(x, maxage_=10)

        For stability reasons, for a cache hit, we require item age to be
        strictly less than `maxage`.

    """

    _disabled = False

    def disable(self):
        """Disable/bypass cache on the decorated function."""

        # set on instance
        self._disabled = True

    def enable(self):
        """Enable cache on the decorated function."""

        # revert to class attr
        try:
            del self._disabled
        except:
            pass

    def _argshash(self, args: abc.Sequence[Any], kwargs: abc.Mapping[Any, Any]):
        """Hash mutable arguments' containers with immutable keys and values."""
        if self.key is None:
            # the default: use all args and kwargs for cache key
            tokens = (repr(args),
                      repr(sorted((repr(k), repr(v)) for k, v in kwargs.items())))
        else:
            # use a single named argument (required!) as the cache key
            tokens = (repr(kwargs[self.key]), )

        return '-'.join((self.bucket, *tokens))

    def __init__(self, *,
                 maxage: Optional[float] = None,
                 store: Union[abc.Mapping, abc.Callable[[], abc.Mapping], None] = None,
                 key: Optional[str] = None,
                 bucket: Optional[str] = None):

        self.default_maxage = maxage

        if store is None:
            store = {}
        if callable(store):
            store = store()
        self.store = store

        self.key = key
        self.bucket = bucket

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # pop additional params before calling the fn
            refresh = kwargs.pop('refresh_', False)
            maxage = kwargs.pop('maxage_', self.default_maxage)
            if maxage is None:
                maxage = math.inf

            if self._disabled:
                return fn(*args, **kwargs)

            now = epochnow()
            key = self._argshash(args, kwargs)
            data = self.store.get(key)

            callee = type(self).__name__
            logger.trace("[%s] call(refresh=%r, maxage=%r, now=%r, store=%r, key=%r, data=%r)",
                         callee, refresh, maxage, now, self.store, key, data)
            found = False
            if not refresh and data and (now - data['created'] < maxage):
                val = data['val']
                found = True
            else:
                val = fn(*args, **kwargs)
                self.store[key] = dict(created=now, val=val)

            logger.trace("[%s] call(...) = %r (cache %s)", callee, val,
                         'hit' if found else 'miss')
            return val

        # expose @cached internals for testing, debugging and cache control via
        # :meth:`.disable()`
        wrapper.cached = self

        # set bucket prefix
        if self.bucket is None:
            self.bucket = f"{fn.__name__}-{token_hex(8)}"

        return wrapper

    @classmethod
    def ondisk(cls, **kwargs):
        """@cached backed by an on-disk sqlite3-based cache."""

        # defer to break circular imports
        from dwave.cloud.config import get_cache_dir

        directory = kwargs.pop('directory', get_cache_dir())
        compression_level = kwargs.pop('compression_level', 6)

        # defer diskcache construction (sqlite db access!) until actually needed.
        # also, by postponing sqlite3 connect we enable easier post-fork re-init
        # (see https://www.sqlite.org/faq.html#q6)
        def store_initializer():
            import diskcache
            return diskcache.Cache(disk=diskcache.JSONDisk, directory=directory,
                                   disk_compress_level=compression_level)

        return cls(store=store_initializer, **kwargs)

    class disabled:
        """Context manager and decorator that disables the cache within the
        context or the decorated function.

        Decorator use example::
            @cached()
            def f(x):
                return x**2

            @cached.disabled()
            def no_cache(x):
                return f(x)

            f(1)            # cache miss
            f(1)            # cache hit
            no_cache(1)     # identical to the undecorated f(x) call; cache untouched

        Context manager use example::
            @cached()
            def f(x):
                return x**2

            with cached.disabled():
                f(1)        # identical to the undecorated f(x) call; cache untouched

            f(1)            # cache miss

        """

        def start(self):
            # lazy import: cached.disabled is rarely used, yet mock import
            # adds 50ms to root package import
            from unittest import mock

            self.patcher = mock.patch.object(cached, '_disabled', True)
            self.patcher.start()

        def stop(self):
            self.patcher.stop()

        def __enter__(self):
            return self.start()

        def __exit__(self, exc_type, exc_value, traceback):
            self.stop()

        def __call__(self, fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                self.start()
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.stop()

            return wrapper


class deprecated:
    """Decorator that issues a deprecation message on each call of the
    decorated function.
    """

    def __init__(self, msg: Optional[str] = None, stacklevel: int = 2):
        self.msg = msg
        self.stacklevel = stacklevel

    def __call__(self, fn: abc.Callable) -> abc.Callable:
        if not callable(fn):
            raise TypeError("decorated object must be callable")

        @wraps(fn)
        def wrapped(*args, **kwargs):
            msg = self.msg
            if not msg:
                fn_name = getattr(fn, '__name__', 'unnamed')
                msg = "{}() has been deprecated".format(fn_name)
            warnings.warn(msg, DeprecationWarning, stacklevel=self.stacklevel)

            return fn(*args, **kwargs)

        return wrapped


class retried:
    """Decorator that retries running the wrapped function `retries` times,
    logging exceptions along the way.

    Args:
        retries (int, default=1):
            Decorated function is allowed to fail `retries` times.

        backoff (number/List[number]/callable, default=0):
            Delay (in seconds) before a retry.

    Example:
        Retry up to three times::

            import random

            def f(thresh):
                r = random.random()
                if r < thresh:
                    raise ValueError
                return r

            retried_f = retried(3)(f)

            retried_f(0.5)
    """

    def __init__(self, retries: int = 1, backoff: float = 0):
        self.retries = retries

        # normalize `backoff` to callable
        if isinstance(backoff, numbers.Number):
            self.backoff = lambda retry: backoff
        elif isinstance(backoff, abc.Sequence):
            it = iter(backoff)
            self.backoff = lambda retry: next(it)
        else:
            self.backoff = backoff

    def __call__(self, fn: abc.Callable) -> abc.Callable:
        if not callable(fn):
            raise TypeError("decorated object must be callable")

        @wraps(fn)
        def wrapped(*args, **kwargs):
            for retries_left in range(self.retries, -1, -1):
                try:
                    return fn(*args, **kwargs)

                except Exception as exc:
                    fn_name = getattr(fn, '__name__', 'unnamed')
                    logger.debug(
                        "Running %s(*%r, **%r) failed with %r. Retries left: %d",
                        fn_name, args, kwargs, exc, retries_left)

                    if retries_left == 0:
                        raise exc

                retry = self.retries - retries_left + 1
                delay = self.backoff(retry)
                logger.debug("Sleeping for %s seconds before retrying.", delay)
                time.sleep(delay)

        return wrapped

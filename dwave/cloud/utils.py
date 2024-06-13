# Copyright 2017 D-Wave Systems Inc.
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

import json
import math
import time
import logging
import warnings
import inspect
import numbers

from collections import OrderedDict
from functools import wraps
from importlib.metadata import Distribution, PackageNotFoundError
from secrets import token_hex
from typing import Any, Optional, Union, List, Dict, Mapping, Sequence
from unittest import mock

import diskcache
from importlib_metadata import entry_points
from packaging.requirements import Requirement

# Use numpy if available for fast decoding
try:
    import numpy
except ImportError:  # pragma: no cover
    pass

__all__ = ['hasinstance', 'exception_chain', 'is_caused_by',
           'NumpyEncoder', 'coerce_numpy_to_python',
           'get_distribution', 'PackageNotFoundError', 'VersionNotFoundError',
           ]

logger = logging.getLogger(__name__)


def coerce_numpy_to_python(obj):
    """Numpy object serializer with support for basic scalar types and ndarrays."""

    if isinstance(obj, numpy.integer):
        return int(obj)
    elif isinstance(obj, numpy.floating):
        return float(obj)
    elif isinstance(obj, numpy.bool_):
        return bool(obj)
    elif isinstance(obj, numpy.ndarray):
        return [coerce_numpy_to_python(v) for v in obj.tolist()]
    elif isinstance(obj, (list, tuple)):    # be explicit to avoid recursing over string et al
        return type(obj)(coerce_numpy_to_python(v) for v in obj)
    elif isinstance(obj, dict):
        return {coerce_numpy_to_python(k): coerce_numpy_to_python(v) for k, v in obj.items()}
    return obj


# copied from dwave-hybrid utils
# (https://github.com/dwavesystems/dwave-hybrid/blob/b9025b5bb3d88dce98ec70e28cfdb25400a10e4a/hybrid/utils.py#L43-L61)
# TODO: switch to `dwave.common` if and when we create it
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types.

    Supported types:
     - basic numeric types: booleans, integers, floats
     - arrays: ndarray, recarray
    """

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()

        return super().default(obj)


def hasinstance(iterable, class_or_tuple):
    """Extension of ``isinstance`` to iterables/sequences. Returns True iff the
    sequence contains at least one object which is instance of ``class_or_tuple``.
    """

    return any(isinstance(e, class_or_tuple) for e in iterable)


def exception_chain(exception):
    """Traverse the chain of embedded exceptions, yielding one at the time.

    Args:
        exception (:class:`Exception`): Chained exception.

    Yields:
        :class:`Exception`: The next exception in the input exception's chain.

    Examples:
        def f():
            try:
                1/0
            except ZeroDivisionError:
                raise ValueError

        try:
            f()
        except Exception as e:
            assert(hasinstance(exception_chain(e), ZeroDivisionError))

    See: PEP-3134.
    """

    while exception:
        yield exception

        # explicit exception chaining, i.e `raise .. from ..`
        if exception.__cause__:
            exception = exception.__cause__

        # implicit exception chaining
        elif exception.__context__:
            exception = exception.__context__

        else:
            return


def is_caused_by(exception, exception_types):
    """Check if any of ``exception_types`` is causing the ``exception``.
    Equivalently, check if any of ``exception_types`` is contained in the
    exception chain rooted at ``exception``.

    Args:
        exception (:class:`Exception`):
            Chained exception.

        exception_types (:class:`Exception` or tuple of :class:`Exception`):
            Exception type or a tuple of exception types to check for.

    Returns:
        bool:
            True when ``exception`` is caused by any of the exceptions in
            ``exception_types``.
    """

    return hasinstance(exception_chain(exception), exception_types)


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

    def _argshash(self, args: List[Any], kwargs: Dict[Any, Any]):
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
                 store: Optional[Mapping] = None,
                 key: Optional[str] = None,
                 bucket: Optional[str] = None):

        self.default_maxage = maxage

        if store is None:
            store = {}
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
        from dwave.cloud.config import get_cache_dir
        directory = kwargs.pop('directory', get_cache_dir())
        compression_level = kwargs.pop('compression_level', 6)
        cache = diskcache.Cache(disk=diskcache.JSONDisk, directory=directory,
                                disk_compress_level=compression_level)
        return cls(store=cache, **kwargs)

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


class retried(object):
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

    def __init__(self, retries=1, backoff=0):
        self.retries = retries

        # normalize `backoff` to callable
        if isinstance(backoff, numbers.Number):
            self.backoff = lambda retry: backoff
        elif isinstance(backoff, Sequence):
            it = iter(backoff)
            self.backoff = lambda retry: next(it)
        else:
            self.backoff = backoff

    def __call__(self, fn):
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


class deprecated(object):
    """Decorator that issues a deprecation message on each call of the
    decorated function.
    """

    def __init__(self, msg=None, stacklevel=2):
        self.msg = msg
        self.stacklevel = stacklevel

    def __call__(self, fn):
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



def pretty_argvalues():
    """Pretty-formatted function call arguments, from the caller's frame."""
    return inspect.formatargvalues(*inspect.getargvalues(inspect.currentframe().f_back))


def get_contrib_config():
    """Return all registered contrib (non-open-source) Ocean packages."""

    # Note: we use `entry_points` from `importlib_metadata` to simplify access
    # and use py312 semantics. See "compatibility note" in `importlib.metadata`
    # docs for entry points.
    contrib = [ep.load() for ep in entry_points(group='dwave_contrib')]
    return contrib


def get_contrib_packages():
    """Combine all contrib packages in an ordered dict. Assumes package names
    are unique.
    """

    contrib = get_contrib_config()

    packages = OrderedDict()
    for dist in contrib:
        for pkg in dist:
            packages[pkg['name']] = pkg

    return packages


class VersionNotFoundError(Exception):
    """Package version requirement not satisfied."""


def get_distribution(requirement: Union[str, Requirement],
                     prereleases: bool = True) -> Distribution:
    """Returns :class:`~importlib.metadata.Distribution` for a matching
    `requirement` specification.

    Note: this function re-implements :func:`pkg_resources.get_distribution`
    functionality for py38+ (including py312, where setuptools/pkg_resources
    is not available by default).

    Args:
        requirement:
            Package dependency requirement according to PEP-508, given as string,
            or :class:`~packaging.requirements.Requirement`.
        prereleases:
            Boolean flag to control if installed prereleases are allowed.

    Raises:
        :class:`~importlib.metadata.PackageNotFoundError`:
            Package by name not found.
        :class:`~dwave.cloud.utils.VersionNotFoundError`:
            Version of the package found (distribution) does not match the
            requirement.
    """

    if not isinstance(requirement, Requirement):
        requirement = Requirement(requirement)

    dist = Distribution.from_name(requirement.name)

    if not requirement.specifier.contains(dist.version, prereleases=prereleases):
        raise VersionNotFoundError(
            f"Package {dist.name!r} version {dist.version} "
            f"does not match {requirement.specifier!s}")

    return dist


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
        >>> from dwave.cloud.utils import aliasdict

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


def bqm_to_dqm(bqm):
    """Represent a :class:`dimod.BQM` as a :class:`dimod.DQM`."""
    try:
        from dimod import DiscreteQuadraticModel
    except ImportError: # pragma: no cover
        raise RuntimeError(
            "dimod package with support for DiscreteQuadraticModel required."
            "Re-install the library with 'dqm' support.")

    dqm = DiscreteQuadraticModel()

    ising = bqm.spin

    for v, bias in ising.linear.items():
        dqm.add_variable(2, label=v)
        dqm.set_linear(v, [-bias, bias])

    for (u, v), bias in ising.quadratic.items():
        biases = numpy.array([[bias, -bias], [-bias, bias]], dtype=numpy.float64)
        dqm.set_quadratic(u, v, biases)

    return dqm

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

import sys
import json
import time
import random
import logging
import platform
import itertools
import numbers
import warnings

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from urllib.parse import urljoin
from datetime import datetime, timedelta
from dateutil.tz import UTC
from functools import partial, wraps
from pkg_resources import iter_entry_points
from typing import Any, Optional, Sequence, Union

import requests
import diskcache

# Use numpy if available for fast decoding
try:
    import numpy
    _numpy = True
except ImportError:  # pragma: no cover
    _numpy = False

__all__ = ['evaluate_ising', 'uniform_iterator', 'uniform_get',
           'default_text_input', 'datetime_to_timestamp',
           'datetime_to_timestamp', 'utcnow', 'epochnow', 'tictoc',
           'hasinstance', 'exception_chain', 'is_caused_by', 'NumpyEncoder',
           ]

logger = logging.getLogger(__name__)


def evaluate_ising(linear, quad, state, offset=0):
    """Calculate the energy of a state given the Hamiltonian.

    Args:
        linear: Linear Hamiltonian terms.
        quad: Quadratic Hamiltonian terms.
        offset: Energy offset.
        state: Vector of spins describing the system state.

    Returns:
        Energy of the state evaluated by the given energy function.
    """

    # If we were given a numpy array cast to list
    if _numpy and isinstance(state, numpy.ndarray):
        return evaluate_ising(linear, quad, state.tolist(), offset=offset)

    # Accumulate the linear and quadratic values
    energy = offset
    for index, value in uniform_iterator(linear):
        energy += state[index] * value
    for (index_a, index_b), value in quad.items():
        energy += value * state[index_a] * state[index_b]
    return energy


def active_qubits(linear, quadratic):
    """Calculate a set of all active qubits. Qubit is "active" if it has
    bias or coupling attached.

    Args:
        linear (dict[variable, bias]/list[variable, bias]):
            Linear terms of the model.

        quadratic (dict[(variable, variable), bias]):
            Quadratic terms of the model.

    Returns:
        set:
            Active qubits' indices.
    """

    active = {idx for idx,bias in uniform_iterator(linear)}
    for edge, _ in quadratic.items():
        active.update(edge)
    return active


def generate_random_ising_problem(solver, h_range=None, j_range=None):
    """Generates an Ising problem formulation valid for a particular solver,
    using all qubits and all couplings and linear/quadratic biases sampled
    uniformly from `h_range`/`j_range`.
    """

    if h_range is None:
        h_range = solver.properties.get('h_range', [-1, 1])
    if j_range is None:
        j_range = solver.properties.get('j_range', [-1, 1])

    lin = {qubit: random.uniform(*h_range) for qubit in solver.nodes}
    quad = {edge: random.uniform(*j_range) for edge in solver.undirected_edges}

    return lin, quad


def generate_const_ising_problem(solver, h=1, j=-1):
    return generate_random_ising_problem(solver, h_range=[h, h], j_range=[j, j])


def uniform_iterator(sequence):
    """Uniform (key, value) iteration on a `dict`,
    or (idx, value) on a `list`."""

    if isinstance(sequence, Mapping):
        return sequence.items()
    else:
        return enumerate(sequence)


def uniform_get(sequence, index, default=None):
    """Uniform `dict`/`list` item getter, where `index` is interpreted as a key
    for maps and as numeric index for lists."""

    if isinstance(sequence, Mapping):
        return sequence.get(index, default)
    else:
        return sequence[index] if index < len(sequence) else default


def reformat_qubo_as_ising(qubo):
    """Split QUBO coefficients into linear and quadratic terms (the Ising form).

    Args:
        qubo (dict[(int, int), float]):
            Coefficients of a quadratic unconstrained binary optimization
            (QUBO) model.

    Returns:
        (dict[int, float], dict[(int, int), float])

    """

    lin = {u: bias for (u, v), bias in qubo.items() if u == v}
    quad = {(u, v): bias for (u, v), bias in qubo.items() if u != v}

    return lin, quad


def strip_head(sequence, values):
    """Strips elements of `values` from the beginning of `sequence`."""
    values = set(values)
    return list(itertools.dropwhile(lambda x: x in values, sequence))


def strip_tail(sequence, values):
    """Strip `values` from the end of `sequence`."""
    return list(reversed(list(strip_head(reversed(sequence), values))))


def default_text_input(prompt: str, default: Optional[Any] = None, *,
                       optional: bool = True,
                       choices: Optional[Sequence[Any]] = None) -> Union[str, None]:
    # CLI util; defer click import until actually needed (see #473)
    import click
    _skip = 'skip'
    kwargs = dict(text=prompt)
    if default:
        kwargs.update(default=default)
    else:
        # make click print [skip] next to prompt
        if optional:
            kwargs.update(default=_skip)
    if choices:
        _type = click.Choice(choices)
        kwargs.update(type=_type)
        # a special case to skip user input instead of forcing input
        if optional:
            def allow_skip(value):
                if value == _skip:
                    return value
                return click.types.convert_type(_type)(value)

            kwargs.update(value_proc=allow_skip)

    value = click.prompt(**kwargs)
    if optional and value == _skip:
        value = None
    return value


def datetime_to_timestamp(dt):
    """Convert timezone-aware `datetime` to POSIX timestamp and
    return seconds since UNIX epoch.

    Note: similar to `datetime.timestamp()` in Python 3.3+.
    """

    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=UTC)
    return (dt - epoch).total_seconds()


def utcnow():
    """Returns tz-aware now in UTC."""
    return datetime.utcnow().replace(tzinfo=UTC)


def epochnow():
    """Returns now as UNIX timestamp.

    Invariant:
        epochnow() ~= datetime_to_timestamp(utcnow())

    """
    return time.time()


def utcrel(offset):
    """Return a timezone-aware `datetime` relative to now (UTC), shifted by
    `offset` seconds in to the future.

    Example:
        a_minute_from_now = utcrel(60)
    """
    return utcnow() + timedelta(seconds=offset)


def strtrunc(s, maxlen=60):
    s = str(s)
    return s[:(maxlen-3)]+'...' if len(s) > maxlen else s


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


class TimeoutingHTTPAdapter(requests.adapters.HTTPAdapter):
    """Sets a default timeout for all adapter (think session) requests. It is
    overridden with per-request timeout. But it can not be reset back to
    infinite wait (``None``).

    Usage:

        s = requests.Session()
        s.mount("http://", TimeoutingHTTPAdapter(timeout=5))
        s.mount("https://", TimeoutingHTTPAdapter(timeout=5))

        s.get('http://httpbin.org/delay/6')                 # -> timeouts after 5sec
        s.get('http://httpbin.org/delay/6', timeout=10)     # -> completes after 6sec

    The alternative is to set ``timeout`` on each request manually/explicitly,
    subclass ``Session``, or monkeypatch ``Session.request()``.
    """

    def __init__(self, timeout=None, *args, **kwargs):
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, *args, **kwargs):
        # can't use setdefault because caller always sets timeout kwarg
        kwargs['timeout'] = self.timeout
        return super().send(*args, **kwargs)


# Note: BaseUrlSession is taken from https://github.com/requests/toolbelt under
# an Apache 2 license. This simple extension didn't warrant a new dependency.
# If we later decide to use additional features from `requests-toolbelt`,
# remove it from here.

class BaseUrlSession(requests.Session):
    """A Session with a URL that all requests will use as a base."""

    base_url = None

    def __init__(self, base_url=None):
        if base_url:
            self.base_url = base_url
        super().__init__()

    def request(self, method, url, *args, **kwargs):
        """Send the request after generating the complete URL."""
        url = self.create_url(url)
        return super().request(method, url, *args, **kwargs)

    def create_url(self, url):
        """Create the URL based off this partial path."""
        return urljoin(self.base_url, url)


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


def user_agent(name=None, version=None):
    """Return User-Agent ~ "name/version language/version interpreter/version os/version"."""

    def _interpreter():
        name = platform.python_implementation()
        version = platform.python_version()
        bitness = platform.architecture()[0]
        if name == 'PyPy':
            version = '.'.join(map(str, sys.pypy_version_info[:3]))
        full_version = [version]
        if bitness:
            full_version.append(bitness)
        return name, "-".join(full_version)

    tags = []

    if name and version:
        tags.append((name, version))

    tags.extend([
        ("python", platform.python_version()),
        _interpreter(),
        ("machine", platform.machine() or 'unknown'),
        ("system", platform.system() or 'unknown'),
        ("platform", platform.platform() or 'unknown'),
    ])

    # add platform-specific tags
    tags.extend(get_platform_tags())

    return ' '.join("{}/{}".format(name, version) for name, version in tags)


class CLIError(Exception):
    """CLI command error that includes the error code in addition to the
    standard error message."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class cached:
    """Caching decorator with max-age/expiry, forced refresh, and
    per-arguments-combo keys.

    Example:
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

    """

    def argshash(self, args, kwargs):
        "Hash mutable arguments' containers with immutable keys and values."
        a = repr(args)
        b = repr(sorted((repr(k), repr(v)) for k, v in kwargs.items()))
        return a + b

    def __init__(self, maxage=None, cache=None):
        if maxage is None:
            maxage = 0
        self.maxage = maxage

        if cache is None:
            cache = {}
        self.cache = cache

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            refresh_ = kwargs.pop('refresh_', False)
            now = epochnow()

            key = self.argshash(args, kwargs)
            data = self.cache.get(key, {})

            logger.trace("cached: refresh=%r, store=%r key=%r, data=%r",
                         refresh_, self.cache, key, data)
            if not refresh_ and data.get('expires', 0) > now:
                val = data.get('val')
            else:
                val = fn(*args, **kwargs)
                self.cache[key] = dict(expires=now+self.maxage, val=val)

            return val

        # expose @cached internals for testing and debugging
        wrapper._cached = self

        return wrapper

    @classmethod
    def ondisk(cls, **kwargs):
        """@cached backed by an on-disk sqlite3-based cache."""
        from dwave.cloud.config import get_cache_dir
        directory = kwargs.pop('directory', get_cache_dir())
        # NOTE: use pickle v4 to support <py38
        # TODO: consider using `diskcache.JSONDisk` if we can serialize `api.models`
        cache = diskcache.Cache(directory=directory, disk_pickle_protocol=4)
        return cls(cache=cache, **kwargs)


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


class tictoc(object):
    """Timer as a context manager."""

    def __enter__(self):
        self.tick = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dt = time.perf_counter() - self.tick


class deprecated(object):
    """Decorator that issues a deprecation message on each call of the
    decorated function.
    """

    def __init__(self, msg=None):
        self.msg = msg

    def __call__(self, fn):
        if not callable(fn):
            raise TypeError("decorated object must be callable")

        @wraps(fn)
        def wrapped(*args, **kwargs):
            msg = self.msg
            if not msg:
                fn_name = getattr(fn, '__name__', 'unnamed')
                msg = "{}() has been deprecated".format(fn_name)
            warnings.warn(msg, DeprecationWarning)

            return fn(*args, **kwargs)

        return wrapped


def deprecated_option(msg=None, update=None):
    """Generate a Click callback function that will print a deprecation notice
    to stderr with a customized message and copy option value to new option.

    Note: if you provide the ``update`` option name, make sure that option is
    processed before the deprecated one (set ``is_eager``).

    Example::

        @click.option('--config-file', '-f', default=None, is_eager=True)
        @click.option(
            '-c', default=None, expose_value=False,
            help="[Deprecated in favor of '-f']",
            callback=deprecated_option(DEPRECATION_MSG, update='config_file'))
        ...
        def ping(config_file, ...):
            pass

    """
    # CLI util; defer click import until actually needed (see #473)
    import click

    def _print_deprecation(ctx, param, value, msg=None, update=None):
        if msg is None:
            msg = "DeprecationWarning: The following options are deprecated: {opts!r}."
        if value and not ctx.resilient_parsing:
            click.echo(click.style(msg.format(opts=param.opts), fg="red"), err=True)
            if update:
                ctx.params[update] = value

    # click seems to strip closure variables in calls to `callback`,
    # so we pass `msg` and `update` via partial application
    return partial(_print_deprecation, msg=msg, update=update)


def parse_loglevel(level_name, default=logging.NOTSET):
    """Resolve numeric and symbolic log level names to numeric levels."""

    try:
        level_name = str(level_name or '').strip().lower()
    except:
        return default

    # note: make sure `TRACE` level is added to `logging` before calling this
    known_levels = {
        'notset': logging.NOTSET,
        'trace': logging.TRACE,
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'warn': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
        'fatal': logging.CRITICAL
    }

    try:
        level = int(level_name)
    except ValueError:
        level = known_levels.get(level_name, default)

    return level


def set_loglevel(logger, level_name):
    level = parse_loglevel(level_name)
    logger.setLevel(level)
    logger.info("Log level for %r namespace set to %r", logger.name, level)


def get_contrib_config():
    """Return all registered contrib (non-open-source) Ocean packages."""

    contrib = [ep.load() for ep in iter_entry_points('dwave_contrib')]
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


def get_platform_tags():
    """Return a list of platform tags generated from registered entry points."""

    fs = [ep.load() for ep in iter_entry_points('dwave.common.platform.tags')]
    tags = list(filter(None, [f() for f in fs]))
    return tags


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

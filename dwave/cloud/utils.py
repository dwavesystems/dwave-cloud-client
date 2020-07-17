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
import time
import random
import logging
import platform
import itertools
import numbers

from collections import abc, OrderedDict
from urllib.parse import urljoin
from datetime import datetime
from dateutil.tz import UTC
from functools import wraps
from pkg_resources import iter_entry_points

import click
import requests

# Use numpy if available for fast decoding
try:
    import numpy as np
    _numpy = True
except ImportError:  # pragma: no cover
    _numpy = False

__all__ = ['evaluate_ising', 'uniform_iterator', 'uniform_get',
           'default_text_input', 'click_info_switch', 'datetime_to_timestamp',
           'datetime_to_timestamp', 'utcnow', 'epochnow', 'tictoc']

logger = logging.getLogger(__name__)


def evaluate_ising(linear, quad, state):
    """Calculate the energy of a state given the Hamiltonian.

    Args:
        linear: Linear Hamiltonian terms.
        quad: Quadratic Hamiltonian terms.
        state: Vector of spins describing the system state.

    Returns:
        Energy of the state evaluated by the given energy function.
    """

    # If we were given a numpy array cast to list
    if _numpy and isinstance(state, np.ndarray):
        return evaluate_ising(linear, quad, state.tolist())

    # Accumulate the linear and quadratic values
    energy = 0.0
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

    if isinstance(sequence, abc.Mapping):
        return sequence.items()
    else:
        return enumerate(sequence)


def uniform_get(sequence, index, default=None):
    """Uniform `dict`/`list` item getter, where `index` is interpreted as a key
    for maps and as numeric index for lists."""

    if isinstance(sequence, abc.Mapping):
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


def input_with_default(prompt, default, optional):
    line = ''
    while not line:
        line = input(prompt)
        if not line:
            line = default
        if not line:
            if optional:
                break
            click.echo("Input required, please try again.")
    return line


def default_text_input(prompt, default=None, optional=True):
    if default:
        prompt = "{} [{}]: ".format(prompt, default)
    else:
        if optional:
            prompt = "{} [skip]: ".format(prompt)
        else:
            prompt = "{}: ".format(prompt)

    return input_with_default(prompt, default, optional)


def click_info_switch(f):
    """Decorator to create eager Click info switch option, as described in:
    http://click.pocoo.org/6/options/#callbacks-and-eager-options.

    Takes a no-argument function and abstracts the boilerplate required by
    Click (value checking, exit on done).

    Example:

        @click.option('--my-option', is_flag=True, callback=my_option,
                    expose_value=False, is_eager=True)
        def test():
            pass

        @click_info_switch
        def my_option()
            click.echo('some info related to my switch')
    """

    @wraps(f)
    def wrapped(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        f()
        ctx.exit()
    return wrapped


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


def strtrunc(s, maxlen=60):
    s = str(s)
    return s[:(maxlen-3)]+'...' if len(s) > maxlen else s


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
        super(TimeoutingHTTPAdapter, self).__init__(*args, **kwargs)

    def send(self, *args, **kwargs):
        # can't use setdefault because caller always sets timeout kwarg
        kwargs['timeout'] = self.timeout
        return super(TimeoutingHTTPAdapter, self).send(*args, **kwargs)


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
        super(BaseUrlSession, self).__init__()

    def request(self, method, url, *args, **kwargs):
        """Send the request after generating the complete URL."""
        url = self.create_url(url)
        return super(BaseUrlSession, self).request(
            method, url, *args, **kwargs
        )

    def create_url(self, url):
        """Create the URL based off this partial path."""
        return urljoin(self.base_url, url)


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

    tags = [
        ("python", platform.python_version()),
        _interpreter(),
        ("machine", platform.machine() or 'unknown'),
        ("system", platform.system() or 'unknown'),
        ("platform", platform.platform() or 'unknown'),
    ]

    # add platform-specific tags
    tags.extend(get_platform_tags())

    return ' '.join("{}/{}".format(name, version) for name, version in tags)


class CLIError(Exception):
    """CLI command error that includes the error code in addition to the
    standard error message."""

    def __init__(self, message, code):
        super(CLIError, self).__init__(message)
        self.code = code


class cached(object):
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

    def __init__(self, maxage=None):
        self.maxage = maxage or 0
        self.cache = {}

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            refresh_ = kwargs.pop('refresh_', False)
            now = epochnow()

            key = self.argshash(args, kwargs)
            data = self.cache.get(key, {})

            if not refresh_ and data.get('expires', 0) > now:
                val = data.get('val')
            else:
                val = fn(*args, **kwargs)
                self.cache[key] = dict(expires=now+self.maxage, val=val)

            return val

        # expose the cache for testing and debugging
        wrapper._cache = self.cache
        wrapper._maxage = self.maxage

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
        elif isinstance(backoff, abc.Sequence):
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

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

from __future__ import division, absolute_import

from datetime import datetime
from dateutil.tz import UTC
from functools import wraps
import itertools
import random
import time

import six
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
           'datetime_to_timestamp', 'utcnow', 'epochnow']


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
    for (index_a, index_b), value in six.iteritems(quad):
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
    for edge, _ in six.iteritems(quadratic):
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


def uniform_iterator(sequence):
    """Uniform (key, value) iteration on a `dict`,
    or (idx, value) on a `list`."""

    if isinstance(sequence, dict):
        return six.iteritems(sequence)
    else:
        return enumerate(sequence)


def uniform_get(sequence, index, default=None):
    """Uniform `dict`/`list` item getter, where `index` is interpreted as a key
    for maps and as numeric index for lists."""

    if isinstance(sequence, dict):
        return sequence.get(index, default)
    else:
        return sequence[index] if index < len(sequence) else default


def strip_head(sequence, values):
    """Strips elements of `values` from the beginning of `sequence`."""
    values = set(values)
    return list(itertools.dropwhile(lambda x: x in values, sequence))


def strip_tail(sequence, values):
    """Strip `values` from the end of `sequence`."""
    return list(reversed(list(strip_head(reversed(sequence), values))))


def default_text_input(prompt, default=None, optional=True):
    if default:
        prompt = "{} [{}]: ".format(prompt, default)
    else:
        if optional:
            prompt = "{} [skip]: ".format(prompt)
        else:
            prompt = "{}: ".format(prompt)

    line = ''
    while not line:
        line = six.moves.input(prompt)
        if not line:
            line = default
        if not line:
            if optional:
                break
            click.echo("Input required, please try again.")
    return line


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

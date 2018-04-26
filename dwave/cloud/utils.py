from __future__ import division, absolute_import

from datetime import datetime
from dateutil.tz import UTC
from functools import wraps

import six
import readline

# Use numpy if available for fast decoding
try:
    import numpy as np
    _numpy = True
except ImportError:  # pragma: no cover
    _numpy = False

__all__ = ['evaluate_ising', 'uniform_iterator', 'uniform_get',
           'readline_input', 'click_info_switch', 'datetime_to_timestamp']


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


def readline_input(prompt, prefill=''):
    """Provide an editable default for ``input()``."""
    # see: https://stackoverflow.com/q/2533120/
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return six.moves.input(prompt)
    finally:
        readline.set_startup_hook()


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

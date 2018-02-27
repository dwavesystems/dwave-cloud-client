from __future__ import division, absolute_import

import six

# Use numpy if available for fast decoding
try:
    import numpy as np
    _numpy = True
except ImportError:
    # If numpy isn't available we can do the encoding slower in native python
    _numpy = False


def _evaluate_ising(linear, quad, state):
    """Calculate the energy of a state given the Hamiltonian.

    This is used to debug energy decoding.

    Args:
        linear: Linear Hamiltonian terms.
        quad: Quadratic Hamiltonian terms.
        state: Vector of spins describing the system state.

    Returns:
        Energy of the state evaluated by the given energy function.
    """
    # If we were given a numpy array cast to list
    if _numpy and isinstance(state, np.ndarray):
        return _evaluate_ising(linear, quad, state.tolist())

    # Accumulate the linear and quadratic values
    energy = 0.0
    for index, value in _uniform_iterator(linear):
        energy += state[index] * value
    for (index_a, index_b), value in six.iteritems(quad):
        energy += value * state[index_a] * state[index_b]
    return energy


def _uniform_iterator(sequence):
    """Key, value iteration on a dict or list."""
    if isinstance(sequence, dict):
        return six.iteritems(sequence)
    else:
        return enumerate(sequence)


def _uniform_get(sequence, index, default=None):
    """Get by key with default value for dict or list."""
    if isinstance(sequence, dict):
        return sequence.get(index, default)
    else:
        return sequence[index] if index < len(sequence) else default


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

"""Old QUBO/Ising utilities for private and Ocean-internal use."""

import random
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

__all__ = []


def uniform_iterator(sequence: Union[Mapping, Sequence]
                     ) -> Iterator[Tuple[int, Any]]:
    """Uniform (key, value) iteration on a `dict`,
    or (idx, value) on a `list`."""

    if isinstance(sequence, Mapping):
        return sequence.items()
    else:
        return enumerate(sequence)


def uniform_get(sequence: Union[Mapping, Sequence],
                index: int,
                default: Optional[Any] = None
                ) -> Any:
    """Uniform `dict`/`list` item getter, where `index` is interpreted as a key
    for maps and as numeric index for lists."""

    if isinstance(sequence, Mapping):
        return sequence.get(index, default)
    else:
        return sequence[index] if index < len(sequence) else default


def evaluate_ising(linear: Union[Mapping, Sequence],
                   quad: Mapping,
                   state: Sequence,
                   offset: float = 0
                   ) -> float:
    """Calculate the energy of a state given the Hamiltonian.

    Args:
        linear: Linear Hamiltonian terms.
        quad: Quadratic Hamiltonian terms.
        offset: Energy offset.
        state: Vector of spins describing the system state

    Returns:
        Energy of the state evaluated by the given energy function.
    """

    # note: we avoid numpy import by tolist() check
    if hasattr(state, 'tolist') and callable(state.tolist):
        return evaluate_ising(linear, quad, state.tolist(), offset=offset)

    # Accumulate the linear and quadratic values
    energy = offset
    for index, value in uniform_iterator(linear):
        energy += state[index] * value
    for (index_a, index_b), value in quad.items():
        energy += value * state[index_a] * state[index_b]
    return energy


def active_qubits(linear: Union[Mapping, Sequence],
                  quadratic: Mapping
                  ) -> Set:
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

    active = {idx for idx, bias in uniform_iterator(linear)}
    for edge, _ in quadratic.items():
        active.update(edge)
    return active


def generate_random_ising_problem(solver: 'dwave.cloud.solver.Solver',
                                  h_range: Optional[List[float]] = None,
                                  j_range: Optional[List[float]] = None
                                  ) -> Tuple[dict]:
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


def generate_const_ising_problem(solver: 'dwave.cloud.solver.Solver',
                                 h: float = 1,
                                 j: float = -1
                                 ) -> Tuple[dict]:

    return generate_random_ising_problem(solver, h_range=[h, h], j_range=[j, j])


def reformat_qubo_as_ising(qubo: Dict[Tuple[int, int], float]) -> Tuple[dict]:
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

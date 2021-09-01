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

import uuid
import itertools
import random
from typing import List, Tuple

__all__ = [
    'solver_configuration_data',
    'structured_solver_data', 'qpu_clique_solver_data',
    'unstructured_solver_data', 'hybrid_bqm_solver_data', 'hybrid_dqm_solver_data',
]


def solver_configuration_data(id: str = None,
                              status: str = None,
                              description: str = None,
                              properties: dict = None,
                              avg_load: float = None) -> dict:
    """Return data dict describing a solver, as returned by SAPI."""

    if id is None:
        id = str(uuid.uuid4())

    if status is None:
        status = "ONLINE"

    if description is None:
        description = "A mock solver"

    if properties is None:
        properties = {}

    if avg_load is None:
        avg_load = random.random()

    return {
        "id": id,
        "status": status,
        "description": description,
        "properties": properties,
        "avg_load": avg_load,
    }


def structured_solver_data(id: str = None,
                           status: str = None,
                           avg_load: float = None,
                           description: str = None,
                           properties: dict = None,
                           # specific properties for convenience, override `properties` dict
                           qubits: List[int] = None,
                           couplers: List[List[int]] = None,
                           **kwargs) -> dict:
    """Return data dict describing a structured solver. By default, solver has
    only one qubit.
    """

    if properties is None:
        # TODO: add more default properties?
        properties = {
            "category": "qpu",
            "tags": ["lower_noise"],
            "topology": {"type": "clique"},
            "supported_problem_types": ["qubo", "ising"],
            "h_range": [-2.0, 2.0],
            "j_range": [-1.0, 1.0],
            "num_reads_range": [1, 10000],
            "annealing_time_range": [1.0, 2000.0],
            "extended_j_range": [-2.0, 1.0],
            "quota_conversion_rate": 1,
            "parameters": {
                "anneal_offsets": "A list of anneal offsets for each working qubit.",
                "anneal_schedule": "A piecewise linear annealing schedule.",
                "label": "Problem label.",
                "num_reads": "Number of samples to return."
            }
        }

    if qubits is None:
        qubits = [0]

    if couplers is None:
        couplers = []

    properties.update({
        "qubits": qubits,
        "couplers": couplers,
        "num_qubits": len(qubits)
    })

    properties.update(kwargs)

    return solver_configuration_data(id=id,
                                     status=status,
                                     description=description,
                                     properties=properties,
                                     avg_load=avg_load)


def qpu_clique_solver_data(size: int, **kwargs) -> dict:
    """Mock QPU solver data with a clique of custom size as topology.

    Args:
        size:
            Clique size. Qubits are sequential integers in range [0, size).
        **kwargs:
            Solver properties passed down to :meth:`.structured_solver_data`.
    """

    qubits = list(range(size))
    couplers = list(itertools.combinations(range(len(qubits)), 2))

    params = dict(
        id="dw_{}q_mock".format(size),
        description="A {}-qubit clique mock QPU solver".format(size),
        qubits=qubits,
        couplers=couplers,
        topology={"type": "clique"},
    )
    params.update(**kwargs)

    return structured_solver_data(**params)


def unstructured_solver_data(id: str = None,
                             status: str = None,
                             avg_load: float = None,
                             description: str = None,
                             properties: dict = None,
                             # specific properties for convenience, override `properties` dict
                             supported_problem_types: List[str] = None,
                             **kwargs) -> dict:
    """Return data dict describing an unstructured solver."""

    if properties is None:
        properties = {
            "category": "hybrid",
            "maximum_number_of_variables": 65536,
            "maximum_time_limit_hrs": 2.0,
            "minimum_time_limit": [
                [1, 1.0], [1024, 2.0], [4096, 4.0], [8192, 16.0]
            ],
            "quota_conversion_rate": 20,
            "parameters": {
                "time_limit": "Hybrid solver execution time limit."
            }
        }

    if supported_problem_types is None:
        supported_problem_types = ["bqm"]

    properties.update({
        "supported_problem_types": supported_problem_types
    })

    properties.update(kwargs)

    return solver_configuration_data(id=id,
                                     status=status,
                                     description=description,
                                     properties=properties,
                                     avg_load=avg_load)


def hybrid_bqm_solver_data(**kwargs) -> dict:
    params = dict(
        id="hybrid_bqm_solver",
        description="Hybrid unstructured BQM mock solver",
        supported_problem_types=["bqm"]
    )
    params.update(**kwargs)
    return unstructured_solver_data(**params)


def hybrid_dqm_solver_data(**kwargs) -> dict:
    params = dict(
        id="hybrid_dqm_solver",
        description="Hybrid unstructured DQM mock solver",
        supported_problem_types=["dqm"]
    )
    params.update(**kwargs)
    return unstructured_solver_data(**params)

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
from typing import Optional

__all__ = [
    'solver_configuration_data',
    'structured_solver_data', 'qpu_clique_solver_data',
    'qpu_chimera_solver_data', 'qpu_pegasus_solver_data', 'qpu_zephyr_solver_data',
    'unstructured_solver_data', 'hybrid_bqm_solver_data', 'hybrid_dqm_solver_data',
    'hybrid_cqm_solver_data', 'hybrid_nl_solver_data', 'qpu_problem_timing_data',
]


def uuid_name():
    return str(uuid.uuid4())


def solver_configuration_data(identity: dict = None,
                              status: str = None,
                              description: str = None,
                              properties: dict = None,
                              avg_load: float = None) -> dict:
    """Return data dict describing a solver, as returned by SAPI."""

    if identity is None:
        identity = dict(name=uuid_name())

    if status is None:
        status = "ONLINE"

    if description is None:
        description = "A mock solver"

    if properties is None:
        properties = {}

    if avg_load is None:
        avg_load = random.random()

    return {
        "identity": identity,
        "status": status,
        "description": description,
        "properties": properties,
        "avg_load": avg_load,
    }


def structured_solver_data(name: str = None,
                           graph_id: str = None,
                           status: str = None,
                           avg_load: float = None,
                           description: str = None,
                           properties: dict = None,
                           # specific properties for convenience, override `properties` dict
                           qubits: list[int] = None,
                           couplers: list[list[int]] = None,
                           **kwargs) -> dict:
    """Return data dict describing a structured solver. By default, solver has
    only one qubit.
    """

    if name is None:
        name = uuid_name()

    if graph_id is None:
        graph_id = uuid_name()[:10]

    identity = dict(name=name, version=dict(graph_id=graph_id))

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

    return solver_configuration_data(identity=identity,
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
        name=f"clique_{size}q_mock",
        graph_id=f"clique_{size}",
        description=f"A {size}-qubit mock QPU solver with clique topology",
        qubits=qubits,
        couplers=couplers,
        topology={"type": "clique", "shape": [size]},
    )
    params.update(**kwargs)

    return structured_solver_data(**params)


def qpu_chimera_solver_data(m: int,
                            n: Optional[int] = None,
                            t: Optional[int] = None,
                            **kwargs) -> dict:
    """Mock QPU solver data with a custom-sized Chimera topology.

    Args:
        m:
            Number of rows in the Chimera lattice. See
            :func:`~dwave_networkx.generators.chimera_graph` for details.
        n:
            Number of columns in the Chimera lattice.
        t:
            Size of the shore within each Chimera tile.
        **kwargs:
            Solver properties passed down to :meth:`.structured_solver_data`.
    """
    try:
        import dwave_networkx as dnx
    except ImportError:     # pragma: no cover
        raise RuntimeError("Can't generate Chimera graph without dwave-networkx. "
                           "Install with 'dwave-cloud-client[mocks]'.")

    graph = dnx.chimera_graph(m, n, t)
    # we need generated graph's values, so we can set topology.shape
    m = graph.graph['rows']
    n = graph.graph['columns']
    t = graph.graph['tile']
    qubits = sorted(list(graph.nodes))
    couplers = sorted(map(sorted, graph.edges))
    num_qubits = len(qubits)

    params = dict(
        name=f"chimera_{num_qubits}q_mock",
        graph_id=f"chimera_{num_qubits}",
        description=f"A {num_qubits}-qubit mock QPU solver with chimera topology",
        qubits=qubits,
        couplers=couplers,
        topology={"type": "chimera", "shape": [m, n, t]},
    )
    params.update(**kwargs)

    return structured_solver_data(**params)


def qpu_pegasus_solver_data(m: int,
                            fabric_only: bool = True,
                            **kwargs) -> dict:
    """Mock QPU solver data with a custom-sized Pegasus topology.

    Args:
        m:
            Size parameter for the Pegasus lattice.
        fabric_only:
            Use only nodes from the largest Pegasus graph component. See
            :func:`~dwave_networkx.generators.pegasus_graph` for details.
        **kwargs:
            Solver properties passed down to :meth:`.structured_solver_data`.

    Note:
        By default, with ``fabric_only=True``, only a subset of Pegasus graph
        is used (fabric qubits only), hence num_active_qubits will be less than
        num_qubits.
    """
    try:
        import dwave_networkx as dnx
    except ImportError:     # pragma: no cover
        raise RuntimeError("Can't generate Pegasus graph without dwave-networkx. "
                           "Install with 'dwave-cloud-client[mocks]'.")

    graph = dnx.pegasus_graph(m, fabric_only=fabric_only)
    qubits = sorted(list(graph.nodes))
    couplers = sorted(map(sorted, graph.edges))
    num_qubits = 24 * m * (m-1)     # includes non-fabric qubits

    params = dict(
        name=f"pegasus_{num_qubits}q_mock",
        graph_id=f"pegasus_{num_qubits}",
        description=f"A {num_qubits}-qubit mock QPU solver with pegasus topology",
        qubits=qubits,
        couplers=couplers,
        topology={"type": "pegasus", "shape": [m]},
        num_qubits=num_qubits,
    )
    params.update(**kwargs)

    return structured_solver_data(**params)


def qpu_zephyr_solver_data(m: int,
                           t: int = 4,
                           **kwargs) -> dict:
    """Mock QPU solver data with a custom-sized Zephyr topology.

    Args:
        m:
            Grid parameter for the Zephyr lattice.
        t:
            Tile parameter for the Zephyr lattice.
        **kwargs:
            Solver properties passed down to :meth:`.structured_solver_data`.
    """
    try:
        import dwave_networkx as dnx
    except ImportError:     # pragma: no cover
        raise RuntimeError("Can't generate Zephyr graph without dwave-networkx. "
                           "Install with 'dwave-cloud-client[mocks]'.")

    graph = dnx.zephyr_graph(m, t)
    qubits = sorted(list(graph.nodes))
    couplers = sorted(map(sorted, graph.edges))
    num_qubits = len(qubits)

    params = dict(
        name=f"zephyr_{num_qubits}q_mock",
        graph_id=f"zephyr_{num_qubits}",
        description=f"A {num_qubits}-qubit mock QPU solver with zephyr topology",
        qubits=qubits,
        couplers=couplers,
        topology={"type": "zephyr", "shape": [m, t]},
        num_qubits=num_qubits,
    )
    params.update(**kwargs)

    return structured_solver_data(**params)


def qpu_problem_timing_data(qpu: str = 'advantage') -> dict:
    """Mock QPU solver proprty problem_timing_data.

    Args:
        qpu:
            Type of QPU that provided timing data, as it was in
            August 2022 on a particular system, used here. Currently supported
            values are: ``advantage`` and ``2000q``.

    """

    timing_data_advantage41_2022_8 = {'version': '1.0.0',
        'typical_programming_time': 14072.88,
        'reverse_annealing_with_reinit_prog_time_delta': 0.0,
        'reverse_annealing_without_reinit_prog_time_delta': 5.55,
        'default_programming_thermalization': 1000.0,
        'default_annealing_time': 20.0,
        'readout_time_model': 'pwl_log_log',
        'readout_time_model_parameters': [0.0, 0.7699665876947938, 1.7242758696010096,
                2.711975459489206, 3.1639057672764026, 3.750276915153992, 1.539131714800995,
                1.8726623164229292, 2.125631787097315, 2.332672340068556, 2.371606651233025,
                2.3716219271760215],
        'qpu_delay_time_per_sample': 20.54,
        'reverse_annealing_with_reinit_delay_time_delta': -4.5,
        'reverse_annealing_without_reinit_delay_time_delta': -1.5,
        'default_readout_thermalization': 0.0,
        'decorrelation_max_nominal_anneal_time': 2000.0,
        'decorrelation_time_range': [500.0, 10000.0]}

    timing_data_2000q6_2022_8 = {'version': '1.0.0',
        'typical_programming_time': 10536.81,
        'reverse_annealing_with_reinit_prog_time_delta': 0.0,
        'reverse_annealing_without_reinit_prog_time_delta': 359.58,
        'default_programming_thermalization': 1000.0,
        'default_annealing_time': 20.0,
        'readout_time_model': 'pwl_log_log',
        'readout_time_model_parameters': [0.0, 3.30984300471607, 2.2975416678181597,
            2.2975416678181597],
        'qpu_delay_time_per_sample': 20.54,
        'reverse_annealing_with_reinit_delay_time_delta': 586.38,
        'reverse_annealing_without_reinit_delay_time_delta': -5.0,
        'default_readout_thermalization': 0.0,
        'decorrelation_max_nominal_anneal_time': 2000.0,
        'decorrelation_time_range': [500.0, 10000.0]}

    name_dict = {'advantage': timing_data_advantage41_2022_8,
                 '2000q': timing_data_2000q6_2022_8}

    return name_dict[qpu]


def unstructured_solver_data(name: str = None,
                             status: str = None,
                             avg_load: float = None,
                             description: str = None,
                             properties: dict = None,
                             # specific properties for convenience, override `properties` dict
                             supported_problem_types: list[str] = None,
                             **kwargs) -> dict:
    """Return data dict describing an unstructured solver."""

    if name is None:
        name = uuid_name()

    if properties is None:
        properties = {
            "category": "hybrid",
            "quota_conversion_rate": 20,
            "parameters": {
                "time_limit": "Hybrid solver execution time limit."
            },
            "version": "1.0",
        }

    if supported_problem_types is None:
        supported_problem_types = ["bqm"]

    properties.update({
        "supported_problem_types": supported_problem_types
    })

    properties.update(kwargs)

    return solver_configuration_data(identity=dict(name=name),
                                     status=status,
                                     description=description,
                                     properties=properties,
                                     avg_load=avg_load)


def hybrid_bqm_solver_data(**kwargs) -> dict:
    params = dict(
        name="hybrid_bqm_solver",
        description="Hybrid unstructured BQM mock solver",
        supported_problem_types=["bqm"],
        # solver-specific properties (mock values)
        maximum_number_of_biases=2**20,
        maximum_number_of_variables=2**16,
        maximum_time_limit_hrs=1.0,
        minimum_time_limit=[[1, 1.0], [1024, 2.0], [4096, 4.0], [8192, 16.0]],
    )
    params.update(**kwargs)
    return unstructured_solver_data(**params)


def hybrid_cqm_solver_data(**kwargs) -> dict:
    params = dict(
        name="hybrid_cqm_solver",
        description="Hybrid unstructured CQM mock solver",
        supported_problem_types=["cqm"],
        # solver-specific properties (mock values)
        maximum_number_of_biases=2**20,
        maximum_number_of_constraints=2**10,
        maximum_number_of_linear_biases_real=2**20,
        maximum_number_of_quadratic_variables=2**20,
        maximum_number_of_quadratic_variables_real=0,
        maximum_number_of_variables=2**16,
        maximum_time_limit_hrs=1.0,
        minimum_time_limit_s=1,
        num_biases_multiplier=1e-06,
        num_constraints_multiplier=1e-09,
        num_variables_multiplier=1e-4,
    )
    params.update(**kwargs)
    return unstructured_solver_data(**params)


def hybrid_dqm_solver_data(**kwargs) -> dict:
    params = dict(
        name="hybrid_dqm_solver",
        description="Hybrid unstructured DQM mock solver",
        supported_problem_types=["dqm"],
        # solver-specific properties (mock values)
        maximum_number_of_biases=2**20,
        maximum_number_of_cases=2**12,
        maximum_number_of_variables=2**10,
        maximum_time_limit_hrs=1.0,
        minimum_time_limit=[[1, 1.0], [1024, 2.0], [4096, 4.0], [8192, 16.0]],
    )
    params.update(**kwargs)
    return unstructured_solver_data(**params)


def hybrid_nl_solver_data(**kwargs) -> dict:
    params = dict(
        name="hybrid_nl_solver",
        description="Hybrid unstructured NL mock solver",
        supported_problem_types=["nl"],
        # solver-specific properties (mock values)
        maximum_decision_state_size=2**20,
        maximum_number_of_nodes=2**16,
        maximum_number_of_states=1,
        maximum_state_size=2**20,
        maximum_time_limit_hrs=1.0,
        num_nodes_multiplier=1e-05,
        num_nodes_state_size_multiplier=1e-12,
        state_size_multiplier=1e-10,
        time_constant=1e-02,
    )
    params.update(**kwargs)
    return unstructured_solver_data(**params)

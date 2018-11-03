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

"""
A :term:`solver` is a resource for solving problems.

Solvers are responsible for:

    - Encoding submitted problems
    - Checking submitted parameters
    - Adding problems to a client's submission queue

You can list all solvers available to a :class:`~dwave.cloud.client.Client` with its
:func:`~dwave.cloud.client.Client.get_solvers` method and select and return one with its
:func:`~dwave.cloud.client.Client.get_solver` method.

"""

from __future__ import division, absolute_import

import json
import logging

from collections import Mapping

from dwave.cloud.exceptions import *
from dwave.cloud.coders import encode_bqm_as_qp
from dwave.cloud.utils import uniform_iterator, uniform_get
from dwave.cloud.computation import Future

__all__ = ['Solver']

_LOGGER = logging.getLogger(__name__)


class Solver(object):
    """
    Class for D-Wave solvers.

    This class provides :term:`Ising` and :term:`QUBO` sampling methods and encapsulates
    the solver description returned from the D-Wave cloud API.

    Args:
        client (:class:`Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.

    Examples:
        This example creates a client using the local system's default D-Wave Cloud
        Client configuration file and checks the identity of its default solver.

        >>> from dwave.cloud import Client
        >>> client = Client.from_config()
        >>> solver = client.get_solver()
        >>> solver.data['id']    # doctest: +SKIP
        u'EXAMPLE_2000Q_SYSTEM'

    """

    # Classes of problems the remote solver has to support (at least one of these)
    # in order for `Solver` to be able to abstract, or use, that solver
    _HANDLED_PROBLEM_TYPES = {"ising", "qubo"}

    def __init__(self, client, data):
        # client handles async api requests (via local thread pool)
        self.client = client

        # data for each solver includes at least: id, description, and properties
        self.data = data

        # Each solver has an ID field
        try:
            self.id = data['id']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'id'")

        # Properties of this solver the server presents: dict
        try:
            self.properties = data['properties']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'properties'")

        # Ensure this remote solver supports at least one of the problem types we know how to handle
        try:
            self.supported_problem_types = set(self.properties['supported_problem_types'])
        except KeyError:
            raise InvalidAPIResponseError(
                "Missing solver property: 'properties.supported_problem_types'")

        if self.supported_problem_types.isdisjoint(self._HANDLED_PROBLEM_TYPES):
            raise UnsupportedSolverError(
                "Remote solver {!r} supports {} problems, but Solver() handles only {}".format(
                    self.id,
                    list(self.supported_problem_types),
                    list(self._HANDLED_PROBLEM_TYPES)))

        # The set of extra parameters this solver will accept in sample_ising or sample_qubo: dict
        try:
            self.parameters = self.properties['parameters']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'parameters'")

        # When True the solution data will be returned as numpy matrices: False
        self.return_matrix = False

        # The exact sequence of nodes/edges is used in encoding problems and must be preserved
        try:
            self._encoding_qubits = self.properties['qubits']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'properties.qubits'")

        try:
            self._encoding_couplers = [tuple(edge) for edge in self.properties['couplers']]
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'properties.couplers'")

        # The nodes in this solver's graph: set(int)
        self.nodes = self.variables = set(self._encoding_qubits)

        # The edges in this solver's graph, every edge will be present as (a, b) and (b, a): set(tuple(int, int))
        self.edges = self.couplers = set(tuple(edge) for edge in self._encoding_couplers) | \
            set((edge[1], edge[0]) for edge in self._encoding_couplers)

        # The edges in this solver's graph, each edge will only be represented once: set(tuple(int, int))
        self.undirected_edges = {edge for edge in self.edges if edge[0] < edge[1]}

        # Create a set of default parameters for the queries
        self._params = {}

    def __repr__(self):
        return "Solver(id={!r})".format(self.id)

    @property
    def is_qpu(self):
        "Is this a QPU-based solver?"
        # TODO: add a field for this in SAPI response; for now base decision on id/name
        return not self.id.startswith('c4-sw_')

    @property
    def is_software(self):
        "Is this a software-based solver?"
        # TODO: add a field for this in SAPI response; for now base decision on id/name
        return self.id.startswith('c4-sw_')

    @property
    def is_vfyc(self):
        "Is this a virtual full-yield chip?"
        return self.properties.get('vfyc') == True

    @property
    def is_online(self):
        "Is this solver online (or offline)?"
        return self.data.get('status', 'online').lower() == 'online'

    @property
    def num_active_qubits(self):
        "The number of active (encoding) qubits."
        return len(self.nodes)

    @property
    def num_qubits(self):
        "Nominal number of qubits on chip (includes active AND inactive)."
        return self.properties.get('num_qubits')

    @property
    def has_flux_biases(self):
        "Solver supports/accepts ``flux_biases``."
        return 'flux_biases' in self.parameters

    @property
    def has_anneal_schedule(self):
        "Solver supports/accepts ``anneal_schedule``."
        return 'anneal_schedule' in self.parameters

    def sample_ising(self, linear, quadratic, **params):
        """Sample from the specified Ising model.

        Args:
            linear (list/dict): Linear terms of the model (h).
            quadratic (dict of (int, int):float): Quadratic terms of the model (J).
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj:`Future`

        Examples:
            This example creates a client using the local system's default D-Wave Cloud Client
            configuration file, which is configured to access a D-Wave 2000Q QPU, submits a
            simple :term:`Ising` problem (opposite linear biases on two coupled qubits), and samples
            5 times.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     computation = solver.sample_ising({u: -1, v: 1},{}, num_reads=5)   # doctest: +SKIP
            ...     for i in range(5):
            ...         print(computation.samples[i][u], computation.samples[i][v])
            ...
            ...
            (1, -1)
            (1, -1)
            (1, -1)
            (1, -1)
            (1, -1)

        """
        # Our linear and quadratic objective terms are already separated in an
        # ising model so we can just directly call `_sample`.
        return self._sample('ising', linear, quadratic, params)

    def sample_qubo(self, qubo, **params):
        """Sample from the specified QUBO.

        Args:
            qubo (dict of (int, int):float): Coefficients of a quadratic unconstrained binary
                optimization (QUBO) model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj:`Future`

        Examples:
            This example creates a client using the local system's default D-Wave Cloud Client
            configuration file, which is configured to access a D-Wave 2000Q QPU, submits
            a :term:`QUBO` problem (a Boolean NOT gate represented by a penalty model), and
            samples 5 times.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            ...     computation = solver.sample_qubo(Q, num_reads=5)
            ...     for i in range(5):
            ...         print(computation.samples[i][u], computation.samples[i][v])
            ...
            ...
            (0, 1)
            (1, 0)
            (1, 0)
            (0, 1)
            (1, 0)

        """
        # In a QUBO the linear and quadratic terms in the objective are mixed into
        # a matrix. For the sake of encoding, we will separate them before calling `_sample`
        linear = {i1: v for (i1, i2), v in uniform_iterator(qubo) if i1 == i2}
        quadratic = {(i1, i2): v for (i1, i2), v in uniform_iterator(qubo) if i1 != i2}
        return self._sample('qubo', linear, quadratic, params)

    def _sample(self, type_, linear, quadratic, params):
        """Internal method for both sample_ising and sample_qubo.

        Args:
            linear (list/dict): Linear terms of the model.
            quadratic (dict of (int, int):float): Quadratic terms of the model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj: `Future`
        """
        # Check the problem
        if not self.check_problem(linear, quadratic):
            raise ValueError("Problem graph incompatible with solver.")

        # Mix the new parameters with the default parameters
        combined_params = dict(self._params)
        combined_params.update(params)

        # Check the parameters before submitting
        for key in combined_params:
            if key not in self.parameters and not key.startswith('x_'):
                raise KeyError("{} is not a parameter of this solver.".format(key))

        # transform some of the parameters in-place
        self._format_params(type_, combined_params)

        body = json.dumps({
            'solver': self.id,
            'data': encode_bqm_as_qp(self, linear, quadratic),
            'type': type_,
            'params': combined_params
        })
        _LOGGER.trace("Encoded sample request: %s", body)

        future = Future(solver=self, id_=None, return_matrix=self.return_matrix,
                        submission_data=(type_, linear, quadratic, params))

        _LOGGER.debug("Submitting new problem to: %s", self.id)
        self.client._submit(body, future)
        return future

    def _format_params(self, type_, params):
        """Reformat some of the parameters for sapi."""
        if 'initial_state' in params:
            # NB: at this moment the error raised when initial_state does not match lin/quad (in
            # active qubits) is not very informative, but there is also no clean way to check here
            # that they match because lin can be either a list or a dict. In the future it would be
            # good to check.
            initial_state = params['initial_state']
            if isinstance(initial_state, Mapping):
                if type_ == 'ising' and any(v == 0 for v in initial_state.values()):
                    # initial_state is in qubo format, coerce to Ising
                    initial_state = [2*initial_state.get(v, 2)-1 for v in range(self.properties['num_qubits'])]
                elif type_ == 'qubo' and any(v == -1 for v in initial_state.values()):
                    # initial_state is in ising format, coerce to QUBO
                    initial_state = [(initial_state.get(v, 5)+1)//2 for v in range(self.properties['num_qubits'])]
                else:
                    initial_state = [initial_state.get(v, 3) for v in range(self.properties['num_qubits'])]

                params['initial_state'] = initial_state
            # else: support old format

    def check_problem(self, linear, quadratic):
        """Test if an Ising model matches the graph provided by the solver.

        Args:
            linear (list/dict): Linear terms of the model (h).
            quadratic (dict of (int, int):float): Quadratic terms of the model (J).

        Returns:
            boolean

        Examples:
            This example creates a client using the local system's default D-Wave Cloud Client
            configuration file, which is configured to access a D-Wave 2000Q QPU, and
            tests a simple :term:`Ising` model for two target embeddings (that is, representations
            of the model's graph by coupled qubits on the QPU's sparsely connected graph),
            where only the second is valid.

            >>> from dwave.cloud import Client
            >>> print((0, 1) in solver.edges)   # doctest: +SKIP
            False
            >>> print((0, 4) in solver.edges)   # doctest: +SKIP
            True
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     print(solver.check_problem({0: -1, 1: 1},{(0, 1):0.5}))
            ...     print(solver.check_problem({0: -1, 4: 1},{(0, 4):0.5}))
            ...
            False
            True
        """
        for key, value in uniform_iterator(linear):
            if value != 0 and key not in self.nodes:
                return False
        for key, value in uniform_iterator(quadratic):
            if value != 0 and tuple(key) not in self.edges:
                return False
        return True

    def _retrieve_problem(self, id_):
        """Resume polling for a problem previously submitted.

        Args:
            id_: Identification of the query.

        Returns:
            :obj: `Future`
        """
        future = Future(self, id_, self.return_matrix, None)
        self.client._poll(future)
        return future

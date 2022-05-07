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
An implementation of the REST client for D-Wave Solver API (SAPI) service.

SAPI servers provide authentication, queuing, and scheduling services,
and provide a network interface to :term:`solver`\ s. This API enables you submit
a binary quadratic (:term:`Ising` or :term:`QUBO`) model
and receive samples from a distribution over the model as defined by a
selected solver.

SAPI server workflow is roughly as follows:

 1. Submitted problems enter an input queue. Each user has an input queue per solver.
 2. Drawing from all input queues for a solver, problems are scheduled.
 3. Results are cached for retrieval by the client.

"""

from dwave.cloud.client import Client as BaseClient
from dwave.cloud.solver import StructuredSolver as Solver

__all__ = ['Client']


class Client(BaseClient):
    """D-Wave Solver API client specialized to work only with QPU solvers.

    This class can be instantiated explicitly, or via (base) Client's factory
    method, :meth:`~dwave.cloud.client.Client.from_config` by supplying
    ``"qpu"`` for ``client``.

    Examples:
        This example explicitly instantiates a :class:`dwave.cloud.qpu.Client`.
        :meth:`~dwave.cloud.client.Client.get_solver` is guaranteed to return a
        QPU solver.

        .. code-block:: python

            from dwave.cloud.qpu import Client

            with Client(token='...') as client:
                solver = client.get_solver()
                response = solver.sample_ising(...)

        The following example instantiates a QPU client indirectly. Again,
        :meth:`~dwave.cloud.client.Client.get_solver`/
        :meth:`~dwave.cloud.client.Client.get_solvers` are guaranteed to return
        only QPU solver(s).

        .. code-block:: python

            from dwave.cloud import Client

            with Client.from_config(client='qpu') as client:
                solver = client.get_solver()
                response = solver.sample_ising(...)

    """

    @staticmethod
    def is_solver_handled(solver):
        """Determine if the specified solver should be handled by this client.

        This predicate function (used from the base class) allows only remote
        QPU solvers.
        """
        return solver and solver.qpu

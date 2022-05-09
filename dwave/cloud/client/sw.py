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
Interface to software :term:`sampler`\ s available through the D-Wave Solver
API (SAPI).

Software samplers have the same interface (response) as QPU samplers, with
classical software resources generating samples.
"""

from dwave.cloud.client import Client as BaseClient
from dwave.cloud.solver import StructuredSolver as Solver

__all__ = ['Client']


class Client(BaseClient):
    """D-Wave Solver API client specialized to work only with remote software
    solvers.

    This class can be instantiated explicitly, or via (base) Client's factory
    method, :meth:`~dwave.cloud.client.Client.from_config` by supplying ``"sw"``
    for ``client``.

    Examples:
        This example explicitly instantiates a :class:`dwave.cloud.sw.Client`.
        :meth:`~dwave.cloud.client.Client.get_solver` is guaranteed to return a
        software solver.

        .. code-block:: python

            from dwave.cloud.sw import Client

            with Client(token='...') as client:
                solver = client.get_solver()
                response = solver.sample_ising(...)

        The following example instantiates a software-solver-only client
        indirectly. Again, :meth:`~dwave.cloud.client.Client.get_solver`/
        :meth:`~dwave.cloud.client.Client.get_solvers` are guaranteed to return
        only software solver(s).

        .. code-block:: python

            from dwave.cloud import Client

            with Client.from_config(client='sw') as client:
                solver = client.get_solver()
                response = solver.sample_ising(...)

    """

    @staticmethod
    def is_solver_handled(solver):
        """Determine if the specified solver should be handled by this client.

        This predicate function (used from the base class) allows only remote
        software solvers.
        """
        return solver and solver.software

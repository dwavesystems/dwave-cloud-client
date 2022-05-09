# Copyright 2020 D-Wave Systems Inc.
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
Interface to hybrid :term:`sampler`\ s available through the D-Wave Solver API
(SAPI).
"""

from dwave.cloud.client import Client as BaseClient
from dwave.cloud.solver import UnstructuredSolver as Solver

__all__ = ['Client']


class Client(BaseClient):
    """D-Wave Solver API client specialized to work only with remote hybrid
    quantum-classical solvers.

    This class can be instantiated explicitly, or via (base) Client's factory
    method, :meth:`~dwave.cloud.client.Client.from_config` by supplying
    ``"hybrid"`` for ``client``.

    Examples:
        This example explicitly instantiates a :class:`dwave.cloud.hybrid.Client`.
        :meth:`~dwave.cloud.client.Client.get_solver` is guaranteed to return a
        hybrid quantum-classical solver.

        .. code-block:: python

            from dwave.cloud.hybrid import Client

            with Client(token='...') as client:
                solver = client.get_solver()
                response = solver.sample_bqm(...)

        The following example instantiates a hybrid client indirectly. Again,
        :meth:`~dwave.cloud.client.Client.get_solver`/
        :meth:`~dwave.cloud.client.Client.get_solvers` are guaranteed to return
        only hybrid solver(s).

        .. code-block:: python

            from dwave.cloud import Client

            with Client.from_config(client='hybrid') as client:
                solver = client.get_solver()
                response = solver.sample_bqm(...)

    """

    @staticmethod
    def is_solver_handled(solver):
        """Determine if the specified solver should be handled by this client.

        This predicate function (used from the base class) allows only remote
        hybrid quantum-classical solvers.
        """
        return solver and solver.hybrid

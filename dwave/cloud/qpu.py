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
An implementation of the REST API for D-Wave Solver API (SAPI) servers.

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

from __future__ import absolute_import

from dwave.cloud.client import Client as BaseClient
from dwave.cloud.solver import Solver
from dwave.cloud.computation import Future

__all__ = ['Client']

class Client(BaseClient):
    """D-Wave API client specialized to work with QPU solvers.

    This class is instantiated by default, or explicitly when `client=qpu`, with the
    typical base client instantiation :code:`with Client.from_config() as client:` of
    a client.

    Examples:
        This example explicitly instantiates a :class:`dwave.cloud.qpu.client` based
        on the local system`s default D-Wave Cloud Client configuration file to sample
        a random Ising problem tailored to fit the client`s default solver`s graph.

        .. code-block:: python

            import random
            from dwave.cloud.qpu import Client

            # Use context manager to ensure resources (thread pools used by Client) are released
            with Client.from_config() as client:

                solver = client.get_solver()

                # Build problem to exactly fit the solver graph
                linear = {index: random.choice([-1, 1]) for index in solver.nodes}
                quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

                # Sample 100 times and print out the first sample
                computation = solver.sample_ising(linear, quad, num_reads=100)
                print(computation.samples[0])

    """

    @staticmethod
    def is_solver_handled(solver):
        """Determine if the specified solver should be handled by this client.

        This predicate function overrides superclass to filter out any non-QPU solvers.

        Current implementation filters out D-Wave software clients with solver IDs
        prefixed with `c4-sw`. If needed, update this method to suit your solver
        naming scheme.

        Examples:
            This example filters solvers for those prefixed 2000Q.

            .. code:: python

                @staticmethod
                def is_solver_handled(solver):
                    return solver and solver.id.startswith('2000Q')

        """
        return solver and solver.is_qpu

"""
An implementation of the REST API for D-Wave Solver API (SAPI) servers.

SAPI servers provide authentication, queuing, and scheduling services,
and provide a network interface to :term:`solver`\ s. This API enables you submit
a binary quadratic (\ :term:`Ising` or :term:QUBO`\ ) model
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


class Client(BaseClient):
    """D-Wave API client specialized to work with the QPU solvers (samplers)."""

An example using the client:

.. code-block:: python
    :linenos:

    import random
    from dwave.cloud.qpu import Client

    # Connect using explicit connection information
    # Also, note the use context manager, which ensures the resources (thread
    # pools used by Client) are freed as soon as we're done with using client.
    with Client('https://sapi-url', 'token-string') as client:

        # Load a solver by name
        solver = client.get_solver('test-solver')

        # Build a random Ising model on +1, -1. Build it to exactly fit the graph the solver provides
        linear = {index: random.choice([-1, 1]) for index in solver.nodes}
        quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

        # Send the problem for sampling, include a solver specific parameter 'num_reads'
        computation = solver.sample_ising(linear, quad, num_reads=100)

        # Print out the first sample (out of a hundred)
        print(computation.samples[0])


    @staticmethod
    def is_solver_handled(solver):
        """Predicate function used from superclass to filter solvers.
        In QPU client we're handling only QPU solvers.
        """
        if not solver:
            return False
        return not solver.id.startswith('c4-sw_')

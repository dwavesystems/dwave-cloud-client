"""
An implementation of the REST API exposed by D-Wave Solver API (SAPI) servers.

This API lets you submit an Ising model and receive samples from a distribution
over the model as defined by the solver you have selected.

 - The SAPI servers provide authentication, queuing, and scheduling services, and
   provide a network interface to the solvers.
 - A solver is a resource that can sample from a discrete quadratic model.
 - This package implements the REST interface these servers provide.

An example using the client:

.. code-block:: python
    :linenos:

    import random
    from dwave.cloud.qpu import Client

    # Connect using explicit connection information
    client = Client('https://sapi-url', 'token-string')

    # Load a solver by name
    solver = client.get_solver('test-solver')

    # Build a random Ising model on +1, -1. Build it to exactly fit the graph the solver provides
    linear = {index: random.choice([-1, 1]) for index in solver.nodes}
    quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

    # Send the problem for sampling, include a solver specific parameter 'num_reads'
    computation = solver.sample_ising(linear, quad, num_reads=100)

    # Print out the first sample
    print(computation.samples[0])

Rough workflow within the SAPI server:
 1. Submitted problems enter an input queue. Each user has an input queue per solver.
 2. Drawing from all input queues for a solver, problems are scheduled.
 3. Results of the server are cached for retrieval by the client.

By default all sampling requests will be processed asynchronously. Reading results from
any future object is a blocking operation.

.. code-block:: python
    :linenos:

    # We can submit several sample requests without blocking
    # (In this specific case we could accomplish the same thing by increasing 'num_reads')
    futures = [solver.sample_ising(linear, quad, num_reads=100) for _ in range(10)]

    # We can check if a set of samples are ready without blocking
    print(futures[0].done())

    # We can wait on a single future
    futures[0].wait()

    # Or we can wait on several futures
    dwave.cloud.qpu.computation.Future.wait_multiple(futures, min_done=3)

"""

from .client import Client
from .solver import Solver
from .computation import Future

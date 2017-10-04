# D-Wave Micro Client

A minimal implementation of the REST interface used to communicate with D-Wave Solver API (SAPI) servers.

SAPI is an application layer built to provide resource discovery, permissions, and scheduling for quantum annealing resources at D-Wave Systems. This package aims to provide a minimal Python interface to that layer that still captures some reasonable practices for interacting with SAPI.

## Example
```python
import dwave_micro_client
import random

# Connect using the default or environment connection information
con = dwave_micro_client.Connection()

# Load the default solver
solver = con.get_solver()

# Build a random Ising model on +1, -1. Build it to exactly fit the graph the solver provides
linear = {index: random.choice([-1, 1]) for index in solver.nodes}
quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

# Send the problem for sampling, include a solver specific parameter 'num_reads'
results = solver.sample_ising(linear, quad, num_reads=100)

# Print out the first sample
print(results.samples[0])
```

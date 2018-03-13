"""
Interface to software samplers available via D-Wave API.

Software samplers have the same interface (response) as the QPU sampler, but
the samples are generated with classical software solvers.
"""
from __future__ import absolute_import

from dwave.cloud.client import Client as BaseClient
from dwave.cloud.solver import Solver
from dwave.cloud.computation import Future


class Client(BaseClient):
    """D-Wave API client specialized to work with the remote software solvers
    (samplers)."""

    @staticmethod
    def is_solver_handled(solver):
        """Predicate function used from superclass to filter solvers.
        In the software client we're handling only remote software solvers.
        """
        if not solver:
            return False
        return solver.id.startswith('c4-sw_')

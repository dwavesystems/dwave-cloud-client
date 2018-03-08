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

    @staticmethod
    def is_solver_handled(solver):
        """Predicate function that determines if the given solver should be
        handled by this client.
        """
        if not solver:
            return False
        return solver.id.startswith('c4-sw_')

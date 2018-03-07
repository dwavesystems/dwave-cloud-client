"""
Interface to software samplers API.

Software samplers emulate the QPU behavior.
"""
from __future__ import absolute_import

from dwave.cloud.client import Client as _QPUClient


class Client(_QPUClient):

    @staticmethod
    def is_solver_handled(solver):
        """Predicate function that determines if the given solver should be
        handled by this client.
        """
        if not solver:
            return False
        return solver.id.startswith('c4-sw_')

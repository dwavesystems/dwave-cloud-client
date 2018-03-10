from __future__ import absolute_import

import unittest

from dwave.cloud.exceptions import *


class TestExceptions(unittest.TestCase):

    def test_solver_auth_error_msg(self):
        try:
            raise SolverAuthenticationError
        except Exception as e:
            self.assertEquals(e.message, "Token not accepted for that action.")

    def test_canceled_future_error_msg(self):
        try:
            raise CanceledFutureError
        except Exception as e:
            self.assertEquals(e.message, "An error occurred reading results from a canceled request")

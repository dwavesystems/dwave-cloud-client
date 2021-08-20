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

import unittest

from dwave.cloud.exceptions import SolverAuthenticationError, CanceledFutureError


class TestExceptions(unittest.TestCase):

    def test_solver_auth_error_msg(self):
        try:
            raise SolverAuthenticationError
        except Exception as e:
            self.assertEqual(str(e), "Invalid token or access denied")

    def test_canceled_future_error_msg(self):
        try:
            raise CanceledFutureError
        except Exception as e:
            self.assertEqual(str(e), "An error occurred reading results from a canceled request")

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

from __future__ import absolute_import, print_function

import base64
import struct
import unittest
import itertools

from dwave.cloud.coders import encode_problem_as_qp
from dwave.cloud.qpu import Solver


def get_solver():
    data = {
        "properties": {
            "supported_problem_types": ["qubo", "ising"],
            "qubits": [0, 1, 2, 3],
            "couplers": [(0, 1), (1, 2), (2, 3), (3, 0)],
            "num_qubits": 4,
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "id": "test-solver",
        "description": "A test solver"
    }
    return Solver(client=None, data=data)


class TestCoders(unittest.TestCase):
    nan = float('nan')

    def encode_doubles(self, values):
        return base64.b64encode(struct.pack('<' + ('d' * len(values)), *values)).decode('utf-8')

    def test_qpu_request_encoding_all_qubits(self):
        """Test biases and coupling strengths are properly encoded (base64 little-endian doubles)."""

        solver = get_solver()
        linear = {index: 1 for index in solver.nodes}
        quadratic = {key: -1 for key in solver.undirected_edges}
        request = encode_problem_as_qp(solver, linear, quadratic)
        self.assertEqual(request['format'], 'qp')
        self.assertEqual(request['lin'],  self.encode_doubles([1, 1, 1, 1]))
        self.assertEqual(request['quad'], self.encode_doubles([-1, -1, -1, -1]))

    def test_qpu_request_encoding_sub_qubits(self):
        """Inactive qubits should be encoded as NaNs. Inactive couplers should be omitted."""

        solver = get_solver()
        linear = {index: 1 for index in sorted(list(solver.nodes))[:2]}
        quadratic = {key: -1 for key in sorted(list(solver.undirected_edges))[:1]}
        request = encode_problem_as_qp(solver, linear, quadratic)
        self.assertEqual(request['format'], 'qp')
        # [1, 1, NaN, NaN]
        self.assertEqual(request['lin'],  self.encode_doubles([1, 1, self.nan, self.nan]))
        # [-1]
        self.assertEqual(request['quad'], self.encode_doubles([-1]))

    def test_qpu_request_encoding_missing_qubits(self):
        """Qubits don't have to be specified with biases only, but also with couplings."""

        solver = get_solver()
        linear = {}
        quadratic = {(0, 1): -1}
        request = encode_problem_as_qp(solver, linear, quadratic)
        self.assertEqual(request['format'], 'qp')
        # [0, 0, NaN, NaN]
        self.assertEqual(request['lin'],  self.encode_doubles([0, 0, self.nan, self.nan]))
        # [-1]
        self.assertEqual(request['quad'], self.encode_doubles([-1]))

    def test_qpu_request_encoding_sub_qubits_implicit_biases(self):
        """Biases don't have to be specified for qubits to be active."""

        solver = get_solver()
        linear = {}
        quadratic = {(0,3): -1}
        request = encode_problem_as_qp(solver, linear, quadratic)
        self.assertEqual(request['format'], 'qp')
        # [0, NaN, NaN, 0]
        self.assertEqual(request['lin'],  self.encode_doubles([0, self.nan, self.nan, 0]))
        # [-1]
        self.assertEqual(request['quad'], self.encode_doubles([-1]))

    def test_qpu_request_encoding_sub_qubits_implicit_couplings(self):
        """Couplings should be zero for active qubits, if not specified."""

        solver = get_solver()
        linear = {0: 0, 3: 0}
        quadratic = {}
        request = encode_problem_as_qp(solver, linear, quadratic)
        self.assertEqual(request['format'], 'qp')
        # [0, NaN, NaN, 0]
        self.assertEqual(request['lin'],  self.encode_doubles([0, self.nan, self.nan, 0]))
        # [0]
        self.assertEqual(request['quad'], self.encode_doubles([0]))

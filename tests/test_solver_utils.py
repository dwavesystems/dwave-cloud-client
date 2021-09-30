# Copyright 2021 D-Wave Systems Inc.
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

import networkx as nx
from parameterized import parameterized_class

try:
    import dimod
except ImportError:
    raise unittest.SkipTest("dimod not installed")

from dwave.cloud.solver import StructuredSolver
from dwave.cloud.testing import mocks


C16 = StructuredSolver(data=mocks.qpu_chimera_solver_data(16), client=None)
P16 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(16), client=None)


@parameterized_class(("solver", ), [
    (C16, ),
    (P16, ),
])
class TestCheckProblem(unittest.TestCase):

    def test_identity(self):
        bqm = dimod.generators.ran_r(1, (self.solver.nodes, self.solver.edges))
        self.assertTrue(self.solver.check_problem(bqm.linear, bqm.quadratic))

    def test_valid_subgraph(self):
        bqm = dimod.generators.ran_r(1, list(self.solver.edges)[:10])
        self.assertTrue(self.solver.check_problem(bqm.linear, bqm.quadratic))

    def test_invalid_subgraph(self):
        bqm = dimod.generators.ran_r(1, 10)
        self.assertFalse(self.solver.check_problem(bqm.linear, bqm.quadratic))

    def test_invalid_small(self):
        bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        self.assertFalse(self.solver.check_problem(bqm.linear, bqm.quadratic))

    def test_supergraph(self):
        n = max(self.solver.nodes)
        edges = list(self.solver.edges) + [(n+i, n+2*i) for i in range(10)]
        bqm = dimod.generators.ran_r(1, edges)
        self.assertFalse(self.solver.check_problem(bqm.linear, bqm.quadratic))

    def test_legacy_format(self):
        bqm = dimod.generators.ran_r(1, (self.solver.nodes, self.solver.edges))
        h = list(bqm.linear.values())   # h as list is still supported
        self.assertTrue(self.solver.check_problem(h, bqm.quadratic))

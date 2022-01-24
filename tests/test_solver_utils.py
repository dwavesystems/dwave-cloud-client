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
    dimod = None

from dwave.cloud.solver import StructuredSolver
from dwave.cloud.testing import mocks


try:
    C16 = StructuredSolver(data=mocks.qpu_chimera_solver_data(16), client=None)
    P16 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(16), client=None)
except RuntimeError:
    raise unittest.SkipTest("missing installs")


@parameterized_class(("solver", ), [
    (C16, ),
    (P16, ),
])
@unittest.skipUnless(dimod, "dimod not installed")
class TestCheckProblem(unittest.TestCase):

    def test_identity(self):
        # NOTE: cast to list can be removed once we drop support for dimod 0.8.x
        bqm = dimod.generators.ran_r(1, list(self.solver.edges))
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
        edges = list(self.solver.edges) + [(n+i, n+2*i) for i in range(1, 10)]
        bqm = dimod.generators.ran_r(1, edges)
        self.assertFalse(self.solver.check_problem(bqm.linear, bqm.quadratic))

    def test_legacy_format(self):
        # NOTE: cast to list can be removed once we drop support for dimod 0.8.x
        bqm = dimod.generators.ran_r(1, list(self.solver.edges))
        h = list(bqm.linear.values())   # h as list is still supported
        self.assertTrue(self.solver.check_problem(h, bqm.quadratic))


class TestReformatParameters(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(StructuredSolver.reformat_parameters('ising', {}, {}), {})

    def test_initial_states(self):
        doc = {'initial_state': {0: 0, 4: 1}}

        self.assertEqual(StructuredSolver.reformat_parameters('ising', doc, dict(num_qubits=9)),
                         dict(initial_state=[-1, 3, 3, 3, 1, 3, 3, 3, 3]))
        self.assertEqual(StructuredSolver.reformat_parameters('qubo', doc, dict(num_qubits=9)),
                         dict(initial_state=[0, 3, 3, 3, 1, 3, 3, 3, 3]))

        if dimod:
            self.assertEqual(StructuredSolver.reformat_parameters('SPIN', doc, dict(num_qubits=9)),
                             dict(initial_state=[-1, 3, 3, 3, 1, 3, 3, 3, 3]))
            self.assertEqual(StructuredSolver.reformat_parameters('BINARY', doc, dict(num_qubits=9)),
                             dict(initial_state=[0, 3, 3, 3, 1, 3, 3, 3, 3]))

        self.assertEqual(doc, {'initial_state': {0: 0, 4: 1}})

    def test_initial_states_inplace(self):
        doc = {'initial_state': {0: 0, 4: 1}}
        StructuredSolver.reformat_parameters('ising', doc, dict(num_qubits=9), inplace=True)
        self.assertEqual(doc, dict(initial_state=[-1, 3, 3, 3, 1, 3, 3, 3, 3]))

    def test_initial_states_sequence(self):
        doc = {'initial_state': [-1, 3, 3, 3, 1, 3, 3, 3, 3]}
        self.assertEqual(StructuredSolver.reformat_parameters('ising', doc, dict(num_qubits=9)),
                         dict(initial_state=[-1, 3, 3, 3, 1, 3, 3, 3, 3]))

    def test_vartype_smoke(self):
        for vt in StructuredSolver._handled_problem_types:
            StructuredSolver.reformat_parameters(vt, {}, {})

        with self.assertRaises(ValueError):
            StructuredSolver.reformat_parameters('not a type', {}, {})

    @unittest.skipUnless(dimod, "dimod not installed")
    def test_vartype_dimod_smoke(self):
        StructuredSolver.reformat_parameters('SPIN', {}, {})
        StructuredSolver.reformat_parameters('BINARY', {}, {})
        StructuredSolver.reformat_parameters(dimod.BINARY, {}, {})
        StructuredSolver.reformat_parameters(dimod.SPIN, {}, {})

        with self.assertRaises(ValueError):
            StructuredSolver.reformat_parameters("INTEGER", {}, {})

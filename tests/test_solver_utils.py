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

import copy
from parameterized import parameterized, parameterized_class

try:
    import dimod
except ImportError:
    dimod = None

from dwave.cloud.client import Client
from dwave.cloud.solver import StructuredSolver
from dwave.cloud.testing import mocks

try:
    C16 = StructuredSolver(data=mocks.qpu_chimera_solver_data(16), client=None)
    P16 = StructuredSolver(data=mocks.qpu_pegasus_solver_data(16), client=None)
    solvers = [(C16,), (P16,)]
except RuntimeError:
    solvers = []

from tests import config

@parameterized_class(("solver", ), solvers)
@unittest.skipUnless(dimod, "dimod not installed")
class TestCheckProblem(unittest.TestCase):

    def test_identity(self):
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


class QpuAccessTimeEstimate(unittest.TestCase):
    """Test QPU access time estimation method."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.solver_mock = StructuredSolver(data=mocks.qpu_pegasus_solver_data(16,
                problem_timing_data=mocks.qpu_problem_timing_data(qpu='advantage')), client=None)
        except RuntimeError:
            raise cls.skipTest(cls, "structured solver mock not available")

    def test_mock_conflicting_params(self):
        # not allowed annealing_time together with anneal_schedule
        with self.assertRaises(ValueError):
            self.solver_mock.estimate_qpu_access_time(num_qubits=1000,
                anneal_schedule=[[0.0, 0.0], [50.0, 0.5]],
                annealing_time=150)
        # reverse anneal without required initial state
        with self.assertRaises(ValueError):
            self.solver_mock.estimate_qpu_access_time(num_qubits=1000,
                anneal_schedule=[[0.0, 1.0], [2.75, 0.45], [82.75, 0.45], [82.761, 1.0]])

    def test_mock_version(self):
        # currently support is for version 1.0.x
        with self.assertRaises(ValueError):
            solver_mock_ver = copy.deepcopy(self.solver_mock)
            solver_mock_ver.properties["problem_timing_data"]["version"] = '1.1.0'
            solver_mock_ver.estimate_qpu_access_time(num_qubits=1000)

    def test_mock_missing_params(self):
        with self.assertRaises(KeyError):
            solver_mock_ver = copy.deepcopy(self.solver_mock)
            solver_mock_ver.properties["problem_timing_data"].pop("version")
            solver_mock_ver.estimate_qpu_access_time(num_qubits=1000)

    @parameterized.expand([({'num_qubits': 1000}, 15341),
                 ({'num_qubits': 1000, 'num_reads': 500}, 149225),
                 ({'num_qubits': 1000, 'annealing_time': 400}, 15721),
                 ({'num_qubits': 1000, 'anneal_schedule': [[0.0, 0.0], [800.0, 0.5]]}, 16121),
                 ({'num_qubits': 1000, 'anneal_schedule': [[0.0, 1.0], [2, 0.45], [102, 0.45], [102, 1.0]], 'initial_state': "dummy"}, 15427),])
    def test_mock_estimations(self, d, t):
        self.assertEqual(int(self.solver_mock.estimate_qpu_access_time(**d)), t)

    def test_mock_estimations_increase_with_reads(self):
        # increase number of reads
        runtime_r500 = self.solver_mock.estimate_qpu_access_time(num_qubits=1000, num_reads=500)
        runtime_r600 = self.solver_mock.estimate_qpu_access_time(num_qubits=1000, num_reads=600)
        self.assertGreater(runtime_r600, runtime_r500)

    @unittest.skipUnless(config, "No live server configuration available.")
    def test_live_qpu_access_time_estimate(self):
        with Client(**config) as client:
            solver = client.get_solver(topology__type='zephyr')

            # test live qpu is in a somewhat reasonable range
            # results for num_qubits=1000: Advantage2_system1.5 ~34700 Advantage4.1 15341, DW_2000Q_6 11775
            runtime_q1000 = solver.estimate_qpu_access_time(num_qubits=1000)
            self.assertGreater(runtime_q1000, 5000)
            self.assertGreater(35000, runtime_q1000)

            # compare for most qubits in use (best model accuracy in version 1.0.0)
            h = {node: 0.5 for node in list(solver.nodes)[:5000]}
            computation = solver.sample_ising(h, {}, num_reads=1000)
            result = computation.result()
            qpu_access_time = result["timing"]["qpu_access_time"]
            runtime_q5000_r1000 = solver.estimate_qpu_access_time(num_qubits=5000, num_reads=1000)
            miss = abs(qpu_access_time - runtime_q5000_r1000)/max(qpu_access_time, runtime_q5000_r1000)
            self.assertLess(miss, 0.05)

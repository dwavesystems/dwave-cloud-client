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

import os
import uuid
import unittest
from unittest import mock

import networkx as nx

try:
    import dimod
except ImportError:
    dimod = None

from dwave.cloud.testing import isolated_environ, iterable_mock_open, mocks


@mock.patch.dict(os.environ)
class TestEnvUtils(unittest.TestCase):
    def setUp(self):
        self.var = str(uuid.uuid4())
        self.val = str(uuid.uuid4())
        self.alt = str(uuid.uuid4())

    def clear(self):
        if self.var in os.environ:
            del os.environ[self.var]

    def setval(self):
        os.environ[self.var] = self.val

    def test_isolation(self):
        # set var to val in "global" env
        self.setval()
        self.assertEqual(os.getenv(self.var), self.val)

        # ensure isolated env stores modified value
        with isolated_environ():
            os.environ[self.var] = self.alt
            self.assertEqual(os.getenv(self.var), self.alt)

        # ensure "global" env is restored
        self.assertEqual(os.getenv(self.var), self.val)

    def test_update(self):
        # set var to val in "global" env
        self.setval()
        self.assertEqual(os.getenv(self.var), self.val)

        # ensure isolated env has modified value
        with isolated_environ({self.var: self.alt}):
            self.assertEqual(os.getenv(self.var), self.alt)

        # ensure "global" env is restored
        self.assertEqual(os.getenv(self.var), self.val)

    def test_add(self):
        # clear var in "global" env
        self.clear()
        self.assertEqual(os.getenv(self.var), None)

        # ensure isolated env adds var
        with isolated_environ({self.var: self.val}):
            self.assertEqual(os.getenv(self.var), self.val)

        # ensure "global" env is restored
        self.assertEqual(os.getenv(self.var), None)

    def test_remove(self):
        # set var to val in "global" env
        self.setval()
        self.assertEqual(os.getenv(self.var), self.val)

        # ensure isolated env removes var
        with isolated_environ(remove={self.var: self.val}):
            self.assertEqual(os.getenv(self.var), None)
        with isolated_environ(remove=[self.var]):
            self.assertEqual(os.getenv(self.var), None)
        with isolated_environ(remove=set([self.var])):
            self.assertEqual(os.getenv(self.var), None)

        # ensure "global" env is restored
        self.assertEqual(os.getenv(self.var), self.val)

    def test_remove_dwave(self):
        # set var to val in "global" env
        dw1, dw2 = 'DWAVE_TEST', 'DW_INTERNAL__TEST'
        val = 'test'
        os.environ[dw1] = val
        os.environ[dw2] = val

        # ensure isolated env removes dwave var
        with isolated_environ(remove_dwave=True):
            self.assertEqual(os.getenv(dw1), None)
            self.assertEqual(os.getenv(dw2), None)

        # ensure "global" env is restored
        self.assertEqual(os.getenv(dw1), val)
        self.assertEqual(os.getenv(dw2), val)

    def test_empty(self):
        # set var to val in "global" env
        self.setval()
        self.assertEqual(os.getenv(self.var), self.val)

        # ensure isolated env starts empty
        with isolated_environ(empty=True):
            self.assertEqual(os.getenv(self.var), None)

        # ensure "global" env is restored
        self.assertEqual(os.getenv(self.var), self.val)


class TestMockUtils(unittest.TestCase):

    def test_iterable_mock_open(self):
        data = '1\n2\n3'
        namespaced_open = '{}.open'.format(__name__)

        # first, verify `mock.mock_open` fails for iteration
        with self.assertRaises(AttributeError):
            with mock.patch(namespaced_open, mock.mock_open(data), create=True):
                for line in open('filename'):
                    pass

        # then verify our `iterable_mock_open` works
        lines = []
        with mock.patch(namespaced_open, iterable_mock_open(data), create=True):
            for line in open('filename'):
                lines.append(line)
        self.assertEqual(''.join(lines), data)


class TestSolverDataMocks(unittest.TestCase):

    def test_solver_configuration_data(self):
        data = mocks.solver_configuration_data()
        self.assertGreater(len(data['id']), 1)
        self.assertGreater(len(data['status']), 1)
        self.assertGreater(len(data['description']), 1)
        self.assertIsInstance(data['properties'], dict)
        self.assertIsInstance(data['avg_load'], float)

    def test_structured_solver_data(self):
        data = mocks.structured_solver_data()
        self.assertEqual(data['properties']['category'], 'qpu')
        self.assertIn('topology', data['properties'])
        self.assertIsInstance(data['properties']['qubits'], list)
        self.assertIsInstance(data['properties']['qubits'], list)
        self.assertGreater(len(data['properties']['qubits']), 0)

    def test_qpu_clique_solver_data(self):
        n = 3
        data = mocks.qpu_clique_solver_data(n)
        self.assertEqual(data['id'], 'clique_{}q_mock'.format(n))
        self.assertEqual(data['properties']['num_qubits'], n)
        self.assertEqual(data['properties']['qubits'], list(range(n)))

    @unittest.skipUnless(dimod, "dimod not installed")
    def test_qpu_chimera_solver_data(self):
        # 2 x 2 chimera tiles of 1-1 bipartite graphs, overall forming a cycle over 8 qubits
        m, n, t = 2, 2, 1
        num_qubits = m * n * 2 * t
        data = mocks.qpu_chimera_solver_data(m, n, t)
        nodes = data['properties']['qubits']
        edges = data['properties']['couplers']
        self.assertEqual(data['id'], f'chimera_{num_qubits}q_mock')
        self.assertEqual(data['properties']['num_qubits'], num_qubits)
        self.assertEqual(set(nodes), set(range(num_qubits)))
        self.assertEqual(len(nx.find_cycle(nx.Graph(edges))), num_qubits)

    @unittest.skipUnless(dimod, "dimod not installed")
    def test_qpu_pegasus_solver_data(self):
        m = 2
        num_qubits = 24 * m * (m-1)     # includes non-fabric qubits
        num_edges = 12 * (15 * (m-1)^2 + m - 3)
        data = mocks.qpu_pegasus_solver_data(m)
        nodes = data['properties']['qubits']
        edges = data['properties']['couplers']
        self.assertEqual(data['id'], f'pegasus_{num_qubits}q_mock')
        self.assertEqual(data['properties']['num_qubits'], num_qubits)
        self.assertLessEqual(len(nodes), num_qubits)
        self.assertLessEqual(len(edges), num_edges)

    def test_unstructured_solver_data(self):
        data = mocks.unstructured_solver_data()
        self.assertEqual(data['properties']['category'], 'hybrid')
        self.assertIn('minimum_time_limit', data['properties'])
        self.assertListEqual(data['properties']['supported_problem_types'], ['bqm'])

        data = mocks.hybrid_bqm_solver_data()
        self.assertListEqual(data['properties']['supported_problem_types'], ['bqm'])

        data = mocks.hybrid_dqm_solver_data()
        self.assertListEqual(data['properties']['supported_problem_types'], ['dqm'])


if __name__ == '__main__':
    unittest.main()

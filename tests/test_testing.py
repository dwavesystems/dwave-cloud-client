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

try:
    import dwave_networkx as dnx
except ImportError:
    dnx = None

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
        def test():
            os.environ[self.var] = self.alt
            self.assertEqual(os.getenv(self.var), self.alt)

        with isolated_environ():
            test()

        isolated_environ()(test)()

        # ensure "global" env is restored
        self.assertEqual(os.getenv(self.var), self.val)

    def test_update(self):
        # set var to val in "global" env
        self.setval()
        self.assertEqual(os.getenv(self.var), self.val)

        # ensure isolated env has modified value
        args = ({self.var: self.alt}, )

        def test():
            self.assertEqual(os.getenv(self.var), self.alt)

        with isolated_environ(*args):
            test()

        isolated_environ(*args)(test)()

        # ensure "global" env is restored
        self.assertEqual(os.getenv(self.var), self.val)

    def test_add(self):
        # clear var in "global" env
        self.clear()
        self.assertEqual(os.getenv(self.var), None)

        # ensure isolated env adds var
        args = ({self.var: self.val}, )

        def test():
            self.assertEqual(os.getenv(self.var), self.val)

        with isolated_environ(*args):
            test()

        isolated_environ(*args)(test)()

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
        var = 'DWAVE_TEST'
        val = 'test'
        os.environ[var] = val

        # ensure isolated env removes dwave var
        with isolated_environ(remove_dwave=True):
            self.assertEqual(os.getenv(var), None)

        # ensure "global" env is restored
        self.assertEqual(os.getenv(var), val)

    def test_empty(self):
        # set var to val in "global" env
        self.setval()
        self.assertEqual(os.getenv(self.var), self.val)

        # ensure isolated env starts empty
        kwargs = dict(empty=True)

        def test():
            self.assertEqual(os.getenv(self.var), None)

        with isolated_environ(**kwargs):
            test()

        isolated_environ(**kwargs)(test)()

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
        self.assertGreater(len(data['identity']['name']), 1)
        self.assertGreater(len(data['status']), 1)
        self.assertGreater(len(data['description']), 1)
        self.assertIsInstance(data['properties'], dict)
        self.assertIsInstance(data['avg_load'], float)

    def test_structured_solver_data(self):
        data = mocks.structured_solver_data()
        self.assertEqual(data['properties']['category'], 'qpu')
        self.assertIn('topology', data['properties'])
        self.assertIn('name', data['identity'])
        self.assertIn('version', data['identity'])
        self.assertIn('graph_id', data['identity']['version'])
        self.assertIsInstance(data['properties']['qubits'], list)
        self.assertIsInstance(data['properties']['qubits'], list)
        self.assertGreater(len(data['properties']['qubits']), 0)

    def test_qpu_clique_solver_data(self):
        n = 3
        data = mocks.qpu_clique_solver_data(n)
        self.assertEqual(data['identity']['name'], 'clique_{}q_mock'.format(n))
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

        self.assertEqual(data['identity']['name'], f'chimera_{num_qubits}q_mock')
        self.assertEqual(data['properties']['num_qubits'], num_qubits)
        self.assertEqual(set(nodes), set(range(num_qubits)))
        self.assertEqual(len(nx.find_cycle(nx.Graph(edges))), num_qubits)

        # verify qubits and couplers are ordered (like on SAPI)
        self.assertEqual(nodes, sorted(nodes))
        self.assertTrue(all(i < j for (i, j) in edges))

    @unittest.skipUnless(dimod, "dimod not installed")
    def test_qpu_pegasus_solver_data(self):
        m = 2
        num_qubits = 24 * m * (m-1)     # includes non-fabric qubits
        num_edges = 12 * (15 * (m-1)^2 + m - 3)

        data = mocks.qpu_pegasus_solver_data(m)
        nodes = data['properties']['qubits']
        edges = data['properties']['couplers']

        self.assertEqual(data['identity']['name'], f'pegasus_{num_qubits}q_mock')
        self.assertEqual(data['properties']['num_qubits'], num_qubits)
        self.assertLessEqual(len(nodes), num_qubits)
        self.assertLessEqual(len(edges), num_edges)

        # verify qubits and couplers are ordered (like on SAPI)
        self.assertEqual(nodes, sorted(nodes))
        self.assertTrue(all(i < j for (i, j) in edges))

    @unittest.skipUnless(dimod and dnx, "dimod/dwave-networkx not installed")
    def test_qpu_zephyr_solver_data(self):
        m, t = 2, 2
        num_qubits = 4 * t * m * (2 * m + 1)
        num_edges = len(dnx.zephyr_graph(m, t).edges)

        data = mocks.qpu_zephyr_solver_data(m, t)
        nodes = data['properties']['qubits']
        edges = data['properties']['couplers']

        self.assertEqual(data['identity']['name'], f'zephyr_{num_qubits}q_mock')
        self.assertEqual(data['properties']['num_qubits'], num_qubits)
        self.assertLessEqual(len(nodes), num_qubits)
        self.assertLessEqual(len(edges), num_edges)

        # verify qubits and couplers are ordered (like on SAPI)
        self.assertEqual(nodes, sorted(nodes))
        self.assertTrue(all(i < j for (i, j) in edges))

    def test_unstructured_solver_data(self):
        data = mocks.unstructured_solver_data()
        self.assertIn('name', data['identity'])
        self.assertNotIn('version', data['identity'])
        self.assertEqual(data['properties']['category'], 'hybrid')
        self.assertListEqual(data['properties']['supported_problem_types'], ['bqm'])

        data = mocks.hybrid_bqm_solver_data()
        self.assertListEqual(data['properties']['supported_problem_types'], ['bqm'])
        self.assertIn('maximum_number_of_variables', data['properties'])

        data = mocks.hybrid_cqm_solver_data()
        self.assertListEqual(data['properties']['supported_problem_types'], ['cqm'])
        self.assertIn('maximum_number_of_variables', data['properties'])
        self.assertIn('maximum_number_of_constraints', data['properties'])

        data = mocks.hybrid_dqm_solver_data()
        self.assertListEqual(data['properties']['supported_problem_types'], ['dqm'])
        self.assertIn('maximum_number_of_variables', data['properties'])
        self.assertIn('maximum_number_of_cases', data['properties'])

        data = mocks.hybrid_nl_solver_data()
        self.assertListEqual(data['properties']['supported_problem_types'], ['nl'])
        self.assertIn('maximum_number_of_nodes', data['properties'])
        self.assertIn('maximum_number_of_states', data['properties'])


if __name__ == '__main__':
    unittest.main()

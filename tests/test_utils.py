import unittest
from collections import OrderedDict
from datetime import datetime

from dwave.cloud.utils import (
    uniform_iterator, uniform_get, strip_head, strip_tail,
    active_qubits, generate_valid_random_problem,
    default_text_input, utcnow)
from dwave.cloud.testing import mock


class TestUtils(unittest.TestCase):

    def test_uniform_iterator(self):
        items = [('a', 1), ('b', 2)]
        self.assertEqual(list(uniform_iterator(OrderedDict(items))), items)
        self.assertEqual(list(uniform_iterator('ab')), list(enumerate('ab')))

    def test_uniform_get(self):
        d = {0: 0, 1: 1}
        self.assertEqual(uniform_get(d, 0), 0)
        self.assertEqual(uniform_get(d, 2), None)
        self.assertEqual(uniform_get(d, 2, default=0), 0)
        l = [0, 1]
        self.assertEqual(uniform_get(l, 0), 0)
        self.assertEqual(uniform_get(l, 2), None)
        self.assertEqual(uniform_get(l, 2, default=0), 0)

    def test_strip_head(self):
        self.assertEqual(strip_head([0, 0, 1, 2], [0]), [1, 2])
        self.assertEqual(strip_head([1], [0]), [1])
        self.assertEqual(strip_head([1], []), [1])
        self.assertEqual(strip_head([0, 0, 1, 2], [0, 1, 2]), [])

    def test_strip_tail(self):
        self.assertEqual(strip_tail([1, 2, 0, 0], [0]), [1, 2])
        self.assertEqual(strip_tail([1], [0]), [1])
        self.assertEqual(strip_tail([1], []), [1])
        self.assertEqual(strip_tail([0, 0, 1, 2], [0, 1, 2]), [])

    def test_active_qubits_dict(self):
        self.assertEqual(active_qubits({}, {}), set())
        self.assertEqual(active_qubits({0: 0}, {}), {0})
        self.assertEqual(active_qubits({}, {(0, 1): 0}), {0, 1})
        self.assertEqual(active_qubits({2: 0}, {(0, 1): 0}), {0, 1, 2})

    def test_active_qubits_list(self):
        self.assertEqual(active_qubits([], {}), set())
        self.assertEqual(active_qubits([2], {}), {0})
        self.assertEqual(active_qubits([2, 2, 0], {}), {0, 1, 2})
        self.assertEqual(active_qubits([], {(0, 1): 0}), {0, 1})
        self.assertEqual(active_qubits([0, 0], {(0, 2): 0}), {0, 1, 2})

    def test_default_text_input(self):
        val = "value"
        with mock.patch("six.moves.input", side_effect=[val], create=True):
            self.assertEqual(default_text_input("prompt", val), val)
        with mock.patch("six.moves.input", side_effect=[val], create=True):
            self.assertEqual(default_text_input("prompt", val+val), val)

    def test_generate_valid_random_problem(self):
        class MockSolver(object):
            nodes = [0, 1, 3]
            undirected_edges = {(0, 1), (1, 3), (0, 4)}
            properties = dict(h_range=[2, 2], j_range=[-1, -1])
        mock_solver = MockSolver()

        lin, quad = generate_valid_random_problem(mock_solver)

        self.assertDictEqual(lin, {0: 2.0, 1: 2.0, 3: 2.0})
        self.assertDictEqual(quad, {(0, 1): -1.0, (1, 3): -1.0, (0, 4): -1.0})

    def test_generate_valid_random_problem_with_user_constrained_ranges(self):
        class MockSolver(object):
            nodes = [0, 1, 3]
            undirected_edges = {(0, 1), (1, 3), (0, 4)}
            properties = dict(h_range=[2, 2], j_range=[-1, -1])
        mock_solver = MockSolver()

        lin, quad = generate_valid_random_problem(mock_solver, h_range=[0,0], j_range=[1,1])

        self.assertDictEqual(lin, {0: 0.0, 1: 0.0, 3: 0.0})
        self.assertDictEqual(quad, {(0, 1): 1.0, (1, 3): 1.0, (0, 4): 1.0})

    def test_utcnow(self):
        t = utcnow()
        now = datetime.utcnow()
        self.assertEqual(t.utcoffset().total_seconds(), 0.0)
        unaware = t.replace(tzinfo=None)
        self.assertLess((now - unaware).total_seconds(), 1.0)


if __name__ == '__main__':
    unittest.main()

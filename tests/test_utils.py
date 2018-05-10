import unittest
from collections import OrderedDict

from dwave.cloud.utils import (
    readline_input, uniform_iterator, uniform_get, strip_head, strip_tail)
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

    def test_readline_input(self):
        val = "value"
        with mock.patch("six.moves.input", side_effect=[val], create=True):
            self.assertEqual(readline_input("prompt", val), val)
        with mock.patch("six.moves.input", side_effect=[val], create=True):
            self.assertEqual(readline_input("prompt", val+val), val)


if __name__ == '__main__':
    unittest.main()

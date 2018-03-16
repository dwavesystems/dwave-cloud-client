import unittest
from collections import OrderedDict

try:
    import unittest.mock as mock
except ImportError:
    import mock

from dwave.cloud.utils import readline_input, uniform_iterator, uniform_get


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

    def test_readline_input(self):
        val = "value"
        with mock.patch("six.moves.input", side_effect=[val], create=True):
            self.assertEqual(readline_input("prompt", val), val)
        with mock.patch("six.moves.input", side_effect=[val], create=True):
            self.assertEqual(readline_input("prompt", val+val), val)


if __name__ == '__main__':
    unittest.main()

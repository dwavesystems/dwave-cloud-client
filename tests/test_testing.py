import os
import uuid
import unittest

from dwave.cloud.testing import mock, isolated_environ


@mock.patch.dict(os.environ)
class TestUtils(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()

"""
Tests the ability to connect to the SAPI server with the `dwave.cloud.qpu.Client`.

test_mock_solver_loading.py duplicates some of these tests against a mock server.
"""
from __future__ import absolute_import

import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock

from dwave.cloud.config import legacy_load_config
from dwave.cloud.qpu import Client
from dwave.cloud.exceptions import SolverAuthenticationError
import dwave.cloud


try:
    config_url, config_token, _, config_solver = legacy_load_config()
    if None in [config_url, config_token, config_solver]:
        raise ValueError()
    skip_live = False
except:
    skip_live = True


class ConnectivityTests(unittest.TestCase):
    """Test connecting and related failure modes."""

    @unittest.skipIf(skip_live, "No live server available.")
    def test_bad_url(self):
        """Connect with a bad URL."""
        with self.assertRaises(IOError):
            client = Client("not-a-url", config_token)
            client.get_solvers()

    @unittest.skipIf(skip_live, "No live server available.")
    def test_bad_token(self):
        """Connect with a bad token."""
        with self.assertRaises(SolverAuthenticationError):
            client = Client(config_url, 'not-a-token')
            client.get_solvers()

    @unittest.skipIf(skip_live, "No live server available.")
    def test_good_connection(self):
        """Connect with a valid URL and token."""
        client = Client(config_url, config_token)
        self.assertTrue(len(client.get_solvers()) > 0)


class SolverLoading(unittest.TestCase):
    """Test loading solvers in a few different configurations."""

    @unittest.skipIf(skip_live, "No live server available.")
    def test_list_all_solvers(self):
        """List all the solvers."""
        client = Client(config_url, config_token)
        self.assertTrue(len(client.get_solvers()) > 0)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_load_all_solvers(self):
        """List and retrieve all the solvers."""
        client = Client(config_url, config_token)
        for name in client.get_solvers():
            client.get_solver(name)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_load_bad_solvers(self):
        """Try to load a nonexistent solver."""
        client = Client(config_url, config_token)
        with self.assertRaises(KeyError):
            client.get_solver("not-a-solver")

    @unittest.skipIf(skip_live, "No live server available.")
    def test_load_any_solver(self):
        """Load a single solver without calling get_solvers (which caches data)."""
        client = Client(config_url, config_token)
        client.get_solver(config_solver)


class ClientFactory(unittest.TestCase):
    """Test client factory."""

    def test_default(self):
        conf = {k: k for k in 'endpoint token'.split()}
        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            client = dwave.cloud.Client.from_config()
            self.assertEqual(client.endpoint, 'endpoint')
            self.assertEqual(client.token, 'token')

if __name__ == '__main__':
    unittest.main()

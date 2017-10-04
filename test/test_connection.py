"""
Tests the ability to connect to the SAPI server with the dwave_micro_client module.

test_mock_solver_loading.py duplicates some of these tests against a mock server.
"""
from __future__ import absolute_import

import unittest

import dwave_micro_client

try:
    config_url, config_token, _, config_solver = dwave_micro_client.load_configuration()
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
            con = dwave_micro_client.Connection("not-a-url", config_token)
            con.solver_names()

    @unittest.skipIf(skip_live, "No live server available.")
    def test_bad_token(self):
        """Connect with a bad token."""
        with self.assertRaises(dwave_micro_client.SolverAuthenticationError):
            con = dwave_micro_client.Connection(config_url, 'not-a-token')
            con.solver_names()

    @unittest.skipIf(skip_live, "No live server available.")
    def test_good_connection(self):
        """Connect with a valid URL and token."""
        con = dwave_micro_client.Connection(config_url, config_token)
        self.assertTrue(len(con.solver_names()) > 0)


class SolverLoading(unittest.TestCase):
    """Test loading solvers in a few different configurations."""

    @unittest.skipIf(skip_live, "No live server available.")
    def test_list_all_solvers(self):
        """List all the solvers."""
        con = dwave_micro_client.Connection(config_url, config_token)
        self.assertTrue(len(con.solver_names()) > 0)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_load_all_solvers(self):
        """List and retrieve all the solvers."""
        con = dwave_micro_client.Connection(config_url, config_token)
        for name in con.solver_names():
            con.get_solver(name)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_load_bad_solvers(self):
        """Try to load a nonexistent solver."""
        con = dwave_micro_client.Connection(config_url, config_token)
        with self.assertRaises(KeyError):
            con.get_solver("not-a-solver")

    @unittest.skipIf(skip_live, "No live server available.")
    def test_load_any_solver(self):
        """Load a single solver without calling solver_names (which caches data)."""
        con = dwave_micro_client.Connection(config_url, config_token)
        con.get_solver(config_solver)


if __name__ == '__main__':
    unittest.main()

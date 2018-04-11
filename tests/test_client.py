"""
Tests the ability to connect to the SAPI server with the `dwave.cloud.qpu.Client`.

test_mock_solver_loading.py duplicates some of these tests against a mock server.
"""
from __future__ import absolute_import

import unittest

from dwave.cloud.config import load_config
from dwave.cloud.qpu import Client
from dwave.cloud.exceptions import SolverAuthenticationError
from dwave.cloud.testing import mock
import dwave.cloud

from tests import config


@unittest.skipUnless(config, "No live server configuration available.")
class ConnectivityTests(unittest.TestCase):
    """Test connecting and related failure modes."""

    def test_bad_url(self):
        """Connect with a bad URL."""
        with self.assertRaises(IOError):
            with Client("not-a-url", config['token']) as client:
                client.get_solvers()

    def test_bad_token(self):
        """Connect with a bad token."""
        with self.assertRaises(SolverAuthenticationError):
            with Client(config['endpoint'], 'not-a-token') as client:
                client.get_solvers()

    def test_good_connection(self):
        """Connect with a valid URL and token."""
        with Client(config['endpoint'], config['token']) as client:
            self.assertTrue(len(client.get_solvers()) > 0)


@unittest.skipUnless(config, "No live server configuration available.")
class SolverLoading(unittest.TestCase):
    """Test loading solvers in a few different configurations."""

    def test_list_all_solvers(self):
        """List all the solvers."""
        with Client(config['endpoint'], config['token']) as client:
            self.assertTrue(len(client.get_solvers()) > 0)

    def test_load_all_solvers(self):
        """List and retrieve all the solvers."""
        with Client(config['endpoint'], config['token']) as client:
            for name in client.get_solvers():
                self.assertEqual(client.get_solver(name).id, name)

    def test_load_bad_solvers(self):
        """Try to load a nonexistent solver."""
        with Client(config['endpoint'], config['token']) as client:
            with self.assertRaises(KeyError):
                client.get_solver("not-a-solver")

    def test_load_any_solver(self):
        """Load a single solver without calling get_solvers (which caches data)."""
        with Client(config['endpoint'], config['token']) as client:
            self.assertEqual(client.get_solver(config['solver']).id, config['solver'])


class ClientFactory(unittest.TestCase):
    """Test client factory."""

    def test_default(self):
        conf = {k: k for k in 'endpoint token'.split()}
        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.endpoint, 'endpoint')
                self.assertEqual(client.token, 'token')
                self.assertIsInstance(client, dwave.cloud.qpu.Client)
                self.assertNotIsInstance(client, dwave.cloud.sw.Client)

    def test_custom_kwargs(self):
        conf = {k: k for k in 'endpoint token'.split()}
        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with mock.patch("dwave.cloud.client.Client.__init__", return_value=None) as init:
                dwave.cloud.Client.from_config(custom='custom')
                init.assert_called_once_with(
                    endpoint='endpoint', token='token', custom='custom')

    def test_custom_kwargs_overrides_config(self):
        conf = {k: k for k in 'endpoint token custom'.split()}
        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with mock.patch("dwave.cloud.client.Client.__init__", return_value=None) as init:
                dwave.cloud.Client.from_config(custom='new-custom')
                init.assert_called_once_with(
                    endpoint='endpoint', token='token', custom='new-custom')

    def test_legacy_config_load_fallback(self):
        conf = {k: k for k in 'endpoint token proxy solver'.split()}
        with mock.patch("dwave.cloud.client.load_config", return_value={}):
            with mock.patch("dwave.cloud.client.legacy_load_config", lambda **kwargs: conf):
                # test fallback works (legacy config is loaded)
                with dwave.cloud.Client.from_config(legacy_config_fallback=True) as client:
                    self.assertEqual(client.endpoint, 'endpoint')
                    self.assertEqual(client.token, 'token')
                    self.assertEqual(client.default_solver, 'solver')
                    self.assertEqual(client.session.proxies['http'], 'proxy')

                # test fallback is avoided (legacy config skipped)
                self.assertRaises(
                    ValueError, dwave.cloud.Client.from_config, legacy_config_fallback=False)


if __name__ == '__main__':
    unittest.main()

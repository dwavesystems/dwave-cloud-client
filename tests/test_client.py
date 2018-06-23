"""
Tests the ability to connect to the SAPI server with the `dwave.cloud.qpu.Client`.

test_mock_solver_loading.py duplicates some of these tests against a mock server.
"""
from __future__ import absolute_import

import unittest

from dwave.cloud.config import load_config
from dwave.cloud.client import Client
from dwave.cloud.solver import Solver
from dwave.cloud.exceptions import SolverAuthenticationError, SolverError
from dwave.cloud.testing import mock
import dwave.cloud

from tests import config


@unittest.skipUnless(config, "No live server configuration available.")
class ConnectivityTests(unittest.TestCase):
    """Test connecting and related failure modes."""

    def test_bad_url(self):
        """Connect with a bad URL."""
        with self.assertRaises(IOError):
            invalid_config = config.copy()
            invalid_config.update(endpoint='invalid-endpoint')
            with Client(**invalid_config) as client:
                client.get_solvers()

    def test_bad_token(self):
        """Connect with a bad token."""
        with self.assertRaises(SolverAuthenticationError):
            invalid_config = config.copy()
            invalid_config.update(token='invalid-token')
            with Client(**invalid_config) as client:
                client.get_solvers()

    def test_good_connection(self):
        """Connect with valid connection settings (url/token/proxy/etc)."""
        with Client(**config) as client:
            self.assertTrue(len(client.get_solvers()) > 0)


@unittest.skipUnless(config, "No live server configuration available.")
class SolverLoading(unittest.TestCase):
    """Test loading solvers in a few different configurations."""

    def test_list_all_solvers(self):
        """List all the solvers."""
        with Client(**config) as client:
            self.assertTrue(len(client.get_solvers()) > 0)

    def test_load_all_solvers(self):
        """List and retrieve all the solvers."""
        with Client(**config) as client:
            for name in client.get_solvers():
                self.assertEqual(client.get_solver(name).id, name)

    def test_load_bad_solvers(self):
        """Try to load a nonexistent solver."""
        with Client(**config) as client:
            with self.assertRaises(KeyError):
                client.get_solver("not-a-solver")

    def test_load_any_solver(self):
        """Load a single solver without calling get_solvers (which caches data)."""
        with Client(**config) as client:
            self.assertEqual(client.get_solver(config['solver']).id, config['solver'])

    def test_get_solver_no_defaults(self):
        from dwave.cloud import qpu, sw
        with qpu.Client(endpoint=config['endpoint'], token=config['token']) as client:
            solvers = client.solvers(qpu=True)
            if solvers:
                self.assertEqual(client.get_solver(), solvers[0])
            else:
                self.assertRaises(SolverError, client.get_solver)

        with sw.Client(endpoint=config['endpoint'], token=config['token']) as client:
            solvers = client.solvers(software=True)
            if solvers:
                self.assertEqual(client.get_solver(), solvers[0])
            else:
                self.assertRaises(SolverError, client.get_solver)


class ClientFactory(unittest.TestCase):
    """Test client factory."""

    def test_default(self):
        conf = {k: k for k in 'endpoint token'.split()}
        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.endpoint, 'endpoint')
                self.assertEqual(client.token, 'token')
                self.assertIsInstance(client, dwave.cloud.client.Client)
                self.assertNotIsInstance(client, dwave.cloud.sw.Client)
                self.assertNotIsInstance(client, dwave.cloud.qpu.Client)

    def test_client_type(self):
        conf = {k: k for k in 'endpoint token'.split()}
        def mocked_load_config(**kwargs):
            kwargs.update(conf)
            return kwargs

        with mock.patch("dwave.cloud.client.load_config", mocked_load_config):
            with dwave.cloud.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.client.Client)

            with dwave.cloud.client.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.client.Client)

            with dwave.cloud.qpu.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.qpu.Client)

            with dwave.cloud.sw.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.sw.Client)

            with dwave.cloud.Client.from_config(client='qpu') as client:
                self.assertIsInstance(client, dwave.cloud.qpu.Client)

            with dwave.cloud.qpu.Client.from_config(client='base') as client:
                self.assertIsInstance(client, dwave.cloud.client.Client)

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


class FeatureBasedSolverSelection(unittest.TestCase):
    """Test Client.solvers()."""

    def setUp(self):
        # mock solvers
        self.solver1 = Solver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1, 2],
                "couplers": [[0, 1], [0, 2], [1, 2]],
                "num_qubits": 3,
                "parameters": {"num_reads": "Number of samples to return."}
            },
            "id": "solver1",
            "description": "A test solver 1"
        })
        self.solver2 = Solver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1, 2, 3, 4],
                "couplers": [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]],
                "num_qubits": 5,
                "parameters": {
                    "num_reads": "Number of samples to return.",
                    "flux_biases": "Supported ..."
                },
                "vfyc": True
            },
            "id": "solver2",
            "description": "A test solver 2"
        })
        self.solvers = [self.solver1, self.solver2]

        # mock client
        self.client = Client('endpoint', 'token')
        self.client._solvers = {
            self.solver1.id: self.solver1,
            self.solver2.id: self.solver2
        }
        self.client._all_solvers_ready = True

    def shutDown(self):
        self.client.close()

    def assertSolvers(self, container, members):
        return set(container) == set(members)

    def test_default(self):
        self.assertSolvers(self.client.solvers(), self.solvers)

    def test_one_boolean(self):
        self.assertSolvers(self.client.solvers(vfyc=False), [self.solver1])
        self.assertSolvers(self.client.solvers(vfyc=True), [self.solver2])
        self.assertSolvers(self.client.solvers(flux_biases=False), [self.solver1])
        self.assertSolvers(self.client.solvers(flux_biases=True), [self.solver2])

    def test_boolean_combo(self):
        self.assertSolvers(self.client.solvers(vfyc=False, flux_biases=True), [])
        self.assertSolvers(self.client.solvers(vfyc=True, flux_biases=True), [self.solver2])

    def test_int_range(self):
        self.assertSolvers(self.client.solvers(num_qubits=3), [self.solver1])
        self.assertSolvers(self.client.solvers(num_qubits=4), [])
        self.assertSolvers(self.client.solvers(num_qubits=5), [self.solver2])

        self.assertSolvers(self.client.solvers(num_qubits=[3, 5]), self.solvers)
        self.assertSolvers(self.client.solvers(num_qubits=[2, None]), self.solvers)
        self.assertSolvers(self.client.solvers(num_qubits=[None, 6]), self.solvers)

        self.assertSolvers(self.client.solvers(num_qubits=[2, 4]), [self.solver1])
        self.assertSolvers(self.client.solvers(num_qubits=[4, None]), [self.solver2])

    def test_range_boolean_combo(self):
        self.assertSolvers(self.client.solvers(num_qubits=3, vfyc=True), [])
        self.assertSolvers(self.client.solvers(num_qubits=[3, None], vfyc=True), [self.solver2])
        self.assertSolvers(self.client.solvers(num_qubits=[None, 4], vfyc=True), [])
        self.assertSolvers(self.client.solvers(num_qubits=[None, 4], flux_biases=False), [self.solver1])
        self.assertSolvers(self.client.solvers(num_qubits=5, flux_biases=True), [self.solver2])


if __name__ == '__main__':
    unittest.main()

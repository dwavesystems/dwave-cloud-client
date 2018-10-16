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

"""
Tests the ability to connect to the SAPI server with the `dwave.cloud.qpu.Client`.

test_mock_solver_loading.py duplicates some of these tests against a mock server.
"""

from __future__ import absolute_import

import unittest
import requests.exceptions

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
class TestTimeouts(unittest.TestCase):
    """Test timeout works for all Client connections."""

    def test_request_timeout(self):
        with self.assertRaises(dwave.cloud.exceptions.RequestTimeout):
            with Client(request_timeout=0.00001, **config) as client:
                client.solvers()

    def test_polling_timeout(self):
        with self.assertRaises(dwave.cloud.exceptions.PollingTimeout):
            with Client(polling_timeout=0.00001, **config) as client:
                client.get_solver().sample_qubo({}).result()


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
                "num_reads_range": [0, 100],
                "parameters": {
                    "num_reads": "Number of samples to return.",
                    "postprocess": "either 'sampling' or 'optimization'"
                }
            },
            "id": "solver1",
            "description": "A test solver 1",
            "status": "online"
        })
        self.solver2 = Solver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1, 2, 3, 4],
                "couplers": [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]],
                "num_qubits": 5,
                "num_reads_range": [0, 200],
                "parameters": {
                    "num_reads": "Number of samples to return.",
                    "flux_biases": "Supported ...",
                    "anneal_schedule": "Supported ..."
                },
                "vfyc": True
            },
            "id": "solver2",
            "description": "A test solver 2"
        })
        self.solver3 = Solver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1],
                "couplers": [[0, 1]],
                "num_qubits": 7,
                "num_reads_range": [0, 1000],
                "parameters": {"num_reads": "Number of samples to return."},
                "vfyc": False
            },
            "id": "c4-sw_solver3",
            "description": "A test of software solver"
        })
        self.solvers = [self.solver1, self.solver2, self.solver3]

        # mock client
        self.client = Client('endpoint', 'token')
        self.client._solvers = {
            self.solver1.id: self.solver1,
            self.solver2.id: self.solver2,
            self.solver3.id: self.solver3
        }
        self.client._all_solvers_ready = True

    def shutDown(self):
        self.client.close()

    def assertSolvers(self, container, members):
        self.assertEqual(set(container), set(members))

    def test_default(self):
        self.assertSolvers(self.client.solvers(), self.solvers)

    def test_online(self):
        self.assertSolvers(self.client.solvers(online=True), self.solvers)
        self.assertSolvers(self.client.solvers(online=False), [])

    def test_qpu_software(self):
        self.assertSolvers(self.client.solvers(qpu=True), [self.solver1, self.solver2])
        self.assertSolvers(self.client.solvers(software=False), [self.solver1, self.solver2])
        self.assertSolvers(self.client.solvers(qpu=False), [self.solver3])
        self.assertSolvers(self.client.solvers(software=True), [self.solver3])

    def test_name(self):
        self.assertSolvers(self.client.solvers(name='solver1'), [self.solver1])
        self.assertSolvers(self.client.solvers(name='solver2'), [self.solver2])
        self.assertSolvers(self.client.solvers(name='solver'), [])
        self.assertSolvers(self.client.solvers(name='olver1'), [])
        self.assertSolvers(self.client.solvers(name__regex='.*1'), [self.solver1])
        self.assertSolvers(self.client.solvers(name__regex='.*[12].*'), [self.solver1, self.solver2])
        self.assertSolvers(self.client.solvers(name__regex='solver[12]'), [self.solver1, self.solver2])
        self.assertSolvers(self.client.solvers(name__regex='^solver(1|2)$'), [self.solver1, self.solver2])

    def test_num_qubits(self):
        self.assertSolvers(self.client.solvers(num_qubits=5), [self.solver2])
        self.assertSolvers(self.client.solvers(num_active_qubits=2), [self.solver3])
        self.assertSolvers(self.client.solvers(num_active_qubits__in=[2, 3]), [self.solver1, self.solver3])

    def test_parameter_availability_check(self):
        self.assertSolvers(self.client.solvers(postprocess__available=True), [self.solver1])
        self.assertSolvers(self.client.solvers(postprocess=True), [self.solver1])
        self.assertSolvers(self.client.solvers(parameters__contains='flux_biases'), [self.solver2])
        self.assertSolvers(self.client.solvers(parameters__contains='num_reads'), self.solvers)

    def test_property_availability_check(self):
        self.assertSolvers(self.client.solvers(vfyc__available=True), [self.solver2, self.solver3])
        self.assertSolvers(self.client.solvers(vfyc__eq=True), [self.solver2])
        self.assertSolvers(self.client.solvers(vfyc=True), [self.solver2])

        # inverse of vfyc=True
        self.assertSolvers(self.client.solvers(vfyc__in=[False, None]), [self.solver1, self.solver3])

        # vfyc unavailable or unadvertized
        self.assertSolvers(self.client.solvers(vfyc__available=False), [self.solver1])
        self.assertSolvers(self.client.solvers(vfyc__eq=False), [self.solver3])
        self.assertSolvers(self.client.solvers(vfyc=False), [self.solver3])

        # non-existing params/props have value of None
        self.assertSolvers(self.client.solvers(vfyc__eq=None), [self.solver1])
        self.assertSolvers(self.client.solvers(vfyc=None), [self.solver1])

    def test_availability_combo(self):
        self.assertSolvers(self.client.solvers(vfyc=False, flux_biases=True), [])
        self.assertSolvers(self.client.solvers(vfyc=True, flux_biases=True), [self.solver2])

    def test_relational_ops(self):
        self.assertSolvers(self.client.solvers(num_qubits=3), [self.solver1])
        self.assertSolvers(self.client.solvers(num_qubits__eq=3), [self.solver1])
        self.assertSolvers(self.client.solvers(num_qubits=4), [])
        self.assertSolvers(self.client.solvers(num_qubits=5), [self.solver2])

        self.assertSolvers(self.client.solvers(num_qubits__gte=2), self.solvers)
        self.assertSolvers(self.client.solvers(num_qubits__gte=5), [self.solver2, self.solver3])
        self.assertSolvers(self.client.solvers(num_qubits__gt=5), [self.solver3])

        self.assertSolvers(self.client.solvers(num_qubits__lte=8), self.solvers)
        self.assertSolvers(self.client.solvers(num_qubits__lte=7), self.solvers)
        self.assertSolvers(self.client.solvers(num_qubits__lt=7), [self.solver1, self.solver2])

    def test_range_ops(self):
        # value within range
        self.assertSolvers(self.client.solvers(num_qubits__within=[3, 7]), self.solvers)
        self.assertSolvers(self.client.solvers(num_qubits__within=[3, 5]), [self.solver1, self.solver2])
        self.assertSolvers(self.client.solvers(num_qubits__within=[2, 4]), [self.solver1])
        self.assertSolvers(self.client.solvers(num_qubits__within=(2, 4)), [self.solver1])
        self.assertSolvers(self.client.solvers(num_qubits__within=(4, 2)), [self.solver1])

        # range within (covered by) range
        self.assertSolvers(self.client.solvers(num_reads_range__within=(0, 500)), [self.solver1, self.solver2])
        self.assertSolvers(self.client.solvers(num_reads_range__within=(1, 500)), [])

        # range covering a value (value included in range)
        self.assertSolvers(self.client.solvers(num_reads_range__covers=0), self.solvers)
        self.assertSolvers(self.client.solvers(num_reads_range__covers=150), [self.solver2, self.solver3])
        self.assertSolvers(self.client.solvers(num_reads_range__covers=550), [self.solver3])
        self.assertSolvers(self.client.solvers(num_reads_range__covers=1000), [self.solver3])
        self.assertSolvers(self.client.solvers(num_reads_range__covers=1001), [])

        # range covering a range
        self.assertSolvers(self.client.solvers(num_reads_range__covers=(10, 90)), self.solvers)
        self.assertSolvers(self.client.solvers(num_reads_range__covers=(110, 200)), [self.solver2, self.solver3])

    def test_membership_ops(self):
        # property contains
        self.assertSolvers(self.client.solvers(supported_problem_types__contains="qubo"), self.solvers)
        self.assertSolvers(self.client.solvers(supported_problem_types__contains="undef"), [])
        self.assertSolvers(self.client.solvers(couplers__contains=[0, 1]), self.solvers)
        self.assertSolvers(self.client.solvers(couplers__contains=[0, 2]), [self.solver1, self.solver2])

        # property in
        self.assertSolvers(self.client.solvers(num_qubits__in=[3, 5]), [self.solver1, self.solver2])
        self.assertSolvers(self.client.solvers(num_qubits__in=[7]), [self.solver3])
        self.assertSolvers(self.client.solvers(num_qubits__in=[]), [])

    def test_regex(self):
        self.assertSolvers(self.client.solvers(num_reads__regex='.*number.*'), [])
        self.assertSolvers(self.client.solvers(num_reads__regex='.*Number.*'), self.solvers)
        self.assertSolvers(self.client.solvers(num_reads__regex='Number.*'), self.solvers)
        self.assertSolvers(self.client.solvers(num_reads__regex='Number'), [])

    def test_range_boolean_combo(self):
        self.assertSolvers(self.client.solvers(num_qubits=3, vfyc=True), [])
        self.assertSolvers(self.client.solvers(num_qubits__gte=3, vfyc=True), [self.solver2])
        self.assertSolvers(self.client.solvers(num_qubits__lte=4, vfyc=True), [])
        self.assertSolvers(self.client.solvers(num_qubits__within=(3, 6), flux_biases=True), [self.solver2])
        self.assertSolvers(self.client.solvers(num_qubits=5, flux_biases=True), [self.solver2])

    def test_anneal_schedule(self):
        self.assertSolvers(self.client.solvers(anneal_schedule__available=True), [self.solver2])
        self.assertSolvers(self.client.solvers(anneal_schedule=True), [self.solver2])


if __name__ == '__main__':
    unittest.main()

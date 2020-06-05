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

import json
import time
import warnings
import unittest
from unittest import mock

import requests.exceptions
from plucky import merge

from dwave.cloud.config import load_config
from dwave.cloud.client import Client
from dwave.cloud.solver import StructuredSolver, UnstructuredSolver
from dwave.cloud.exceptions import (
    SolverAuthenticationError, SolverError, SolverNotFoundError)
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
                client.get_solvers()

    def test_polling_timeout(self):
        with self.assertRaises(dwave.cloud.exceptions.PollingTimeout):
            with Client(polling_timeout=0.00001, **config) as client:
                client.get_solver().sample_qubo({}).result()


@unittest.skipUnless(config, "No live server configuration available.")
class TestRetrieveAnswer(unittest.TestCase):
    """Test loading a problem from its id"""
    def test_retrieve_answer(self):
        with Client(**config) as client:
            solver = client.get_solver()

            h = {v: -1 for v in solver.nodes}

            f = solver.sample_ising(h, {})

            # the id is not set right away
            while f.id is None:
                time.sleep(.01)

            id_ = f.id
        
        with Client(**config) as client:
            # get a "new" client
            f2 = client.retrieve_answer(id_)

            self.assertIn('solutions', f2.result())


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
            for solver in client.get_solvers():
                self.assertEqual(client.get_solver(name=solver.id).id, solver.id)

    def test_load_bad_solvers(self):
        """Try to load a nonexistent solver."""
        with Client(**config) as client:
            with self.assertRaises(SolverNotFoundError):
                client.get_solver("not-a-solver")

    def test_load_any_solver(self):
        """Load a single solver without calling get_solvers."""
        with Client(**config) as client:
            self.assertTrue(client.get_solver(software=True).software)

    def test_get_solver_no_defaults(self):
        """Specialized client returns the correct solver by default."""
        from dwave.cloud import qpu, sw, hybrid

        conf = dict(endpoint=config['endpoint'], token=config['token'])

        with Client(**conf) as base_client:

            with qpu.Client(**conf) as client:
                solvers = base_client.get_solvers(qpu=True)
                if solvers:
                    self.assertEqual(client.get_solver().id, solvers[0].id)
                else:
                    self.assertRaises(SolverError, client.get_solver)

            with sw.Client(**conf) as client:
                solvers = base_client.get_solvers(software=True)
                if solvers:
                    self.assertEqual(client.get_solver().id, solvers[0].id)
                else:
                    self.assertRaises(SolverError, client.get_solver)

            with hybrid.Client(**conf) as client:
                solvers = base_client.get_solvers(hybrid=True)
                if solvers:
                    self.assertEqual(client.get_solver().id, solvers[0].id)
                else:
                    self.assertRaises(SolverError, client.get_solver)


class ClientFactory(unittest.TestCase):
    """Test Client.from_config() factory."""

    def test_default(self):
        conf = {k: k for k in 'endpoint token'.split()}
        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.endpoint, 'endpoint')
                self.assertEqual(client.token, 'token')
                self.assertIsInstance(client, dwave.cloud.client.Client)
                self.assertNotIsInstance(client, dwave.cloud.sw.Client)
                self.assertNotIsInstance(client, dwave.cloud.qpu.Client)
                self.assertNotIsInstance(client, dwave.cloud.hybrid.Client)

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

            with dwave.cloud.hybrid.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.hybrid.Client)

            with dwave.cloud.qpu.Client.from_config(client='base') as client:
                self.assertIsInstance(client, dwave.cloud.client.Client)

            with dwave.cloud.Client.from_config(client='qpu') as client:
                self.assertIsInstance(client, dwave.cloud.qpu.Client)

            with dwave.cloud.Client.from_config(client='sw') as client:
                self.assertIsInstance(client, dwave.cloud.sw.Client)

            with dwave.cloud.Client.from_config(client='hybrid') as client:
                self.assertIsInstance(client, dwave.cloud.hybrid.Client)

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
                    self.assertEqual(client.default_solver, {"name__eq": "solver"})
                    self.assertEqual(client.session.proxies['http'], 'proxy')

                # test fallback is avoided (legacy config skipped)
                self.assertRaises(
                    ValueError, dwave.cloud.Client.from_config, legacy_config_fallback=False)

    def test_solver_features_from_config(self):
        solver_def = {"qpu": True}
        conf = {k: k for k in 'endpoint token'.split()}
        conf.update(solver=json.dumps(solver_def))

        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.default_solver, solver_def)

    def test_solver_name_from_config(self):
        solver_def = {"name__eq": "solver"}
        conf = {k: k for k in 'endpoint token solver'.split()}

        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.default_solver, solver_def)

    def test_solver_features_kwargs_override_config(self):
        new_solver_def = {"software": True}
        conf = {k: k for k in 'endpoint token solver'.split()}

        def load_config(**kwargs):
            return merge(kwargs, conf, op=lambda a, b: a or b)

        with mock.patch("dwave.cloud.client.load_config", load_config):
            with dwave.cloud.Client.from_config(solver=new_solver_def) as client:
                self.assertEqual(client.default_solver, new_solver_def)

    def test_solver_name_overrides_config_features(self):
        conf = {k: k for k in 'endpoint token solver'.split()}
        conf.update(solver=json.dumps({"software": True}))

        def load_config(**kwargs):
            return merge(kwargs, conf, op=lambda a, b: a or b)

        with mock.patch("dwave.cloud.client.load_config", load_config):
            with dwave.cloud.Client.from_config(solver='solver') as client:
                self.assertEqual(client.default_solver, {"name__eq": "solver"})

    def test_headers_from_config(self):
        headers_dict = {"key-1": "value-1", "key-2": "value-2"}
        headers_str = """  key-1:value-1
            key-2: value-2
        """
        conf = dict(token='token', headers=headers_str)

        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertDictEqual(client.headers, headers_dict)

    def test_headers_from_kwargs(self):
        headers_dict = {"key-1": "value-1", "key-2": "value-2"}
        headers_str = "key-2:value-2\nkey-1:value-1"
        conf = dict(token='token')

        def load_config(**kwargs):
            return merge(kwargs, conf, op=lambda a, b: a or b)

        # headers as dict
        with mock.patch("dwave.cloud.client.load_config", load_config):
            with dwave.cloud.Client.from_config(headers=headers_dict) as client:
                self.assertDictEqual(client.headers, headers_dict)

        # headers as str
        with mock.patch("dwave.cloud.client.load_config", load_config):
            with dwave.cloud.Client.from_config(headers=headers_str) as client:
                self.assertDictEqual(client.headers, headers_dict)

    def test_client_cert_from_config(self):
        crt = '/path/to/crt'
        key = '/path/to/key'

        # single file with cert+key
        client_cert = crt
        conf = dict(token='token', client_cert=crt)

        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.client_cert, client_cert)

                session = client.create_session()
                self.assertEqual(session.cert, client_cert)

        # separate cert and key files
        client_cert = (crt, key)
        conf = dict(token='token', client_cert=crt, client_cert_key=key)

        with mock.patch("dwave.cloud.client.load_config", lambda **kwargs: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.client_cert, client_cert)

                session = client.create_session()
                self.assertEqual(session.cert, client_cert)

    def test_client_cert_from_kwargs(self):
        crt = '/path/to/crt'
        key = '/path/to/key'

        def load_config(**kwargs):
            file_conf = dict(token='token')
            return merge(kwargs, file_conf, op=lambda a, b: a or b)

        # single file with cert+key
        client_cert = crt
        conf = dict(client_cert=crt)

        with mock.patch("dwave.cloud.client.load_config", load_config):
            with dwave.cloud.Client.from_config(**conf) as client:
                self.assertEqual(client.client_cert, client_cert)

        # separate cert and key files
        client_cert = (crt, key)
        conf = dict(client_cert=crt, client_cert_key=key)

        with mock.patch("dwave.cloud.client.load_config", load_config):
            with dwave.cloud.Client.from_config(**conf) as client:
                self.assertEqual(client.client_cert, client_cert)

        # client_cert as tuple (direct `requests` format)
        client_cert = (crt, key)
        conf = dict(client_cert=client_cert)

        with mock.patch("dwave.cloud.client.load_config", load_config):
            with dwave.cloud.Client.from_config(**conf) as client:
                self.assertEqual(client.client_cert, client_cert)


class FeatureBasedSolverSelection(unittest.TestCase):
    """Test Client.get_solvers(**filters)."""

    def setUp(self):
        # mock solvers
        self.qpu1 = StructuredSolver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1, 2],
                "couplers": [[0, 1], [0, 2], [1, 2]],
                "num_qubits": 3,
                "num_reads_range": [0, 100],
                "parameters": {
                    "num_reads": "Number of samples to return.",
                    "postprocess": "either 'sampling' or 'optimization'"
                },
                "topology": {
                    "type": "chimera",
                    "shape": [16, 16, 4]
                },
                "category": "qpu",
                "tags": ["lower_noise"]
            },
            "id": "qpu1",
            "description": "QPU Chimera solver",
            "status": "online"
        })
        self.qpu2 = StructuredSolver(client=None, data={
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
                "topology": {
                    "type": "pegasus",
                    "shape": [6, 6, 12]
                },
                "category": "qpu",
                "vfyc": True
            },
            "id": "qpu2",
            "description": "QPU Pegasus solver"
        })
        self.software = StructuredSolver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1],
                "couplers": [[0, 1]],
                "num_qubits": 7,
                "num_reads_range": [0, 1000],
                "parameters": {"num_reads": "Number of samples to return."},
                "vfyc": False,
                "topology": {
                    "type": "chimera",
                    "shape": [4, 4, 4]
                },
                "category": "software",
                # the following are only present in this solver
                "some_set": [1, 2],
                "some_range": [1, 2],
                "some_string": "x",
                "tags": ["tag"]
            },
            "id": "sw_solver1",
            "description": "Software solver",
            "avg_load": 0.7
        })
        self.hybrid = UnstructuredSolver(client=None, data={
            "properties": {
                "supported_problem_types": ["bqm"],
                "maximum_number_of_variables": 10000,
                "maximum_time_limit_hrs": 24.0,
                "minimum_time_limit": [[1, 3.0], [1024, 3.0], [4096, 10.0], [10000, 40.0]],
                "quota_conversion_rate": 20,
                "parameters": {
                    "time_limit": ""
                },
                "category": "hybrid",
            },
            "id": "hybrid_v1",
            "description": "Hybrid solver"
        })

        self.qpu_solvers = [self.qpu1, self.qpu2]
        self.software_solvers = [self.software]
        self.hybrid_solvers = [self.hybrid]

        self.structured_solvers = self.qpu_solvers + self.software_solvers
        self.unstructured_solvers = self.hybrid_solvers

        self.solvers = self.structured_solvers + self.unstructured_solvers

        # mock client
        self.client = Client('endpoint', 'token')
        self.client._fetch_solvers = lambda **kw: self.solvers

    def shutDown(self):
        self.client.close()

    def assertSolvers(self, container, members):
        self.assertEqual(set(container), set(members))

    def test_default(self):
        self.assertSolvers(self.client.get_solvers(), self.solvers)

    def test_online(self):
        self.assertSolvers(self.client.get_solvers(online=True), self.solvers)
        self.assertSolvers(self.client.get_solvers(online=False), [])

    def test_qpu_hybrid_software(self):
        self.assertSolvers(self.client.get_solvers(qpu=True), self.qpu_solvers)
        self.assertSolvers(self.client.get_solvers(qpu=False), self.software_solvers + self.hybrid_solvers)
        self.assertSolvers(self.client.get_solvers(software=True), self.software_solvers)
        self.assertSolvers(self.client.get_solvers(software=False), self.qpu_solvers + self.hybrid_solvers)
        self.assertSolvers(self.client.get_solvers(hybrid=True), self.hybrid_solvers)
        self.assertSolvers(self.client.get_solvers(hybrid=False), self.qpu_solvers + self.software_solvers)

    def test_category(self):
        self.assertSolvers(self.client.get_solvers(category='qpu'), self.qpu_solvers)
        self.assertSolvers(self.client.get_solvers(category='software'), self.software_solvers)
        self.assertSolvers(self.client.get_solvers(category='hybrid'), self.hybrid_solvers)

    def test_name(self):
        self.assertSolvers(self.client.get_solvers(name='qpu1'), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(name='qpu2'), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(name='solver'), [])
        self.assertSolvers(self.client.get_solvers(name='olver1'), [])
        self.assertSolvers(self.client.get_solvers(name__regex='.*1'), [self.qpu1, self.software, self.hybrid])
        self.assertSolvers(self.client.get_solvers(name__regex='.*_v1'), [self.hybrid])
        self.assertSolvers(self.client.get_solvers(name__regex='.*[12].*'), self.solvers)
        self.assertSolvers(self.client.get_solvers(name__regex='qpu[12]'), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(name__regex='^qpu(1|2)$'), [self.qpu1, self.qpu2])

    def test_num_qubits(self):
        self.assertSolvers(self.client.get_solvers(num_qubits=5), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_active_qubits=2), [self.software])
        self.assertSolvers(self.client.get_solvers(num_active_qubits__in=[2, 3]), [self.qpu1, self.software])

    def test_lower_noise_derived_property(self):
        self.assertSolvers(self.client.get_solvers(lower_noise=True), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(lower_noise=False), [self.qpu2, self.software])

    def test_parameter_availability_check(self):
        self.assertSolvers(self.client.get_solvers(postprocess__available=True), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(postprocess=True), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(parameters__contains='flux_biases'), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(parameters__contains='num_reads'), self.structured_solvers)

    def test_property_availability_check(self):
        self.assertSolvers(self.client.get_solvers(vfyc__available=True), [self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(vfyc__eq=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(vfyc=True), [self.qpu2])

        # inverse of vfyc=True
        self.assertSolvers(self.client.get_solvers(vfyc__in=[False, None]), [self.qpu1, self.software, self.hybrid])

        # vfyc unavailable or unadvertized
        self.assertSolvers(self.client.get_solvers(vfyc__available=False), [self.qpu1, self.hybrid])
        self.assertSolvers(self.client.get_solvers(vfyc__eq=False), [self.software])
        self.assertSolvers(self.client.get_solvers(vfyc=False), [self.software])

        # non-existing params/props have value of None
        self.assertSolvers(self.client.get_solvers(vfyc__eq=None), [self.qpu1, self.hybrid])
        self.assertSolvers(self.client.get_solvers(vfyc=None), [self.qpu1, self.hybrid])

    def test_availability_combo(self):
        self.assertSolvers(self.client.get_solvers(vfyc=False, flux_biases=True), [])
        self.assertSolvers(self.client.get_solvers(vfyc=True, flux_biases=True), [self.qpu2])

    def test_relational_ops(self):
        self.assertSolvers(self.client.get_solvers(num_qubits=3), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits__eq=3), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits=4), [])
        self.assertSolvers(self.client.get_solvers(num_qubits=5), [self.qpu2])

        self.assertSolvers(self.client.get_solvers(num_qubits__gte=2), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__gte=5), [self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(num_qubits__gt=5), [self.software])

        self.assertSolvers(self.client.get_solvers(num_qubits__lte=8), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__lte=7), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__lt=7), [self.qpu1, self.qpu2])

        # skip solver if LHS value not defined (None)
        self.assertSolvers(self.client.get_solvers(avg_load__gt=0), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__gte=0), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__lt=1), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__lte=1), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load=0.7), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__eq=0.7), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load=None), [self.qpu1, self.qpu2, self.hybrid])
        self.assertSolvers(self.client.get_solvers(avg_load__eq=None), [self.qpu1, self.qpu2, self.hybrid])

    def test_range_ops(self):
        # value within range
        self.assertSolvers(self.client.get_solvers(num_qubits__within=[3, 7]), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__within=[3, 5]), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=[2, 4]), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=(2, 4)), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=(4, 2)), [self.qpu1])

        # range within (covered by) range
        self.assertSolvers(self.client.get_solvers(num_reads_range__within=(0, 500)), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_reads_range__within=(1, 500)), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_range__within=[0, 2]), [self.software])

        # range covering a value (value included in range)
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=0), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=150), [self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=550), [self.software])
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=1000), [self.software])
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=1001), [])

        # range covering a range
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=(10, 90)), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=(110, 200)), [self.qpu2, self.software])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_range__covers=1.5), [self.software])

    def test_membership_ops(self):
        # property contains
        self.assertSolvers(self.client.get_solvers(supported_problem_types__contains="qubo"), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__contains="undef"), [])
        self.assertSolvers(self.client.get_solvers(supported_problem_types__contains="bqm"), self.unstructured_solvers)
        self.assertSolvers(self.client.get_solvers(couplers__contains=[0, 1]), [self.qpu1, self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(couplers__contains=[0, 2]), [self.qpu1, self.qpu2])

        # property in
        self.assertSolvers(self.client.get_solvers(num_qubits__in=[3, 5]), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits__in=[7]), [self.software])
        self.assertSolvers(self.client.get_solvers(num_qubits__in=[]), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_set__contains=1), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__in=[None]), [self.qpu1, self.qpu2, self.hybrid])
        self.assertSolvers(self.client.get_solvers(avg_load__in=[None, 0.7]), self.solvers)

    def test_set_ops(self):
        # property issubset
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=("qubo", "ising", "bqm", "other")), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=["qubo", "ising", "bqm"]), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=["bqm", "ising", "qubo"]), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset={"ising", "qubo", "bqm"}), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=("unicorn", "ising", "other")), [])

        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset={"ising", "qubo"}), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset={"bqm", "other"}), self.unstructured_solvers)

        # property issuperset
        self.assertSolvers(self.client.get_solvers(qubits__issuperset={0, 1}), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(qubits__issuperset={1, 2}), [self.qpu1, self.qpu2])

        # unhashable types
        self.assertSolvers(self.client.get_solvers(couplers__issuperset=[[0, 1]]), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1)}), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1), (0, 2)}), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1), (0, 2), (2, 3)}), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1), (0, 2), (2, 3), (0, 5)}), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_set__issubset={0, 1, 2}), [self.software])
        self.assertSolvers(self.client.get_solvers(some_set__issuperset={1}), [self.software])

    def test_regex(self):
        self.assertSolvers(self.client.get_solvers(num_reads__regex='.*number.*'), [])
        self.assertSolvers(self.client.get_solvers(num_reads__regex='.*Number.*'), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads__regex='Number.*'), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads__regex='Number'), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_string__regex='x'), [self.software])

    def test_range_boolean_combo(self):
        self.assertSolvers(self.client.get_solvers(num_qubits=3, vfyc=True), [])
        self.assertSolvers(self.client.get_solvers(num_qubits__gte=3, vfyc=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits__lte=4, vfyc=True), [])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=(3, 6), flux_biases=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits=5, flux_biases=True), [self.qpu2])

    def test_nested_properties_leaf_lookup(self):
        self.assertSolvers(self.client.get_solvers(topology__type="chimera"), [self.qpu1, self.software])
        self.assertSolvers(self.client.get_solvers(topology__type="pegasus"), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(topology__type__eq="pegasus"), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(topology__shape=[6,6,12]), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(topology__type="chimera", topology__shape__contains=16), [self.qpu1])

    def test_nested_properties_intermediate_key_lookup(self):
        self.assertSolvers(self.client.get_solvers(topology__contains="shape"), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(topology={"type": "pegasus", "shape": [6, 6, 12]}), [self.qpu2])

    def test_anneal_schedule(self):
        self.assertSolvers(self.client.get_solvers(anneal_schedule__available=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(anneal_schedule=True), [self.qpu2])

    def test_solvers_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.client.solvers()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_order_by_edgecases(self):
        # default: sort by avg_load
        self.assertEqual(self.client.get_solvers(), [self.software, self.qpu1, self.qpu2, self.hybrid])

        # explicit no sort
        self.assertEqual(self.client.get_solvers(order_by=None), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by=''), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by=False), self.solvers)

        # reverse without sorting
        self.assertEqual(self.client.get_solvers(order_by='-'), list(reversed(self.solvers)))

        # invalid type of `order_by`
        with self.assertRaises(TypeError):
            self.client.get_solvers(order_by=1)
        with self.assertRaises(TypeError):
            self.client.get_solvers(order_by=list)

    def test_order_by_string(self):
        # sort by Solver inferred properties
        self.assertEqual(self.client.get_solvers(order_by='id'), [self.hybrid, self.qpu1, self.qpu2, self.software])
        self.assertEqual(self.client.get_solvers(order_by='qpu'), [self.software, self.hybrid, self.qpu1, self.qpu2])
        self.assertEqual(self.client.get_solvers(order_by='num_qubits'), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by='num_active_qubits'), [self.software, self.qpu1, self.qpu2, self.hybrid])

        # sort by solver property
        self.assertEqual(self.client.get_solvers(order_by='properties.num_qubits'), self.solvers)

        # sort (and reverse sort) by upper bound of a range property
        self.assertEqual(self.client.get_solvers(order_by='properties.num_reads_range[1]'), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by='-properties.num_reads_range[1]'), [self.hybrid, self.software, self.qpu2, self.qpu1])

        # check solvers with None values for key end up last
        self.assertEqual(self.client.get_solvers(order_by='properties.vfyc'), [self.software, self.qpu2, self.qpu1, self.hybrid])
        self.assertEqual(self.client.get_solvers(order_by='-properties.vfyc'), [self.hybrid, self.qpu1, self.qpu2, self.software])

        # check invalid keys don't fail, and effectively don't sort the list
        self.assertEqual(self.client.get_solvers(order_by='non_existing_key'), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by='-non_existing_key'), list(reversed(self.solvers)))

    def test_order_by_callable(self):
        # sort by Solver inferred properties
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.id), [self.hybrid, self.qpu1, self.qpu2, self.software])
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.avg_load), [self.software, self.qpu1, self.qpu2, self.hybrid])

        # sort by solver property
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.properties.get('num_qubits')), self.solvers)

        # sort None`s last (here: False, True, None)
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.properties.get('vfyc')), [self.software, self.qpu2, self.qpu1, self.hybrid])

        # test no sort
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: None), self.solvers)


if __name__ == '__main__':
    unittest.main()

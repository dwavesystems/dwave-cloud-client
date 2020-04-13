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

"""Try to load solver data from mock servers."""

import os
import json
import unittest
from unittest import mock

import requests
import requests_mock

from dwave.cloud.client import Client, Solver
from dwave.cloud.qpu import Client as QPUClient
from dwave.cloud.sw import Client as SoftwareClient
from dwave.cloud.exceptions import (
    SolverPropertyMissingError, ConfigFileReadError, ConfigFileParseError,
    SolverError, SolverNotFoundError)
from dwave.cloud.config import legacy_load_config, load_config
from dwave.cloud.testing import iterable_mock_open


url = 'https://dwavesys.com'
token = 'abc123abc123abc123abc123abc123abc123'
solver_name = 'test_solver'
second_solver_name = 'test_solver2'

bad_url = 'https://not-a-subdomain.dwavesys.com'
bad_token = '------------------------'


def structured_solver_data(id_, cat='qpu', incomplete=False):
    """Return string describing a single solver."""
    obj = {
        "properties": {
            "supported_problem_types": ["qubo", "ising"],
            "qubits": [1, 2, 3],
            "couplers": [[1, 2], [1, 3], [2, 3]],
            "num_qubits": 3,
            "category": cat,
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "id": id_,
        "description": "A test solver",
        "status": "ONLINE"
    }

    if incomplete:
        del obj['properties']['supported_problem_types']

    return json.dumps(obj)

def solver_object(id_, cat='qpu', incomplete=False):
    return Solver(client=None, data=json.loads(structured_solver_data(id_, cat, incomplete)))


# Define the endpoints
all_solver_url = '{}/solvers/remote/'.format(url)
solver1_url = '{}/solvers/remote/{}/'.format(url, solver_name)
solver2_url = '{}/solvers/remote/{}/'.format(url, second_solver_name)


def setup_server(m):
    """Add endpoints to the server."""
    # Content strings
    first_solver_response = structured_solver_data(solver_name)
    second_solver_response = structured_solver_data(second_solver_name)
    two_solver_response = '[' + first_solver_response + ',' + second_solver_response + ']'

    # Setup the server
    headers = {'X-Auth-Token': token}
    m.get(requests_mock.ANY, status_code=404)
    m.get(all_solver_url, status_code=403, request_headers={})
    m.get(solver1_url, status_code=403, request_headers={})
    m.get(solver2_url, status_code=403, request_headers={})
    m.get(all_solver_url, text=two_solver_response, request_headers=headers)
    m.get(solver1_url, text=first_solver_response, request_headers=headers)
    m.get(solver2_url, text=second_solver_response, request_headers=headers)


class MockConnectivityTests(unittest.TestCase):
    """Test connecting some related failure modes."""

    def test_bad_url(self):
        """Connect with a bad URL."""
        with requests_mock.mock() as m:
            setup_server(m)
            with self.assertRaises(IOError):
                with Client(bad_url, token) as client:
                    client.get_solvers()

    def test_bad_token(self):
        """Connect with a bad token."""
        with requests_mock.mock() as m:
            setup_server(m)
            with self.assertRaises(IOError):
                with Client(url, bad_token) as client:
                    client.get_solvers()

    def test_good_connection(self):
        """Connect with a valid URL and token."""
        with requests_mock.mock() as m:
            setup_server(m)
            with Client(url, token) as client:
                self.assertTrue(len(client.get_solvers()) > 0)


class MockSolverLoading(unittest.TestCase):
    """Test loading solvers in a few different configurations.

    Note:
        A mock server does not test authentication.

    Expect three responses from the server for /solvers/*:
        - A single solver when a single solver is requested
        - A list of single solver objects when all solvers are requested
        - A 404 error when the requested solver does not exist
    An additional condition to test for is that the server may have configured
    a given solver such that it does not provide all the required information
    about it.
    """

    def test_load_solver(self):
        """Load a single solver."""
        with requests_mock.mock() as m:
            setup_server(m)

            # test default, cached solver get
            with Client(url, token) as client:
                # fetch solver not present in cache
                solver = client.get_solver(solver_name)
                self.assertEqual(solver.id, solver_name)

                # modify cached solver and re-fetch it
                solver.id = 'different-solver'
                # cached solver name doesn't match, so it won't be returned
                with self.assertRaises(SolverError):
                    client.get_solver(solver_name, refresh=False)
                # cache is refreshed?
                self.assertEqual(client.get_solver(solver_name, refresh=True).id, solver_name)

    def test_load_all_solvers(self):
        """Load the list of solver names."""

        def spoof_cache(client, clear_val=False, clear_expires=False):
            cache = client._fetch_solvers._cache
            for args in cache:
                if clear_val:
                    cache[args]['val'] = []
                if clear_expires:
                    cache[args]['expires'] = 0

        with requests_mock.mock() as m:
            setup_server(m)

            # test default case, fetch all solvers for the first time
            with Client(url, token) as client:
                solvers = client.get_solvers()

                self.assertEqual(len(solvers), 2)

                # test default refresh
                spoof_cache(client, clear_expires=True)
                self.assertEqual(len(client.get_solvers()), 2)      # should refresh

                # test no refresh
                spoof_cache(client, clear_val=True)
                self.assertEqual(len(client.get_solvers(refresh=False)), 0)     # should not refresh

                # test refresh
                self.assertEqual(len(client.get_solvers(refresh=True)), 2)      # should refresh

    def test_load_missing_solver(self):
        """Try to load a solver that does not exist."""
        with requests_mock.mock() as m:
            m.get(requests_mock.ANY, status_code=404)
            with Client(url, token) as client:
                with self.assertRaises(SolverNotFoundError):
                    client.get_solver(solver_name)

    def test_load_solver_missing_data(self):
        """Try to load a solver that has incomplete data."""
        with requests_mock.mock() as m:
            m.get(solver1_url, text=structured_solver_data(solver_name, incomplete=True))
            with Client(url, token) as client:
                with self.assertRaises(SolverNotFoundError):
                    client.get_solver(solver_name)

    def test_load_solver_broken_response(self):
        """Try to load a solver for which the server has returned a truncated response."""
        with requests_mock.mock() as m:
            body = structured_solver_data(solver_name)
            m.get(solver1_url, text=body[0:len(body)//2])
            with Client(url, token) as client:
                with self.assertRaises(ValueError):
                    client.get_solver(solver_name)

    def test_solver_filtering_in_client(self):
        # base client
        self.assertTrue(Client.is_solver_handled(solver_object('test', 'qpu')))
        self.assertTrue(Client.is_solver_handled(solver_object('test', 'software')))
        self.assertTrue(Client.is_solver_handled(solver_object('test', 'hybrid')))
        self.assertTrue(Client.is_solver_handled(solver_object('test', 'whatever')))
        self.assertTrue(Client.is_solver_handled(None))
        # qpu client
        self.assertTrue(QPUClient.is_solver_handled(solver_object('test', 'qpu')))
        self.assertFalse(QPUClient.is_solver_handled(solver_object('test', 'software')))
        self.assertFalse(QPUClient.is_solver_handled(solver_object('test', 'hybrid')))
        self.assertFalse(QPUClient.is_solver_handled(solver_object('test', 'whatever')))
        self.assertFalse(QPUClient.is_solver_handled(None))
        # sw client
        self.assertFalse(SoftwareClient.is_solver_handled(solver_object('test', 'qpu')))
        self.assertTrue(SoftwareClient.is_solver_handled(solver_object('test', 'software')))
        self.assertFalse(SoftwareClient.is_solver_handled(solver_object('test', 'hybrid')))
        self.assertFalse(SoftwareClient.is_solver_handled(solver_object('test', 'whatever')))
        self.assertFalse(SoftwareClient.is_solver_handled(None))

    def test_solver_feature_properties(self):
        self.assertTrue(solver_object('solver', 'qpu').qpu)
        self.assertTrue(solver_object('solver', 'QPU').qpu)
        self.assertFalse(solver_object('solver', 'qpu').software)
        self.assertFalse(solver_object('solver', 'qpu').hybrid)
        self.assertFalse(solver_object('solver', 'software').qpu)
        self.assertTrue(solver_object('solver', 'software').software)
        self.assertFalse(solver_object('solver', 'software').hybrid)
        self.assertTrue(solver_object('solver', 'hybrid').hybrid)
        self.assertFalse(solver_object('solver', 'hybrid').qpu)
        self.assertFalse(solver_object('solver', 'hybrid').software)

        self.assertFalse(solver_object('solver').is_vfyc)
        self.assertEqual(solver_object('solver').num_qubits, 3)
        self.assertFalse(solver_object('solver').has_flux_biases)

        # test .num_qubits vs .num_actual_qubits
        data = json.loads(structured_solver_data('test'))
        data['properties']['num_qubits'] = 7
        solver = Solver(None, data)
        self.assertEqual(solver.num_qubits, 7)
        self.assertEqual(solver.num_active_qubits, 3)

        # test .is_vfyc
        data = json.loads(structured_solver_data('test'))
        data['properties']['vfyc'] = 'error'
        self.assertFalse(Solver(None, data).is_vfyc)
        data['properties']['vfyc'] = True
        self.assertTrue(Solver(None, data).is_vfyc)

        # test .has_flux_biases
        self.assertFalse(Solver(None, data).has_flux_biases)
        data['properties']['parameters']['flux_biases'] = '...'
        self.assertTrue(Solver(None, data).has_flux_biases)

        # test .has_anneal_schedule
        self.assertFalse(Solver(None, data).has_anneal_schedule)
        data['properties']['parameters']['anneal_schedule'] = '...'
        self.assertTrue(Solver(None, data).has_anneal_schedule)

        # test `.online` property
        self.assertTrue(solver_object('solver').online)
        data = json.loads(structured_solver_data('test'))
        data['status'] = 'offline'
        self.assertFalse(Solver(None, data).online)
        del data['status']
        self.assertTrue(Solver(None, data).online)


class RequestEvent(Exception):
    """Throws exception when mocked client submits an HTTP request."""

    def __init__(self, method, url, *args, **kwargs):
        """Return the URL of the request with the exception for test verification."""
        self.method = method
        self.url = url
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def request(session, method, url, *args, **kwargs):
        """Callback function that can be inserted into a mock."""
        raise RequestEvent(method, url, *args, **kwargs)


legacy_config_body = """
prod|file-prod-url,file-prod-token
alpha|file-alpha-url,file-alpha-token,,alpha-solver
"""

config_body = """
[prod]
endpoint = http://file-prod.url
token = file-prod-token

[alpha]
endpoint = http://file-alpha.url
token = file-alpha-token
solver = alpha-solver

[custom]
endpoint = http://httpbin.org/delay/10
token = 123
permissive_ssl = True
request_timeout = 15
polling_timeout = 180
"""


# patch the new config loading mechanism, to test only legacy config loading
@mock.patch("dwave.cloud.config.get_configfile_paths", lambda: [])
# patch Session.request to raise RequestEvent with the URL requested
@mock.patch.object(requests.Session, 'request', RequestEvent.request)
class MockLegacyConfiguration(unittest.TestCase):
    """Ensure that the precedence of configuration sources is followed."""

    endpoint = "http://custom-endpoint.url"

    def setUp(self):
        # clear `config_load`-relevant environment variables before testing, so
        # we only need to patch the ones that we are currently testing
        for key in frozenset(os.environ.keys()):
            if key.startswith("DWAVE_") or key.startswith("DW_INTERNAL__"):
                os.environ.pop(key, None)

    def test_explicit_only(self):
        """Specify information only through function arguments."""
        with Client.from_config(endpoint=self.endpoint, token='arg-token') as client:
            try:
                client.get_solver('arg-solver')
            except RequestEvent as event:
                self.assertTrue(event.url.startswith(self.endpoint))
                return
        self.fail()

    def test_nonexisting_file(self):
        """With no values set, we should get an error when trying to create Client."""
        with self.assertRaises(ConfigFileReadError):
            with Client.from_config(config_file='nonexisting', legacy_config_fallback=False) as client:
                pass

    def test_explicit_with_file(self):
        """With arguments and a config file, the config file should be ignored."""
        with mock.patch("dwave.cloud.config.open", iterable_mock_open(config_body), create=True):
            with Client.from_config(endpoint=self.endpoint, token='arg-token') as client:
                try:
                    client.get_solver('arg-solver')
                except RequestEvent as event:
                    self.assertTrue(event.url.startswith(self.endpoint))
                    return
                self.fail()

    def test_only_file(self):
        """With no arguments or environment variables, the default connection from the config file should be used."""
        with mock.patch("dwave.cloud.config.open", iterable_mock_open(config_body), create=True):
            with Client.from_config('config_file') as client:
                try:
                    client.get_solver('arg-solver')
                except RequestEvent as event:
                    self.assertTrue(event.url.startswith('http://file-prod.url/'))
                    return
                self.fail()

    def test_only_file_key(self):
        """If give a name from the config file the proper URL should be loaded."""
        with mock.patch("dwave.cloud.config.open", iterable_mock_open(config_body), create=True):
            with mock.patch("dwave.cloud.config.get_configfile_paths", lambda *x: ['file']):
                with Client.from_config(profile='alpha') as client:
                    try:
                        client.get_solver('arg-solver')
                    except RequestEvent as event:
                        self.assertTrue(event.url.startswith('http://file-alpha.url/'))
                        return
                    self.fail()

    def test_env_with_file_set(self):
        """With environment variables and a config file, the config file should be ignored."""
        with mock.patch("dwave.cloud.config.open", iterable_mock_open(legacy_config_body), create=True):
            with mock.patch.dict(os.environ, {'DW_INTERNAL__HTTPLINK': 'http://env.url', 'DW_INTERNAL__TOKEN': 'env-token'}):
                with Client.from_config(config_file=False, legacy_config_fallback=True) as client:
                    try:
                        client.get_solver('arg-solver')
                    except RequestEvent as event:
                        self.assertTrue(event.url.startswith('http://env.url/'))
                        return
                    self.fail()

    def test_env_args_set(self):
        """With arguments and environment variables, the environment variables should be ignored."""
        with mock.patch.dict(os.environ, {'DW_INTERNAL__HTTPLINK': 'http://env.url', 'DW_INTERNAL__TOKEN': 'env-token'}):
            with Client.from_config(endpoint=self.endpoint, token='args-token') as client:
                try:
                    client.get_solver('arg-solver')
                except RequestEvent as event:
                    self.assertTrue(event.url.startswith(self.endpoint))
                    return
                self.fail()

    def test_file_read_error(self):
        """On config file read error, we should fail with `ConfigFileReadError`,
        but only if .dwrc actually exists on disk."""
        with mock.patch("dwave.cloud.config.open", side_effect=OSError, create=True):
            with mock.patch("os.path.exists", lambda fn: True):
                self.assertRaises(ConfigFileReadError, legacy_load_config)


class MockConfiguration(unittest.TestCase):

    def test_custom_options(self):
        """Test custom options (request_timeout, polling_timeout, permissive_ssl) are propagated to Client."""
        request_timeout = 15
        polling_timeout = 180

        with mock.patch("dwave.cloud.config.open", iterable_mock_open(config_body), create=True):
            with Client.from_config('config_file', profile='custom') as client:
                # check permissive_ssl and timeouts custom params passed-thru
                self.assertFalse(client.session.verify)
                self.assertEqual(client.request_timeout, request_timeout)
                self.assertEqual(client.polling_timeout, polling_timeout)

                # verify client uses those properly
                def mock_send(*args, **kwargs):
                    self.assertEqual(kwargs.get('timeout'), request_timeout)
                    response = requests.Response()
                    response.status_code = 200
                    response._content = b'{}'
                    return response

                with mock.patch("requests.adapters.HTTPAdapter.send", mock_send):
                    client.get_solvers()

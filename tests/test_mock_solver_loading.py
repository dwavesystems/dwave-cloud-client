"""Try to load solver data from mock servers."""
from __future__ import division, absolute_import, print_function, unicode_literals

import os
import json
import unittest
import requests_mock

from dwave.cloud.qpu import Client, Solver
from dwave.cloud.exceptions import (
    InvalidAPIResponseError, ConfigFileReadError, ConfigFileParseError)
from dwave.cloud.config import legacy_load_config, load_config
from dwave.cloud.testing import mock, iterable_mock_open


url = 'https://dwavesys.com'
token = 'abc123abc123abc123abc123abc123abc123'
solver_name = 'test_solver'
second_solver_name = 'test_solver2'

bad_url = 'https://not-a-subdomain.dwavesys.com'
bad_token = '------------------------'


def solver_data(id_, incomplete=False):
    """Return string describing a single solver."""
    obj = {
        "properties": {
            "supported_problem_types": ["qubo", "ising"],
            "qubits": [0, 1, 2],
            "couplers": [[0, 1], [0, 2], [1, 2]],
            "num_qubits": 3,
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "id": id_,
        "description": "A test solver"
    }

    if incomplete:
        del obj['properties']['parameters']

    return json.dumps(obj)

def solver_object(id_, incomplete=False):
    return Solver(client=None, data=json.loads(solver_data(id_, incomplete)))


# Define the endpoinds
all_solver_url = '{}/solvers/remote/'.format(url)
solver1_url = '{}/solvers/remote/{}/'.format(url, solver_name)
solver2_url = '{}/solvers/remote/{}/'.format(url, second_solver_name)


def setup_server(m):
    """Add endpoints to the server."""
    # Content strings
    first_solver_response = solver_data(solver_name)
    second_solver_response = solver_data(second_solver_name)
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
                solver = client.get_solver(solver_name)
                self.assertEqual(solver.id, solver_name)

                # fetch solver not present in cache
                client._solvers = {}
                self.assertEqual(client.get_solver(solver_name).id, solver_name)

                # re-fetch solver present in cache
                solver = client.get_solver(solver_name)
                solver.id = 'different-solver'
                self.assertEqual(client.get_solver(solver_name, refresh=True).id, solver_name)

    def test_load_all_solvers(self):
        """Load the list of solver names."""
        with requests_mock.mock() as m:
            setup_server(m)

            # test default case, fetch all solvers for the first time
            with Client(url, token) as client:
                self.assertEqual(len(client.get_solvers()), 2)

                # test default refresh
                client._solvers = {}
                self.assertEqual(len(client.get_solvers()), 0)

                # test no refresh
                client._solvers = {}
                self.assertEqual(len(client.get_solvers(refresh=False)), 0)

                # test refresh
                client._solvers = {}
                self.assertEqual(len(client.get_solvers(refresh=True)), 2)

    def test_load_missing_solver(self):
        """Try to load a solver that does not exist."""
        with requests_mock.mock() as m:
            m.get(requests_mock.ANY, status_code=404)
            with Client(url, token) as client:
                with self.assertRaises(KeyError):
                    client.get_solver(solver_name)

    def test_load_solver_missing_data(self):
        """Try to load a solver that has incomplete data."""
        with requests_mock.mock() as m:
            m.get(solver1_url, text=solver_data(solver_name, True))
            with Client(url, token) as client:
                with self.assertRaises(InvalidAPIResponseError):
                    client.get_solver(solver_name)

    def test_load_solver_broken_response(self):
        """Try to load a solver for which the server has returned a truncated response."""
        with requests_mock.mock() as m:
            body = solver_data(solver_name)
            m.get(solver1_url, text=body[0:len(body)//2])
            with Client(url, token) as client:
                with self.assertRaises(ValueError):
                    client.get_solver(solver_name)

    def test_solver_filtering_in_client(self):
        self.assertTrue(Client.is_solver_handled(solver_object('test')))
        self.assertFalse(Client.is_solver_handled(solver_object('c4-sw_')))
        self.assertFalse(Client.is_solver_handled(None))

    def test_solver_feature_properties(self):
        self.assertTrue(solver_object('dw2000').is_qpu)
        self.assertFalse(solver_object('dw2000').is_software)
        self.assertFalse(solver_object('c4-sw_x').is_qpu)
        self.assertTrue(solver_object('c4-sw_x').is_software)

        self.assertFalse(solver_object('dw2000').is_vfyc)
        self.assertEqual(solver_object('dw2000').num_qubits, 3)
        self.assertFalse(solver_object('dw2000').has_flux_biases)

        data = json.loads(solver_data('test'))
        data['properties']['vfyc'] = 'error'
        self.assertFalse(Solver(None, data).is_vfyc)
        data['properties']['vfyc'] = True
        self.assertTrue(Solver(None, data).is_vfyc)

        data['properties']['parameters']['flux_biases'] = '...'
        self.assertTrue(Solver(None, data).has_flux_biases)


class GetEvent(Exception):
    """Throws exception when mocked client submits an HTTP GET request."""

    def __init__(self, url):
        """Return the URL of the request with the exception for test verification."""
        self.url = url

    @staticmethod
    def handle(path, *args, **kwargs):
        """Callback function that can be inserted into a mock."""
        raise GetEvent(path)


legacy_config_body = """
prod|file-prod-url,file-prod-token
alpha|file-alpha-url,file-alpha-token,,alpha-solver
"""

config_body = """
[prod]
endpoint = file-prod-url
token = file-prod-token

[alpha]
endpoint = file-alpha-url
token = file-alpha-token
solver = alpha-solver
"""


# patch the new config loading mechanism, to test only legacy config loading
@mock.patch("dwave.cloud.config.get_configfile_paths", lambda: [])
class MockLegacyConfiguration(unittest.TestCase):
    """Ensure that the precedence of configuration sources is followed."""

    def setUp(self):
        # clear `config_load`-relevant environment variables before testing, so
        # we only need to patch the ones that we are currently testing
        for key in frozenset(os.environ.keys()):
            if key.startswith("DWAVE_") or key.startswith("DW_INTERNAL__"):
                os.environ.pop(key, None)

    def test_explicit_only(self):
        """Specify information only through function arguments."""
        with Client.from_config(endpoint='arg-url', token='arg-token') as client:
            client.session.get = GetEvent.handle
            try:
                client.get_solver('arg-solver')
            except GetEvent as event:
                self.assertTrue(event.url.startswith('arg-url'))
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
            with Client.from_config(endpoint='arg-url', token='arg-token') as client:
                client.session.get = GetEvent.handle
                try:
                    client.get_solver('arg-solver')
                except GetEvent as event:
                    self.assertTrue(event.url.startswith('arg-url'))
                    return
                self.fail()

    def test_only_file(self):
        """With no arguments or environment variables, the default connection from the config file should be used."""
        with mock.patch("dwave.cloud.config.open", iterable_mock_open(config_body), create=True):
            with Client.from_config('config_file') as client:
                client.session.get = GetEvent.handle
                try:
                    client.get_solver('arg-solver')
                except GetEvent as event:
                    self.assertTrue(event.url.startswith('file-prod-url'))
                    return
                self.fail()

    def test_only_file_key(self):
        """If give a name from the config file the proper URL should be loaded."""
        with mock.patch("dwave.cloud.config.open", iterable_mock_open(config_body), create=True):
            with mock.patch("dwave.cloud.config.get_configfile_paths", lambda *x: ['file']):
                with Client.from_config(profile='alpha') as client:
                    client.session.get = GetEvent.handle
                    try:
                        client.get_solver('arg-solver')
                    except GetEvent as event:
                        self.assertTrue(event.url.startswith('file-alpha-url'))
                        return
                    self.fail()

    def test_env_with_file_set(self):
        """With environment variables and a config file, the config file should be ignored."""
        with mock.patch("dwave.cloud.config.open", iterable_mock_open(legacy_config_body), create=True):
            with mock.patch.dict(os.environ, {'DW_INTERNAL__HTTPLINK': 'env-url', 'DW_INTERNAL__TOKEN': 'env-token'}):
                with Client.from_config(False) as client:
                    client.session.get = GetEvent.handle
                    try:
                        client.get_solver('arg-solver')
                    except GetEvent as event:
                        self.assertTrue(event.url.startswith('env-url'))
                        return
                    self.fail()

    def test_env_args_set(self):
        """With arguments and environment variables, the environment variables should be ignored."""
        with mock.patch.dict(os.environ, {'DW_INTERNAL__HTTPLINK': 'env-url', 'DW_INTERNAL__TOKEN': 'env-token'}):
            with Client.from_config(endpoint='args-url', token='args-token') as client:
                client.session.get = GetEvent.handle
                try:
                    client.get_solver('arg-solver')
                except GetEvent as event:
                    self.assertTrue(event.url.startswith('args-url'))
                    return
                self.fail()

    def test_file_read_error(self):
        """On config file read error, we should fail with `ConfigFileReadError`,
        but only if .dwrc actually exists on disk."""
        with mock.patch("dwave.cloud.config.open", side_effect=OSError, create=True):
            with mock.patch("os.path.exists", lambda fn: True):
                self.assertRaises(ConfigFileReadError, legacy_load_config)

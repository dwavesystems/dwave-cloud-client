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

import requests_mock

from dwave.cloud.client import Client
from dwave.cloud.qpu import Client as QPUClient
from dwave.cloud.sw import Client as SoftwareClient
from dwave.cloud.hybrid import Client as HybridClient
from dwave.cloud.solver import Solver
from dwave.cloud.exceptions import *
from dwave.cloud.config import load_config


url = 'https://dwavesys.com'
token = 'abc123abc123abc123abc123abc123abc123'
solver1_name = 'first_solver'
solver2_name = 'second_solver'

bad_url = 'https://not-a-subdomain.dwavesys.com'
bad_token = '------------------------'


def structured_solver_data(id_, cat='qpu', incomplete=False):
    """Return data dict describing a single solver."""
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

    return obj

def solver_object(id_, cat='qpu', incomplete=False):
    return Solver(client=None, data=structured_solver_data(id_, cat, incomplete))


# Define the endpoints
solver1_url = '{}/solvers/remote/{}/'.format(url, solver1_name)
solver2_url = '{}/solvers/remote/{}/'.format(url, solver2_name)
all_solver_url = '{}/solvers/remote/'.format(url)


def setup_server(m):
    """Add endpoints to the server."""

    solver1_data = structured_solver_data(solver1_name)
    solver2_data = structured_solver_data(solver2_name)
    all_solver_data = [solver1_data, solver2_data]

    # Setup the server
    valid_token_headers = {'X-Auth-Token': token}
    invalid_token_headers = {'X-Auth-Token': bad_token}

    m.get(requests_mock.ANY, status_code=404)
    m.get(requests_mock.ANY, status_code=401, request_headers=invalid_token_headers)

    m.get(solver1_url, json=solver1_data, request_headers=valid_token_headers)
    m.get(solver2_url, json=solver2_data, request_headers=valid_token_headers)
    m.get(all_solver_url, json=all_solver_data, request_headers=valid_token_headers)


class MockConnectivityTests(unittest.TestCase):
    """Test connecting some related failure modes."""

    def test_bad_url(self):
        """Connect with a bad URL."""
        with requests_mock.Mocker() as m:
            setup_server(m)
            with self.assertRaises(SAPIError) as err:
                with Client(bad_url, token) as client:
                    client.get_solvers()
            # TODO: fix when exceptions/sapi call generalized
            self.assertEqual(err.exception.error_code, 404)

    def test_bad_token(self):
        """Connect with a bad token."""
        with requests_mock.Mocker() as m:
            setup_server(m)
            with self.assertRaises(SolverAuthenticationError) as err:
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
                solver = client.get_solver(solver1_name)
                self.assertEqual(solver.id, solver1_name)

                # modify cached solver and re-fetch it
                solver.id = 'different-solver'
                # cached solver name doesn't match, so it won't be returned
                with self.assertRaises(SolverError):
                    client.get_solver(solver1_name, refresh=False)
                # cache is refreshed?
                self.assertEqual(client.get_solver(solver1_name, refresh=True).id, solver1_name)

    def test_load_all_solvers(self):
        """Load the list of solver names."""

        def spoof_cache(client, clear_val=False, clear_expires=False):
            cache = client._fetch_solvers._cached.cache
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
                    client.get_solver(solver1_name)

    def test_load_solver_missing_data(self):
        """Try to load a solver that has incomplete data."""
        with requests_mock.mock() as m:
            m.get(solver1_url, json=structured_solver_data(solver1_name, incomplete=True))
            with Client(url, token) as client:
                with self.assertRaises(SolverNotFoundError):
                    client.get_solver(solver1_name)

    def test_load_solver_broken_response(self):
        """Try to load a solver for which the server has returned a truncated response."""
        with requests_mock.mock() as m:
            body = json.dumps(structured_solver_data(solver1_name))
            m.get(solver1_url, text=body[0:len(body)//2])
            with Client(url, token) as client:
                with self.assertRaises(InvalidAPIResponseError):
                    client.get_solver(solver1_name)

    def test_get_solver_reproducible(self):
        """get_solver should return same solver (assuming cache hasn't changed)"""

        with requests_mock.mock() as m:
            setup_server(m)

            # prefer solvers with longer name: that's our second solver
            defaults = dict(solver=dict(order_by=lambda s: -len(s.id)))

            with Client(url, token, defaults=defaults) as client:
                solver = client.get_solver()
                self.assertEqual(solver.id, solver2_name)

                solver = client.get_solver()
                self.assertEqual(solver.id, solver2_name)

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
        # hybrid client
        self.assertFalse(HybridClient.is_solver_handled(solver_object('test', 'qpu')))
        self.assertFalse(HybridClient.is_solver_handled(solver_object('test', 'software')))
        self.assertTrue(HybridClient.is_solver_handled(solver_object('test', 'hybrid')))
        self.assertFalse(HybridClient.is_solver_handled(solver_object('test', 'whatever')))
        self.assertFalse(HybridClient.is_solver_handled(None))

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
        data = structured_solver_data('test')
        data['properties']['num_qubits'] = 7
        solver = Solver(None, data)
        self.assertEqual(solver.num_qubits, 7)
        self.assertEqual(solver.num_active_qubits, 3)

        # test .is_vfyc
        data = structured_solver_data('test')
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
        data = structured_solver_data('test')
        data['status'] = 'offline'
        self.assertFalse(Solver(None, data).online)
        del data['status']
        self.assertTrue(Solver(None, data).online)

    # Test fallback for legacy solvers without the `category` property
    # TODO: remove when all production solvers are updated
    def test_solver_with_category_missing(self):

        # client type filtering support
        self.assertTrue(QPUClient.is_solver_handled(solver_object('solver', cat='')))
        self.assertTrue(SoftwareClient.is_solver_handled(solver_object('c4-sw_x', cat='')))
        self.assertTrue(HybridClient.is_solver_handled(solver_object('hybrid_x', cat='')))

        # derived properties are correct
        self.assertTrue(solver_object('solver', cat='').qpu)
        self.assertFalse(solver_object('solver', cat='').software)
        self.assertFalse(solver_object('solver', cat='').hybrid)
        self.assertFalse(solver_object('c4-sw_x', cat='').qpu)
        self.assertTrue(solver_object('c4-sw_x', cat='').software)
        self.assertFalse(solver_object('c4-sw_x', cat='').hybrid)
        self.assertFalse(solver_object('hybrid_x', cat='').qpu)
        self.assertFalse(solver_object('hybrid_x', cat='').software)
        self.assertTrue(solver_object('hybrid_x', cat='').hybrid)

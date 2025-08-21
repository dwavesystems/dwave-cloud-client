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

import time
import unittest
import unittest.mock
from urllib.parse import urlencode

import requests_mock

from dwave.cloud.client import Client
from dwave.cloud.client.qpu import Client as QPUClient
from dwave.cloud.client.sw import Client as SoftwareClient
from dwave.cloud.client.hybrid import Client as HybridClient
from dwave.cloud.solver import Solver
from dwave.cloud.exceptions import *


def solver_data(name, status="ONLINE", avg_load=0.1, cat='qpu', incomplete=False, subset='all', graph_id='1'):
    """Return data dict describing a single solver."""

    # solver identity format depends on solver category now
    identity = dict(name=name)
    if cat.lower() in ('qpu', 'software'):
        identity.update(version=dict(graph_id=graph_id))

    obj = {
        "properties": {
            "supported_problem_types": ["qubo", "ising"],
            "qubits": [1, 2, 3],
            "couplers": [[1, 2], [1, 3], [2, 3]],
            "num_qubits": 3,
            "category": cat,
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "identity": identity,
        "description": "A test solver",
        "status": status,
        "avg_load": avg_load,
    }

    if incomplete:
        del obj['properties']['supported_problem_types']

    if subset == 'static':
        del obj['status']
        del obj['avg_load']

    elif subset == 'dynamic':
        obj = {
            "identity": identity,
            "status": status,
            "avg_load": avg_load
        }

    elif subset != 'all':
        raise ValueError(f'unknown subset {subset!r}')

    return obj


def solver_object(name, **kwargs):
    return Solver(client=None, data=solver_data(name, **kwargs))


class MockSolverLoading(unittest.TestCase):

    endpoint = 'https://mock'
    token = 'abc123abc123abc123abc123abc123abc123'

    solver1_name = 'first_solver'
    solver2_name = 'second_solver'
    solver3_incomplete_name = 'incomplete_properties'
    solver4_truncated_name = 'invalid_data_format'

    bad_endpoint = 'https://not-a-subdomain.dwavesys.com'
    bad_token = '------------------------'

    def setUp(self):
        m = self.mocker = requests_mock.Mocker()

        base = self.endpoint

        solver1_url = f'{base}/solvers/remote/{self.solver1_name}'
        solver2_url = f'{base}/solvers/remote/{self.solver2_name}'
        solver3_incomplete_url = f'{base}/solvers/remote/{self.solver3_incomplete_name}'
        solver4_truncated_url = f'{base}/solvers/remote/{self.solver4_truncated_name}'
        all_solvers_url = f'{base}/solvers/remote/'

        valid_token_headers = {'X-Auth-Token': self.token}
        invalid_token_headers = {'X-Auth-Token': self.bad_token}

        solver_content_type = {
            'Content-Type': 'application/vnd.dwave.sapi.solver-definition+json; version=3.0.0'}
        solvers_content_type = {
            'Content-Type': 'application/vnd.dwave.sapi.solver-definition-list+json; version=3.0.0'}

        m.get(requests_mock.ANY, status_code=404)
        m.get(requests_mock.ANY, status_code=401, request_headers=invalid_token_headers)

        def add_mock_solver(m, url, name, incomplete=False):
            m.get(url, complete_qs=True, json=solver_data(name, incomplete=incomplete),
                  request_headers=valid_token_headers, headers=solver_content_type)
            m.get(f"{url}?{urlencode(dict(filter='all,-status,-avg_load'))}",
                  json=solver_data(name, subset='static', incomplete=incomplete),
                  request_headers=valid_token_headers, headers=solver_content_type)
            m.get(f"{url}?{urlencode(dict(filter='none,+identity,+status,+avg_load'))}",
                  json=solver_data(name, subset='dynamic', incomplete=incomplete),
                  request_headers=valid_token_headers, headers=solver_content_type)

        add_mock_solver(m, solver1_url, self.solver1_name)
        add_mock_solver(m, solver2_url, self.solver2_name)
        add_mock_solver(m, solver3_incomplete_url, self.solver3_incomplete_name, incomplete=True)

        m.get(solver4_truncated_url,
              text='{"id', request_headers=valid_token_headers, headers=solver_content_type)
        m.get(f"{solver4_truncated_url}?{urlencode(dict(filter='all,-status,-avg_load'))}",
              text='{"id', request_headers=valid_token_headers, headers=solver_content_type)
        m.get(f"{solver4_truncated_url}?{urlencode(dict(filter='none,+id,+status,+avg_load'))}",
              text='{"id', request_headers=valid_token_headers, headers=solver_content_type)

        m.get(all_solvers_url,
              json=[solver_data(self.solver1_name), solver_data(self.solver2_name)],
              request_headers=valid_token_headers, headers=solvers_content_type)
        m.get(f"{all_solvers_url}?{urlencode(dict(filter='all,-status,-avg_load'))}",
              json=[solver_data(self.solver1_name, subset='static'),
                    solver_data(self.solver2_name, subset='static')],
              request_headers=valid_token_headers, headers=solvers_content_type)
        m.get(f"{all_solvers_url}?{urlencode(dict(filter='none,+id,+status,+avg_load'))}",
              json=[solver_data(self.solver1_name, subset='dynamic'),
                    solver_data(self.solver2_name, subset='dynamic')],
              request_headers=valid_token_headers, headers=solvers_content_type)

        m.start()

    def tearDown(self):
        self.mocker.stop()

    def test_bad_endpoint(self):
        with self.assertRaises(SAPIError):
            with Client(endpoint=self.bad_endpoint, token=self.token) as client:
                client.get_solvers()

    def test_bad_token(self):
        with self.assertRaises(SolverAuthenticationError) as err:
            with Client(endpoint=self.endpoint, token=self.bad_token) as client:
                client.get_solvers()

    def test_good_connection(self):
        with Client(endpoint=self.endpoint, token=self.token) as client:
            self.assertTrue(len(client.get_solvers()) > 0)

    def test_load_solver(self):
        ref = solver_object(self.solver1_name)

        with Client(endpoint=self.endpoint, token=self.token) as client:
            solver = client.get_solver(self.solver1_name)

            with self.subTest("static properties initialized"):
                self.assertEqual(solver.identity, ref.identity)
                self.assertEqual(solver.properties, ref.properties)

            with self.subTest("dynamic properties initialized"):
                self.assertEqual(solver.data['status'], ref.data['status'])
                self.assertEqual(solver.data['avg_load'], ref.data['avg_load'])

    def test_client_ref(self):
        with Client(endpoint=self.endpoint, token=self.token) as client:
            solver = client.get_solver(self.solver1_name)
            self.assertEqual(solver.client, client)

        del client

        with self.assertRaises(RuntimeError):
            solver.client

    def test_load_all_solvers(self):
        refs = [solver_object(self.solver1_name), solver_object(self.solver2_name)]

        with Client(endpoint=self.endpoint, token=self.token) as client:
            solvers = client.get_solvers()

            self.assertEqual(len(solvers), 2)

            _f = lambda attr, solvers: [getattr(s, attr) for s in solvers]
            with self.subTest("static properties initialized"):
                self.assertEqual(_f('id', solvers), _f('id', refs))
                self.assertEqual(_f('identity', solvers), _f('identity', refs))
                self.assertEqual(_f('properties', solvers), _f('properties', refs))

            _f = lambda key, solvers: [s.data[key] for s in solvers]
            with self.subTest("dynamic properties initialized"):
                self.assertEqual(_f('status', solvers), _f('status', refs))
                self.assertEqual(_f('avg_load', solvers), _f('avg_load', refs))

    def test_solvers_cache(self):
        with unittest.mock.patch.multiple(
                Client,
                _DEFAULT_SOLVERS_STATIC_PART_MAXAGE=60,
                _DEFAULT_SOLVERS_DYNAMIC_PART_MAXAGE=10,
                _DEFAULT_SOLVERS_CACHE_CONFIG=dict(maxage=60, store={})):

            with Client(endpoint=self.endpoint, token=self.token) as client:

                with self.subTest("initial solver fetch"):
                    self.mocker.reset_mock()

                    solvers = client.get_solvers()

                    self.assertEqual(len(solvers), 2)
                    self.assertEqual(self.mocker.call_count, 2)

                with self.subTest("cache hit"):
                    self.mocker.reset_mock()

                    solvers = client.get_solvers()

                    self.assertEqual(len(solvers), 2)
                    self.assertEqual(self.mocker.call_count, 0)

                with self.subTest("cache refresh forced"):
                    self.mocker.reset_mock()

                    solvers = client.get_solvers(refresh=True)

                    self.assertEqual(len(solvers), 2)
                    self.assertEqual(self.mocker.call_count, 2)

                with self.subTest("cache updated for dynamic parts"):
                    with unittest.mock.patch(
                        'dwave.cloud.api.client.epochnow',
                        lambda: time.time() + client._DEFAULT_SOLVERS_DYNAMIC_PART_MAXAGE + 1
                    ):
                        self.mocker.reset_mock()

                        solvers = client.get_solvers()

                        self.assertEqual(len(solvers), 2)
                        self.assertEqual(self.mocker.call_count, 1)


    def test_load_missing_solver(self):
        with Client(endpoint=self.endpoint, token=self.token) as client:
            with self.assertRaises(SolverNotFoundError):
                client.get_solver("non-existing")

    def test_load_solver_missing_data(self):
        with Client(endpoint=self.endpoint, token=self.token) as client:
            with self.assertRaises(SolverNotFoundError):
                client.get_solver(self.solver3_incomplete_name)

    def test_load_solver_broken_response(self):
        with Client(endpoint=self.endpoint, token=self.token) as client:
            with self.assertRaises(InvalidAPIResponseError):
                client.get_solver(self.solver4_truncated_name)

    def test_get_solver_reproducible(self):
        # prefer solvers with longer name: that's our second solver
        defaults = dict(solver=dict(order_by=lambda s: -len(s.name)))

        with Client(endpoint=self.endpoint, token=self.token, defaults=defaults) as client:
            solver = client.get_solver()
            self.assertEqual(solver.name, self.solver2_name)

            solver = client.get_solver(refresh=True)
            self.assertEqual(solver.name, self.solver2_name)

    def test_solver_filtering_in_client(self):
        # base client
        self.assertTrue(Client.is_solver_handled(solver_object('test', cat='qpu')))
        self.assertTrue(Client.is_solver_handled(solver_object('test', cat='software')))
        self.assertTrue(Client.is_solver_handled(solver_object('test', cat='hybrid')))
        self.assertTrue(Client.is_solver_handled(solver_object('test', cat='whatever')))
        self.assertTrue(Client.is_solver_handled(None))
        # qpu client
        self.assertTrue(QPUClient.is_solver_handled(solver_object('test', cat='qpu')))
        self.assertFalse(QPUClient.is_solver_handled(solver_object('test', cat='software')))
        self.assertFalse(QPUClient.is_solver_handled(solver_object('test', cat='hybrid')))
        self.assertFalse(QPUClient.is_solver_handled(solver_object('test', cat='whatever')))
        self.assertFalse(QPUClient.is_solver_handled(None))
        # sw client
        self.assertFalse(SoftwareClient.is_solver_handled(solver_object('test', cat='qpu')))
        self.assertTrue(SoftwareClient.is_solver_handled(solver_object('test', cat='software')))
        self.assertFalse(SoftwareClient.is_solver_handled(solver_object('test', cat='hybrid')))
        self.assertFalse(SoftwareClient.is_solver_handled(solver_object('test', cat='whatever')))
        self.assertFalse(SoftwareClient.is_solver_handled(None))
        # hybrid client
        self.assertFalse(HybridClient.is_solver_handled(solver_object('test', cat='qpu')))
        self.assertFalse(HybridClient.is_solver_handled(solver_object('test', cat='software')))
        self.assertTrue(HybridClient.is_solver_handled(solver_object('test', cat='hybrid')))
        self.assertFalse(HybridClient.is_solver_handled(solver_object('test', cat='whatever')))
        self.assertFalse(HybridClient.is_solver_handled(None))

    def test_solver_feature_properties(self):
        self.assertTrue(solver_object('solver', cat='qpu').qpu)
        self.assertTrue(solver_object('solver', cat='QPU').qpu)
        self.assertFalse(solver_object('solver', cat='qpu').software)
        self.assertFalse(solver_object('solver', cat='qpu').hybrid)
        self.assertFalse(solver_object('solver', cat='software').qpu)
        self.assertTrue(solver_object('solver', cat='software').software)
        self.assertFalse(solver_object('solver', cat='software').hybrid)
        self.assertTrue(solver_object('solver', cat='hybrid').hybrid)
        self.assertFalse(solver_object('solver', cat='hybrid').qpu)
        self.assertFalse(solver_object('solver', cat='hybrid').software)

        self.assertFalse(solver_object('solver').is_vfyc)
        self.assertEqual(solver_object('solver').num_qubits, 3)
        self.assertFalse(solver_object('solver').has_flux_biases)

        # identity and derived properties
        name, graph_id = 'solver', '1234'
        version = {'graph_id': graph_id}
        identity = {'name': name, 'version': version}
        self.assertEqual(solver_object(name).name, name)
        self.assertEqual(solver_object(name, graph_id=graph_id).id, f'{name};graph_id={graph_id}')
        self.assertEqual(solver_object(name, graph_id=graph_id).graph_id, graph_id)
        self.assertEqual(solver_object(name, graph_id=graph_id).version, version)
        self.assertEqual(solver_object(name, graph_id=graph_id).identity, identity)
        self.assertEqual(solver_object(name, graph_id=graph_id).identity.dict(), identity)
        self.assertEqual(solver_object(name, graph_id=graph_id).identity.version, version)
        self.assertEqual(solver_object(name, graph_id=graph_id).identity.version.dict(), version)

        # test .num_qubits vs .num_actual_qubits
        data = solver_data('test')
        data['properties']['num_qubits'] = 7
        solver = Solver(None, data)
        self.assertEqual(solver.num_qubits, 7)
        self.assertEqual(solver.num_active_qubits, 3)

        # test .is_vfyc
        data = solver_data('test')
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
        data = solver_data('test')
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

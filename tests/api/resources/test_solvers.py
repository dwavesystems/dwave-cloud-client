# Copyright 2021 D-Wave Systems Inc.
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

import io
import contextlib
import tempfile
import uuid
import unittest
from functools import partial
from urllib.parse import urljoin, urlparse, parse_qsl

import json
import requests
import requests_mock

from dwave.cloud.api import exceptions, models
from dwave.cloud.api.client import CachingSessionMixin
from dwave.cloud.api.resources import Solvers
from dwave.cloud.config import validate_config_v1
from dwave.cloud.testing.mocks import qpu_clique_solver_data

from tests import config


class TestMockSolvers(unittest.TestCase):
    """Test request formation and response parsing (including error handling)
    works correctly for all :class:`dwave.cloud.api.resources.Solvers` methods.
    """

    token = str(uuid.uuid4())
    endpoint = 'http://test.com/path/'

    def setUp(self):
        self.mocker = requests_mock.Mocker()

        solver1_data = qpu_clique_solver_data(3)
        solver1_name = solver1_data['identity']['name']
        solver1_uri = urljoin(self.endpoint, 'solvers/remote/{}'.format(solver1_name))

        solver2_data = qpu_clique_solver_data(5)
        solver2_name = solver2_data['identity']['name']
        solver2_uri = urljoin(self.endpoint, 'solvers/remote/{}'.format(solver2_name))

        all_solver_data = [solver1_data, solver2_data]
        all_solver_data_uri = urljoin(self.endpoint, 'solvers/remote/')
        self.solver_names = [solver1_name, solver2_name]
        self.solver_sizes = [3, 5]

        headers = {'X-Auth-Token': self.token}

        self.mocker.get(requests_mock.ANY, status_code=401)
        self.mocker.get(requests_mock.ANY, status_code=404, request_headers=headers)

        self.mocker.get(solver1_uri, json=solver1_data, request_headers=headers)
        self.mocker.get(solver2_uri, json=solver2_data, request_headers=headers)
        self.mocker.get(all_solver_data_uri, json=all_solver_data, request_headers=headers)

        self.mocker.start()

        self.api = Solvers(token=self.token, endpoint=self.endpoint, version_strict_mode=False)

    def tearDown(self):
        self.api.close()
        self.mocker.stop()

    def test_list_solvers(self):
        """List of solver configurations fetched."""

        solvers = self.api.list_solvers()

        self.assertEqual(len(solvers), 2)
        self.assertEqual(solvers[0].properties['num_qubits'], self.solver_sizes[0])
        self.assertEqual(solvers[1].properties['num_qubits'], self.solver_sizes[1])

    def test_get_solver(self):
        """Specific solver config retrieved (by id)."""

        solver_name = self.solver_names[0]
        num_qubits = self.solver_sizes[0]

        solver = self.api.get_solver(solver_name)

        self.assertEqual(solver.identity.name, solver_name)
        self.assertEqual(solver.properties['num_qubits'], num_qubits)

    def test_nonexisting_solver(self):
        """Not found error is raised when trying to fetch a non-existing solver."""

        with self.assertRaises(exceptions.ResourceNotFoundError):
            self.api.get_solver('non-existing-solver')

    def test_invalid_token(self):
        """Auth error is raised when request not authorized with token."""

        api = Solvers(token='invalid-token', endpoint=self.endpoint, version_strict_mode=False)
        with self.assertRaises(exceptions.ResourceAuthenticationError):
            api.list_solvers()


class FilteringTestsMixin:
    # XXX: solver=identity
    additive_filter = 'none,+solver'
    subtractive_filter = 'all,-properties.couplers'

    # assume self.api is initialized

    def test_solver_collection_property_filtering(self):
        with self.subTest('SAPI additive filtering'):
            solvers = self.api.list_solvers(filter=self.additive_filter)
            for item in solvers:
                self.assertIsInstance(item.root, models.SolverFilteredConfiguration)
                self.assertEqual(item.model_dump().keys(), {'solver'})

        with self.subTest('SAPI subtractive filtering'):
            solvers = self.api.list_solvers(filter=self.subtractive_filter)
            for item in solvers:
                self.assertIsInstance(item.root, models.SolverCompleteConfiguration)
                self.assertNotIn('couplers', item.properties)
                if item.properties.get('category') == 'qpu':
                    self.assertIn('qubits', item.properties)

    def test_solver_property_filtering(self):
        # find a QPU solver to query
        qpu = next(iter(solver for solver in self.api.list_solvers()
                        if solver.properties.get('category') == 'qpu'))
        name = qpu.identity.name
        graph_id = qpu.identity.version.graph_id

        with self.subTest('SAPI additive filtering'):
            solver = self.api.get_solver(solver_name=name, filter=self.additive_filter)
            self.assertIsInstance(solver.root, models.SolverFilteredConfiguration)
            self.assertEqual(solver.identity.name, name)
            self.assertEqual(solver.identity.version.graph_id, graph_id)

        with self.subTest('SAPI subtractive filtering'):
            solver = self.api.get_solver(solver_name=name, filter=self.subtractive_filter)
            self.assertIsInstance(solver.root, models.SolverCompleteConfiguration)
            self.assertEqual(solver.identity.name, name)
            self.assertEqual(solver.identity.version.graph_id, graph_id)
            self.assertNotIn('couplers', solver.properties)
            self.assertIn('qubits', solver.properties)

    def test_solver_config_model_is_drop_in_for_dict(self):
        solver = models.SolverConfiguration(
            identity=dict(name='name', version=dict(graph_id='gid')))

        # getters
        self.assertEqual(solver['identity'], solver.identity)
        self.assertEqual(solver.get('identity'), solver.identity)
        self.assertEqual(solver.get('nonexisting', 'x'), 'x')

        # setters
        solver['nonexisting'] = 'x'
        self.assertEqual(solver.nonexisting, 'x')
        solver.another = 'y'
        self.assertEqual(solver['another'], 'y')


class TestFiltering(FilteringTestsMixin, unittest.TestCase):

    token = str(uuid.uuid4())
    endpoint = 'http://test.com/path/'

    def setUp(self):
        self.mocker = requests_mock.Mocker()

        self.solver_data = qpu_clique_solver_data(3)
        self.solver_name = self.solver_data['identity']['name']

        self.solver_uri = urljoin(self.endpoint, 'solvers/remote/{}'.format(self.solver_name))
        self.list_uri = urljoin(self.endpoint, 'solvers/remote/')

        # XXX: solver=identity
        self.additive_filter_data = {"solver": self.solver_data['identity']}

        self.subtractive_filter_data = self.solver_data.copy()
        del self.subtractive_filter_data['properties']['couplers']

        def custom_matcher(request):
            url = urlparse(request.path_url)
            filter = dict(parse_qsl(url.query)).get('filter', '')

            if filter == self.additive_filter:
                data = self.additive_filter_data
            elif filter == self.subtractive_filter:
                data = self.subtractive_filter_data
            elif filter == '':
                data = self.solver_data
            else:
                return None

            def reply(data):
                resp = requests.Response()
                resp.status_code = 200
                resp.raw = io.BytesIO(json.dumps(data).encode('ascii'))
                return resp

            if url.path == urlparse(self.solver_uri).path:
                return reply(data)
            elif url.path == urlparse(self.list_uri).path:
                return reply([data])
            else:
                return None

        self.mocker.add_matcher(custom_matcher)

        self.mocker.start()
        self.api = Solvers(token=self.token, endpoint=self.endpoint, version_strict_mode=False)

    def tearDown(self):
        self.api.close()
        self.mocker.stop()


@unittest.skipUnless(config, "SAPI access not configured.")
class TestLiveSolvers(FilteringTestsMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.api = Solvers.from_config(validate_config_v1(config))

    @classmethod
    def tearDownClass(cls):
        cls.api.close()

    def test_list_solvers(self):
        """List of all available solvers retrieved."""

        solvers = self.api.list_solvers()

        self.assertIsInstance(solvers, list)
        self.assertGreater(len(solvers), 0)

        for solver in solvers:
            self.assertIsInstance(solver, models.SolverConfiguration)

    def test_get_solver(self):
        """Specific solver config retrieved (by id)."""

        # don't assume solver availability, instead try fetching one from the list
        solvers = self.api.list_solvers()
        solver_name = solvers.pop().identity.name

        solver = self.api.get_solver(solver_name)
        self.assertIsInstance(solver, models.SolverConfiguration)
        self.assertEqual(solver.identity.name, solver_name)

    def test_nonexisting_solver(self):
        """Not found error is raised when trying to fetch a non-existing solver."""

        with self.assertRaises(exceptions.ResourceNotFoundError):
            self.api.get_solver('non-existing-solver')


@unittest.skipUnless(config, "SAPI access not configured.")
class TestLiveSolversCaching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.store = partial(CachingSessionMixin._default_cache_config['store'],
                            directory=cls.tmpdir.name)
        # XXX: solver=identity
        cls.filter_ids = 'none,+solver'

    @classmethod
    def tearDownClass(cls):
        with contextlib.suppress(OSError):
            cls.tmpdir.cleanup()

    def test_default(self):
        # caching is disabled by default
        # sapi is queried every time
        with Solvers.from_config(validate_config_v1(config), history_size=2) as resource:
            resource.list_solvers(filter=self.filter_ids)
            self.assertEqual(len(resource.session.history), 1)
            self.assertEqual(resource.session.history[-1].response.status_code, 200)

            resource.list_solvers(filter=self.filter_ids)
            self.assertEqual(len(resource.session.history), 2)
            self.assertEqual(resource.session.history[-1].response.status_code, 200)

    def test_cache(self):
        with Solvers.from_config(validate_config_v1(config),
                                 cache=dict(store=self.store, maxage=60),
                                 history_size=2,
                                 ) as resource:

            with self.subTest("cache miss, solvers fetched from sapi"):
                solvers1 = resource.list_solvers(filter=self.filter_ids)
                self.assertGreater(len(solvers1), 0)
                self.assertEqual(len(resource.session.history), 1)
                self.assertEqual(resource.session.history[-1].response.status_code, 200)

            with self.subTest("cache hit"):
                solvers2 = resource.list_solvers(filter=self.filter_ids)
                self.assertEqual(solvers1, solvers2)
                self.assertEqual(len(resource.session.history), 1)

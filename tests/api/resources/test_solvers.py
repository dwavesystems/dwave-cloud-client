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

import uuid
import unittest
from urllib.parse import urljoin

import requests_mock

from dwave.cloud.api.resources import Solvers
from dwave.cloud.api import exceptions, models
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
        solver1_id = solver1_data['id']
        solver1_uri = urljoin(self.endpoint, 'solvers/remote/{}'.format(solver1_id))

        solver2_data = qpu_clique_solver_data(5)
        solver2_id = solver2_data['id']
        solver2_uri = urljoin(self.endpoint, 'solvers/remote/{}'.format(solver2_id))

        all_solver_data = [solver1_data, solver2_data]
        all_solver_data_uri = urljoin(self.endpoint, 'solvers/remote/')
        self.solver_ids = [solver1_id, solver2_id]
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

        solver_id = self.solver_ids[0]
        num_qubits = self.solver_sizes[0]

        solver = self.api.get_solver(solver_id)

        self.assertEqual(solver.id, solver_id)
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


@unittest.skipUnless(config, "SAPI access not configured.")
class TestCloudSolvers(unittest.TestCase):

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
        solver_id = solvers.pop().id

        solver = self.api.get_solver(solver_id)
        self.assertIsInstance(solver, models.SolverConfiguration)
        self.assertEqual(solver.id, solver_id)

    def test_nonexisting_solver(self):
        """Not found error is raised when trying to fetch a non-existing solver."""

        with self.assertRaises(exceptions.ResourceNotFoundError):
            self.api.get_solver('non-existing-solver')

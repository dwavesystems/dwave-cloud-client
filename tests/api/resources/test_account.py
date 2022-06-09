# Copyright 2022 D-Wave Systems Inc.
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
from dwave.cloud.api.client import LeapAPIClient

import requests_mock

from dwave.cloud.api.resources import LeapAccount
from dwave.cloud.api import exceptions

from tests import config


class TestMockAccount(unittest.TestCase):
    """Test request formation and response parsing (including error handling)
    works correctly for all :class:`dwave.cloud.api.resources.LeapAccount`
    methods.
    """

    token = str(uuid.uuid4())
    endpoint = 'http://test.com/path/'

    def setUp(self):
        self.mocker = requests_mock.Mocker()

        self.p1 = {"id": 1, "name": "Project I", "code": "ONE"}
        self.p2 = {"id": 2, "name": "Project II", "code": "TWO"}
        headers = {'Authorization': f'Bearer {self.token}'}

        self.mocker.get(requests_mock.ANY, status_code=401)
        self.mocker.get(requests_mock.ANY, status_code=404, request_headers=headers)

        active_project_uri = urljoin(self.endpoint, 'api/account/active_project/oauth/')
        active_project_data = {"data": {"project": self.p1}}
        self.mocker.get(active_project_uri, json=active_project_data, request_headers=headers)

        projects_uri = urljoin(self.endpoint, 'api/account/projects/oauth/')
        projects_data = {"data": {"projects": [{"project": self.p1}, {"project": self.p2}]}}
        self.mocker.get(projects_uri, json=projects_data, request_headers=headers)

        token_uri = urljoin(self.endpoint, 'api/account/token/oauth/')
        self.project_token = {1: "ONE-123", 2: "TWO-234"}
        self.mocker.get(f"{token_uri}?project_id={self.p1['id']}",
                        json={"data": {"token": self.project_token[self.p1['id']]}},
                        request_headers=headers)
        self.mocker.get(f"{token_uri}?project_id={self.p2['id']}",
                        json={"data": {"token": self.project_token[self.p2['id']]}},
                        request_headers=headers)

        self.mocker.start()

    def tearDown(self):
        self.mocker.stop()

    def test_get_active_project(self):
        """Active Leap project fetched."""

        resource = LeapAccount(token=self.token, endpoint=self.endpoint)
        project = resource.get_active_project()

        self.assertEqual(project.id, self.p1['id'])
        self.assertEqual(project.name, self.p1['name'])
        self.assertEqual(project.code, self.p1['code'])

    def test_get_project(self):
        """All Leap projects fetched."""

        resource = LeapAccount(token=self.token, endpoint=self.endpoint)
        projects = resource.get_projects()

        ref = [self.p1, self.p2]
        self.assertEqual(len(projects), len(ref))
        for i in range(len(ref)):
            self.assertEqual(projects[i].id, ref[i]['id'])
            self.assertEqual(projects[i].name, ref[i]['name'])
            self.assertEqual(projects[i].code, ref[i]['code'])

    def test_get_project_token(self):
        """Correct token for Leap project fetched."""

        resource = LeapAccount(token=self.token, endpoint=self.endpoint)
        project = resource.get_active_project()

        token = resource.get_project_token(project_id=project.id)
        self.assertEqual(token, self.project_token[project.id])

        token = resource.get_project_token(project=project)
        self.assertEqual(token, self.project_token[project.id])

    def test_get_project_token__for_nonexisting_project(self):
        """Not found error is raised when trying to fetch a non-existing
        project's token."""

        resource = LeapAccount(token=self.token, endpoint=self.endpoint)
        with self.assertRaises(exceptions.ResourceNotFoundError):
            resource.get_project_token(project_id=42)

    def test_invalid_token(self):
        """Auth error is raised when request not authorized with token."""

        resource = LeapAccount(token='invalid-token', endpoint=self.endpoint)
        with self.assertRaises(exceptions.ResourceAuthenticationError):
            resource.get_active_project()


@unittest.skipUnless(config, "LeapAPI access not configured.")
class TestCloudAccount(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with LeapAPIClient(**config) as client:
            cls.api = LeapAccount.from_client_config(client)

    @classmethod
    def tearDownClass(cls):
        cls.api.close()

    # TODO: we need to implement generalized config before we can implement
    # live tests (access_token has to be stored)

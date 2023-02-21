# Copyright 2023 D-Wave Systems Inc.
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

import unittest
from urllib.parse import urljoin

import requests_mock
from pydantic import parse_obj_as, BaseModel

from dwave.cloud.api import exceptions
from dwave.cloud.api.resources import ResourceBase


class MathResource(ResourceBase):
    """A prototypical resource, implementing an interface to a 2-integer adder."""

    resource_path = 'math/'

    class AddResult(BaseModel):
        out: int

    def add(self, in1: int, in2: int) -> int:
        path = 'add/'
        response = self.session.get(path, params=dict(in1=in1, in2=in2))
        result = response.json()
        return parse_obj_as(MathResource.AddResult, result).out

    def nonexisting(self):
        return self.session.get('nonexisting')


class TestMockResource(unittest.TestCase):
    """Test request formation and response parsing (including error
    handling and api version validation) works correctly for all
    :class:`dwave.cloud.api.resources.ResourceBase` methods.
    """

    endpoint = 'http://test.com/path/'

    def setUp(self):
        self.add_uri = urljoin(urljoin(self.endpoint, MathResource.resource_path), 'add/')

    @requests_mock.Mocker()
    def test_basic_flow(self, m):
        m.get(urljoin(self.add_uri, '?in1=1&in2=2'), json=dict(out=3))

        resource = MathResource(endpoint=self.endpoint)
        res = resource.add(1, 2)
        self.assertEqual(res, 3)

    @requests_mock.Mocker()
    def test_nonexisting_path(self, m):
        m.get(requests_mock.ANY, status_code=404)

        resource = MathResource(endpoint=self.endpoint)
        with self.assertRaises(exceptions.ResourceNotFoundError):
            resource.nonexisting()

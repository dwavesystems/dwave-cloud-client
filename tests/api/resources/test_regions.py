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

import unittest
from urllib.parse import urljoin

import requests_mock

from dwave.cloud.api.resources import Regions
from dwave.cloud.api import exceptions, models

from tests import config


class TestMockRegions(unittest.TestCase):
    """Test request formation and response parsing (including error handling)
    works correctly for all :class:`dwave.cloud.api.resources.Regions` methods.
    """

    endpoint = 'http://test.com/path/'

    def setUp(self):
        self.mocker = requests_mock.Mocker()

        region1_data = {"code": "A", "name": "Region A", "endpoint": urljoin(self.endpoint, 'regions/A')}
        region1_code = region1_data['code']
        region1_uri = region1_data['endpoint']

        region2_data = {"code": "B", "name": "Region B", "endpoint": urljoin(self.endpoint, 'regions/B')}
        region2_code = region2_data['code']
        region2_uri = region2_data['endpoint']

        all_region_data = [region1_data, region2_data]
        all_region_data_uri = urljoin(self.endpoint, 'regions/')
        self.region_codes = [region1_code, region2_code]

        self.mocker.get(requests_mock.ANY, status_code=404)

        self.mocker.get(region1_uri, json=region1_data)
        self.mocker.get(region2_uri, json=region2_data)
        self.mocker.get(all_region_data_uri, json=all_region_data)

        self.mocker.start()

    def tearDown(self):
        self.mocker.stop()

    def test_list_regions(self):
        resource = Regions(endpoint=self.endpoint)
        regions = resource.list_regions()

        self.assertEqual(len(regions), len(self.region_codes))
        self.assertEqual(regions[0].code, self.region_codes[0])
        self.assertEqual(regions[1].code, self.region_codes[1])

    def test_get_region(self):
        code = self.region_codes[0]

        regions = Regions(endpoint=self.endpoint)
        region = regions.get_region(code)

        self.assertEqual(region.code, code)
        self.assertEqual(region.endpoint, urljoin(self.endpoint, f"regions/{code}"))

    def test_nonexisting_region(self):
        resource = Regions(endpoint=self.endpoint)
        with self.assertRaises(exceptions.ResourceNotFoundError):
            resource.get_region('non-existing-region')


@unittest.skipUnless(config, "API access not configured.")
class TestCloudRegions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.api = Regions()

    @classmethod
    def tearDownClass(cls):
        cls.api.close()

    def test_list_solvers(self):
        regions = self.api.list_regions()

        self.assertIsInstance(regions, list)
        self.assertGreater(len(regions), 0)

        for region in regions:
            self.assertIsInstance(region, models.Region)

    def test_get_region(self):
        # don't assume solver availability, instead try fetching one from the list
        regions = self.api.list_regions()
        region_code = regions.pop().code

        region = self.api.get_region(region_code)
        self.assertIsInstance(region, models.Region)
        self.assertEqual(region.code, region_code)

    def test_nonexisting_region(self):
        with self.assertRaises(exceptions.ResourceNotFoundError):
            self.api.get_region('non-existing-region')

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
from unittest import mock

import requests_mock
from parameterized import parameterized
from pydantic import TypeAdapter

from dwave.cloud.api.models import Region
from dwave.cloud.api.exceptions import ResourceAccessForbiddenError
from dwave.cloud.config.constants import DEFAULT_METADATA_API_ENDPOINT
from dwave.cloud.config.models import ClientConfig
from dwave.cloud.regions import get_regions, resolve_endpoints


class GetRegionsInit(unittest.TestCase):

    mock_endpoint = "https://mock.dwavesys.com/metadata/api/"

    @parameterized.expand([
        ("string-endpoint", mock_endpoint),
        ("config-dict", {"metadata_api_endpoint": mock_endpoint}),
        ("config-model", ClientConfig(metadata_api_endpoint=mock_endpoint)),
    ])
    @mock.patch("dwave.cloud.regions._fetch_available_regions")
    def test_config(self, name, value, fetch_mock: mock.Mock):

        get_regions(value)

        self.assertEqual(fetch_mock.call_count, 1)

        config = fetch_mock.call_args.kwargs.get('config')
        self.assertEqual(config.metadata_api_endpoint, self.mock_endpoint)

    @mock.patch("dwave.cloud.regions._fetch_available_regions")
    def test_default_config(self, fetch_mock: mock.Mock):

        get_regions()

        self.assertEqual(fetch_mock.call_count, 1)

        config = fetch_mock.call_args.kwargs.get('config')
        self.assertEqual(config.metadata_api_endpoint, ClientConfig().metadata_api_endpoint)

    def test_invalid_config(self):
        with self.assertRaises(TypeError):
            get_regions(config=1)


class GetRegionsFunctionality(unittest.TestCase):

    def test_config(self):
        # get_regions() caching params are passed down to CachingSession

        with mock.patch("dwave.cloud.regions.api.Regions.from_config") as m:
            ls = m.return_value.__enter__.return_value.list_regions

            with self.subTest('default call'):
                get_regions()
                ls.assert_called()

            with self.subTest('cache refresh'):
                m.reset_mock()
                get_regions(refresh=True)
                self.assertTrue(ls.call_args.kwargs.get('refresh_'))

            with self.subTest('cache maxage'):
                m.reset_mock()
                get_regions(maxage=0)
                self.assertEqual(ls.call_args.kwargs.get('maxage_'), 0)

            with self.subTest('cache bypass'):
                m.reset_mock()
                get_regions(no_cache=True)
                self.assertTrue(ls.call_args.kwargs.get('no_cache_'))

    @mock.patch("dwave.cloud.regions._fetch_available_regions",
                side_effect=ResourceAccessForbiddenError)
    def test_fetch_error(self, fetch_mock: mock.Mock):
        with self.assertRaises(ValueError):
            get_regions()

    @requests_mock.Mocker()
    def test_end_to_end_mocked(self, m):

        mock_endpoint = 'https://mock'

        mock_regions = [
            Region(code='A', name='Region A', endpoint='http://a/'),
            Region(code='B', name='Region B', endpoint='http://b/'),
        ]
        mock_json = TypeAdapter(list[Region]).dump_python(mock_regions)

        m.get(f"{mock_endpoint}/regions/", json=mock_json,
              headers={'Content-Type': 'application/vnd.dwave.metadata.regions+json; version=1.0'})

        regions = get_regions(mock_endpoint)
        self.assertEqual(regions, mock_regions)


class ResolveEndpointsMocked(unittest.TestCase):

    def setUp(self):
        self.mock_regions = [
            Region(code='A', name='Region A', endpoint='https://a.mock.dwavesys.com/path'),
            Region(code='B', name='Region B', endpoint='https://b.mock.dwavesys.com/path'),
        ]

        self.mocker = mock.patch('dwave.cloud.regions.get_regions',
                                 return_value=self.mock_regions)
        self.mocker.start()

    def tearDown(self):
        self.mocker.stop()

    def test_null(self):
        config = ClientConfig(metadata_api_endpoint=None, region=None,
                              endpoint=None, leap_api_endpoint=None)

        with mock.patch('dwave.cloud.regions.DEFAULT_REGION',
                        self.mock_regions[0].code):
            resolve_endpoints(config, inplace=True, shortcircuit=False)

        self.assertEqual(config.metadata_api_endpoint, DEFAULT_METADATA_API_ENDPOINT)
        self.assertEqual(config.region, self.mock_regions[0].code)
        self.assertEqual(config.endpoint, self.mock_regions[0].endpoint)
        self.assertEqual(config.leap_api_endpoint, self.mock_regions[0].leap_api_endpoint)

    def test_endpoint_override(self):
        endpoint = 'https://mock.dwavesys.com/sapi'
        config = ClientConfig(endpoint=endpoint)

        resolve_endpoints(config, inplace=True)

        self.assertEqual(config.metadata_api_endpoint, DEFAULT_METADATA_API_ENDPOINT)
        self.assertIsNone(config.region)
        self.assertEqual(config.endpoint, endpoint)
        self.assertIsNotNone(config.leap_api_endpoint)

    def test_leap_api_endpoint_override(self):
        leap_api_endpoint = 'https://mock.dwavesys.com/leap'
        config = ClientConfig(leap_api_endpoint=leap_api_endpoint)

        resolve_endpoints(config, inplace=True)

        self.assertEqual(config.metadata_api_endpoint, DEFAULT_METADATA_API_ENDPOINT)
        self.assertIsNone(config.region)
        self.assertEqual(config.leap_api_endpoint, leap_api_endpoint)
        self.assertIsNotNone(config.endpoint)

    def test_region_given(self):
        mock_region = self.mock_regions[1].code
        config = ClientConfig(region=mock_region)

        resolve_endpoints(config, inplace=True)

        self.assertEqual(config.region, mock_region)
        self.assertEqual(config.endpoint, self.mock_regions[1].endpoint)
        self.assertEqual(config.leap_api_endpoint, self.mock_regions[1].leap_api_endpoint)

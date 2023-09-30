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
import contextlib

from parameterized import parameterized

from dwave.cloud.api.models import Region
from dwave.cloud.api import constants
from dwave.cloud.api.exceptions import ResourceAccessForbiddenError
from dwave.cloud.config.models import ClientConfig
from dwave.cloud.regions import get_regions, resolve_endpoints


class GetRegionsInit(unittest.TestCase):

    mock_endpoint = "https://example.com/metadata/api/"

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
        mock_regions = [
            Region(code='A', name='Region A', endpoint='http://a/'),
            Region(code='B', name='Region B', endpoint='http://b/'),
        ]
        # use a mock metadata api to avoid main cache contamination with mock data
        # XXX: alternatively, we flush the cache after this test
        mock_metadata_api_endpoint = 'https://example.com/metadata/'

        # mock `api.Regions.from_config().list_regions()` call
        @contextlib.contextmanager
        def mock_regions_from_config(*args, **kwargs):
            class mock_regions_obj:
                def list_regions(_):
                    return mock_regions
            yield mock_regions_obj()

        with mock.patch("dwave.cloud.regions.api.Regions.from_config",
                        side_effect=mock_regions_from_config) as m:

            with self.subTest('default call'):
                ret = get_regions(mock_metadata_api_endpoint)
                m.assert_called()
                self.assertEqual(ret, mock_regions)

            with self.subTest('caching'):
                m.reset_mock()
                ret = get_regions(mock_metadata_api_endpoint)
                m.assert_not_called()
                self.assertEqual(ret, mock_regions)

            with self.subTest('cache refresh'):
                m.reset_mock()
                ret = get_regions(mock_metadata_api_endpoint, refresh=True)
                m.assert_called()

            with self.subTest('cache refresh'):
                m.reset_mock()
                ret = get_regions(mock_metadata_api_endpoint, maxage=0)
                m.assert_called()

            with self.subTest('cache bypass'):
                m.reset_mock()
                ret = get_regions(mock_metadata_api_endpoint, no_cache=True)
                m.assert_called()

    @mock.patch("dwave.cloud.regions._fetch_available_regions",
                side_effect=ResourceAccessForbiddenError)
    def test_fetch_error(self, fetch_mock: mock.Mock):
        with self.assertRaises(ValueError):
            get_regions()


class ResolveEndpointsMocked(unittest.TestCase):

    def setUp(self):
        self.mock_regions = [
            Region(code='A', name='Region A', endpoint='https://a.example.com/path'),
            Region(code='B', name='Region B', endpoint='https://b.example.com/path'),
        ]

        self.mocker = mock.patch('dwave.cloud.regions.get_regions',
                                 return_value=self.mock_regions)
        self.mocker.start()

    def tearDown(self):
        self.mocker.stop()

    def test_null(self):
        config = ClientConfig(metadata_api_endpoint=None, region=None,
                              endpoint=None, leap_api_endpoint=None)

        mock_default_region = self.mock_regions[0].code

        with mock.patch('dwave.cloud.regions.DEFAULT_REGION', mock_default_region):
            resolve_endpoints(config, inplace=True)

        self.assertEqual(config.metadata_api_endpoint, constants.DEFAULT_METADATA_API_ENDPOINT)
        self.assertEqual(config.region, mock_default_region)
        self.assertEqual(config.endpoint, self.mock_regions[0].endpoint)
        self.assertEqual(config.leap_api_endpoint, self.mock_regions[0].leap_api_endpoint)

    def test_endpoint_override(self):
        endpoint = 'https://example.com/sapi'
        config = ClientConfig(endpoint=endpoint)

        resolve_endpoints(config, inplace=True)

        self.assertEqual(config.metadata_api_endpoint, constants.DEFAULT_METADATA_API_ENDPOINT)
        self.assertIsNone(config.region)
        self.assertEqual(config.endpoint, endpoint)
        self.assertIsNotNone(config.leap_api_endpoint)

    def test_leap_api_endpoint_override(self):
        leap_api_endpoint = 'https://example.com/leap'
        config = ClientConfig(leap_api_endpoint=leap_api_endpoint)

        resolve_endpoints(config, inplace=True)

        self.assertEqual(config.metadata_api_endpoint, constants.DEFAULT_METADATA_API_ENDPOINT)
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


class ResolveEndpointsLive(unittest.TestCase):

    def test_null(self):
        config = ClientConfig(metadata_api_endpoint=None, region=None,
                              endpoint=None, leap_api_endpoint=None)

        resolve_endpoints(config, inplace=True)

        self.assertEqual(config.metadata_api_endpoint, constants.DEFAULT_METADATA_API_ENDPOINT)
        self.assertEqual(config.region, constants.DEFAULT_REGION)

        regions = get_regions(config)
        regions = {r.code: r for r in regions}

        region = regions[constants.DEFAULT_REGION]
        self.assertEqual(config.endpoint, region.endpoint)
        self.assertEqual(config.leap_api_endpoint, region.leap_api_endpoint)

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

import json
import time
import uuid
import unittest
import zlib

import diskcache
import requests
import requests_mock
from parameterized import parameterized

from dwave.cloud.api import exceptions
from dwave.cloud.api.client import (
    DWaveAPIClient, SolverAPIClient, MetadataAPIClient, LeapAPIClient)
from dwave.cloud.config import ClientConfig, constants
from dwave.cloud.package_info import __packagename__, __version__


class TestConfig(unittest.TestCase):
    """Session is initiated from config."""

    def test_defaults(self):
        with self.assertRaises(ValueError):
            DWaveAPIClient()

        endpoint = constants.DEFAULT_METADATA_API_ENDPOINT
        client = DWaveAPIClient(endpoint=endpoint)

        defaults = DWaveAPIClient.DEFAULTS.copy()
        defaults.update(endpoint=endpoint)

        self.assertEqual(client.config, defaults)
        self.assertIsInstance(client.session, requests.Session)

        # verify Retry object config
        retry = client.session.get_adapter('https://').max_retries
        conf = DWaveAPIClient.DEFAULTS['retry']
        client.close()
        self.assertEqual(retry.total, conf['total'])

    def test_init(self):
        config = dict(endpoint='https://test.com/path/',
                      token=str(uuid.uuid4()),
                      timeout=1,
                      retry=dict(total=3),
                      headers={'Custom': 'Field 123'},
                      verify=False,
                      proxies={'https': 'http://proxy.com'})

        with DWaveAPIClient(**config) as client:
            session = client.session
            self.assertIsInstance(session, requests.Session)

            self.assertEqual(session.base_url, config['endpoint'])
            self.assertEqual(session.cert, None)
            self.assertEqual(session.headers['X-Auth-Token'], config['token'])
            self.assertEqual(session.headers['Custom'], config['headers']['Custom'])
            self.assertIn(__packagename__, session.headers['User-Agent'])
            self.assertIn(__version__, session.headers['User-Agent'])
            self.assertEqual(session.verify, config['verify'])
            self.assertEqual(session.proxies, config['proxies'])

            # verify Retry object config
            retry = session.get_adapter('https://').max_retries
            self.assertEqual(retry.total, config['retry']['total'])

    def test_sapi_client(self):
        with SolverAPIClient() as client:
            self.assertEqual(client.session.base_url,
                             constants.DEFAULT_SOLVER_API_ENDPOINT)

    def test_metadata_client(self):
        with MetadataAPIClient() as client:
            self.assertEqual(client.session.base_url,
                             constants.DEFAULT_METADATA_API_ENDPOINT)

    def test_leap_client(self):
        with LeapAPIClient() as client:
            self.assertEqual(client.session.base_url,
                             constants.DEFAULT_LEAP_API_ENDPOINT)

    def test_from_config_model_factory(self):
        config = ClientConfig(region='ignored', connection_close=True,
                              permissive_ssl=True)
        kwargs = dict(endpoint='https://test.com/path/')

        def _verify(client):
            # note: region is removed due to endpoint kwarg update
            self.assertIsNone(client.config.get('region'))
            self.assertEqual(client.session.base_url, kwargs['endpoint'])
            self.assertIn('Connection', client.session.headers)
            self.assertFalse(client.session.verify)

        with DWaveAPIClient.from_config_model(config, **kwargs) as client:
            _verify(client)

        # also test .from_config dispatch
        with DWaveAPIClient.from_config(config, **kwargs) as client:
            _verify(client)

    def test_from_config_file(self):
        # `load_config` is already tested thoroughly in `tests.test_config`,
        # so it's ok to just mock it here
        config = dict(endpoint='https://test.com/path/')
        with unittest.mock.patch("dwave.cloud.config.load_config",
                                 lambda *a, **kw: config):
            with DWaveAPIClient.from_config_file() as client:
                self.assertEqual(client.session.base_url, config['endpoint'])

            # also test .from_config dispatch
            with DWaveAPIClient.from_config() as client:
                self.assertEqual(client.session.base_url, config['endpoint'])


class TestRequests(unittest.TestCase):

    @requests_mock.Mocker()
    def test_request(self, m):
        """Config options are respected when making requests."""

        config = dict(endpoint='https://test.com/path/',
                      token=str(uuid.uuid4()),
                      headers={'Custom': 'Field 123'})

        auth_headers = {'X-Auth-Token': config['token']}
        data = dict(answer=123)

        m.get(requests_mock.ANY, status_code=401)
        m.get(requests_mock.ANY, status_code=404, request_headers=auth_headers)
        m.get(config['endpoint'], json=data, request_headers=config['headers'])

        with DWaveAPIClient(**config) as client:
            self.assertEqual(client.session.get('').json(), data)

    @requests_mock.Mocker()
    def test_paths(self, m):
        """Path translation works."""

        baseurl = 'https://test.com'
        config = dict(endpoint=baseurl)

        path_a, path_b = 'a', 'b'
        data_a, data_b = dict(answer='a'), dict(answer='b')

        m.get(requests_mock.ANY, status_code=404)
        m.get(f"{baseurl}/{path_a}", json=data_a)
        m.get(f"{baseurl}/{path_b}", json=data_b)

        with DWaveAPIClient(**config) as client:
            self.assertEqual(client.session.get(path_a).json(), data_a)
            self.assertEqual(client.session.get(path_b).json(), data_b)

    @requests_mock.Mocker()
    def test_session_history(self, m):
        """Session history is available."""

        baseurl = 'https://test.com'
        config = dict(endpoint=baseurl, history_size=1)

        m.get(requests_mock.ANY, status_code=404)
        m.get(f"{baseurl}/path", json=dict(data=True))

        with DWaveAPIClient(**config) as client:
            client.session.get('path')
            self.assertEqual(client.session.history[-1].request.path_url, '/path')

            with self.assertRaises(exceptions.ResourceNotFoundError):
                client.session.get('unknown')
                self.assertEqual(client.session.history[-1].exception.error_code, 404)

            client.session.get('/path')
            self.assertEqual(client.session.history[-1].request.path_url, '/path')

    @requests_mock.Mocker()
    def test_payload_compression(self, m):
        """Payload compression on upload is supported."""

        baseurl = 'https://test.com'
        config = dict(endpoint=baseurl)

        data = b'test payload'
        data_compressed = zlib.compress(data)

        def match_data(request):
            body = request.body
            if not isinstance(body, bytes):
                body = b''.join(request.body)
            return body == data_compressed

        m.get(requests_mock.ANY, status_code=404)
        m.post(baseurl, additional_matcher=match_data, text='ok')

        with self.subTest('compress all'):
            with DWaveAPIClient(compress=True, **config) as client:
                resp = client.session.post('', data=data)
                self.assertEqual(resp.text, 'ok')

        with self.subTest('compress request'):
            with DWaveAPIClient(**config) as client:
                resp = client.session.post('', data=data, compress=True)
                self.assertEqual(resp.text, 'ok')


class TestVersionValidation(unittest.TestCase):

    version_strict_mode = True

    def setUp(self):
        self.mocker = requests_mock.Mocker()

        endpoint = 'http://test.com/path'
        self.media_type = 'application/vnd.dwave.api.mock+json'

        version = '1.2.3'
        self.data = dict(v=version)

        self.mocker.get(requests_mock.ANY, status_code=404)
        self.mocker.get(f"{endpoint}/no-type", json=self.data)
        self.mocker.get(f"{endpoint}/no-version", json=self.data,
                        headers={'Content-Type': f'{self.media_type}'})
        self.mocker.get(f"{endpoint}/version", json=self.data,
                        headers={'Content-Type': f'{self.media_type}; version={version}'})

        self.mocker.start()

        self.client = DWaveAPIClient(
            endpoint=endpoint, version_strict_mode=self.version_strict_mode)

    def tearDown(self):
        self.client.close()
        self.mocker.stop()

    def test_nominal(self):
        self.client.session.set_accept(media_type=self.media_type, accept_version='~=1.2.0')
        self.assertEqual(self.client.session.get('version').json(), self.data)

    def test_no_type(self):
        with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'^Media type not present'):
            self.client.session.set_accept(media_type=self.media_type, accept_version='~=1.2.0')
            self.assertEqual(self.client.session.get('no-type').json(), self.data)

    def test_no_type_when_not_expected(self):
        self.client.session.set_accept()
        self.assertEqual(self.client.session.get('no-type').json(), self.data)

    def test_wrong_type(self):
        self.client.session.set_accept(media_type='wrong')
        with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'^Received media type'):
            self.assertEqual(self.client.session.get('version').json(), self.data)

    def test_no_version(self):
        self.client.session.set_accept(media_type=self.media_type, accept_version='~=1.2.0')
        with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'version undefined in the response'):
            self.assertEqual(self.client.session.get('no-version').json(), self.data)

    def test_wrong_version(self):
        self.client.session.set_accept(media_type=self.media_type, accept_version='>2')
        with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'version .* not compliant'):
            self.assertEqual(self.client.session.get('version').json(), self.data)


class TestNonStrictVersionValidation(TestVersionValidation):

    version_strict_mode = False

    def test_no_type(self):
        self.client.session.set_accept(media_type=self.media_type, accept_version='~=1.2.0')
        self.assertEqual(self.client.session.get('no-type').json(), self.data)


class TestResponseParsing(unittest.TestCase):

    @requests_mock.Mocker()
    def test_json(self, m):
        """Non-JSON response is OK if consistent with media type."""

        mock_response = {"a": 1}

        m.get(requests_mock.ANY, content=json.dumps(mock_response).encode('ascii'),
              status_code=200, headers={'Content-Type': 'application/octet-stream'})

        with DWaveAPIClient(endpoint='https://mock') as client:

            with self.subTest("binary response when expecting json"):
                client.session.set_accept(media_type="application/json")
                with self.assertRaises(exceptions.ResourceBadResponseError):
                    client.session.get('')

            with self.subTest("response matches the expected media type"):
                client.session.set_accept(media_type="application/octet-stream", media_type_params={"case": 2})
                self.assertEqual(client.session.get('').json(), mock_response)

            with self.subTest("everything goes when there's no expectation on media type"):
                client.session.unset_accept()
                self.assertTrue(len(client.session.get('').content))

    @requests_mock.Mocker()
    def test_structured_error_response(self, m):
        """Error response dict correctly initializes exc."""

        error_msg = "I looked, but couldn't find."
        error_code = 404
        error = dict(error_msg=error_msg, error_code=error_code)

        m.get(requests_mock.ANY, json=error, status_code=error_code)

        with DWaveAPIClient(endpoint='https://mock') as client:

            with self.assertRaisesRegex(exceptions.ResourceNotFoundError, error_msg) as exc:
                client.session.get('test')

                self.assertEqual(exc.error_msg, error_msg)
                self.assertEqual(exc.error_code, error_code)

    @requests_mock.Mocker()
    def test_plain_text_error(self, m):
        """Error messages in plain text/body correctly initialize exc."""

        error_msg = "I looked, but couldn't find."
        error_code = 404

        m.get(requests_mock.ANY, text=error_msg, status_code=error_code)

        with DWaveAPIClient(endpoint='https://mock') as client:

            with self.assertRaisesRegex(exceptions.ResourceNotFoundError, error_msg) as exc:
                client.session.get('test')

                self.assertEqual(exc.error_msg, error_msg)
                self.assertEqual(exc.error_code, error_code)

    @requests_mock.Mocker()
    def test_unknown_errors(self, m):
        """Unknown status code with plain text msg raised as general req exc."""

        error_msg = "I'm a teapot"
        error_code = 418

        m.get(requests_mock.ANY, text=error_msg, status_code=error_code)

        with DWaveAPIClient(endpoint='https://mock') as client:

            with self.assertRaisesRegex(exceptions.RequestError, error_msg) as exc:
                client.session.get('test')

                self.assertEqual(exc.error_msg, error_msg)
                self.assertEqual(exc.error_code, error_code)


class TestResponseCaching(unittest.TestCase):

    def test_config(self):
        with self.subTest("cache disabled by default"):
            with DWaveAPIClient(endpoint='https://mock') as client:
                self.assertFalse(client.session._cache_enabled)

        with self.subTest("cache disabled with config"):
            with DWaveAPIClient(endpoint='https://mock', cache=False) as client:
                self.assertFalse(client.session._cache_enabled)

        with self.subTest("cache enabled with config"):
            with DWaveAPIClient(endpoint='https://mock', cache=True) as client:
                self.assertTrue(client.session._cache_enabled)
                self.assertIsNotNone(client.session._maxage)
                self.assertIsInstance(client.session._store, diskcache.Cache)

        with self.subTest("cache configured"):
            with DWaveAPIClient(endpoint='https://mock',
                                cache=dict(maxage=5, store={})) as client:
                self.assertTrue(client.session._cache_enabled)
                self.assertEqual(client.session._maxage, 5)
                self.assertIsInstance(client.session._store, dict)

    @parameterized.expand([
        (None, ),
        (2j, ),
        (-1.0, ),
    ])
    def test_invalid_maxage(self, maxage):
        with self.assertRaises(ValueError):
            DWaveAPIClient(
                endpoint='https://mock', cache=dict(maxage=maxage, store={}))

    @parameterized.expand([
        (0, ),
        (1, ),
        (2.0, ),
    ])
    def test_valid_maxage(self, maxage):
        with DWaveAPIClient(endpoint='https://mock',
                            cache=dict(maxage=maxage, store={})) as client:
            self.assertEqual(client.session._maxage, maxage)

    @requests_mock.Mocker()
    def test_conditional_requests_with_no_cache_control(self, m):
        endpoint = 'https://mock'

        path_a = 'path-a'
        etag_a = 'etag-a'
        data_a = {"a": 1}

        path_b = 'path-b'
        etag_b = 'etag-b'
        data_b = {"b": 1}
        etag_bb = 'etag-bb'
        data_bb = {"bb": 1}

        # mock: resource not modified
        m.get(f"{endpoint}/{path_a}", json=data_a, headers={'ETag': etag_a})
        m.get(f"{endpoint}/{path_a}",
              request_headers={'If-None-Match': etag_a}, status_code=304,
              headers={'ETag': etag_a})

        # mock: resource modified between requests
        m.get(f"{endpoint}/{path_b}", json=data_b, headers={'ETag': etag_b})
        m.get(f"{endpoint}/{path_b}",
              request_headers={'If-None-Match': etag_b}, json=data_bb,
              headers={'ETag': etag_bb})

        store = {}

        with DWaveAPIClient(endpoint=endpoint, cache=dict(store=store), history_size=1) as client:

            with self.subTest("cache miss"):
                self.assertEqual(len(store), 0)

                r = client.session.get(path_a)
                self.assertEqual(r.json(), data_a)

                self.assertTrue(m.called)
                self.assertEqual(len(store), 2)     # data + meta

            with self.subTest("cache hit"):
                m.reset_mock()

                r = client.session.get(path_a, maxage_=10)
                self.assertEqual(r.json(), data_a)

                # if mock not called, data came from cache
                self.assertFalse(m.called)

            with self.subTest("cache validate, not modified"):
                m.reset_mock()

                r = client.session.get(path_a)
                self.assertEqual(r.json(), data_a)

                self.assertTrue(m.called)
                self.assertEqual(client.session.history[-1].response.status_code, 304)

            with self.subTest("cache validate, modified"):
                m.reset_mock()

                r = client.session.get(path_b)
                self.assertEqual(r.json(), data_b)

                r = client.session.get(path_b)
                self.assertEqual(r.json(), data_bb)

                self.assertEqual(m.call_count, 2)
                self.assertEqual(client.session.history[-1].response.status_code, 200)

            with self.subTest("cache skipped"):
                m.reset_mock()

                r = client.session.get(path_a, no_cache_=True)
                self.assertEqual(r.json(), data_a)

                self.assertTrue(m.called)

            with self.subTest("cache refreshed"):
                m.reset_mock()

                r = client.session.get(path_a, maxage_=10, refresh_=True)
                self.assertEqual(r.json(), data_a)

                self.assertTrue(m.called)

    @requests_mock.Mocker()
    def test_cache_control_maxage(self, m):
        # verify max-age from cache-control header is respected

        endpoint = 'https://mock'

        path = 'path'
        etag = 'etag'
        data = {"a": 1}

        maxage = 10

        # mock: resource not modified
        m.get(f"{endpoint}/{path}", json=data,
              headers={'ETag': etag, 'Cache-Control': f'public, max-age={maxage}'})
        m.get(f"{endpoint}/{path}",
              request_headers={'If-None-Match': etag}, status_code=304,
              headers={'ETag': etag, 'Cache-Control': f'public, max-age={maxage}'})

        store = {}

        with DWaveAPIClient(endpoint=endpoint, cache=dict(store=store), history_size=1) as client:

            with self.subTest("cache miss"):
                self.assertEqual(len(store), 0)

                r = client.session.get(path)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)
                self.assertEqual(len(store), 2)     # data + meta

            with self.subTest("cache hit"):
                m.reset_mock()

                with unittest.mock.patch('dwave.cloud.api.client.epochnow',
                                         lambda: time.time() + maxage - 1):
                    r = client.session.get(path)
                self.assertEqual(r.json(), data)

                # if mock not called, data came from cache
                self.assertFalse(m.called)

            with self.subTest("cache expired, validated"):
                m.reset_mock()

                with unittest.mock.patch('dwave.cloud.api.client.epochnow',
                                         lambda: time.time() + maxage + 1):
                    r = client.session.get(path)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)
                self.assertEqual(client.session.history[-1].response.status_code, 304)

    @requests_mock.Mocker()
    def test_cache_control_no_store(self, m):
        # verify we don't cache when server forbids caching

        endpoint = 'https://mock'

        path = 'path'
        etag = 'etag'
        data = {"a": 1}

        m.get(f"{endpoint}/{path}", json=data,
              headers={'ETag': etag, 'Cache-Control': 'no-store'})

        store = {}

        with DWaveAPIClient(endpoint=endpoint, cache=dict(store=store)) as client:

            with self.subTest("cache miss"):
                self.assertEqual(len(store), 0)

                r = client.session.get(path)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)
                self.assertEqual(len(store), 0)

    @requests_mock.Mocker()
    def test_cache_control_no_cache(self, m):
        # verify we cache but always validate when cache-control is no-cache

        endpoint = 'https://mock'

        path = 'path'
        etag = 'etag'
        data = {"a": 1}

        m.get(f"{endpoint}/{path}", json=data,
              headers={'ETag': etag, 'Cache-Control': f'no-cache, max-age=100'})
        m.get(f"{endpoint}/{path}",
              request_headers={'If-None-Match': etag}, status_code=304,
              headers={'ETag': etag, 'Cache-Control': f'no-cache, max-age=100'})

        store = {}

        with DWaveAPIClient(endpoint=endpoint, cache=dict(store=store), history_size=1) as client:

            with self.subTest("cache miss"):
                self.assertEqual(len(store), 0)

                r = client.session.get(path)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)
                self.assertEqual(len(store), 2)

            with self.subTest("cache validated"):
                m.reset_mock()

                r = client.session.get(path)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)
                self.assertEqual(client.session.history[-1].response.status_code, 304)

    @requests_mock.Mocker()
    def test_time_based_validation_only(self, m):
        # when etag is not available, cache validation is based on time/maxage

        endpoint = 'https://mock'
        path = 'path'
        data = {"a": 1}

        m.get(f"{endpoint}/{path}", json=data)

        store = {}
        maxage = 10

        with DWaveAPIClient(endpoint=endpoint,
                            cache=dict(store=store, maxage=maxage)) as client:

            with self.subTest("cache miss"):
                self.assertEqual(len(store), 0)

                r = client.session.get(path)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)
                self.assertEqual(len(store), 2)     # data + meta

            with self.subTest("cache hit"):
                m.reset_mock()

                with unittest.mock.patch('dwave.cloud.api.client.epochnow',
                                         lambda: time.time() + maxage - 1):
                    r = client.session.get(path)
                self.assertEqual(r.json(), data)

                # if mock not called, data came from cache
                self.assertFalse(m.called)

            with self.subTest("cache expired"):
                m.reset_mock()

                with unittest.mock.patch('dwave.cloud.api.client.epochnow',
                                         lambda: time.time() + maxage + 1):
                    r = client.session.get(path)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)

            with self.subTest("ignored maxage=None"):
                m.reset_mock()
                store.clear()

                r = client.session.get(path, maxage_=None)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)

            with self.subTest("warn about invalid maxage"):
                m.reset_mock()
                store.clear()

                with self.assertWarns(UserWarning):
                    r = client.session.get(path, maxage_=-1)
                self.assertEqual(r.json(), data)

                self.assertTrue(m.called)

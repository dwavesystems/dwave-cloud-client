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

import requests
import requests_mock

from dwave.cloud.api import exceptions, constants
from dwave.cloud.api.client import DWaveAPIClient, SolverAPIClient, MetadataAPIClient
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
    def test_api_version_validation(self, m):
        """API response version validation works."""

        baseurl = 'https://test.com'
        config = dict(endpoint=baseurl)
        media_type = 'application/vnd.dwave.api.mock+json'
        version = '1.2.3'

        path, data = 'version', dict(v=version)

        m.get(requests_mock.ANY, status_code=404)
        m.get(f"{baseurl}/{path}", json=data,
              headers={'Content-Type': f'{media_type}; version={version}'})

        with DWaveAPIClient(**config) as client:
            # nominal
            client.session.set_accept(media_type=media_type, accept_version='~=1.2.0')
            self.assertEqual(client.session.get(path).json(), data)

            # wrong type
            client.session.set_accept(media_type='wrong', accept_version='~=1.2.0')
            with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'^Received media type'):
                self.assertEqual(client.session.get(path).json(), data)

            # wrong version
            client.session.set_accept(media_type=media_type, accept_version='>2')
            with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'^API response format version'):
                self.assertEqual(client.session.get(path).json(), data)


class TestResponseParsing(unittest.TestCase):

    @requests_mock.Mocker()
    def test_non_json(self, m):
        """Non-JSON OK response is unexpected."""

        m.get(requests_mock.ANY, text='text', status_code=200)

        with DWaveAPIClient(endpoint='https://mock') as client:

            with self.assertRaises(exceptions.ResourceBadResponseError) as exc:
                client.session.get('test')

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

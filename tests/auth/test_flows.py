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
from urllib.parse import urlsplit, parse_qsl

import requests_mock

from dwave.cloud.auth.flows import AuthFlow, LeapAuthFlow
from dwave.cloud.auth.config import OCEAN_SDK_CLIENT_ID, OCEAN_SDK_SCOPES
from dwave.cloud.config import ClientConfig


class TestAuthFlow(unittest.TestCase):

    def setUp(self):
        self.client_id = '123'
        self.scopes = ('scope-a', 'scope-b')
        self.redirect_uri_oob = 'oob'
        self.authorization_endpoint = 'https://example.com/authorize'
        self.token_endpoint = 'https://example.com/token'

        self.test_args = dict(
            client_id=self.client_id,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri_oob,
            authorization_endpoint=self.authorization_endpoint,
            token_endpoint=self.token_endpoint)

    def test_auth_url(self):
        flow = AuthFlow(**self.test_args)

        url = flow.get_authorization_url()
        q = dict(parse_qsl(urlsplit(url).query))

        self.assertTrue(url.startswith(self.authorization_endpoint))
        self.assertEqual(q['response_type'], 'code')
        self.assertEqual(q['client_id'], self.client_id)
        self.assertEqual(q['redirect_uri'], self.redirect_uri_oob)
        self.assertEqual(q['scope'], ' '.join(self.scopes))
        self.assertIn('state', q)
        # pkce
        self.assertIn('code_challenge', q)
        self.assertEqual(q['code_challenge_method'], 'S256')

    def test_fetch_token_state(self):
        flow = AuthFlow(**self.test_args)

        # generate auth request (state)
        _ = flow.get_authorization_url()

        # try exchanging the code with wrong state
        with self.assertRaisesRegex(ValueError, "State mismatch"):
            flow.fetch_token(code='not important', state='invalid')

    @requests_mock.Mocker()
    def test_fetch_token(self, m):
        # mock the token_endpoint
        code = '123456'
        token = dict(access_token='123', refresh_token='456', id_token='789')
        expected_params = dict(
            grant_type='authorization_code', client_id=self.client_id,
            redirect_uri=self.redirect_uri_oob, code=code)

        def post_body_matcher(request):
            params = dict(parse_qsl(request.text))
            return params == expected_params

        m.get(requests_mock.ANY, status_code=404)
        m.post(requests_mock.ANY, status_code=404)
        m.post(self.token_endpoint, additional_matcher=post_body_matcher, json=token)

        # verify token fetch flow
        flow = AuthFlow(**self.test_args)

        response = flow.fetch_token(code=code)
        self.assertEqual(response, token)

    def test_session_config(self):
        config = dict(
            cert='/path/to/cert',
            headers={'X-Field': 'Value'},
            proxies=dict(https='socks5://localhost'),
            timeout=60,
            verify=False)

        def verify(session, config):
            self.assertEqual(session.cert, config['cert'])
            self.assertEqual({**session.headers, **config['headers']}, session.headers)
            self.assertEqual(session.proxies, config['proxies'])
            self.assertEqual(session.default_timeout, config['timeout'])
            self.assertEqual(session.verify, config['verify'])

        with self.subTest('on construction'):
            flow = AuthFlow(session_config=config, **self.test_args)
            verify(flow.session, config)

        with self.subTest('post-construction'):
            flow = AuthFlow(**self.test_args)
            flow.update_session(config)
            verify(flow.session, config)


class TestLeapAuthFlow(unittest.TestCase):

    def test_from_minimal_config(self):
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')

        flow = LeapAuthFlow.from_config_model(config)

        # endpoint urls are generated?
        self.assertTrue(flow.authorization_endpoint.startswith(config.leap_api_endpoint))
        self.assertTrue(flow.token_endpoint.startswith(config.leap_api_endpoint))
        # Leap-specific:
        self.assertTrue(flow.authorization_endpoint.endswith('authorize'))
        self.assertTrue(flow.token_endpoint.endswith('token'))

        self.assertEqual(flow.client_id, OCEAN_SDK_CLIENT_ID)
        self.assertEqual(flow.scopes, ' '.join(OCEAN_SDK_SCOPES))
        self.assertEqual(flow.redirect_uri, LeapAuthFlow._OOB_REDIRECT_URI)

    def test_from_minimal_config_with_overrides(self):
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')
        client_id = '123'
        scopes = ['email']
        redirect_uri = 'https://example.com/callback'

        flow = LeapAuthFlow.from_config_model(
            config=config, client_id=client_id,
            scopes=scopes, redirect_uri=redirect_uri)

        self.assertEqual(flow.client_id, client_id)
        self.assertEqual(flow.scopes, ' '.join(scopes))
        self.assertEqual(flow.redirect_uri, redirect_uri)

    def test_from_common_config(self):
        config = ClientConfig(leap_api_endpoint='https://example.com/leap',
                              headers=dict(injected='value'), request_timeout=10)

        flow = LeapAuthFlow.from_config_model(config)

        self.assertEqual(flow.session.headers.get('injected'), 'value')
        self.assertEqual(flow.session.default_timeout, 10)

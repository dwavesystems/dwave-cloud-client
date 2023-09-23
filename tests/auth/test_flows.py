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
from urllib.parse import urlsplit, parse_qs, parse_qsl

import requests_mock

from dwave.cloud.auth.flows import AuthFlow


class TestAuthFlow(unittest.TestCase):

    def setUp(self):
        self.client_id = '123'
        self.scopes = ('scope-a', 'scope-b')
        self.redirect_uri_oob = 'oob'
        self.authorization_endpoint = 'https://example.com/authorize'
        self.token_endpoint = 'https://example.com/token'

    def test_auth_url(self):
        flow = AuthFlow(client_id=self.client_id,
                        scopes=self.scopes,
                        redirect_uri=self.redirect_uri_oob,
                        authorization_endpoint=self.authorization_endpoint,
                        token_endpoint=self.token_endpoint)

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
        flow = AuthFlow(client_id=self.client_id,
                        scopes=self.scopes,
                        redirect_uri=self.redirect_uri_oob,
                        authorization_endpoint=self.authorization_endpoint,
                        token_endpoint=self.token_endpoint)

        response = flow.fetch_token(code=code)
        self.assertEqual(response, token)

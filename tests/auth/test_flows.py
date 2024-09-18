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

import sys
import time
import threading
import unittest
from functools import partial
from unittest import mock
from urllib.parse import urlsplit, parse_qsl, urljoin

import requests
import requests_mock

from dwave.cloud.auth.flows import AuthFlow, LeapAuthFlow, OAuthError
from dwave.cloud.auth.config import OCEAN_SDK_CLIENT_ID, OCEAN_SDK_SCOPES
from dwave.cloud.auth.creds import Credentials
from dwave.cloud.config import ClientConfig, DEFAULT_LEAP_API_ENDPOINT


class TestAuthFlow(unittest.TestCase):

    def setUp(self):
        self.client_id = '123'
        self.scopes = ('scope-a', 'scope-b')
        self.redirect_uri_oob = 'oob'
        self.authorization_endpoint = 'https://example.com/authorize'
        self.token_endpoint = 'https://example.com/token'
        self.revocation_endpoint = 'https://example.com/revoke'
        self.token = dict(access_token='123', refresh_token='456', id_token='789')
        self.creds = Credentials(create=False)
        self.leap_api_endpoint = 'https://example.com/leap/api'

        self.test_args = dict(
            client_id=self.client_id,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri_oob,
            authorization_endpoint=self.authorization_endpoint,
            token_endpoint=self.token_endpoint,
            revocation_endpoint=self.revocation_endpoint,
            leap_api_endpoint=self.leap_api_endpoint,
            creds=self.creds)

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
        expected_params = dict(
            grant_type='authorization_code', client_id=self.client_id,
            redirect_uri=self.redirect_uri_oob, code=code)

        def post_body_matcher(request):
            params = dict(parse_qsl(request.text))
            return params == expected_params

        m.get(requests_mock.ANY, status_code=404)
        m.post(requests_mock.ANY, status_code=404)
        m.post(self.token_endpoint, json=dict(error="error", error_description="bad request"))
        m.post(self.token_endpoint, additional_matcher=post_body_matcher, json=self.token)

        # reset creds
        self.creds.clear()

        # make auth request to generate all request params (like PKCE's verifier)
        flow = AuthFlow(**self.test_args)
        _ = flow.get_authorization_url()
        expected_params.update(code_verifier=flow.code_verifier)

        # verify token fetch flow
        response = flow.fetch_token(code=code)
        self.assertEqual(response, self.token)

        # verify token proxy to oauth2 session
        self.assertEqual(flow.token, self.token)

        # verify token saved to creds
        self.assertEqual(flow.creds[flow.leap_api_endpoint], self.token)

    def test_token_setter(self):
        flow = AuthFlow(**self.test_args)

        self.assertIsNone(flow.session.token)

        self.creds.clear()

        flow.token = self.token

        self.assertIsNotNone(flow.session.token)
        self.assertEqual(flow.session.token['access_token'], self.token['access_token'])

        # verify token is persisted
        self.assertEqual(self.creds[self.leap_api_endpoint], self.token)

    def test_refresh_token(self):
        flow = AuthFlow(**self.test_args)
        self.creds.clear()

        with mock.patch.object(flow.session, 'refresh_token',
                               return_value=self.token) as m:
            flow.refresh_token()

        m.assert_called_once_with(url=flow.token_endpoint)

        # verify token is persisted
        self.assertEqual(self.creds[self.leap_api_endpoint], self.token)

    def test_ensure_active_token(self):
        flow = AuthFlow(**self.test_args)
        flow.token = self.token

        with mock.patch.object(flow.session, 'ensure_active_token') as m:
            flow.ensure_active_token()

        m.assert_called_once_with(token=self.token)

    def test_revoke_token(self):
        flow = AuthFlow(**self.test_args)

        ok = requests.Response()
        ok.status_code = 200

        with mock.patch.object(flow.session, 'revoke_token',
                               return_value=ok) as m:
            status = flow.revoke_token()
            self.assertTrue(status)

        m.assert_called_once_with(url=flow.revocation_endpoint, token=None,
                                  token_type_hint=None)

    def test_revoke_token_is_conditionally_available(self):
        # when: `revocation_endpoint` is undefined
        args = self.test_args.copy()
        del args['revocation_endpoint']

        # then: init works
        flow = AuthFlow(**args)

        # but: revoke_token() is unavailable
        with self.assertRaisesRegex(TypeError, 'endpoint undefined'):
            flow.revoke_token()

    @requests_mock.Mocker()
    def test_revoke_token_request_formation(self, m):
        # when we want to revoke a specific token, outgoing request conforms to RFC 7009

        # mock the revocation_endpoint
        expected_params = dict(
            token=self.token['access_token'],
            token_type_hint='access_token')

        def post_body_matcher(request):
            params = dict(parse_qsl(request.text))
            return params == expected_params

        m.get(requests_mock.ANY, status_code=404)
        m.post(requests_mock.ANY, status_code=404)
        m.post(self.revocation_endpoint, json=dict(error="error", error_description="bad request"))
        m.post(self.revocation_endpoint, additional_matcher=post_body_matcher, status_code=200)

        # reset creds
        self.creds.clear()

        # initialize flow with token
        flow = AuthFlow(**self.test_args)
        flow.token = self.token

        # verify revoke token flow
        status = flow.revoke_token(token=self.token['access_token'],
                                   token_type_hint='access_token')
        self.assertTrue(status)

    @requests_mock.Mocker()
    def test_revoke_token_default_request_formation(self, m):
        # when token is unspecified, refresh token should be revoked

        # mock the revocation_endpoint:
        expected_params = dict(token=self.token['refresh_token'])

        def post_body_matcher(request):
            params = dict(parse_qsl(request.text))
            return params == expected_params

        m.get(requests_mock.ANY, status_code=404)
        m.post(requests_mock.ANY, status_code=404)
        m.post(self.revocation_endpoint, json=dict(error="error", error_description="bad request"))
        m.post(self.revocation_endpoint, additional_matcher=post_body_matcher, status_code=200)

        # reset creds
        self.creds.clear()

        # initialize flow with token
        flow = AuthFlow(**self.test_args)
        flow.token = self.token

        # verify revoke token flow
        status = flow.revoke_token()
        self.assertTrue(status)

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

    def test_token_expires_soon(self):
        flow = AuthFlow(**self.test_args)
        now = time.time()

        with mock.patch.object(flow.session.token_auth, 'token',
                               dict(expires_at=now + 59)):
            self.assertTrue(flow.token_expires_soon())
            self.assertTrue(flow.token_expires_soon(within=60))
            self.assertFalse(flow.token_expires_soon(within=0))


class TestLeapAuthFlow(unittest.TestCase):

    def test_from_default_config(self):
        config = ClientConfig()

        flow = LeapAuthFlow.from_config_model(config)

        # endpoint urls are generated?
        prefix = urljoin(DEFAULT_LEAP_API_ENDPOINT, '/')
        self.assertTrue(flow.authorization_endpoint.startswith(prefix))
        self.assertTrue(flow.token_endpoint.startswith(prefix))

    def test_from_minimal_config(self):
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')

        flow = LeapAuthFlow.from_config_model(config)

        # endpoint urls are generated?
        self.assertTrue(flow.authorization_endpoint.startswith(config.leap_api_endpoint))
        self.assertTrue(flow.token_endpoint.startswith(config.leap_api_endpoint))
        # Leap-specific:
        self.assertTrue(flow.authorization_endpoint.endswith('authorize'))
        self.assertTrue(flow.token_endpoint.endswith('token'))
        self.assertEqual(flow.leap_api_endpoint, config.leap_api_endpoint)

        self.assertEqual(flow.client_id, OCEAN_SDK_CLIENT_ID)
        self.assertEqual(flow.scopes, ' '.join(OCEAN_SDK_SCOPES))
        self.assertEqual(flow.redirect_uri, LeapAuthFlow._OOB_REDIRECT_URI)
        self.assertIsNotNone(flow.creds)

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

    def test_client_id_from_config(self):
        client_id = '123'
        config = ClientConfig(leap_api_endpoint='https://example.com/leap',
                              leap_client_id=client_id)

        flow = LeapAuthFlow.from_config_model(config)

        self.assertEqual(flow.client_id, client_id)


class TestLeapAuthFlowOOB(unittest.TestCase):

    @mock.patch('click.echo', return_value=None)
    def test_oob(self, m):
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')
        flow = LeapAuthFlow.from_config_model(config)

        mock_code = '1234'

        with mock.patch('click.prompt', return_value=mock_code):
            with mock.patch.object(flow, 'fetch_token') as fetch_token:
                flow.run_oob_flow()
                fetch_token.assert_called_once_with(code=mock_code)

class TestLeapAuthFlowRedirect(unittest.TestCase):

    @mock.patch('click.echo', return_value=None)
    def test_success(self, m):
        # success case: access authorized, token fetched
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')
        flow = LeapAuthFlow.from_config_model(config)

        mock_code = '1234'

        ctx = {}
        ready = threading.Event()
        def url_open(url, *args, **kwargs):
            ctx.update(parse_qsl(urlsplit(url).query))
            ready.set()

        with mock.patch.object(flow, 'fetch_token') as fetch_token:
            f = threading.Thread(
                target=partial(flow.run_redirect_flow, open_browser=url_open))
            f.start()

            ready.wait()
            response = requests.get(ctx['redirect_uri'],
                                    params=dict(code=mock_code, state=ctx['state']))
            self.assertEqual(len(response.history), 1)
            self.assertEqual(response.history[0].status_code, 302)
            location = response.history[0].headers.get('Location')
            self.assertTrue(location.startswith(config.leap_api_endpoint))
            self.assertIn('/success', location)
            self.assertEqual(urlsplit(location).query, '')

            f.join()
            fetch_token.assert_called_once_with(code=mock_code, state=ctx['state'])

    class FlowRunner(threading.Thread):
        def __init__(self, *, flow, authorize_response_params, **kwargs):
            super().__init__(**kwargs)

            self.flow = flow

            # set when authorize url is "opened" in browser
            self.authorize_query = {}
            self.authorize_ready = threading.Event()
            self.authorize_response_params = authorize_response_params

        def open_browser(self, url, *args, **kwargs):
            self.authorize_query.update(parse_qsl(urlsplit(url).query))
            self.authorize_ready.set()

        def authorize(self):
            # fake authorize request from Leap to our redirect_uri
            # `params` determine if auth was denied or approved
            self.authorize_ready.wait()

            query = dict(state=self.authorize_query['state'])
            query.update(self.authorize_response_params)
            return requests.get(self.authorize_query['redirect_uri'], params=query)

        def run(self):
            try:
                self.flow.run_redirect_flow(open_browser=self.open_browser)
            except:
                # make exception discoverable from the main thread
                self._exc_info = sys.exc_info()

        def exception(self):
            # see: https://stackoverflow.com/a/1854263, published under CC BY-SA 4.0.
            # to be called from main thread
            if hasattr(self, '_exc_info') and self._exc_info:
                raise self._exc_info[1].with_traceback(self._exc_info[2])

    def assert_redirect_to_error_page(self, response, flow):
        # check that `response` is a 302 redirect with redirect location pointing to leap error page
        error_uri = flow._infer_leap_error_uri(flow.leap_api_endpoint)
        self.assertEqual(len(response.history), 1)
        self.assertEqual(response.history[0].status_code, 302)
        location = response.history[0].headers.get('Location')
        self.assertTrue(location.startswith(error_uri))

    def assert_redirect_params(self, response, expected_params):
        # check that redirect location params match `expected params`
        location = response.history[0].headers.get('Location')
        location_params = dict(parse_qsl(urlsplit(location).query))
        self.assertEqual(location_params, expected_params)

    @mock.patch('click.echo', return_value=None)
    def test_auth_denied(self, m):
        # error case: access not authorized (e.g. user clicks "Decline")
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')
        flow = LeapAuthFlow.from_config_model(config)

        error = 'access_denied'
        error_description = 'Authorization request denied'

        with mock.patch.object(flow, 'fetch_token') as fetch_token:
            authorize_response_params = dict(error=error, error_description=error_description)

            t = self.FlowRunner(flow=flow, authorize_response_params=authorize_response_params)
            t.start()

            response = t.authorize()

            # check response is a redirect to leap error page
            self.assert_redirect_to_error_page(response, flow)

            # and all request query params are propagated
            request = response.history[0].request.url
            request_params = dict(parse_qsl(urlsplit(request).query))
            self.assert_redirect_params(response, request_params)

            # token fetch shouldn't even be attempted
            t.join()
            fetch_token.assert_not_called()

            # oauth exception raised by the redirect flow runner
            with self.assertRaisesRegex(OAuthError, error_description):
                t.exception()

    @mock.patch('click.echo', return_value=None)
    def test_exchange_fails(self, m):
        # error case: access authorized, but code exchange fails with oauth error
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')
        flow = LeapAuthFlow.from_config_model(config)

        code = '1234'

        # mock `fetch_token` to raise an oauth error (e.g. code expired)
        error = OAuthError(error='invalid_grant',
                           description='The provided authorization grant is invalid')

        with mock.patch.object(flow, 'fetch_token', side_effect=error) as fetch_token:
            authorize_response_params = dict(code=code)     # state appended by FlowRunner.authorize()

            t = self.FlowRunner(flow=flow, authorize_response_params=authorize_response_params)
            t.start()

            response = t.authorize()

            state = t.authorize_query['state']

            # check response is a redirect to leap error page
            self.assert_redirect_to_error_page(response, flow)

            # and error page params are constructed properly
            expected_params = dict(error=error.error,
                                   error_description=error.description,
                                   state=state)
            self.assert_redirect_params(response, expected_params)

            # token fetch is attempted (but it fails with `error`)
            t.join()
            fetch_token.assert_called_once_with(code=code, state=state)

            # oauth exception raised by the redirect flow runner
            with self.assertRaisesRegex(OAuthError, str(error)):
                t.exception()

    @mock.patch('click.echo', return_value=None)
    def test_non_auth_failure_during_code_exchange(self, m):
        # error case: access authorized, but code exchange fails with unexpected non-auth error
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')
        flow = LeapAuthFlow.from_config_model(config)

        code = '1234'

        # mock `fetch_token` to raise a non-auth error (e.g. file error)
        error = OSError("credential store unavailable")

        with mock.patch.object(flow, 'fetch_token', side_effect=error) as fetch_token:
            authorize_response_params = dict(code=code)     # state appended by FlowRunner.authorize()

            t = self.FlowRunner(flow=flow, authorize_response_params=authorize_response_params)
            t.start()

            response = t.authorize()

            state = t.authorize_query['state']

            # check response is a redirect to leap error page
            self.assert_redirect_to_error_page(response, flow)

            # and error page params are constructed properly
            expected_params = dict(error=type(error).__name__,
                                   error_description=str(error),
                                   state=state)
            self.assert_redirect_params(response, expected_params)

            # token fetch is attempted (but it fails with `error`)
            t.join()
            fetch_token.assert_called_once_with(code=code, state=state)

            # oauth exception raised by the redirect flow runner
            with self.assertRaisesRegex(type(error), str(error)):
                t.exception()

    @mock.patch('click.echo', return_value=None)
    def test_csrf_failure_during_code_exchange(self, m):
        # error case: access authorized, but code exchange fails due to state mismatch
        config = ClientConfig(leap_api_endpoint='https://example.com/leap')
        flow = LeapAuthFlow.from_config_model(config)

        code = '1234'

        error = ValueError("State mismatch")

        with mock.patch.object(flow, 'fetch_token') as fetch_token:
            authorize_response_params = dict(code=code, state='spoofed')

            t = self.FlowRunner(flow=flow, authorize_response_params=authorize_response_params)
            t.start()

            response = t.authorize()

            # check response is a redirect to leap error page
            self.assert_redirect_to_error_page(response, flow)

            # and error page params are constructed properly
            expected_params = dict(error=type(error).__name__,
                                   error_description=str(error),
                                   state=authorize_response_params['state'])
            self.assert_redirect_params(response, expected_params)

            # token fetch is never attempted (state precondition failed)
            t.join()
            fetch_token.assert_not_called()

            # oauth exception raised by the redirect flow runner
            with self.assertRaisesRegex(type(error), str(error)):
                t.exception()

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

from __future__ import annotations

import logging
import webbrowser
from operator import sub
from typing import Any, Callable, Dict, Optional, Union, Sequence
from urllib.parse import urljoin, parse_qsl

from authlib.integrations.requests_client import OAuth2Session
from authlib.common.security import generate_token

from dwave.cloud.auth.config import OCEAN_SDK_CLIENT_ID, OCEAN_SDK_SCOPES
from dwave.cloud.auth.server import SingleRequestAppServer, RequestCaptureApp
from dwave.cloud.config.models import ClientConfig
from dwave.cloud.regions import resolve_endpoints
from dwave.cloud.utils import pretty_argvalues

__all__ = ['AuthFlow', 'LeapAuthFlow']

logger = logging.getLogger(__name__)


class AuthFlow:
    """`OAuth 2.0 Authorization Code`_ exchange flow with `PKCE`_ extension
    for public (secretless) clients.

    Args:
        client_id:
            OAuth 2.0 Client ID.
        scopes:
            List of requested scopes.
        redirect_uri:
            Redirect URI that was registered as callback for client identified
            with ``client_id``.
        authorization_endpoint:
            URL of the authorization server's authorization endpoint.
        token_endpoint:
            URL of the authorization server's token endpoint.
        session_config:
            Configuration options for the low-level ``requests.Session`` used
            for all OAuth 2 requests. Supported options are: ``cert``,
            ``cookies``, ``headers``, ``proxies``, ``timeout``, ``verify``.

    .. _OAuth 2.0 Authorization Code:
        https://datatracker.ietf.org/doc/html/rfc6749#section-4.1
    .. _PKCE:
        https://datatracker.ietf.org/doc/html/rfc7636
    """

    def __init__(self, *,
                 client_id: str,
                 scopes: Sequence[str],
                 redirect_uri: str,
                 authorization_endpoint: str,
                 token_endpoint: str,
                 session_config: Optional[Dict[str, Any]] = None,
                 ):
        self.client_id = client_id
        self.scopes = ' '.join(scopes)
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint

        self.session = OAuth2Session(
            client_id=client_id, scope=scopes, redirect_uri=redirect_uri,
            code_challenge_method='S256',
            # metadata set via kwargs
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint)

        if session_config is not None:
            self.update_session(session_config)

        logger.debug(f"{type(self).__name__} initialized with: {pretty_argvalues()}")

    def update_session(self, config: Dict[str, Any]) -> None:
        """Update OAuth2Session/requests.Session with config values for:
        ``cert``, ``cookies``, ``headers``, ``proxies``, ``timeout``, ``verify``.
        """
        logger.trace(f"{type(self).__name__}.update_session(config={config!r})")

        self.session.headers.update(config.get('headers') or {})
        self.session.default_timeout = config.get('timeout', None)

        for key in ('cert', 'cookies', 'proxies', 'verify'):
            if key in config:
                setattr(self.session, key, config[key])

    @property
    def redirect_uri(self):
        return self.session.redirect_uri

    @redirect_uri.setter
    def redirect_uri(self, value):
        self.session.redirect_uri = value

    @property
    def token(self):
        return self.session.token

    @token.setter
    def token(self, value):
        self.session.token = value

    def get_authorization_url(self) -> str:
        self.state = generate_token(30)
        self.code_verifier = generate_token(48)

        url, _ = self.session.create_authorization_url(
            url=self.authorization_endpoint,
            state=self.state,
            code_verifier=self.code_verifier)

        logger.debug(f"{type(self).__name__}.get_authorization_url() = {url!r}")
        return url

    def fetch_token(self, *, code: str, **kwargs) -> dict:
        """Exchange ``code`` for access token.

        Args:
            code:
                OAuth 2.0 authorization code to be exchanged for access token.
            state:
                Authorization state; should match the one generated when creating
                the authorize URL.

                Note: except for the out-of-band flow, be sure to pass in the
                ``state`` parameter received on the ``redirect_uri`` for added
                security.
        """
        # make sure states match before proceeding, we shouldn't be tricked
        # into exchanging arbitrary auth codes.
        state = kwargs.get('state', None)
        if state is not None:
            if state != self.state:
                raise ValueError('State mismatch')

        token = self.session.fetch_token(
            url=self.token_endpoint,
            grant_type='authorization_code',
            code=code,
            **kwargs)

        logger.debug(f"{type(self).__name__}.fetch_token() = {token!r}")
        return token

    def refresh_token(self):
        return self.session.refresh_token(url=self.token_endpoint)

    def ensure_active_token(self):
        return self.session.ensure_active_token(token=self.session.token)


class LeapAuthFlow(AuthFlow):
    """:class:`.AuthFlow` specialized for Ocean and Leap.

    Example::
        from dwave.cloud.auth import LeapAuthFlow
        from dwave.cloud.config import load_config, validate_config_v1

        config = validate_config_v1(load_config(profile='eu'))
        flow = LeapAuthFlow.from_config_model(config)

        url = flow.get_authorization_url()
        print('Visit and authorize:', url)

        code = input('code: ')
        token = flow.fetch_token(code=code)
    """

    _OOB_REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob'

    _VISIT_AUTH_MSG = 'Please visit the following URL to authorize Ocean: '
    _INPUT_CODE_MSG = 'Authorization code: '

    _REDIRECT_HOST = '127.0.0.1'
    _REDIRECT_PORT_RANGE = (36000, 36050)
    _REDIRECT_DONE_MSG = ('The authorization code exchange flow has completed. '
                          'You can now close this browser tab.')

    _AUTH_TIMEOUT_MSG = 'Authorization flow did not complete in the allotted time.'

    # note: in the future we might want to replace these url resolvers with a
    # OpenID Provider Metadata server query
    @staticmethod
    def _infer_auth_endpoint(leap_api_endpoint: str) -> str:
        return urljoin(leap_api_endpoint, '/leap/openid/authorize')

    @staticmethod
    def _infer_token_endpoint(leap_api_endpoint: str) -> str:
        return urljoin(leap_api_endpoint, '/leap/openid/token')

    @classmethod
    def from_config_model(cls, config: ClientConfig, **kwargs) -> LeapAuthFlow:
        """Construct a :class:`.LeapAuthModel` initialized with Ocean SDK's
        ``client_id``, and Leap authorization/token endpoint.

        Args:
            config:
                Client configuration model. Only generic request properties
                are used: ``cert``, ``headers``, ``proxy``, ``request_timeout``,
                and ``permissive_ssl``.
            **kwargs:
                Keyword arguments to optionally override: ``client_id``,
                ``scopes`` and ``redirect_uri``. For description of these
                parameters, see :class:`.AuthFlow`.
        """
        logger.trace(f"{cls.__name__}.from_config_model("
                     f"config={config!r}, **kwargs={kwargs!r})")

        config = resolve_endpoints(config, inplace=False)

        authorization_endpoint = cls._infer_auth_endpoint(config.leap_api_endpoint)
        token_endpoint = cls._infer_token_endpoint(config.leap_api_endpoint)

        # note: possible to de-dup via `DWaveAPIClient.from_config_model`, but
        # currently not worth the extra complexity (indirection layer).
        session_config = dict(
            cert=config.cert,
            headers=config.headers,
            proxies=dict(http=config.proxy, https=config.proxy),
            timeout=config.request_timeout,
            verify=not config.permissive_ssl)

        return cls(
            client_id=kwargs.pop('client_id', config.leap_client_id or OCEAN_SDK_CLIENT_ID),
            scopes=kwargs.pop('scopes', OCEAN_SDK_SCOPES),
            redirect_uri=kwargs.pop('redirect_uri', cls._OOB_REDIRECT_URI),
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            session_config=session_config)

    def run_oob_flow(self):
        """Run OAuth 2.0 code exchange (out-of-band flow.) 
        
        Runs the OAuth 2.0 Authorization Code exchange flow using the out-of-band code 
        exchange.

        After authorizing access by visiting the authorization URL displayed,
        user has to copy the authorization code and paste it back in their
        terminal. On successful completion Leap API access and refresh tokens
        are returned.
        """
        self.redirect_uri = self._OOB_REDIRECT_URI

        url = self.get_authorization_url()
        print(self._VISIT_AUTH_MSG, url)

        code = input(self._INPUT_CODE_MSG)
        return self.fetch_token(code=code)

    def run_redirect_flow(self, *, open_browser: Union[bool,Callable] = False,
                          timeout: Optional[float] = None):
        """Run the OAuth 2.0 code exchange (using locally hosted redirect URI). 
        
        Run the OAuth 2.0 Authorization Code exchange flow, using a redirect
        URI hosted on a local server.

        This flow is preferred to :meth:`.run_oob_flow` if you can access 
        localhost addresses serviced by Ocean running in your (development)
        environment from your browser.

        After authorizing access by visiting the authorization URL displayed,
        the flow will continue on the localhost redirect URI. The authorization
        code is exchanged automatically, and on successful completion, Leap API
        access and refresh tokens are returned.

        Example::
            from dwave.cloud.auth.flows import LeapAuthFlow
            from dwave.cloud.config import load_config, validate_config_v1

            config = validate_config_v1(load_config())
            flow = LeapAuthFlow.from_config_model(config)

            flow.run_redirect_flow(open_browser=True)

        """
        app = RequestCaptureApp(message=self._REDIRECT_DONE_MSG)
        srv = SingleRequestAppServer(
            host=self._REDIRECT_HOST,
            base_port=self._REDIRECT_PORT_RANGE[0],
            max_port=self._REDIRECT_PORT_RANGE[1],
            linear_tries=max(1, -sub(*self._REDIRECT_PORT_RANGE) // 10),
            app=app)
        srv.start()

        self.redirect_uri = srv.root_url

        url = self.get_authorization_url()
        print(self._VISIT_AUTH_MSG, url)
        if open_browser:
            if callable(open_browser):
                open_browser(url)
            else:
                webbrowser.open(url)

        try:
            srv.wait_shutdown(timeout)
        except TimeoutError:
            print(self._AUTH_TIMEOUT_MSG)
            return

        q = dict(parse_qsl(app.query))
        return self.fetch_token(code=q.get('code'), state=q.get('state'))

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
import time
import webbrowser
from collections import abc
from operator import sub
from typing import Any, Optional, Union, Literal, TYPE_CHECKING
from urllib.parse import urljoin

import click
import requests
from authlib.integrations.requests_client import OAuth2Session, OAuthError
from authlib.oauth2.rfc6749 import OAuth2Token
from authlib.common.security import generate_token
from authlib.common.urls import add_params_to_uri

from dwave.cloud.auth.config import OCEAN_SDK_CLIENT_ID, OCEAN_SDK_SCOPES
from dwave.cloud.auth.creds import Credentials
from dwave.cloud.auth.server import SingleRequestAppServer, RequestCaptureAndRedirectApp
from dwave.cloud.regions import resolve_endpoints
from dwave.cloud.utils.http import default_user_agent
from dwave.cloud.utils.logging import pretty_argvalues

if TYPE_CHECKING:
    from dwave.cloud.config.models import ClientConfig

__all__ = ['AuthFlow', 'LeapAuthFlow', 'OAuthError']

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
        revocation_endpoint:
            URL of the authorization server's OAuth 2.0 revocation endpoint.
        session_config:
            Configuration options for the low-level ``requests.Session`` used
            for all OAuth 2 requests. Supported options are: ``cert``,
            ``cookies``, ``headers``, ``proxies``, ``timeout``, ``verify``.
        leap_api_endpoint:
            Leap API endpoint, optional. Used to scope the token in credentials
            store.
        creds:
            :class:`~dwave.cloud.auth.creds.Credentials` store used to persist
            the token fetched.

    .. _OAuth 2.0 Authorization Code:
        https://datatracker.ietf.org/doc/html/rfc6749#section-4.1
    .. _PKCE:
        https://datatracker.ietf.org/doc/html/rfc7636
    """

    def __init__(self, *,
                 client_id: str,
                 scopes: abc.Sequence[str],
                 redirect_uri: str,
                 authorization_endpoint: str,
                 token_endpoint: str,
                 revocation_endpoint: Optional[str] = None,
                 session_config: Optional[abc.Mapping[str, Any]] = None,
                 leap_api_endpoint: Optional[str] = None,
                 creds: Optional[Credentials] = None
                 ):
        self.client_id = client_id
        self.scopes = ' '.join(scopes)
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint
        self.revocation_endpoint = revocation_endpoint
        self.leap_api_endpoint = leap_api_endpoint
        self.creds = creds

        # TODO: verify authorization/token/revocation endpoints are https

        self.session = OAuth2Session(
            client_id=client_id, scope=scopes, redirect_uri=redirect_uri,
            code_challenge_method='S256',
            # metadata set via kwargs
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint)

        # setup response logging
        def log_response(response, **kwargs):
            callee = type(self.session).__name__

            req = getattr(response, 'request', None)
            if req is not None:
                logger.trace("[%s] request=%r", callee,
                             dict(method=req.method, url=req.url,
                                  headers=req.headers, body=req.body))

            logger.trace("[%s] request(...) = (code=%r, body=%r, headers=%r, %r)",
                         callee, response.status_code, response.text, response.headers, kwargs)
        self.session.hooks["response"].append(log_response)

        # set default headers (overwritten by ``session_config['headers']``)
        self.session.headers.update({'User-Agent': default_user_agent()})

        if session_config is not None:
            self.update_session(session_config)

        self.token = self._load_token_from_creds()

        logger.debug(f"{type(self).__name__} initialized with: {pretty_argvalues()}")

    def update_session(self, config: abc.Mapping[str, Any]) -> None:
        """Update OAuth2Session/requests.Session with config values for:
        ``cert``, ``cookies``, ``headers``, ``proxies``, ``timeout``, ``verify``.
        """
        logger.trace(f"{type(self).__name__}.update_session(config={config!r})")

        self.session.headers.update(config.get('headers') or {})
        self.session.default_timeout = config.get('timeout', None)

        for key in ('cert', 'cookies', 'proxies', 'verify'):
            if key in config:
                setattr(self.session, key, config[key])

    def _load_token_from_creds(self) -> Optional[dict]:
        """Update session token with value from creds."""
        if not isinstance(self.creds, Credentials) or not self.leap_api_endpoint:
            return

        token = self.creds.get(self.leap_api_endpoint)
        logger.debug(f"{type(self).__name__} loaded token {token!r} from {self.creds}")

        return token

    def _save_token_to_creds(self, token: Union[OAuth2Token, dict]):
        """Persist session token to creds file."""
        if not isinstance(self.creds, Credentials) or not self.leap_api_endpoint:
            return

        if token:
            self.creds[self.leap_api_endpoint] = token
            logger.debug(f"{type(self).__name__} saved token {token!r} to {self.creds}")

    @property
    def redirect_uri(self) -> str:
        return self.session.redirect_uri

    @redirect_uri.setter
    def redirect_uri(self, value: str):
        self.session.redirect_uri = value

    @property
    def token(self) -> OAuth2Token:
        return self.session.token

    @token.setter
    def token(self, value: Union[OAuth2Token, dict]):
        self.session.token = value
        self._save_token_to_creds(value)

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
            code_verifier=self.code_verifier,
            **kwargs)

        logger.debug(f"{type(self).__name__}.fetch_token() = {token!r}")
        self._save_token_to_creds(token)

        return token

    def refresh_token(self):
        token = self.session.refresh_token(url=self.token_endpoint)
        self._save_token_to_creds(token)
        return token

    def revoke_token(self, *,
                     token: Optional[str] = None,
                     token_type_hint: Optional[
                         Literal['access_token', 'refresh_token']] = None
                     ) -> bool:
        """Revoke OAuth 2.0 token.

        Args:
            token:
                OAuth 2.0 access or refresh token to be revoked. If refresh token
                is specified, the authorization server should also revoke the
                corresponding access tokens. If ``token`` is not specified, then
                current refresh token is revoked.
            token_type_hint:
                Type of the token to be revoked.

        Returns:
            Token revocation status. True if revocation was successful, or token
            was invalid already. False in case of an error.

        Note:
            Token revocation requires that a ``revocation_endpoint`` is defined
            in :class:`~dwave.cloud.auth.flows.AuthFlow` construction.
        """
        if self.revocation_endpoint is None:
            raise TypeError("Revocation endpoint undefined.")

        response: requests.Response = self.session.revoke_token(
            url=self.revocation_endpoint, token=token,
            token_type_hint=token_type_hint)

        return response.ok

    def ensure_active_token(self):
        if not self.token:
            return False
        is_active = self.session.ensure_active_token(token=self.token)
        self._save_token_to_creds(self.token)
        return is_active

    def token_expires_soon(self, within: int = 60) -> Optional[bool]:
        """Is the token expired, or expires soon (within the next ``within`` seconds)?"""
        expires_at = self.token.get('expires_at')
        if not expires_at:
            return None
        return (expires_at - within) < time.time()


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
    _INPUT_CODE_MSG = 'Authorization code'

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

    @staticmethod
    def _infer_revocation_endpoint(leap_api_endpoint: str) -> str:
        return urljoin(leap_api_endpoint, '/leap/openid/revoke_token/')

    @staticmethod
    def _infer_leap_success_uri(leap_api_endpoint: str) -> str:
        return urljoin(leap_api_endpoint, '/leap/openid/success/')

    @staticmethod
    def _infer_leap_error_uri(leap_api_endpoint: str) -> str:
        return urljoin(leap_api_endpoint, '/leap/openid/error/')

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
        creds = Credentials()

        authorization_endpoint = cls._infer_auth_endpoint(config.leap_api_endpoint)
        token_endpoint = cls._infer_token_endpoint(config.leap_api_endpoint)
        revocation_endpoint = cls._infer_revocation_endpoint(config.leap_api_endpoint)

        # note: possible to de-dup via `DWaveAPIClient.from_config_model`, but
        # currently not worth the extra complexity (indirection layer).
        session_config = dict(
            cert=config.cert,
            headers=config.headers,
            proxies=dict(http=config.proxy, https=config.proxy),
            timeout=config.request_timeout,
            verify=not config.permissive_ssl)

        flow = cls(
            client_id=kwargs.pop('client_id', config.leap_client_id or OCEAN_SDK_CLIENT_ID),
            scopes=kwargs.pop('scopes', OCEAN_SDK_SCOPES),
            redirect_uri=kwargs.pop('redirect_uri', cls._OOB_REDIRECT_URI),
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            revocation_endpoint=revocation_endpoint,
            session_config=session_config,
            leap_api_endpoint=config.leap_api_endpoint,
            creds=creds)

        # save full config if needed later
        flow.config = config

        return flow

    def run_oob_flow(self, *, open_browser: Union[bool, abc.Callable] = False):
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
        click.echo(click.style(self._VISIT_AUTH_MSG, bold=True), nl=False)
        click.echo(click.style(url, underline=True))
        click.echo()
        if open_browser:
            if callable(open_browser):
                open_browser(url)
            else:
                webbrowser.open(url)

        code = click.prompt(click.style(self._INPUT_CODE_MSG, bold=True))
        click.echo()
        return self.fetch_token(code=code)

    def run_redirect_flow(self, *, open_browser: Union[bool, abc.Callable] = False,
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
        error_uri = self._infer_leap_error_uri(self.leap_api_endpoint)
        success_uri = self._infer_leap_success_uri(self.leap_api_endpoint)

        def exchange_code(app: RequestCaptureAndRedirectApp) -> str:
            # error responses redirect immediately
            if 'error' in app.query:
                app.set_exception(
                    OAuthError(error=app.query['error'],
                               description=app.query.get('error_description')))
                return add_params_to_uri(error_uri, app.query)

            # when code received, exchange it for token
            try:
                # verify state (Cross-Site Request Forgery protection)
                state = app.query.get('state')
                if state != self.state:
                    raise ValueError('State mismatch')

                self.fetch_token(code=app.query.get('code'), state=state)

            except OAuthError as exc:
                # store for main thread
                app.set_exception(exc)

                # redirect to leap error page
                query = dict(error=exc.error, error_description=exc.description,
                             state=app.query.get('state'))
                return add_params_to_uri(error_uri, query)

            except Exception as exc:
                # store for main thread
                app.set_exception(exc)

                # redirect to leap error page
                query = dict(error=type(exc).__name__, error_description=str(exc),
                             state=app.query.get('state'))
                return add_params_to_uri(error_uri, query)

            # redirect to leap success page
            return success_uri

        app = RequestCaptureAndRedirectApp(
            message=self._REDIRECT_DONE_MSG,
            redirect_uri=exchange_code,
            include_query=False)

        srv = SingleRequestAppServer(
            host=self._REDIRECT_HOST,
            base_port=self._REDIRECT_PORT_RANGE[0],
            max_port=self._REDIRECT_PORT_RANGE[1],
            linear_tries=max(1, -sub(*self._REDIRECT_PORT_RANGE) // 10),
            app=app)
        srv.start()

        self.redirect_uri = srv.root_url

        url = self.get_authorization_url()
        click.echo(click.style(self._VISIT_AUTH_MSG, bold=True), nl=False)
        click.echo(click.style(url, underline=True))
        click.echo()
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

        # now raise auth exception if set by the app during code exchange
        app.exception()

        return self.token

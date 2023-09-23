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
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urljoin

from authlib.integrations.requests_client import OAuth2Session
from authlib.common.security import generate_token

from dwave.cloud.auth.config import OCEAN_SDK_CLIENT_ID, OCEAN_SDK_SCOPES
from dwave.cloud.config.models import ClientConfig


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

    def __init__(self,
                 client_id: str,
                 scopes: Sequence[str],
                 redirect_uri: str,
                 authorization_endpoint: str,
                 token_endpoint: str,
                 session_config: Optional[Dict[str, Any]] = None,
                 ):
        self.client_id = client_id
        self.scopes = ' '.join(scopes)
        self.redirect_uri = redirect_uri
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint

        self.session = OAuth2Session(
            client_id=client_id, scope=scopes, redirect_uri=redirect_uri,
            code_challenge_method='S256')

        if session_config is not None:
            self.update_session(session_config)

    def update_session(self, config: Dict[str, Any]) -> None:
        """Update OAuth2Session/requests.Session with config values for:
        ``cert``, ``cookies``, ``headers``, ``proxies``, ``timeout``, ``verify``.
        """
        self.session.headers.update(config.get('headers', {}))
        self.session.default_timeout = config.get('timeout', None)

        for key in ('cert', 'cookies', 'proxies', 'verify'):
            if key in config:
                setattr(self.session, key, config[key])

    def get_authorization_url(self) -> str:
        self.state = generate_token(30)
        self.code_verifier = generate_token(48)

        url, _ = self.session.create_authorization_url(
            url=self.authorization_endpoint,
            state=self.state,
            code_verifier=self.code_verifier)

        return url

    # todo: propagate headers, currently via kwargs
    def fetch_token(self, code: str, **kwargs) -> dict:
        return self.session.fetch_token(
            url=self.token_endpoint,
            grant_type='authorization_code',
            code=code,
            **kwargs)


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
            client_id=kwargs.pop('client_id', OCEAN_SDK_CLIENT_ID),
            scopes=kwargs.pop('scopes', OCEAN_SDK_SCOPES),
            redirect_uri=kwargs.pop('redirect_uri', cls._OOB_REDIRECT_URI),
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            session_config=session_config)

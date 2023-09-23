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

from authlib.integrations.requests_client import OAuth2Session
from authlib.common.security import generate_token


__all__ = ['AuthFlow']

logger = logging.getLogger(__name__)


class AuthFlow:
    """`OAuth 2.0 Authorization Code`_ exchange flow with `PKCE`_ extension
    for public (secretless) clients.

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
        `cert`, `cookies`, `headers`, `proxies`, `timeout`, `verify`.
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

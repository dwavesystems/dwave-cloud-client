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

from dwave.cloud.api import exceptions
from dwave.cloud.api.client import SAPIClient
from dwave.cloud.package_info import __packagename__, __version__


class TestConfig(unittest.TestCase):
    """Session is initiated from config."""

    def test_defaults(self):
        client = SAPIClient()

        self.assertEqual(client.config, SAPIClient.DEFAULTS)
        self.assertIsInstance(client.session, requests.Session)

        # verify Retry object config
        retry = client.session.get_adapter('https://').max_retries
        conf = SAPIClient.DEFAULTS['retry']
        self.assertEqual(retry.total, conf['total'])

    def test_init(self):
        config = dict(endpoint='https://test.com/path/',
                      token=str(uuid.uuid4()),
                      timeout=1,
                      retry=dict(total=3),
                      headers={'Custom': 'Field 123'},
                      verify=False,
                      proxies={'https': 'http://proxy.com'})

        client = SAPIClient(**config)

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

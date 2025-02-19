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

import re
import unittest
import zlib
from urllib.parse import urljoin

import requests
import requests_mock
from pydantic import TypeAdapter, BaseModel

from dwave.cloud.api import exceptions, constants
from dwave.cloud.api.resources import ResourceBase, accepts, compress_if


class MathResource(ResourceBase):
    """A prototypical resource, implementing an interface to a 2-integer adder."""

    resource_path = 'math/'

    class AddResult(BaseModel):
        out: int

    def add(self, in1: int, in2: int) -> int:
        path = 'add/'
        response = self.session.get(path, params=dict(in1=in1, in2=in2))
        result = response.json()
        return TypeAdapter(MathResource.AddResult).validate_python(result).out

    def nonexisting(self):
        return self.session.get('nonexisting')

    @accepts(media_type='application/vnd.dwave.api.mock+json', accept_version='>0')
    def format(self, subpath: str) -> dict:
        return self.session.get(f'format/{subpath}').json()

    @accepts(media_type='application/vnd.dwave.api.mock+json',
             ask_version='1.1', accept_version='>=1.1,<2')
    def version(self, v: str = '') -> dict:
        return self.session.get(f'version/{v}').json()

    @accepts(media_type_params=dict(a='1', b='2'))
    def params(self) -> dict:
        return self.session.get(f'params/').json()

    @accepts(media_type='application/vnd.dwave.api.mock+json', version='>0', strict_mode=False)
    def nonstrict(self) -> dict:
        return self.session.get(f'nonstrict/').json()

    @compress_if(lambda _, __, compress: compress)
    def zip(self, data: str, compress: bool = False) -> dict:
        headers = dict(compress=str(compress))
        return self.session.post('zip/', headers=headers, data=data).json()


class TestMockResource(unittest.TestCase):
    """Test request formation and response parsing (including error
    handling and api version validation) works correctly for all
    :class:`dwave.cloud.api.resources.ResourceBase` methods.
    """

    endpoint = 'http://test.com/path/'

    def setUp(self):
        self.base_uri = urljoin(self.endpoint, MathResource.resource_path)

    @requests_mock.Mocker()
    def test_basic_flow(self, m):
        m.get(urljoin(self.base_uri, 'add/?in1=1&in2=2'), json=dict(out=3))

        resource = MathResource(endpoint=self.endpoint)
        res = resource.add(1, 2)
        self.assertEqual(res, 3)

    @requests_mock.Mocker()
    def test_nonexisting_path(self, m):
        m.get(requests_mock.ANY, status_code=404)

        resource = MathResource(endpoint=self.endpoint)
        with self.assertRaises(exceptions.ResourceNotFoundError):
            resource.nonexisting()

    @requests_mock.Mocker()
    def test_media_type(self, m):
        m.get(urljoin(self.base_uri, 'format/ok'), json=dict(works=True),
              headers={'Content-Type': 'application/vnd.dwave.api.mock+json; version=1'})
        m.get(urljoin(self.base_uri, 'format/wrong'), json={},
              headers={'Content-Type': 'unknown'})

        resource = MathResource(endpoint=self.endpoint)

        # media_type is set and correct
        res = resource.format(subpath='ok')
        self.assertEqual(res['works'], True)

        # media_type is set and incorrect
        with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'^Received media type'):
            resource.format(subpath='wrong')

    @requests_mock.Mocker()
    def test_accept_version(self, m):
        m.get(urljoin(self.base_uri, 'version/1.0'), json=dict(v='1.0'),
              headers={'Content-Type': 'application/vnd.dwave.api.mock+json; version=1.0'})
        m.get(urljoin(self.base_uri, 'version/1.1'), json=dict(v='1.1'),
              headers={'Content-Type': 'application/vnd.dwave.api.mock+json; version=1.1'})
        m.get(urljoin(self.base_uri, 'version/1.2'), json=dict(v='1.2'),
              headers={'Content-Type': 'application/vnd.dwave.api.mock+json; version=1.2'})
        m.get(urljoin(self.base_uri, 'version/2.0'), json=dict(v='2.0'),
              headers={'Content-Type': 'application/vnd.dwave.api.mock+json; version=2.0'})

        resource = MathResource(endpoint=self.endpoint)

        # accept_version is `>=1.1,<2`
        # versions 1.1 and 1.2 work
        self.assertEqual(resource.version('1.1')['v'], '1.1')
        self.assertEqual(resource.version('1.2')['v'], '1.2')

        # versions 1.0 and 2.0 fail
        with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'^API response format version'):
            resource.version('1.0')
        with self.assertRaisesRegex(exceptions.ResourceBadResponseError, r'^API response format version'):
            resource.version('2.0')

    @requests_mock.Mocker()
    def test_ask_version(self, m):
        # return exactly version asked for
        def json_callback(request: requests.Request, context):
            accept = request.headers['Accept']
            version = re.search(r'version=(\d+(\.\d+)?)', accept).group(1)
            context.headers['Content-Type'] = f'application/vnd.dwave.api.mock+json; version={version}'
            return dict(v=version)
        m.get(urljoin(self.base_uri, 'version/'), json=json_callback)

        resource = MathResource(endpoint=self.endpoint)

        # ask_version is set to 1.1; ensure that's actually sent
        self.assertEqual(resource.version()['v'], '1.1')

    @requests_mock.Mocker()
    def test_accepts_params(self, m):
        # return exactly version asked for
        def json_callback(request: requests.Request, context):
            return dict(accept=request.headers['Accept'])
        m.get(urljoin(self.base_uri, 'params/'), json=json_callback,
              headers={'Content-Type': constants.DEFAULT_API_MEDIA_TYPE})

        resource = MathResource(endpoint=self.endpoint)

        # ask_version is set to 1.1; ensure that's actually sent
        expected_accept = f"{constants.DEFAULT_API_MEDIA_TYPE}; a=1; b=2"
        self.assertEqual(resource.params()['accept'], expected_accept)

    @requests_mock.Mocker()
    def test_resource_partial_decoration_with_accepts(self, m):
        m.get(urljoin(self.base_uri, 'add/?in1=1&in2=2'), json=dict(out=3))
        m.get(urljoin(self.base_uri, 'format/ok'), json=dict(works=True),
              headers={'Content-Type': 'application/vnd.dwave.api.mock+json; version=1'})

        resource = MathResource(endpoint=self.endpoint)

        # test validation works on the decorated endpoint (`format/ok`)
        res = resource.format(subpath='ok')
        self.assertEqual(res['works'], True)

        # test undecorated endpoint is not validated
        res = resource.add(1, 2)
        self.assertEqual(res, 3)

    @requests_mock.Mocker()
    def test_nonstrict_mode(self, m):
        m.get(urljoin(self.base_uri, 'nonstrict/'), json=dict(works=True))

        resource = MathResource(endpoint=self.endpoint)

        # media_type is not enforced (because response lacks Content-Type)
        res = resource.nonstrict()
        self.assertEqual(res['works'], True)

    @requests_mock.Mocker()
    def test_compression(self, m):
        data = b'test payload'
        data_compressed = zlib.compress(data)

        def match_data(request):
            if request.headers.get('compress') == 'True':
                self.assertEqual(request.headers.get('Content-Encoding'), 'deflate')
                expected = data_compressed
            else:
                self.assertNotIn('Content-Encoding', request.headers)
                expected = data

            body = request.body
            if not isinstance(body, bytes):
                body = b''.join(request.body)

            return body == expected

        m.post(requests_mock.ANY, status_code=404)
        m.post(urljoin(self.base_uri, 'zip/'), additional_matcher=match_data, json=dict(ok=True))

        resource = MathResource(endpoint=self.endpoint)

        res = resource.zip(data, compress=True)
        self.assertTrue(res['ok'])

        res = resource.zip(data, compress=False)
        self.assertTrue(res['ok'])

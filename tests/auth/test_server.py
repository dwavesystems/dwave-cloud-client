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

import concurrent.futures
import socket
import time
import unittest
from secrets import token_hex
from urllib.parse import urljoin

import requests
from parameterized import parameterized

from dwave.cloud.auth.server import (
    iterports, BackgroundAppServer, SingleRequestAppServer,
    RequestCaptureApp, RequestCaptureAndRedirectApp)


class TestPortsGenerator(unittest.TestCase):

    @parameterized.expand([
        (-1, 0, 0, None),
        (0, -1, 0, None),
        (0, 0, -1, None),
        (0, 0, 0, -1),
        (0, 1, 3, None),
        (0, 1, 2, 1),
    ])
    def test_edge_cases(self, lb, ub, n_lin, n_rand):
        with self.assertRaises(ValueError):
            next(iterports(start=lb, end=ub, n_lin=n_lin, n_rand=n_rand))

    @parameterized.expand([
        (0, 1, 0, None, []),
        (0, 1, 1, None, [0]),
        (0, 1, 2, None, [0, 1]),
        (0, 9, 3, None, [0, 1, 2]),
        (0, 9, 0, None, []),
        (0, 9, 0, 10, []),
    ])
    def test_sequences(self, lb, ub, n_lin, n_rand, lin):
        ports = list(iterports(start=lb, end=ub, n_lin=n_lin, n_rand=n_rand))
        self.assertEqual(ports[:n_lin], lin)
        self.assertEqual(set(ports[n_lin:]), set(range(lb, ub+1)).difference(lin))


class TestBackgroundAppServer(unittest.TestCase):

    @unittest.mock.patch('threading.excepthook', lambda *a, **kw: None)
    def test_port_search(self):
        base_port = 64000

        first = BackgroundAppServer(
            host='', base_port=base_port, max_port=base_port, app=None)
        first.start()
        self.assertEqual(first.server.server_port, base_port)

        second = BackgroundAppServer(
            host='', base_port=base_port, max_port=base_port+9, linear_tries=2, app=None)
        second.start()
        self.assertEqual(second.server.server_port, base_port+1)

        # test port search exhaustion
        with self.assertRaises(RuntimeError):
            third = BackgroundAppServer(
                host='', base_port=base_port, max_port=base_port+1, app=None)
            third.start()
            third.wait_shutdown()
            third.exception()

        first.stop()
        second.stop()

    def test_basics(self):
        base_port = 64010
        response = 'It works!'

        def app(environ, start_response):
            start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
            return [response.encode('utf-8')]

        srv = BackgroundAppServer(
            host='127.0.0.1', base_port=base_port, max_port=base_port+9, app=app)
        srv.start()

        # test root url
        self.assertEqual(srv.root_url, f'http://127.0.0.1:{srv.server.server_port}/')

        # test response
        self.assertEqual(requests.get(srv.root_url).text, response)

        srv.stop()

    def test_multithreading(self):
        base_port = 64020
        response = 'It works!'
        response_delay = 2
        n_requests = 2

        def app(environ, start_response):
            start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
            time.sleep(response_delay)
            return [response.encode('utf-8')]

        srv = BackgroundAppServer(
            host='127.0.0.1', base_port=base_port, max_port=base_port+9, app=app)
        srv.start()

        url = srv.root_url
        def get_url(url, timeout=10):
            return requests.get(url, timeout=timeout).text

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as executor:
            fs = {executor.submit(get_url, url): url for _ in range(n_requests)}
            done, not_done = concurrent.futures.wait(
                fs, timeout=response_delay*1.3, return_when=concurrent.futures.ALL_COMPLETED)

        # verify all requests finished within (1.3 * response_delay) < (response_delay * n_request)
        self.assertEqual(len(done), n_requests)

        # all responses correct
        for f in fs:
            self.assertEqual(f.result(), response)

        srv.stop()

    def test_wait_shutdown(self):
        base_port = 64030
        srv = BackgroundAppServer(
            host='127.0.0.1', base_port=base_port, max_port=base_port+9, app=None)
        srv.start()

        with self.assertRaises(TimeoutError):
            srv.wait_shutdown(0.5)

        srv.stop()
        srv.wait_shutdown()
        self.assertFalse(srv.is_alive())

    def test_timeout(self):
        # XXX: not the most elegant method of checking if server closes the
        # connection after `timeout`, so we should revisit this at some point.
        # I have manually verified the behavior is correct, thought, with
        # telnet, curl and wireshark.

        base_port = 64040
        timeout = 0.5

        def app(environ, start_response):
            start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
            return ['It works!'.encode('utf-8')]

        srv = BackgroundAppServer(
            host='127.0.0.1', base_port=base_port, max_port=base_port+9,
            app=app, timeout=timeout)
        srv.start()

        def do_get(timeout=0):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(srv.server.server_address)

            time.sleep(timeout)

            try:
                sock.send('GET /\n\n'.encode('utf-8'))
                data = sock.recv(1024)
                sock.close()
            except ConnectionError:
                # seems to work only on macos and win
                data = ''

            return data

        # base case, no timeout, server returns data
        self.assertTrue(len(do_get(timeout=0)) > 0)

        # client waits 2*connection_timeout, so receives nothing
        self.assertTrue(len(do_get(timeout=timeout*2)) == 0)

        srv.stop()


class TestSingleRequestAppServer(unittest.TestCase):

    def test_function(self):
        base_port = 64050
        response = 'It works!'

        def app(environ, start_response):
            start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
            return [response.encode('utf-8')]

        srv = SingleRequestAppServer(
            host='127.0.0.1', base_port=base_port, max_port=base_port+9, app=app)
        srv.start()

        # test response
        self.assertEqual(requests.get(srv.root_url).text, response)

        # test server shuts down within reasonable time
        srv.wait_shutdown(timeout=0.5)
        self.assertFalse(srv.is_alive())


class TestRequestCaptureApp(unittest.TestCase):

    def test_function(self):
        base_port = 64060
        response = 'Got it!'

        app = RequestCaptureApp(response)
        srv = SingleRequestAppServer(
            host='127.0.0.1', base_port=base_port, max_port=base_port+9, app=app)
        srv.start()

        query = dict(a=token_hex(8), b=token_hex(16))

        # test response
        self.assertEqual(requests.get(srv.root_url, params=query).text, response)

        # wait for server shutdown to make sure request is done
        srv.wait_shutdown()

        # test url/query capture
        self.assertTrue(app.uri.startswith(f"{srv.root_url}?"))
        self.assertEqual(app.query, query)


class TestRequestCaptureAndRedirectApp(unittest.TestCase):
    base_port = 64070

    def setUp(self):
        self.success_text = 'success'
        self.success_app = RequestCaptureApp(message=self.success_text)
        self.success_srv = SingleRequestAppServer(
            host='127.0.0.1', base_port=self.base_port, max_port=self.base_port+9,
            app=self.success_app)
        self.success_srv.start()
        self.success_uri = urljoin(self.success_srv.root_url, '/success')

        self.error_text = 'error'
        self.error_app = RequestCaptureApp(message=self.error_text)
        self.error_srv = SingleRequestAppServer(
            host='127.0.0.1', base_port=self.base_port, max_port=self.base_port+9,
            app=self.error_app)
        self.error_srv.start()
        self.error_uri = urljoin(self.error_srv.root_url, '/error')

        def redirect_uri(app):
            return self.error_uri if 'error' in app.query else self.success_uri

        self.redirect_app = RequestCaptureAndRedirectApp(
            message='redirecting', redirect_uri=redirect_uri, include_query=True)
        self.redirect_srv = SingleRequestAppServer(
            host='127.0.0.1', base_port=self.base_port, max_port=self.base_port+9,
            app=self.redirect_app)
        self.redirect_srv.start()

    def test_redirect_on_success(self):
        query = dict(a=token_hex(8), b=token_hex(16))

        # test response
        req = requests.get(self.redirect_srv.root_url, params=query)
        self.assertEqual(req.text, self.success_text)

        # wait for server shutdown to make sure request is done
        self.redirect_srv.wait_shutdown()

        # test url/query capture
        self.assertTrue(self.success_app.uri.startswith(self.success_uri))
        self.assertEqual(self.success_app.query, query)

    def test_redirect_on_error(self):
        query = dict(error='error-id', error_description='error message')

        # test response
        req = requests.get(self.redirect_srv.root_url, params=query)
        self.assertEqual(req.text, self.error_text)

        # wait for server shutdown to make sure request is done
        self.redirect_srv.wait_shutdown()

        # test url/query capture
        self.assertTrue(self.error_app.uri.startswith(self.error_uri))
        self.assertEqual(self.error_app.query, query)

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

import logging
import random
import sys
import threading
import traceback
from socketserver import ThreadingMixIn
from typing import Callable, Iterator, Optional, Union
from urllib.parse import urljoin, urlsplit, parse_qsl, SplitResult
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer
from wsgiref.util import request_uri

logger = logging.getLogger(__name__)


class LoggingStream:
    """Provide file-like interface to a logger."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        for line in message.split('\n'):
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        pass

# stream interface to our local logger
request_logging_stream = LoggingStream(logger, logging.DEBUG)


class LoggingWSGIRequestHandler(WSGIRequestHandler):
    """WSGIRequestHandler subclass that logs to our logger, instead of to
    ``sys.stderr`` (as hardcoded in ``http.server.BaseHTTPRequestHandler``).
    """

    def log_message(self, format, *args):
        logger.info(format, *args)

    def get_stderr(self):
        return request_logging_stream


class ErrorLoggingTCPServerMixin:
    """Extend :class:`http.server.HTTPServer`/:class:`socketserver.TCPServer`
    to log errors to our logger, instead of to a hard-coded ``sys.stderr`` stream.
    """

    def handle_error(self, request, client_address):
        traceback.print_exception(*sys.exc_info(), file=request_logging_stream)


class SocketTimeoutTCPServerMixin:
    """Extend :class:`http.server.HTTPServer`/:class:`socketserver.TCPServer`
    by setting a timeout on each client connection socket.

    This prevents hanging connections waiting for client data.
    """
    connection_timeout = None

    def get_request(self):
        conn, addr = super().get_request()
        if self.connection_timeout is not None:
            conn.settimeout(self.connection_timeout)
        return conn, addr


class NoAddrReuseTCPServerMixin:
    """Disable address reuse for a TCPServer while socket in TIME_WAIT on Linux,
    or unless SO_EXCLUSIVEADDRUSE is set on Windows. It's set to truthy value by
    HTTPServer subclass, so this mixin reverts the TCPServer default.

    See https://learn.microsoft.com/en-us/windows/win32/winsock/so-exclusiveaddruse.
    """
    allow_reuse_address = False


class ThreadingWSGIServer(ThreadingMixIn, SocketTimeoutTCPServerMixin,
                          ErrorLoggingTCPServerMixin, NoAddrReuseTCPServerMixin,
                          WSGIServer):
    """:class:`~wsgiref.simple_server.WSGIServer` subclass that:
    - supports multithreading, i.e. handles each request in a new thread,
    - set a timeout on socket connections, and
    - logs errors to our configured logger, instead of to stderr.
    """


def iterports(start: int, end: int,
              n_lin: int, n_rand: Optional[int] = None) -> Iterator[int]:
    """Server port proposal generator. Starts with a linear search, then
    switches to a randomized search (random permutation of remaining ports).
    """
    # sanity checks
    if n_rand is None:
        n_rand = end - start + 1 - n_lin
    if start < 0 or end < 0 or n_lin < 0 or (n_rand is not None and n_rand < 0):
        raise ValueError("Non-negative integers required for all parameters")
    if n_lin + n_rand > end - start + 1:
        raise ValueError("Sum of tries must be less or equal to population size")

    # linear search
    yield from range(start, start + n_lin)

    # randomized search
    yield from random.sample(range(start + n_lin, end + 1), k=n_rand)


class BackgroundAppServer(threading.Thread):
    """WSGI application server container that runs in a background thread,
    handling each request in its own thread.
    """

    def _make_server(self) -> ThreadingWSGIServer:
        """Instantiate a http server, similarly to :func:`~wsgiref.simple_server.make_server`,
        but bounding it to the first port available, instead of failing if the specified
        port is unavailable.

        Port search is a combination of linear and randomized search, starting at
        ``base_port`` and ending with ``max_port`` (both ends inclusive).
        """

        for port in iterports(start=self.base_port, end=self.max_port,
                              n_lin=self.linear_tries, n_rand=self.randomized_tries):
            try:
                server = make_server(self.host, port, self.app,
                                     server_class=ThreadingWSGIServer,
                                     handler_class=LoggingWSGIRequestHandler)
                server.connection_timeout = self.timeout
                return server
            except OSError:
                # linux: "OSError: [Errno 98] Address already in use"
                # macos: "OSError: [Errno 48] Address already in use"
                # win: "OSError: [WinError 10048] Only one usage of each socket address ... permitted"
                pass

        raise RuntimeError("Unable to find available port in range: "
                           f"[{self.base_port}, {self.max_port}].")

    @property
    def server(self) -> ThreadingWSGIServer:
        """HTTP server accessor that creates the actual server instance
        (and binds it to host:port) on first access.
        """

        with self._server_lock:
            self._server = getattr(self, '_server', None)

            if self._server is None:
                self._server = self._make_server()
                self._root_url = 'http://{}:{}/'.format(*self._server.server_address)

            return self._server

    def __init__(self, *, host: str, base_port: int, max_port: int, app: Callable,
                 linear_tries: int = 1, randomized_tries: Optional[int] = None,
                 timeout: Optional[float] = 10):
        super().__init__(daemon=True)

        # store config, but start the web server (and bind to address) on run()
        self.host = host
        self.base_port = base_port
        self.max_port = max_port
        self.linear_tries = linear_tries
        self.randomized_tries = randomized_tries
        self.app = app
        self.timeout = timeout
        self._server_lock = threading.RLock()
        self._server_ready = threading.Event()

    def run(self):
        """Don't call this method directly. Instead call `.start()`."""
        logger.debug(f"Running {type(self).__name__} worker thread")
        try:
            srv = self.server
            self._server_ready.set()
            srv.serve_forever()
        except:
            # make exception discoverable from the main thread
            self._exc_info = sys.exc_info()
        finally:
            self.server.server_close()
        logger.debug(f"{type(self).__name__} worker thread done.")

    def exception(self):
        """Raises an exception that was uncaught in the worker thread.
        Intended to be called from your main thread.
        """

        # this idea for exception propagation from thread to main comes from
        # https://stackoverflow.com/a/1854263, published under CC BY-SA 4.0.
        if hasattr(self, '_exc_info') and self._exc_info:
            raise self._exc_info[1].with_traceback(self._exc_info[2])

    def stop(self):
        logger.debug(f"{type(self).__name__}.stop()")
        self.server.shutdown()
        self.join()

    @property
    def root_url(self):
        """Server root URL, or None if server not started yet."""
        self.wait_ready()
        return self._root_url

    def wait_ready(self, timeout: Optional[float] = None):
        """Waits for ``timeout`` in seconds for server to become ready (thread
        is spawned, http server is created, address is bound, etc).
        """
        self._server_ready.wait(timeout)
        if not self._server_ready.is_set():
            raise TimeoutError("Server has not become ready in the allotted period.")

    def wait_shutdown(self, timeout: Optional[float] = None):
        """Waits for ``timeout`` in seconds for server to shutdown before
        raising a ``TimeoutError``.
        """
        logger.debug(f"{type(self).__name__}.wait_shutdown(timeout={timeout})")
        self.join(timeout)
        if self.is_alive():
            raise TimeoutError("Server has not shut down in the allotted timeout.")


class SingleRequestAppServer(BackgroundAppServer):
    """An extension of :class:`.BackgroundAppServer` that terminates after a
    single request has completed.
    """

    # note: we can't simply use `server.handle_request()` to handle a single request
    # because of the problem with some browsers (pre-)opening "ghost" connection (not
    # sending any data), as it is described in python's `ThreadingHTTPServer` docs
    # (https://docs.python.org/3/library/http.server.html#http.server.ThreadingHTTPServer)

    def __init__(self, **kwargs):
        app = kwargs.pop('app')

        def single_request_app(*args, **kwargs):
            try:
                return app(*args, **kwargs)
            finally:
                # we're done after the first request
                self.server.shutdown()

        super().__init__(app=single_request_app, **kwargs)


class RequestCaptureApp:
    """A simple WSGI application that stores request data (currently only URL),
    and displays a static message in response.
    """

    def __init__(self, message: str):
        self.message = message
        self.uri: str = None
        self.parts: SplitResult = None
        self.query: dict = None

    def store_request(self, environ: dict):
        # store the URI accessed
        self.uri = request_uri(environ, include_query=True)
        self.parts = urlsplit(self.uri)
        self.query = dict(parse_qsl(self.parts.query))
        # in the future, we might also store: method, body data, headers, etc,
        # but we don't need that for now

    def set_exception(self, exc):
        self._exc = exc

    def exception(self):
        if hasattr(self, '_exc') and self._exc:
            raise self._exc

    def __call__(self, environ: dict, start_response: Callable):
        self.store_request(environ)

        start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
        return [self.message.encode('utf-8')]


class RequestCaptureAndRedirectApp(RequestCaptureApp):
    """A simple WSGI application that stores request data (currently only URL),
    and redirects to `redirect_uri`, possibly dynamically generated.
    """

    def __init__(self, message: str,
                 redirect_uri: Union[str, Callable],
                 include_query: bool = True):
        super().__init__(message)

        self.redirect_uri = redirect_uri
        self.include_query = include_query

    def __call__(self, environ: dict, start_response: Callable):
        self.store_request(environ)

        if callable(self.redirect_uri):
            uri = self.redirect_uri(self)
        else:
            uri = self.redirect_uri

        if self.include_query:
            uri = urljoin(uri, f'?{self.parts.query}')

        start_response('302 Found', [
            ('Content-Type', 'text/plain; charset=utf-8'),
            ('Location', uri),
        ])

        # just in case, include the message in body
        return [self.message.encode('utf-8')]

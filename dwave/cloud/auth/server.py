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
from typing import Optional, Callable
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer

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


class LoggingWSGIServer(WSGIServer):
    """WSGIServer subclass that logs to our logger, instead of to ``sys.stderr``
    (as hardcoded in ``socketserver.BaseServer.handle_error``).
    """

    def handle_error(self, request, client_address):
        traceback.print_exception(*sys.exc_info(), file=request_logging_stream)


class ThreadingWSGIServer(ThreadingMixIn, LoggingWSGIServer):
    daemon_threads = True


def _ports(start, end, n_lin, n_rand=None):
    """Server port proposal generator. Starts with a linear search, then
    switches to a randomized search (random permutation of remaining ports).
    """
    # sanity checks
    if start < 0 or end < 0 or n_lin < 0 or (n_rand is not None and n_rand < 0):
        raise ValueError("Non-negative integers required for all parameters")
    if n_rand is None:
        n_rand = end - start + 1 - n_lin
    if n_lin + n_rand > end - start + 1:
        raise ValueError("Sum of tries must be less or equal to population size")

    # linear search
    yield from range(start, start + n_lin)

    # randomized search
    yield from random.sample(range(start + n_lin, end + 1), k=n_rand)


def _adaptive_make_server(
        app: Callable,
        host: str,
        base_port: int,
        max_port: int,
        linear_tries: int = 1,
        randomized_tries: Optional[int] = None
        ) -> ThreadingWSGIServer:
    """Instantiate a http server, similarly to :func:`~wsgiref.simple_server.make_server`,
    but bounding it to the first port available, instead of failing if the specified
    port is unavailable.

    Port search is a combination of linear and randomized search, starting at
    ``base_port`` and ending with ``max_port`` (both ends inclusive).
    """
    for port in _ports(start=base_port, end=max_port,
                       n_lin=linear_tries, n_rand=randomized_tries):
        try:
            return make_server(host, port, app,
                               server_class=ThreadingWSGIServer,
                               handler_class=LoggingWSGIRequestHandler)
        except OSError as exc:
            # handle only "[Errno 98] Address already in use"
            if exc.errno != 98:
                raise

    raise RuntimeError("Unable to find available port in range: "
                       f"[{base_port}, {max_port}].")


class BackgroundAppServer(threading.Thread):
    """WSGI server container for a wsgi app that runs asynchronously (in a
    separate thread).
    """

    def _make_server(self):
        # create http server, and bind it to first available port >= base_port
        return _adaptive_make_server(host=self.host, base_port=self.base_port,
                                     max_port=self.max_port, app=self.app)

    @property
    def server(self):
        """HTTP server accessor that creates the actual server instance
        (and binds it to host:port) on first access.
        """

        with self._server_lock:
            self._server = getattr(self, '_server', None)

            if self._server is None:
                self._server = self._make_server()

            return self._server

    def __init__(self, host, base_port, max_port, app):
        super().__init__(daemon=True)

        # store config, but start the web server (and bind to address) on run()
        self.host = host
        self.base_port = base_port
        self.max_port = max_port
        self.app = app
        self._server_lock = threading.RLock()

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.join()

    def get_callback_url(self):
        return 'http://{}:{}/'.format(*self.server.server_address)

    def ensure_started(self):
        if not self.is_alive():
            self.start()
        return True

    def ensure_stopped(self):
        if self.is_alive():
            self.stop()

    def wait_shutdown(self, timeout=None):
        logger.debug('%s.wait_shutdown(timeout=%r)', type(self).__name__, timeout)
        self.join(timeout)

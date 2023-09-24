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
import random
import logging
import traceback
import threading
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


class WSGIAsyncServer(threading.Thread):
    """WSGI server container for a wsgi app that runs asynchronously (in a
    separate thread).
    """

    def _safe_make_server(self, host, base_port, app, tries=20):
        """Instantiate a http server. Discover available port starting with
        `base_port` (use linear and random search).
        """

        def ports(start, linear=5):
            """Server port proposal generator. Starts with a linear search, then
            converts to a random look up.
            """
            for port in range(start, start + linear):
                yield port
            while True:
                yield random.randint(port + 1, (1<<16) - 1)

        for _, port in zip(range(tries), ports(start=base_port)):
            try:
                return make_server(host, port, app,
                                   server_class=LoggingWSGIServer,
                                   handler_class=LoggingWSGIRequestHandler)
            except OSError as exc:
                # handle only "[Errno 98] Address already in use"
                if exc.errno != 98:
                    raise

        raise RuntimeError("unable to find available port to bind local "
                           "webserver to even after {} tries".format(tries))

    def _make_server(self):
        # create http server, and bind it to first available port >= base_port
        return self._safe_make_server(self.host, self.base_port, self.app)

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

    def __init__(self, host, base_port, app):
        super(WSGIAsyncServer, self).__init__(daemon=True)

        # store config, but start the web server (and bind to address) on run()
        self.host = host
        self.base_port = base_port
        self.app = app
        self._server_lock = threading.RLock()

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.join()

    def get_callback_url(self):
        return 'http://{}:{}/callback'.format(*self.server.server_address)

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

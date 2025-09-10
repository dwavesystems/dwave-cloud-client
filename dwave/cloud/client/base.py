# Copyright 2017 D-Wave Systems Inc.
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

"""
D-Wave API clients handle communications with :term:`solver` resources: problem submittal,
monitoring, samples retrieval, etc.

Examples:
    This example creates a client using the local system's default D-Wave Cloud 
    Client configuration file, which provide the API token, submits to a Leap
    quantum-classical hybrid binary quadratic model (BQM) sampler a :term:`QUBO` 
    problem representing a Boolean NOT gate as a penalty model, and prints the 
    returned samples.

    >>> from dwave.cloud import Client
    >>> Q = {(0, 0): -1, (0, 4): 0, (4, 0): 2, (4, 4): -1}
    >>> with Client.from_config(client="hybrid") as client:  # doctest: +SKIP
    ...     solver = client.get_solver(supported_problem_types__issubset={"bqm"})
    ...     computation = solver.sample_qubo(Q, time_limit=5)
    ...
    >>> print(computation.samples)     # doctest: +SKIP
    [[1 0]]

"""

import io
import re
import time
import copy
import queue
import logging
import operator
import threading

import base64
import hashlib
import codecs
import concurrent.futures
import zlib

from itertools import chain, zip_longest
from functools import partial, wraps
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import orjson
from dateutil.parser import parse as parse_datetime
from plucky import pluck

from dwave.cloud import api
from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.exceptions import *    # TODO: fix
from dwave.cloud.exceptions import UseAfterCloseError
from dwave.cloud.computation import Future
from dwave.cloud.config import load_config, update_config, validate_config_v1
from dwave.cloud.config import constants as config_constants
from dwave.cloud.config.models import ClientConfig, PollingStrategy
from dwave.cloud.solver import available_solvers, StructuredSolver, UnstructuredSolver
from dwave.cloud.concurrency import PriorityThreadPoolExecutor
from dwave.cloud.regions import resolve_endpoints
from dwave.cloud.upload import ChunkedData
from dwave.cloud.events import dispatches_events
from dwave.cloud.utils.decorators import retried
from dwave.cloud.utils.http import PretimedHTTPAdapter, BaseUrlSession, default_user_agent
from dwave.cloud.utils.time import datetime_to_timestamp, utcnow

__all__ = ['Client']

logger = logging.getLogger(__name__)


def _ensure_active(*, allow_while_closing=False):
    def _decorator(method):
        @wraps(method)
        def _method(obj, *args, **kwargs):
            with obj._close_lock:
                if obj._closed or (not allow_while_closing and obj._closing):
                    raise UseAfterCloseError(
                        f"{method.__name__} cannot be called after client has been closed")
                return method(obj, *args, **kwargs)
        return _method
    return _decorator


class Client(object):
    """Base client class for all D-Wave API clients. Used by QPU, software and
    hybrid :term:`sampler` classes.

    Manages workers and handles thread pools for submitting problems, cancelling
    tasks, polling problem status, and retrieving results.

    Args:
        region (str, optional, default='na-west-1'):
            D-Wave Solver API region. To see available regions use
            :func:`~dwave.cloud.regions.get_regions`.

        endpoint (str, optional):
            D-Wave Solver API endpoint URL. If undefined, inferred from
            ``region`` code.

        token (str):
            Authentication token for the D-Wave API.

        solver (dict/str, optional):
            Default solver features (or simply solver name) to use in
            :meth:`.Client.get_solver`.

            Defined via dictionary of solver feature constraints
            (see :meth:`.Client.get_solvers`).
            For backward compatibility, a solver name, as a string, is also
            accepted and converted to ``{"name": <solver name>}``.

        proxy (str, optional):
            Proxy URL to be used for accessing the D-Wave API.

        permissive_ssl (bool, default=False):
            Disables SSL verification.

        request_timeout (float, default=60):
            Connect and read timeout, in seconds, for all requests to the
            D-Wave API.

        polling_timeout (float, optional):
            Problem status polling timeout, in seconds, after which polling is
            aborted.

        connection_close (bool, default=False):
            Force HTTP(S) connection close after each request. Set to ``True``
            to prevent intermediate network equipment closing idle connections.

        compress_qpu_problem_data (bool, default=True):
            Enable QPU problem data compression on upload to SAPI. Enabled by
            default. Set to ``False`` to disable compression.

        headers (dict/str, optional):
            Newline-separated additional HTTP headers to include with each
            API request, or a dictionary of (key, value) pairs.

        client_cert (str, optional):
            Path to client side certificate file.

        client_cert_key (str, optional):
            Path to client side certificate key file.

        poll_strategy (str, 'backoff' | 'long-polling', default='long-polling'):
            Polling strategy for problem status. Supported options are short polling 
            with exponential back-off, configured with ``poll_backoff_*`` parameters,
            and long polling, configured with ``poll_wait_time`` and ``poll_pause``
            parameters.

            .. versionadded:: 0.13.0
                Added support for long polling strategy. Long polling is the new
                default, but the ``"backoff"`` strategy can still be used if
                desired.

        poll_backoff_min (float, default=0.05):
            When problem status is polled with exponential back-off schedule, the
            duration of the first interval (between first and second poll) is
            set to ``poll_backoff_min`` seconds.

        poll_backoff_max (float, default=60):
            When problem status is polled with exponential back-off schedule, the
            maximum back-off period is limited to ``poll_backoff_max`` seconds.

        poll_backoff_base (float, default=1.3):
            When problem status is polled with exponential back-off schedule,
            the exponential function base is defined with ``poll_backoff_base``.
            Interval between ``poll_idx`` and ``poll_idx + 1`` is given with::

                poll_backoff_min * poll_backoff_base ** poll_idx

            with upper bound set to ``poll_backoff_max``.

        poll_wait_time (int, in range [0, 30], default=30):
            When problem status is polled using long polling, this value sets the 
            maximum time, in seconds, a long polling connection waits for the API 
            response.

            .. versionadded:: 0.13.0

        poll_pause (float, default=0.0):
            When problem status is polled using long polling, this value limits the 
            delay, in seconds, between two successive long polling connections.
            It's preferred to keep this value at zero.

            .. versionadded:: 0.13.0

        http_retry_total (int, default=10):
            Total number of retries of failing idempotent HTTP requests to
            allow. Takes precedence over other counts.
            See ``total`` in :class:`~urllib3.util.retry.Retry` for details.

        http_retry_connect (int, default=None):
            How many connection-related errors to retry on.
            See ``connect`` in :class:`~urllib3.util.retry.Retry` for details.

        http_retry_read (int, default=None):
            How many times to retry on read errors.
            See ``read`` in :class:`~urllib3.util.retry.Retry` for details.

        http_retry_redirect (int, default=None):
            How many redirects to perform.
            See ``redirect`` in :class:`~urllib3.util.retry.Retry` for details.

        http_retry_status (int, default=None):
            How many times to retry on bad status codes.
            See ``status`` in :class:`~urllib3.util.retry.Retry` for details.

        http_retry_backoff_factor (float, default=0.01):
            A backoff factor to apply between attempts after the second try.
            Sleep between retries, in seconds::

                {backoff factor} * (2 ** ({number of total retries} - 1))

            See ``backoff_factor`` in :class:`~urllib3.util.retry.Retry` for
            details.

        http_retry_backoff_max (float, default=60):
            Maximum backoff time in seconds.
            See :attr:`~urllib3.util.retry.Retry.BACKOFF_MAX` for details.

        metadata_api_endpoint (str, optional):
            D-Wave Metadata API endpoint. Central for all regions, used for
            regional SAPI endpoint discovery.

        leap_api_endpoint (str, optional):
            Leap API endpoint.

        leap_client_id (str, optional):
            Leap OAuth 2.0 Ocean client id. Reserved for testing, otherwise
            don't override.

        defaults (dict, optional):
            Defaults for the client instance that override the class
            :attr:`.Client.DEFAULTS`.

    .. versionchanged:: 0.11.0
        Added the ``leap_api_endpoint`` parameter and config option (also
        available via environment variable ``DWAVE_LEAP_API_ENDPOINT``).

        Added the ``leap_client_id`` parameter and config option (also available
        via environment variable ``DWAVE_LEAP_CLIENT_ID``). This option is
        reserved for testing.

    Note:
        Default values of all constructor arguments listed above are kept in
        a class variable :attr:`.Client.DEFAULTS`.

        Instance-level defaults can be specified via ``defaults`` argument.

    .. versionremoved:: 0.12.0
        Positional arguments in :class:`.Client` constructor, deprecated in 0.10.0,
        are removed in 0.12.0. Use keyword arguments instead.

    .. versionremoved:: 0.13.0
        Config attributes on :class:`.Client`, deprecated in 0.11.0, are removed
        in 0.13.0. Use config model attributes on ``Client.config`` instead.

    Examples:
        This example directly initializes a :class:`.Client`.
        Direct initialization uses class constructor arguments, the minimum
        being a value for ``token``.

        >>> from dwave.cloud import Client
        >>> client = Client(token='secret')     # doctest: +SKIP
        >>> # code that uses client
        >>> client.close()       # doctest: +SKIP

    """

    # The status flags that a problem can have
    STATUS_IN_PROGRESS = 'IN_PROGRESS'
    STATUS_PENDING = 'PENDING'
    STATUS_COMPLETE = 'COMPLETED'
    STATUS_FAILED = 'FAILED'
    STATUS_CANCELLED = 'CANCELLED'

    # Cases when multiple status flags qualify
    ANY_STATUS_ONGOING = [STATUS_IN_PROGRESS, STATUS_PENDING]
    ANY_STATUS_NO_RESULT = [STATUS_FAILED, STATUS_CANCELLED]

    # Default API endpoint
    # TODO: remove when refactored to use `dwave.cloud.api`?
    DEFAULT_API_ENDPOINT = config_constants.DEFAULT_SOLVER_API_ENDPOINT
    DEFAULT_API_REGION = config_constants.DEFAULT_REGION

    # Should we wait for all jobs to complete on client.close() by default?
    DEFAULT_WAIT_ON_CLOSE = True

    # Class-level defaults for all constructor and factory arguments
    DEFAULTS = {
        # factory only
        'config_file': None,
        'profile': None,
        'client': 'base',
        # constructor (and factory)
        'metadata_api_endpoint': config_constants.DEFAULT_METADATA_API_ENDPOINT,
        'region': DEFAULT_API_REGION,
        'leap_api_endpoint': None,
        'leap_client_id': None,
        'endpoint': None,       # defined via region, resolved on client init
        'token': None,
        'solver': None,
        'proxy': None,
        'permissive_ssl': False,
        'request_timeout': 60,
        'polling_timeout': None,
        'connection_close': False,
        'compress_qpu_problem_data': True,
        'headers': None,
        'client_cert': None,
        'client_cert_key': None,
        'poll_strategy': 'long-polling',
        # poll back-off schedule defaults [sec]
        'poll_backoff_min': 0.05,
        'poll_backoff_max': 60,
        'poll_backoff_base': 1.3,
        # long polling parameters
        'poll_wait_time': 30,
        'poll_pause': 0,
        # idempotent http requests retry params
        'http_retry_total': 10,
        'http_retry_connect': None,
        'http_retry_read': None,
        'http_retry_redirect': None,
        'http_retry_status': None,
        'http_retry_backoff_factor': 0.01,
        'http_retry_backoff_max': 60,
    }

    # Number of problems to include in a submit/status query
    _SUBMIT_BATCH_SIZE = 20
    _STATUS_QUERY_SIZE = 100

    # Number of worker threads for each problem processing task
    _SUBMISSION_THREAD_COUNT = 5
    _UPLOAD_PROBLEM_THREAD_COUNT = 1
    _UPLOAD_PART_THREAD_COUNT = 10
    _ENCODE_PROBLEM_THREAD_COUNT = _UPLOAD_PROBLEM_THREAD_COUNT
    _CANCEL_THREAD_COUNT = 1
    _POLL_THREAD_COUNT = 5
    _LOAD_THREAD_COUNT = 5

    # Poll grouping time frame; two scheduled polls are grouped if closer than [sec]:
    _POLL_GROUP_TIMEFRAME = 2

    # Downloaded solver definition cache config
    _DEFAULT_SOLVERS_STATIC_PART_MAXAGE = 3600  # 1 hour
    _DEFAULT_SOLVERS_DYNAMIC_PART_MAXAGE = 900  # 15 min
    _DEFAULT_SOLVERS_CACHE_CONFIG = dict(
        enabled=True,
        # heuristic maxage (cache-control in response overrides it)
        maxage=_DEFAULT_SOLVERS_STATIC_PART_MAXAGE,
    )

    # Downloaded region metadata cache maxage [sec]
    _REGIONS_CACHE_MAXAGE = 7 * 86400   # 7 days

    # Multipart upload parameters
    _UPLOAD_PART_SIZE_BYTES = 5 * 1024 * 1024
    _UPLOAD_PART_RETRIES = 2
    _UPLOAD_REQUEST_RETRIES = 2
    _UPLOAD_RETRIES_BACKOFF = lambda retry: 2 ** retry

    # Binary-ref answer download parameters
    _DOWNLOAD_ANSWER_THREAD_COUNT = 2

    # SAPI deprecation message handling; keep private for now
    _DEFAULT_ON_DEPRECATION_CONFIG = dict(log=True, warn=True, store=True)

    @classmethod
    def from_config(cls, config_file=None, profile=None, client=None, **kwargs):
        """Client factory method to instantiate a client instance from configuration.

        Configuration values can be specified in multiple ways, ranked in the following
        order (with 1 the highest ranked):

        1. Values specified as keyword arguments in :func:`from_config()`
        2. Values specified as environment variables
        3. Values specified in the configuration file
        4. Values specified as :class:`.Client` instance defaults
        5. Values specified in :class:`.Client` class :attr:`.Client.DEFAULTS`

        Configuration-file format and environment variables are described in
        :mod:`dwave.cloud.config`.

        File/environment configuration loading mechanism is described in
        :func:`~dwave.cloud.config.load_config`.

        Args:
            config_file (str/[str]/None/False/True, default=None):
                Path to configuration file. For interpretation, see
                :func:`~dwave.cloud.config.load_config`.

            profile (str, default=None):
                Profile name. For interpretation, see
                :func:`~dwave.cloud.config.load_config`.

            client (str, default=None):
                Client type used for accessing the API. Supported values are
                ``qpu`` for :class:`dwave.cloud.qpu.Client`,
                ``sw`` for :class:`dwave.cloud.sw.Client` and
                ``hybrid`` for :class:`dwave.cloud.hybrid.Client`.

            **kwargs (dict):
                :class:`.Client` constructor options.

        Returns:
            :class:`~dwave.cloud.client.Client` subclass:
                Appropriate instance of a QPU/software/hybrid client.

        Raises:
            :exc:`~dwave.cloud.config.exceptions.ConfigFileReadError`:
                Config file specified or detected could not be opened or read.

            :exc:`~dwave.cloud.config.exceptions.ConfigFileParseError`:
                Config file parse failed.

            :exc:`ValueError`:
                Invalid (non-existing) profile name.

        """

        # load configuration from config file(s) and environment
        config = load_config(config_file=config_file, profile=profile,
                             client=client, **kwargs)
        logger.debug("Config loaded: %r", config)

        from dwave.cloud.client import qpu, sw, hybrid
        _clients = {
            'base': cls,
            'qpu': qpu.Client,
            'sw': sw.Client,
            'hybrid': hybrid.Client,
        }
        _client = config.pop('client', None) or 'base'

        logger.debug("Creating %s.Client() with: %r", _client, config)
        return _clients[_client](**config)

    class _WaitableCounter:
        """A thread-safe counter, with an ability to block until the count
        reaches zero (use :meth:`.wait`).
        """

        def __init__(self, value: int = 0):
            self._cond = threading.Condition()
            self._value = value

        def __repr__(self):
            return f"{type(self).__name__}(value={self._value})"

        def inc(self):
            with self._cond:
                self._value += 1

        def dec(self) -> bool:
            with self._cond:
                if self._value > 0:
                    self._value -= 1
                if self._value == 0:
                    self._cond.notify_all()

        def wait(self):
            with self._cond:
                if self._value > 0:
                    self._cond.wait()

    @dispatches_events('client_init')
    def __init__(self, **kwargs):
        logger.debug("Client init called with: %r", kwargs)

        self._closed = False
        self._closing = False
        self._close_lock = threading.Lock()
        self._jobs = self._WaitableCounter()

        # derive instance-level defaults from class defaults and init defaults
        self.defaults = copy.deepcopy(self.DEFAULTS)
        user_defaults = kwargs.pop('defaults', None)
        if user_defaults is None:
            user_defaults = {}
        update_config(self.defaults, user_defaults)

        # combine instance-level defaults with file/env/kwarg option values
        # note: treat empty string values (e.g. from file/env) as undefined/None
        options = copy.deepcopy(self.defaults)
        update_config(options, kwargs)
        logger.debug("Client options over defaults: %r", options)

        self.config: ClientConfig = validate_config_v1(options)
        logger.debug("Validated client config=%r", self.config)

        # resolve endpoints using region
        resolve_endpoints(self.config, inplace=True)
        logger.debug("Final client config=%r", self.config)

        # sanity check
        if not self.config.endpoint:
            raise ValueError("API endpoint not defined")

        if not self.config.token:
            raise ValueError("API token not defined")

        # Build the problem submission queue, start its workers
        self._submission_queue = queue.Queue()
        self._submission_workers = []
        for _ in range(self._SUBMISSION_THREAD_COUNT):
            worker = threading.Thread(target=self._do_submit_problems)
            worker.daemon = True
            worker.start()
            self._submission_workers.append(worker)

        # Build the cancel problem queue, start its workers
        self._cancel_queue = queue.Queue()
        self._cancel_workers = []
        for _ in range(self._CANCEL_THREAD_COUNT):
            worker = threading.Thread(target=self._do_cancel_problems)
            worker.daemon = True
            worker.start()
            self._cancel_workers.append(worker)

        # Build the problem status polling queue, start its workers
        self._poll_queue = queue.PriorityQueue()
        self._poll_workers = []
        for _ in range(self._POLL_THREAD_COUNT):
            worker = threading.Thread(target=self._do_poll_problems)
            worker.daemon = True
            worker.start()
            self._poll_workers.append(worker)

        # Build the result loading queue, start its workers
        self._load_queue = queue.Queue()
        self._load_workers = []
        for _ in range(self._LOAD_THREAD_COUNT):
            worker = threading.Thread(target=self._do_load_results)
            worker.daemon = True
            worker.start()
            self._load_workers.append(worker)

        # Setup multipart upload executors
        self._upload_problem_executor = \
            ThreadPoolExecutor(self._UPLOAD_PROBLEM_THREAD_COUNT)

        self._upload_part_executor = \
            PriorityThreadPoolExecutor(self._UPLOAD_PART_THREAD_COUNT)

        self._encode_problem_executor = \
            ThreadPoolExecutor(self._ENCODE_PROBLEM_THREAD_COUNT)

        # Setup binary-ref answer download executors
        self._download_answer_executor = \
            ThreadPoolExecutor(self._DOWNLOAD_ANSWER_THREAD_COUNT)

    class _Session(api.client.VersionedAPISessionMixin,
                   api.client.DeprecationAwareSessionMixin,
                   api.client.LoggingSessionMixin,
                   BaseUrlSession):
        pass

    def create_session(self):
        """Create a new requests session based on client's (self) params.

        Note: since `requests.Session` is NOT thread-safe, every thread should
        create and use an isolated session.
        """

        # allow endpoint path to not end with /
        endpoint = self.config.endpoint
        if not endpoint.endswith('/'):
            endpoint += '/'

        session = self._Session(base_url=endpoint, strict_mode=True,
                                on_deprecation=self._DEFAULT_ON_DEPRECATION_CONFIG)
        session.mount('http://', PretimedHTTPAdapter(
            timeout=self.config.request_timeout,
            max_retries=self.config.request_retry.to_urllib3_retry()))
        session.mount('https://', PretimedHTTPAdapter(
            timeout=self.config.request_timeout,
            max_retries=self.config.request_retry.to_urllib3_retry()))

        session.headers.update({'User-Agent': default_user_agent()})
        if self.config.headers:
            session.headers.update(self.config.headers)
        if self.config.token:
            session.headers.update({'X-Auth-Token': self.config.token})
        if self.config.cert:
            session.cert = self.config.cert

        session.proxies = {'http': self.config.proxy, 'https': self.config.proxy}
        if self.config.permissive_ssl:
            session.verify = False
        if self.config.connection_close:
            session.headers.update({'Connection': 'close'})

        # Debug-log headers
        logger.trace("create_session(session.headers=%r)", session.headers)

        return session

    def _shutdown_threads(self, wait: bool = True):
        # Finish all the work that requires the connection
        logger.debug("Joining submission queue")
        self._submission_queue.join()
        logger.debug("Joining cancel queue")
        self._cancel_queue.join()
        logger.debug("Joining poll queue")
        self._poll_queue.join()
        logger.debug("Joining load queue")
        self._load_queue.join()

        logger.debug("Shutting down problem upload executor")
        self._upload_problem_executor.shutdown(wait=True)
        logger.debug("Shutting down problem part upload executor")
        self._upload_part_executor.shutdown(wait=True)
        logger.debug("Shutting down problem encoder executor")
        self._encode_problem_executor.shutdown(wait=True)
        logger.debug("Shutting down answer download executor")
        self._download_answer_executor.shutdown(wait=True)

        # Send kill-task to all worker threads
        # Note: threads can't be 'killed' in Python, they have to die by
        # natural causes
        for _ in self._submission_workers:
            self._submission_queue.put(None)
        for _ in self._cancel_workers:
            self._cancel_queue.put(None)
        for _ in self._poll_workers:
            self._poll_queue.put((-1, None))
        for _ in self._load_workers:
            self._load_queue.put(None)

        # Wait for threads to die
        if wait:
            for worker in chain(self._submission_workers, self._cancel_workers,
                                self._poll_workers, self._load_workers):
                worker.join()

    def close(self, wait: Optional[bool] = None):
        """Perform a clean shutdown.

        Waits for all the currently scheduled work to finish, kills the workers,
        and closes the connection pool.

        Args:
            wait:
                When set to true (the default), allow all (remote) jobs to finish
                and their results to be downloaded before shutting down the client.
                New jobs are never accepted while closing.

                .. versionadded:: 0.14.0
                    The ``wait`` parameter. By default, we now wait for all jobs
                    to finish on client close.

        Where possible, it is recommended you use a context manager
        (a :code:`with Client.from_config(...) as` construct) to ensure your
        code properly closes all resources.

        Examples:
            This example creates a client, executes some code (represented by
            a placeholder comment), and then closes the client.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()    # doctest: +SKIP
            >>> # code that uses client
            >>> client.close()    # doctest: +SKIP

        """
        if wait is None:
            wait = self.DEFAULT_WAIT_ON_CLOSE

        logger.debug("Client.close(wait=%r) initiated while active jobs: %r",
                     wait, self._jobs)

        with self._close_lock:
            if self._closing or self._closed:
                return

            if wait:
                # by marking client as closing, we allow *only some* job submissions
                # (like poll and answer download)
                self._closing = True
            else:
                # this client can't be used anymore
                # note: mark it closed early to prevent job submission while closing!
                self._closed = True

        # if wait was True, we have to wait for organic shutdown to mark the client closed
        if wait:
            self._jobs.wait()

        solvers_session = getattr(self, '_solvers_session', None)
        if solvers_session:
            solvers_session.close()

        self._shutdown_threads(wait=wait)

        with self._close_lock:
            self._closed = True

    def __enter__(self):
        """Let connections be used in with blocks."""
        return self

    def __exit__(self, *args):
        """At the end of a with block perform a clean shutdown of the connection."""
        self.close(wait=self.DEFAULT_WAIT_ON_CLOSE)
        return False

    @staticmethod
    def is_solver_handled(solver):
        """Determine if the specified solver should be handled by this client.

        Default implementation accepts all solvers (always returns True). Override this
        predicate function with a subclass if you want to specialize your client for a
        particular type of solvers.

        Examples:
            This function accepts only solvers named "My_Solver_*".

            .. code:: python

                @staticmethod
                def is_solver_handled(solver):
                    return solver and solver.name.startswith('My_Solver_')

        """
        return True

    @property
    @_ensure_active(allow_while_closing=False)
    def solvers_session(self) -> api.resources.Solvers:
        session = getattr(self, '_solvers_session', None)

        # init on first use
        if session is None:
            session = self._solvers_session = \
                api.Solvers.from_config(
                    config=self.config,
                    cache=self._DEFAULT_SOLVERS_CACHE_CONFIG
                )

        return session

    def _fetch_solvers(self,
                       name: Optional[str] = None,
                       refresh_: Optional[bool] = False,
                       ) -> list[Union[StructuredSolver, UnstructuredSolver]]:

        static_fields = 'all,-status,-avg_load'
        dynamic_fields = 'none,+identity,+status,+avg_load'

        if name is not None:
            logger.info("Fetching definition of a solver with name=%r", name)

            try:
                solver = self.solvers_session.get_solver(
                    solver_name=name, filter=static_fields, refresh_=refresh_)
                status = self.solvers_session.get_solver(
                    solver_name=name, filter=dynamic_fields, refresh_=refresh_,
                    maxage_=self._DEFAULT_SOLVERS_DYNAMIC_PART_MAXAGE)

                # merge static and dynamic properties
                solver.status = status.get('status', 'offline')
                solver.avg_load = status.get('avg_load', 0)

                solvers = [solver]

            except api.exceptions.ResourceNotFoundError as exc:
                raise SolverNotFoundError(f"No solver with name={name!r} available") from exc

        else:
            logger.info("Fetching definitions of all available solvers")

            solvers = self.solvers_session.list_solvers(
                filter=static_fields, refresh_=refresh_)
            dynamic = self.solvers_session.list_solvers(
                filter=dynamic_fields, refresh_=refresh_,
                maxage_=self._DEFAULT_SOLVERS_DYNAMIC_PART_MAXAGE)

            # add dynamic properties to static solvers
            # note: allow solver list mismatch
            statuses = {solver.identity.name: solver for solver in dynamic}
            for solver in solvers:
                solver.status = statuses.get(solver.identity.name, {}).get('status', 'offline')
                solver.avg_load = statuses.get(solver.identity.name, {}).get('avg_load', 0)

        logger.info("Received solver data for %d solver(s).", len(solvers))
        if logger.isEnabledFor(logging.TRACE):
            logger.trace("Solver data received for solver %r: %r", name, solvers)

        instantiated_solvers = []
        for solver_desc in solvers:
            for solver_class in available_solvers:
                try:
                    logger.debug("Trying to instantiate %r", solver_class.__name__)
                    solver = solver_class(client=self, data=solver_desc)
                    if self.is_solver_handled(solver):
                        instantiated_solvers.append(solver)
                        logger.info("Adding solver %r", solver)
                        break
                    else:
                        logger.debug("Skipping solver %r (not handled by this client)", solver)

                except UnsupportedSolverError as e:
                    logger.debug("Skipping solver due to %r", e)

            # propagate all other/decoding errors, like InvalidAPIResponseError, etc.

        return instantiated_solvers

    def retrieve_answer(self, id_):
        """Retrieve a problem by id.

        Args:
            id_ (str):
                As returned by :attr:`Future.id`.

        Returns:
            :class:`Future`

        """
        future = Future(None, id_)
        self._jobs.inc()
        self._load(future)
        return future

    @dispatches_events('get_solvers')
    def get_solvers(self, refresh=False, order_by='avg_load', **filters):
        """Return a filtered list of solvers handled by this client.

        Args:
            refresh (bool, default=False):
                Force refresh of cached list of solvers/properties.

            order_by (callable/str/None, default='avg_load'):
                Solver sorting key function (or :class:`~dwave.cloud.solver.Solver`
                attribute/item dot-separated path). By default, solvers are sorted
                by average load. To explicitly not sort the solvers (and use the
                API-returned order), set ``order_by=None``.

                Signature of the `key` `callable` is::

                    key :: (Solver s, Ord k) => s -> k

                Basic structure of the `key` string path is::

                    "-"? (attr|item) ( "." (attr|item) )*

                For example, to use solver property named ``max_anneal_schedule_points``,
                available in ``Solver.properties`` dict, you can either specify a
                callable `key`::

                    key=lambda solver: solver.properties['max_anneal_schedule_points']

                or, you can use a short string path based key::

                    key='properties.max_anneal_schedule_points'

                Solver derived properties, available as :class:`Solver` properties
                can also be used (e.g. ``num_active_qubits``, ``online``,
                ``avg_load``, etc).

                Ascending sort order is implied, unless the key string path does
                not start with ``-``, in which case descending sort is used.

                Note: the sort used for ordering solvers by `key` is **stable**,
                meaning that if multiple solvers have the same value for the
                key, their relative order is preserved, and effectively they are
                in the same order as returned by the API.

                Note: solvers with ``None`` for key appear last in the list of
                solvers. When providing a key callable, ensure all values returned
                are of the same type (particularly in Python 3). For solvers with
                undefined key value, return ``None``.

            **filters:
                See `Filtering forms` and `Operators` below.

        Solver filters are defined, similarly to Django QuerySet filters, with
        keyword arguments of form `<key1>__...__<keyN>[__<operator>]=<value>`.
        Each `<operator>` is a predicate (boolean) function that acts on two
        arguments: value of feature `<name>` (described with keys path
        `<key1.key2...keyN>`) and the required `<value>`.

        Feature `<name>` can be:

        1) a derived solver property, available as an identically named
           :class:`Solver`'s property (`name`, `qpu`, `hybrid`, `software`,
           `online`, `num_active_qubits`, `avg_load`)
        2) a solver parameter, available in :obj:`Solver.parameters`
        3) a solver property, available in :obj:`Solver.properties`
        4) a path describing a property in nested dictionaries

        Filtering forms are:

        * <derived_property>__<operator> (object <value>)
        * <derived_property> (bool)

          This form ensures the value of solver's property bound to `derived_property`,
          after applying `operator` equals the `value`. The default operator is `eq`.

          For example::

            >>> client.get_solvers(avg_load__gt=0.5)

          but also::

            >>> client.get_solvers(online=True)
            >>> # identical to:
            >>> client.get_solvers(online__eq=True)

        * <parameter>__<operator> (object <value>)
        * <parameter> (bool)

          This form ensures that the solver supports `parameter`. General operator form can
          be used but usually does not make sense for parameters, since values are human-readable
          descriptions. The default operator is `available`.

          Example::

            >>> client.get_solvers(flux_biases=True)
            >>> # identical to:
            >>> client.get_solvers(flux_biases__available=True)

        * <property>__<operator> (object <value>)
        * <property> (bool)

          This form ensures the value of the solver's `property`, after applying `operator`
          equals the righthand side `value`. The default operator is `eq`.

        Note: if a non-existing parameter/property name/key given, the default operator is `eq`.

        Operators are:

        * `available` (<name>: str, <value>: bool):
            Test availability of <name> feature.
        * `eq`, `lt`, `lte`, `gt`, `gte` (<name>: str, <value>: any):
            Standard relational operators that compare feature <name> value with <value>.
        * `regex` (<name>: str, <value>: str):
            Test regular expression matching feature value.
        * `covers` (<name>: str, <value>: single value or range expressed as 2-tuple/list):
            Test feature <name> value (which should be a *range*) covers a given value or a subrange.
        * `within` (<name>: str, <value>: range expressed as 2-tuple/list):
            Test feature <name> value (which can be a *single value* or a *range*) is within a given range.
        * `in` (<name>: str, <value>: container type):
            Test feature <name> value is *in* <value> container.
        * `contains` (<name>: str, <value>: any):
            Test feature <name> value (container type) *contains* <value>.
        * `issubset` (<name>: str, <value>: container type):
            Test feature <name> value (container type) is a subset of <value>.
        * `issuperset` (<name>: str, <value>: container type):
            Test feature <name> value (container type) is a superset of <value>.

        Derived properies are:

        * `identity` (str): Solver identity dict. Includes a name, and possibly version(s).
        * `name` (str): Solver name.
        * `version` (dict): QPU solver version dict (contains at least `graph_id`)
        * `graph_id` (str): QPU solver working graph id
        * `qpu` (bool): Solver is a QPU?
        * `hybrid` (bool): Solver is a hybrid quantum-classical solver?
        * `software` (bool): Solver is a software solver?
        * `online` (bool, default=True): Is solver online?
        * `num_active_qubits` (int): Number of active qubits. Less then or equal to `num_qubits`.
        * `avg_load` (float): Solver's average load (similar to Unix load average).

        Common solver parameters are:

        * `flux_biases`: Should solver accept flux biases?
        * `anneal_schedule`: Should solver accept anneal schedule?

        Common solver properties are:

        * `num_qubits` (int): Number of qubits available.
        * `vfyc` (bool): Should solver work on "virtual full-yield chip"?
        * `max_anneal_schedule_points` (int): Piecewise linear annealing schedule points.
        * `h_range` ([int,int]), j_range ([int,int]): Biases/couplings values range.
        * `num_reads_range` ([int,int]): Range of allowed values for `num_reads` parameter.

        Returns:
            list[Solver]: List of all solvers that satisfy the conditions.

        Note:
            Client subclasses (e.g. :class:`dwave.cloud.qpu.Client` or
            :class:`dwave.cloud.hybrid.Client`) already filter solvers by resource
            type, so for `qpu` and `hybrid` filters to have effect, call :meth:`.get_solvers`
            on base :class:`~dwave.cloud.client.Client` class.

        Examples::

            client.get_solvers(
                num_qubits__gt=5000,                # we need more than 5000 qubits
                num_qubits__lt=6000,                # ... but fewer than 6000 qubits
                num_qubits__within=(5000, 6000),    # an alternative to the previous two lines
                num_active_qubits=5627,             # we want a particular number of active qubits
                anneal_schedule=True,               # we need support for custom anneal schedule
                max_anneal_schedule_points__gte=4,  # we need at least 4 points for our anneal schedule
                num_reads_range__covers=1000,       # our solver must support returning 1000 reads
                extended_j_range__covers=[-2, 1],   # we need extended J range to contain subrange [-2,1]
                couplers__contains=[30, 31],        # coupler (edge between) qubits (30,31) must exist
                couplers__issuperset=[[30,31], [30,45]],
                                                    # two couplers required: (30,31) and (30,45)
                qubits__issuperset={30, 31, 32},    # qubits 30, 31 and 32 must exist
                supported_problem_types__issubset={'ising', 'qubo'},
                                                    # require Ising, QUBO or both to be supported
                name='Advantage_system4.1',         # solver name match
                name__regex='Advantage.*',          # partial/regex-based solver name match
                graph_id='01abcd1234',              # QPU solver working graph id match
                chip_id__regex='Advantage_.*',      # chip ID prefix must be Advantage_
                topology__type__eq="pegasus"        # topology.type must be Pegasus
                topology__type="pegasus"            # same as above, `eq` implied even for nested properties
            )
        """
        logger.debug('get_solvers(refresh=%r, order_by=%r, **filters=%r)',
                     refresh, order_by, filters)

        def covers_op(prop, val):
            """Does LHS `prop` (range) fully cover RHS `val` (range or item)?"""

            # `prop` must be a 2-element list/tuple range.
            if not isinstance(prop, (list, tuple)) or not len(prop) == 2:
                raise ValueError("2-element list/tuple range required for LHS value")
            llo, lhi = min(prop), max(prop)

            # `val` can be a single value, or a range (2-list/2-tuple).
            if isinstance(val, (list, tuple)) and len(val) == 2:
                # val range within prop range?
                rlo, rhi = min(val), max(val)
                return llo <= rlo and lhi >= rhi
            else:
                # val item within prop range?
                return llo <= val <= lhi

        def within_op(prop, val):
            """Is LHS `prop` (range or item) fully covered by RHS `val` (range)?"""
            try:
                return covers_op(val, prop)
            except ValueError:
                raise ValueError("2-element list/tuple range required for RHS value")

        def _set(iterable):
            """Like set(iterable), but works for lists as items in iterable.
            Before constructing a set, lists are converted to tuples.
            """
            first = next(iter(iterable))
            if isinstance(first, list):
                return set(tuple(x) for x in iterable)
            return set(iterable)

        def with_valid_lhs(op):
            @wraps(op)
            def _wrapper(prop, val):
                if prop is None:
                    return False
                return op(prop, val)
            return _wrapper

        # available filtering operators
        ops = {
            'lt': with_valid_lhs(operator.lt),
            'lte': with_valid_lhs(operator.le),
            'gt': with_valid_lhs(operator.gt),
            'gte': with_valid_lhs(operator.ge),
            'eq': operator.eq,
            'available': lambda prop, val: prop is not None if val else prop is None,
            'regex': with_valid_lhs(lambda prop, val: re.match("^{}$".format(val), prop)),
            # range operations
            'covers': with_valid_lhs(covers_op),
            'within': with_valid_lhs(within_op),
            # membership tests
            'in': lambda prop, val: prop in val,
            'contains': with_valid_lhs(lambda prop, val: val in prop),
            # set tests
            'issubset': with_valid_lhs(lambda prop, val: _set(prop).issubset(_set(val))),
            'issuperset': with_valid_lhs(lambda prop, val: _set(prop).issuperset(_set(val))),
        }

        def predicate(solver, query, val):
            # needs to handle kwargs like these:
            #  key=val
            #  key__op=val
            #  key__key=val
            #  key__key__op=val
            # LHS is split on __ in `query`
            assert len(query) >= 1

            potential_path, potential_op_name = query[:-1], query[-1]

            if potential_op_name in ops:
                # op is explicit, and potential path is correct
                op_name = potential_op_name
            else:
                # op is implied and depends on property type, path is the whole query
                op_name = None
                potential_path = query

            path = '.'.join(potential_path)

            if path in solver.derived_properties:
                op = ops[op_name or 'eq']
                return op(getattr(solver, path), val)
            elif pluck(solver.parameters, path, None) is not None:
                op = ops[op_name or 'available']
                return op(pluck(solver.parameters, path), val)
            elif pluck(solver.properties, path, None) is not None:
                op = ops[op_name or 'eq']
                return op(pluck(solver.properties, path), val)
            else:
                op = ops[op_name or 'eq']
                return op(pluck(solver, path), val)

        # param validation
        sort_reverse = False
        if not order_by:
            sort_key = None
        elif isinstance(order_by, str):
            if order_by[0] == '-':
                sort_reverse = True
                order_by = order_by[1:]
            if not order_by:
                sort_key = None
            else:
                sort_key = lambda solver: pluck(solver, order_by, None)
        elif callable(order_by):
            sort_key = order_by
        else:
            raise TypeError("expected string or callable for 'order_by'")

        # default filters:
        filters.setdefault('online', True)

        predicates = []
        for lhs, val in filters.items():
            query = lhs.split('__')
            predicates.append(partial(predicate, query=query, val=val))

        logger.debug("Filtering solvers with predicates=%r", predicates)

        # optimization for case when exact solver name/id is known:
        # we can fetch only that solver
        # NOTE: in future, complete feature-based filtering will be on server-side
        query = dict(refresh_=refresh)
        if 'name' in filters:
            query['name'] = filters['name']
        if 'name__eq' in filters:
            query['name'] = filters['name__eq']

        # shortcircuit lookup if identity defines a name
        identity = filters.get('identity__eq', filters.get('identity', {}))
        if 'name' in identity:
            query['name'] = identity['name']

        # filter
        try:
            solvers = self._fetch_solvers(**query)

        # wrap exceptions for backwards-compatibility
        except api.exceptions.ResourceAuthenticationError as e:
            raise SolverAuthenticationError from e
        except api.exceptions.ResourceBadResponseError as e:
            raise InvalidAPIResponseError(e) from e
        except orjson.JSONDecodeError as e:
            raise InvalidAPIResponseError("JSON response expected") from e

        solvers = [s for s in solvers if all(p(s) for p in predicates)]

        # sort: undefined (None) key values go last
        if sort_key is not None:
            solvers_with_keys = [(sort_key(solver), solver) for solver in solvers]
            solvers_with_invalid_keys = [(key, solver) for key, solver in solvers_with_keys if key is None]
            solvers_with_valid_keys = [(key, solver) for key, solver in solvers_with_keys if key is not None]
            solvers_with_valid_keys.sort(key=operator.itemgetter(0))
            solvers = [solver for key, solver in chain(solvers_with_valid_keys, solvers_with_invalid_keys)]

        # reverse if necessary (as a separate step from sorting, so it works for invalid keys
        # and plain list reverse without sorting)
        if sort_reverse:
            solvers.reverse()

        return solvers

    def get_solver(self, name=None, refresh=False, **filters):
        """Load the configuration for a single solver.

        Makes a blocking web call to `{endpoint}/solvers/remote/{solver_name}/`, where `{endpoint}`
        is a URL configured for the client, and returns a :class:`.Solver` instance
        that can be used to submit sampling problems to the D-Wave API and retrieve results.

        Args:
            name (str):
                ID of the requested solver. ``None`` returns the default solver.
                If default solver is not configured, ``None`` returns the first available
                solver in ``Client``'s class (QPU/software/base).

            **filters (keyword arguments, optional):
                Dictionary of filters over features this solver has to have. For a list of
                feature names and values, see: :meth:`~dwave.cloud.client.Client.get_solvers`.

            order_by (callable/str/None, default='avg_load'):
                Solver sorting key function (or :class:`~dwave.cloud.solver.Solver`
                attribute/item dot-separated path). By default, solvers are sorted by average
                load. For details, see :meth:`~dwave.cloud.client.Client.get_solvers`.

            refresh (bool):
                Return solver from cache (if cached with
                :meth:`~dwave.cloud.client.Client.get_solvers`), unless set to
                ``True``.

        Returns:
            :class:`.Solver`

        Examples:
            This example creates two solvers for a client instantiated from
            a local system's auto-detected default configuration file, which configures
            a connection to a D-Wave resource that provides two solvers. The first
            uses the default solver, the second selects a hybrid CQM solver.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()    # doctest: +SKIP
            >>> client.get_solvers()   # doctest: +SKIP
            BQMSolver(name='hybrid_binary_quadratic_model_version2p'),
            DQMSolver(name='hybrid_discrete_quadratic_model_version1p'),
            CQMSolver(name='hybrid_constrained_quadratic_model_version1p'),
            StructuredSolver(name='Advantage_system6.4', graph_id='01dae5a273')]
            >>> solver1 = client.get_solver()    # doctest: +SKIP
            >>> solver2 = client.get_solver(supported_problem_types__issubset={'cqm'})    # doctest: +SKIP
            >>> solver1.name  # doctest: +SKIP
            'Advantage_system6.4'
            >>> solver2.name   # doctest: +SKIP
            'hybrid_constrained_quadratic_model_version1p'
            >>> # code that uses client
            >>> client.close() # doctest: +SKIP

        """
        logger.debug(f"get_solver({name=}, {refresh=}, {filters=})")

        # backward compatibility: name as the first feature
        if name is not None:
            filters.setdefault('name', name)

        # allow `order_by` to be specified as part of solver features dict
        order_by = filters.pop('order_by', None)

        # in absence of other filters, config/env solver filters/name are used
        if not filters and self.config.solver:
            filters = copy.deepcopy(self.config.solver)

        # allow `order_by` from default config/init override
        if order_by is None:
            order_by = filters.pop('order_by', 'avg_load')
        else:
            filters.pop('order_by', None)

        # get the first solver that satisfies all filters
        try:
            logger.info("Fetching solvers according to filters=%r, order_by=%r",
                        filters, order_by)
            solvers = self.get_solvers(refresh=refresh, order_by=order_by, **filters)
            logger.info("Filtered solvers=%r", solvers)
            return solvers[0]
        except IndexError:
            raise SolverNotFoundError("Solver with the requested features not available")

    @_ensure_active(allow_while_closing=False)
    def _submit(self, body, future):
        """Enqueue a problem for submission to the server.

        This method is thread safe.
        """
        self._jobs.inc()
        self._submission_queue.put(self._submit.Message(body, future))

    _submit.Message = namedtuple('Message', ['body', 'future'])

    def _do_submit_problems(self):
        """Pull problems from the submission queue and submit them.

        Note:
            This method is always run inside of a daemon thread.
        """

        def task_done():
            self._submission_queue.task_done()

        def filter_ready(item):
            """Pass-through ready (encoded) problems, re-enqueue ones for which
            the encoding is in progress, and fail the ones for which encoding
            failed.
            """

            # body is a `concurrent.futures.Future`, so make sure
            # it's ready for submitting
            if item.body.done():
                exc = item.body.exception()
                if exc:
                    # encoding failed, submit should fail as well
                    logger.info("Problem encoding prior to submit "
                                "failed with: %r", exc)
                    item.future._set_exception(exc)
                    self._jobs.dec()
                    task_done()

                else:
                    # problem ready for submit
                    return [item]

            else:
                # body not ready, return the item to queue
                self._submission_queue.put(item)
                task_done()

            return []

        session = self.create_session()
        session.set_accept(media_type='application/vnd.dwave.sapi.problems+json',
                           accept_version='~=3.0', ask_version='3.0.0')
        try:
            while True:
                # Pull as many problems as we can, block on the first one,
                # but once we have one problem, switch to non-blocking then
                # submit without blocking again.

                # `None` task is used to signal thread termination
                item = self._submission_queue.get()

                if item is None:
                    task_done()
                    break

                ready_problems = filter_ready(item)
                while len(ready_problems) < self._SUBMIT_BATCH_SIZE:
                    try:
                        item = self._submission_queue.get_nowait()
                    except queue.Empty:
                        break

                    ready_problems.extend(filter_ready(item))

                if not ready_problems:
                    continue

                # Submit the problems
                logger.debug("Submitting %d problems", len(ready_problems))
                try:
                    data = orjson.dumps([
                        orjson.Fragment(msg.body.result()) for msg in ready_problems
                    ])
                    logger.debug('Size of problems/jobs data = %d', len(data))

                    headers = {}
                    if self.config.compress_qpu_problem_data:
                        data = zlib.compress(data)
                        headers['Content-Encoding'] = 'deflate'
                        logger.debug("Compressed with 'deflate', new size = %d", len(data))

                    message = Client._sapi_request(
                        session.post, 'problems/', data=data, headers=headers)
                    logger.debug("Finished submitting %d problems", len(ready_problems))

                except Exception as exc:
                    logger.debug("Submit failed for %d problems with %r",
                                 len(ready_problems), exc)
                    for msg in ready_problems:
                        msg.future._set_exception(exc)
                        self._jobs.dec()
                        task_done()
                    continue

                # Pass on the information
                for submission, msg in zip_longest(ready_problems, message):
                    try:
                        self._handle_problem_status(msg, submission.future)
                    except Exception as exc:
                        submission.future._set_exception(exc)
                        self._jobs.dec()
                    finally:
                        task_done()

        except BaseException as err:
            logger.exception(err)

        finally:
            session.close()

    def _handle_problem_status(self, message, future):
        """Handle the results of a problem submission or results request.

        This method checks the status of the problem and puts it in the correct
        queue.

        Args:
            message (dict):
                Update message from the SAPI server wrt. this problem.
            future (:class:`dwave.cloud.computation.Future`:
                future corresponding to the problem

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            logger.trace("Handling response: %r", message)
            if not isinstance(message, dict):
                raise InvalidAPIResponseError("Unexpected format of problem description response")
            logger.debug("Handling response for %s with status %s",
                         message.get('id'), message.get('status'))

            # Handle errors in batch mode
            if 'error_code' in message and 'error_msg' in message:
                logger.debug("Error response received: %r", message)
                raise SolverFailureError(message['error_msg'])

            if 'status' not in message:
                raise InvalidAPIResponseError("'status' missing in problem description response")
            if 'id' not in message:
                raise InvalidAPIResponseError("'id' missing in problem description response")

            future.id = message['id']
            future.label = message.get('label')
            future.remote_status = status = message['status']

            # The future may not have the ID set yet
            with future._single_cancel_lock:
                # This handles the case where cancel has been called on a future
                # before that future received the problem id
                if future._cancel_requested:
                    if not future._cancel_sent and status == self.STATUS_PENDING:
                        # The problem has been canceled but the status says its still in queue
                        # try to cancel it
                        self._cancel(message['id'], future)
                    # If a cancel request could meaningfully be sent it has been now
                    future._cancel_sent = True

            if not future.time_received and message.get('submitted_on'):
                future.time_received = parse_datetime(message['submitted_on'])

            if not future.time_solved and message.get('solved_on'):
                future.time_solved = parse_datetime(message['solved_on'])

            if status == self.STATUS_COMPLETE:
                # TODO: find a better way to differentiate between
                # `completed-on-submit` and `completed-on-poll`.
                # Loading should happen only once, not every time when response
                # doesn't contain 'answer'.

                # If the message is complete, forward it to the future object
                if 'answer' in message:

                    # If the future does not know which solver it's associated
                    # with, we get it from the info provided from the server.
                    # An alternative to making this call here would be to pass
                    # self in with the message
                    if future.solver is None:
                        future.solver = self.get_solver(identity=message['solver'])

                    future._set_message(message)

                    # decode the result as part of download (i.e. resolve answer refs)
                    # note: this is (possibly) blocking, but that's OK because
                    # we're running from a daemon thread (likely from `_do_load_results`)
                    future.result()

                    self._jobs.dec()

                # If the problem is complete, but we don't have the result data
                # put the problem in the queue for loading results.
                else:
                    self._load(future)
            elif status in self.ANY_STATUS_ONGOING:
                # If the response is pending add it to the queue.
                self._poll(future)
            elif status == self.STATUS_CANCELLED:
                # If canceled return error
                raise CanceledFutureError()
            else:
                # Return an error to the future object
                errmsg = message.get('error_message', 'An unknown error has occurred.')
                if 'solver is offline' in errmsg.lower():
                    raise SolverOfflineError(errmsg)
                else:
                    raise SolverFailureError(errmsg)

        except Exception as exc:
            # If there were any unhandled errors we need to release the
            # lock in the future, otherwise deadlock occurs.
            future._set_exception(exc)
            self._jobs.dec()

    @_ensure_active(allow_while_closing=False)
    def _cancel(self, id_, future):
        """Enqueue a problem to be canceled.

        This method is thread safe.
        """
        self._cancel_queue.put((id_, future))

    def _do_cancel_problems(self):
        """Pull ids from the cancel queue and submit them.

        Note:
            This method is always run inside of a daemon thread.
        """
        session = self.create_session()
        session.set_accept(media_type='application/vnd.dwave.sapi.problems+json',
                           accept_version='~=3.0', ask_version='3.0.0')
        try:
            while True:
                # Pull as many problems as we can, block when none are available.

                # `None` task is used to signal thread termination
                item = self._cancel_queue.get()
                if item is None:
                    break

                item_list = [item]
                while True:
                    try:
                        item_list.append(self._cancel_queue.get_nowait())
                    except queue.Empty:
                        break

                # Submit the problems, attach the ids as a json list in the
                # body of the delete query.
                try:
                    ids = orjson.dumps([item[0] for item in item_list])
                    Client._sapi_request(session.delete, 'problems/', data=ids)

                except Exception as exc:
                    for _, future in item_list:
                        if future is not None:
                            future._set_exception(exc)
                            self._jobs.dec()

                # Mark all the ids as processed regardless of success or failure.
                for _ in item_list:
                    self._cancel_queue.task_done()

        except Exception as err:
            logger.exception(err)

        finally:
            session.close()

    @_ensure_active(allow_while_closing=True)
    def _poll(self, future: Future) -> None:
        """Enqueue a problem to poll the server for status."""

        # split for simplicity
        if self.config.polling_schedule.strategy == PollingStrategy.BACKOFF:
            return self._poll_using_backoff(future)
        elif self.config.polling_schedule.strategy == PollingStrategy.LONG_POLLING:
            return self._poll_using_long_polling(future)
        else:
            raise RuntimeError("unexpected polling strategy")

    def _poll_using_backoff(self, future: Future) -> None:
        if future._poll_backoff is None:
            # on first poll, start with minimal back-off
            future._poll_backoff = self.config.polling_schedule.backoff_min
        else:
            # on subsequent polls, do exponential back-off, clipped to a range
            future._poll_backoff = \
                max(self.config.polling_schedule.backoff_min,
                    min(future._poll_backoff * self.config.polling_schedule.backoff_base,
                        self.config.polling_schedule.backoff_max))

        # for poll priority we use timestamp of next scheduled poll
        at = time.time() + future._poll_backoff

        now = utcnow()
        future_age = (now - future.time_created).total_seconds()
        logger.debug("Polling scheduled at %.2f with %.2f sec new back-off for: %s (future's age: %.2f sec)",
                     at, future._poll_backoff, future.id, future_age)

        # don't enqueue for next poll if polling_timeout is exceeded by then
        future_age_on_next_poll = future_age + (at - datetime_to_timestamp(now))
        if self.config.polling_timeout is not None and future_age_on_next_poll > self.config.polling_timeout:
            logger.debug("Polling timeout exceeded before next poll: %.2f sec > %.2f sec, aborting polling!",
                         future_age_on_next_poll, self.config.polling_timeout)
            raise PollingTimeout

        self._poll_queue.put((at, future))

    def _poll_using_long_polling(self, future: Future) -> None:
        # we use problem submit time to prioritize polling of jobs submitted earlier
        created_at = datetime_to_timestamp(future.time_created)

        # don't enqueue for next poll if polling_timeout is exceeded by then
        future_age = time.time() - created_at
        if self.config.polling_timeout is not None and future_age > self.config.polling_timeout:
            logger.debug("Polling timeout exceeded before next poll: %.2f sec > %.2f sec, aborting polling!",
                         future_age, self.config.polling_timeout)
            raise PollingTimeout

        # use the same priority queue as for backoff polling
        self._poll_queue.put((created_at, future))

    def _do_poll_problems(self):
        """Poll the server for the status of a set of problems.

        Note:
            This method is always run inside of a daemon thread.
        """
        session = self.create_session()
        session.set_accept(media_type='application/vnd.dwave.sapi.problems+json',
                           accept_version='~=3.0', ask_version='3.0.0')
        try:
            # grouped futures (all scheduled within _POLL_GROUP_TIMEFRAME)
            # and/or up to _STATUS_QUERY_SIZE (depending on strategy)
            frame_futures = {}

            use_long_polling = (
                self.config.polling_schedule.strategy == PollingStrategy.LONG_POLLING)

            def task_done():
                self._poll_queue.task_done()

            def add(future):
                # add future to query frame_futures
                # returns: worker lives on?

                # `None` task signifies thread termination
                if future is None:
                    task_done()
                    return False

                if future.id not in frame_futures and not future.done():
                    frame_futures[future.id] = future
                else:
                    task_done()

                return True

            while True:
                frame_futures.clear()

                # blocking add first scheduled
                frame_earliest, future = self._poll_queue.get()
                if not add(future):
                    return

                # try grouping if scheduled within grouping timeframe
                # (or in the long polling case, add up to _STATUS_QUERY_SIZE available futures)
                while len(frame_futures) < self._STATUS_QUERY_SIZE:
                    try:
                        task = self._poll_queue.get_nowait()
                    except queue.Empty:
                        break

                    at, future = task
                    if use_long_polling or (at - frame_earliest <= self._POLL_GROUP_TIMEFRAME):
                        if not add(future):
                            return
                    else:
                        self._poll_queue.put(task)
                        task_done()
                        break

                # build a query string with ids of all futures in this frame
                ids = [future.id for future in frame_futures.values()]
                logger.debug("Polling for status of futures: %s", ids)
                query_string = 'problems/?id=' + ','.join(ids)
                if use_long_polling:
                    query_string += f'&timeout={self.config.polling_schedule.wait_time}'

                # if futures were cancelled while `add`ing, skip empty frame
                if not ids:
                    continue

                if not use_long_polling:
                    # wait until `frame_earliest` before polling
                    delay = frame_earliest - time.time()
                    if delay > 0:
                        logger.debug("Pausing polling %.2f sec for futures: %s", delay, ids)
                        time.sleep(delay)
                    else:
                        logger.trace("Skipping non-positive delay of %.2f sec", delay)

                # execute and handle the polling request
                try:
                    logger.trace("Executing poll API request")

                    try:
                        statuses = Client._sapi_request(session.get, query_string)

                    except SAPIError as exc:
                        # assume 5xx errors are transient, and don't abort polling
                        if exc.error_code and 500 <= exc.error_code < 600:
                            logger.warning(
                                "Received an internal server error response on "
                                "problem status polling request (%s). Assuming "
                                "error is transient, and resuming polling.",
                                exc.error_code)
                            # add all futures in this frame back to the polling queue
                            # XXX: logic split between `_handle_problem_status` and here
                            for future in frame_futures.values():
                                self._poll(future)
                        else:
                            raise

                    else:
                        # handle a successful request
                        for status in statuses:
                            self._handle_problem_status(status, frame_futures[status['id']])

                except Exception as exc:
                    for id_ in frame_futures.keys():
                        frame_futures[id_]._set_exception(exc)
                        self._jobs.dec()

                for id_ in frame_futures.keys():
                    task_done()

                if use_long_polling and (pause := self.config.polling_schedule.pause) > 0:
                    logger.debug("Pausing %.3f sec between long polling requests", pause)
                    time.sleep(pause)

        except Exception as err:
            logger.exception(err)

        finally:
            session.close()

    @_ensure_active(allow_while_closing=True)
    def _load(self, future):
        """Enqueue a problem to download results from the server.

        Args:
            future (:class:`~dwave.cloud.computation.Future`):
                Future object corresponding to the remote computation.

        This method is thread-safe.
        """
        self._load_queue.put(future)

    def _do_load_results(self):
        """Submit a query asking for the results for a particular problem.

        To request the results of a problem: ``GET /problems/{problem_id}/``

        Note:
            This method is always run inside of a daemon thread.
        """
        session = self.create_session()
        session.set_accept(media_type='application/vnd.dwave.sapi.problem+json',
                           accept_version='~=3.0', ask_version='3.0.0')
        try:
            while True:
                # Select a problem
                future = self._load_queue.get()
                # `None` task signifies thread termination
                if future is None:
                    break
                logger.debug("Loading results of: %s", future.id)

                # Submit the query
                query_string = 'problems/{}/'.format(future.id)

                try:
                    message = Client._sapi_request(session.get, query_string)

                except Exception as exc:
                    logger.debug("Answer load request failed with %r", exc)
                    future._set_exception(exc)
                    self._jobs.dec()
                    self._load_queue.task_done()
                    continue

                # Dispatch the results, mark the task complete
                self._handle_problem_status(message, future)
                self._load_queue.task_done()

        except Exception as err:
            logger.error('Load result error: ' + str(err))

        finally:
            session.close()

    @_ensure_active(allow_while_closing=True)
    def _download_answer_binary_ref(self, *, auth_method: str, url: str,
                                    output: Optional[io.IOBase] = None) -> concurrent.futures.Future:
        """Initiate binary-ref answer download, returning the binary data in
        :class:`~concurrent.futures.Future`.

        Args:
            auth_method:
                Authentication method used to access data at ``url``.

            url:
                Answer binary data.

        Returns:
            :class:`concurrent.futures.Future`[bytes | io.IOBase]:
                Answer data in a Future.
        """
        return self._download_answer_executor.submit(
            self._download_answer_worker, auth_method=auth_method, url=url, output=output)

    def _download_answer_worker(self, *, auth_method: str, url: str,
                                output: Optional[io.IOBase] = None) -> io.IOBase:
        if auth_method != api.constants.BinaryRefAuthMethod.SAPI_TOKEN:
            raise ValueError(f"Authentication method {auth_method!r} not supported.")
        if output is None:
            output = io.BytesIO()

        logger.debug("Downloading binary-ref answer from %r using %r method.",
                     url, auth_method)

        with self.create_session() as session:
            size = 0
            for chunk in session.get(url, stream=True).iter_content(chunk_size=8192):
                size += output.write(chunk)
            output.seek(0)

        logger.debug("Answer data downloaded from %r. Written %r bytes.", url, size)

        return output

    @_ensure_active(allow_while_closing=False)
    def upload_problem_encoded(self, problem, problem_id=None, **kwargs):
        """Initiate multipart problem upload, returning the Problem ID in a
        :class:`~concurrent.futures.Future`.

        Args:
            problem (bytes-like/file-like):
                Encoded problem data to upload.

            problem_id (str, optional):
                Problem ID. If provided, problem will be re-uploaded. Previously
                uploaded parts, with a matching checksum, are skipped.

        Returns:
            :class:`concurrent.futures.Future`[str]:
                Problem ID in a Future. Problem ID can be used to submit
                problems by reference.

        Note:
            For a higher-level interface, use upload/submit solver methods.
        """
        return self._upload_problem_executor.submit(
            self._upload_problem_worker, problem=problem, problem_id=problem_id)

    @staticmethod
    def _sapi_request(meth, *args, **kwargs):
        """Execute an HTTP request defined with the ``meth`` callable and
        parse the response and interpret errors in compliance with SAPI REST
        interface.

        Note:
            For internal use only.

        Args:
            meth (callable):
                Callable object to be called with args and kwargs supplied, with
                expected behavior consistent with one of ``requests.Session()``
                request methods.

            *args, **kwargs (list, dict):
                Arguments to the ``meth`` callable.

        Returns:
            dict: JSON decoded body.

        Raises:
            A :class:`~dwave.cloud.exceptions.SAPIError` subclass, or
            :class:`~dwave.cloud.exceptions.RequestTimeout`.
        """

        # execute request
        try:
            response = meth(*args, **kwargs)
            # note: LoggingSessionMixin will wrap timeout exceptions as RequestTimeout
        except api.exceptions.ResourceBadResponseError as e:
            # returned by VersionedAPISessionMixin; wrap for backwards-compatibility
            raise InvalidAPIResponseError(e) from e

        # workaround for charset_normalizer episode in requests>=2.26.0,
        # where decoding of an empty json object '{}' fails.
        # see: https://github.com/psf/requests/issues/5871,
        # https://github.com/dwavesystems/dwave-cloud-client/pull/471, and
        # https://github.com/dwavesystems/dwave-cloud-client/pull/476.
        response.encoding = 'utf-8'

        # NOTE: the expected behavior is for SAPI to return JSON error on
        # failure. However, that is currently not the case. We need to work
        # around this until it's fixed.

        # no error -> body is json
        # error -> body can be json or plain text error message
        if response.ok:
            try:
                return orjson.loads(response.content)
            except:
                raise InvalidAPIResponseError("JSON response expected")

        else:
            if response.status_code == 401:
                raise SolverAuthenticationError(error_code=401)

            try:
                msg = orjson.loads(response.content)
                error_msg = msg['error_msg']
                error_code = msg['error_code']
            except:
                error_msg = response.text
                error_code = response.status_code

            # NOTE: for backwards compat only. Change to: SAPIError
            raise SolverError(error_msg=error_msg, error_code=error_code)

    @staticmethod
    @retried(_UPLOAD_REQUEST_RETRIES, backoff=_UPLOAD_RETRIES_BACKOFF)
    def _initiate_multipart_upload(session, size):
        """Sync http request using `session`."""

        logger.debug("Initiating problem multipart upload (size=%r)", size)

        path = 'bqm/multipart'
        body = dict(size=size)
        msg = Client._sapi_request(session.post, path, json=body)

        try:
            problem_id = msg['id']
        except KeyError:
            raise InvalidAPIResponseError("problem ID missing")

        logger.debug("Multipart upload initiated (problem_id=%r)", problem_id)

        return problem_id

    @staticmethod
    def _digest(data):
        # data: bytes => md5(data): bytes
        return hashlib.md5(data).digest()

    @staticmethod
    def _checksum_b64(digest):
        # digest: bytes => base64(digest): str
        return base64.b64encode(digest).decode('ascii')

    @staticmethod
    def _checksum_hex(digest):
        # digest: bytes => hex(digest): str
        return codecs.encode(digest, 'hex').decode('ascii')

    @staticmethod
    def _combined_checksum(checksums):
        # TODO: drop this requirement server-side
        # checksums: dict[int, str] => hex(md5(cat(digests))): str
        combined = ''.join(h for _, h in sorted(checksums.items()))
        digest = codecs.decode(combined, 'hex')
        return Client._checksum_hex(Client._digest(digest))

    @staticmethod
    @retried(_UPLOAD_PART_RETRIES, backoff=_UPLOAD_RETRIES_BACKOFF)
    def _upload_multipart_part(session, problem_id, part_id, part_generator,
                               uploaded_part_checksum=None):
        """Upload one problem part. Sync http request.

        Args:
            session (:class:`requests.Session`):
                Session used for all API requests.
            problem_id (str):
                Problem id.
            part_id (int):
                Part number/id.
            part_generator (generator of :class:`io.BufferedIOBase`/binary-stream-like):
                Callable that produces problem part data container that supports
                `read` and `seek` operations.
            uploaded_part_checksum (str/None):
                Checksum of previously uploaded part. Optional, but if specified
                checksum is verified, and part is uploaded only if checksums
                don't match.

        Returns:
            Hex digest of part data MD5 checksum.
        """

        logger.debug("Uploading part_id=%r of problem_id=%r", part_id, problem_id)

        # generate the mutable part stream from immutable stream generator
        part_stream = part_generator()

        # TODO: work-around to get a checksum of a binary stream (avoid 2x read)
        data = part_stream.read()
        digest = Client._digest(data)
        b64digest = Client._checksum_b64(digest)
        hexdigest = Client._checksum_hex(digest)
        del data

        if uploaded_part_checksum is not None:
            if hexdigest == uploaded_part_checksum:
                logger.debug("Uploaded part checksum matches. "
                             "Skipping upload for part_id=%r.", part_id)
                return hexdigest
            else:
                logger.debug("Uploaded part checksum does not match. "
                             "Re-uploading part_id=%r.", part_id)

        # rewind the stream after read
        part_stream.seek(0)

        path = 'bqm/multipart/{problem_id}/part/{part_id}'.format(
            problem_id=problem_id, part_id=part_id)
        headers = {
            'Content-MD5': b64digest,
            'Content-Type': 'application/octet-stream',
        }

        msg = Client._sapi_request(session.put, path, data=part_stream, headers=headers)

        logger.debug("Uploaded part_id=%r of problem_id=%r", part_id, problem_id)

        return hexdigest

    @staticmethod
    @retried(_UPLOAD_REQUEST_RETRIES, backoff=_UPLOAD_RETRIES_BACKOFF)
    def _get_multipart_upload_status(session, problem_id):
        logger.debug("Checking upload status of problem_id=%r", problem_id)

        path = 'bqm/multipart/{problem_id}/status'.format(problem_id=problem_id)

        msg = Client._sapi_request(session.get, path)

        try:
            msg['status']
            msg['parts']
        except KeyError:
            raise InvalidAPIResponseError("'status' and/or 'parts' missing")

        logger.debug("Got upload status=%r for problem_id=%r",
                     msg['status'], problem_id)

        return msg

    @staticmethod
    def _failsafe_get_multipart_upload_status(session, problem_id):
        try:
            return Client._get_multipart_upload_status(session, problem_id)
        except Exception as e:
            logger.debug("Upload status check failed with %r", e)

        return {"status": "UNDEFINED", "parts": []}

    @staticmethod
    @retried(_UPLOAD_REQUEST_RETRIES, backoff=_UPLOAD_RETRIES_BACKOFF)
    def _combine_uploaded_parts(session, problem_id, checksum):
        logger.debug("Combining uploaded parts of problem_id=%r", problem_id)

        path = 'bqm/multipart/{problem_id}/combine'.format(problem_id=problem_id)
        body = dict(checksum=checksum)

        msg = Client._sapi_request(session.post, path, json=body)

        logger.debug("Issued a combine command for problem_id=%r", problem_id)

    @staticmethod
    def _uploaded_parts_from_problem_status(problem_status):
        uploaded_parts = {}
        if problem_status.get('status') == 'UPLOAD_IN_PROGRESS':
            for part in problem_status.get('parts', ()):
                part_no = part.get('part_number')
                checksum = part.get('checksum', '').strip('"')  # fix double-quoting bug
                uploaded_parts[part_no] = checksum
        return uploaded_parts

    def _upload_part_worker(self, problem_id, part_no, chunk_generator,
                            uploaded_part_checksum=None):

        with self.create_session() as session:
            part_checksum = self._upload_multipart_part(
                session, problem_id, part_id=part_no, part_generator=chunk_generator,
                uploaded_part_checksum=uploaded_part_checksum)

            return part_no, part_checksum

    def _upload_problem_worker(self, problem, problem_id=None):
        """Upload a problem to SAPI using multipart upload interface.

        Args:
            problem (bytes/str/file-like):
                Problem description.

            problem_id (str, optional):
                Problem ID under which to upload the problem. If omitted, a new
                problem is created.

        """

        # in python 3.7+ we could create the session once, on thread init,
        # via executor initializer
        with self.create_session() as session:
            chunks = ChunkedData(problem, chunk_size=self._UPLOAD_PART_SIZE_BYTES)
            size = chunks.total_size

            if problem_id is None:
                try:
                    problem_id = self._initiate_multipart_upload(session, size)
                except Exception as e:
                    errmsg = ("Multipart upload initialization failed "
                              "with {!r}.".format(e))
                    logger.error(errmsg)
                    raise ProblemUploadError(errmsg) from e

            # check problem status, so we only upload parts missing or invalid
            problem_status = \
                self._failsafe_get_multipart_upload_status(session, problem_id)

            if problem_status.get('status') == 'UPLOAD_COMPLETED':
                logger.debug("Problem already uploaded.")
                return problem_id

            uploaded_parts = \
                self._uploaded_parts_from_problem_status(problem_status)

            # enqueue all parts, worker skips if checksum matches
            parts = {}
            for chunk_no, chunk_generator in enumerate(chunks.generators()):
                part_no = chunk_no + 1
                part_future = self._upload_part_executor.submit(
                    self._upload_part_worker,
                    problem_id, part_no, chunk_generator,
                    uploaded_part_checksum=uploaded_parts.get(part_no))

                parts[part_no] = part_future

            # wait for parts to upload/fail
            concurrent.futures.wait(parts.values())

            # verify all parts uploaded without error
            for part_no, part_future in parts.items():
                try:
                    part_future.result()
                except Exception as e:
                    errmsg = ("Multipart upload of problem_id={!r} failed for "
                              "part_no={!r} with {!r}.".format(problem_id, part_no, e))
                    logger.error(errmsg)
                    raise ProblemUploadError(errmsg) from e

            # verify all parts uploaded via status call
            # (check remote checksum matches the local one)
            final_problem_status = \
                self._failsafe_get_multipart_upload_status(session, problem_id)

            final_uploaded_parts = \
                self._uploaded_parts_from_problem_status(final_problem_status)

            if len(final_uploaded_parts) != len(parts):
                errmsg = "Multipart upload unexpectedly failed for some parts."
                logger.error(errmsg)
                logger.debug("problem_id=%r, expected_parts=%r, uploaded_parts=%r",
                             problem_id, parts.keys(), final_uploaded_parts.keys())
                raise ProblemUploadError(errmsg)

            for part_no, part_future in parts.items():
                _, part_checksum = part_future.result()
                remote_checksum = final_uploaded_parts[part_no]
                if part_checksum != remote_checksum:
                    errmsg = ("Checksum mismatch for part_no={!r} "
                              "(local {!r} != remote {!r})".format(
                                  part_no, part_checksum, remote_checksum))
                    logger.error(errmsg)
                    raise ProblemUploadError(errmsg)

            # send parts combine request
            combine_checksum = Client._combined_checksum(final_uploaded_parts)
            try:
                self._combine_uploaded_parts(session, problem_id, combine_checksum)
            except Exception as e:
                errmsg = ("Multipart upload of problem_id={!r} failed on parts "
                          "combine with {!r}".format(problem_id, e))
                logger.error(errmsg)
                raise ProblemUploadError(errmsg) from e

            return problem_id

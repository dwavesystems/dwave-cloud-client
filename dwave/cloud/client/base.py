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
    This example creates a client using the local system's default D-Wave Cloud Client
    configuration file, which is configured to access a D-Wave 2000Q QPU, submits
    a :term:`QUBO` problem (a Boolean NOT gate represented by a penalty model), and
    samples 5 times.

    >>> from dwave.cloud import Client
    >>> Q = {(0, 0): -1, (0, 4): 0, (4, 0): 2, (4, 4): -1}
    >>> with Client.from_config() as client:  # doctest: +SKIP
    ...     solver = client.get_solver()
    ...     computation = solver.sample_qubo(Q, num_reads=5)
    ...
    >>> for i in range(5):     # doctest: +SKIP
    ...     print(computation.samples[i][0], computation.samples[i][4])
    ...
    (1, 0)
    (1, 0)
    (0, 1)
    (0, 1)
    (0, 1)

"""

import re
import time
import json
import copy
import queue
import logging
import inspect
import warnings
import operator
import threading

import base64
import hashlib
import codecs
import concurrent.futures

from itertools import chain, zip_longest
from functools import partial, wraps, lru_cache
from collections import abc, namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict

import requests
import urllib3
from dateutil.parser import parse as parse_datetime
from plucky import pluck

from dwave.cloud import api
from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.exceptions import *    # TODO: fix
from dwave.cloud.computation import Future
from dwave.cloud.config import (
    load_config, parse_float, parse_int, parse_boolean, update_config)
from dwave.cloud.solver import Solver, available_solvers
from dwave.cloud.concurrency import PriorityThreadPoolExecutor
from dwave.cloud.upload import ChunkedData
from dwave.cloud.events import dispatches_events
from dwave.cloud.utils import (
    TimeoutingHTTPAdapter, BaseUrlSession, user_agent,
    datetime_to_timestamp, utcnow, cached, retried, is_caused_by)

__all__ = ['Client']

logger = logging.getLogger(__name__)


class Client(object):
    """Base client class for all D-Wave API clients. Used by QPU, software and
    hybrid :term:`sampler` classes.

    Manages workers and handles thread pools for submitting problems, cancelling
    tasks, polling problem status, and retrieving results.

    Args:
        region (str, optional, default='na-west-1'):
            D-Wave Solver API region. To see available regions use
            :meth:`.Client.get_regions`.

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

        headers (dict/str, optional):
            Newline-separated additional HTTP headers to include with each
            API request, or a dictionary of (key, value) pairs.

        client_cert (str, optional):
            Path to client side certificate file.

        client_cert_key (str, optional):
            Path to client side certificate key file.

        poll_backoff_min (float, default=0.05):
            Problem status is polled with exponential back-off schedule.
            Duration of the first interval (between first and second poll) is
            set to ``poll_backoff_min`` seconds.

        poll_backoff_max (float, default=60):
            Problem status is polled with exponential back-off schedule.
            Maximum back-off period is limited to ``poll_backoff_max`` seconds.

        poll_backoff_base (float, default=1.3):
            Problem status is polled with exponential back-off schedule.
            The exponential function base is defined with ``poll_backoff_base``.
            Interval between ``poll_idx`` and ``poll_idx + 1`` is given with::

                poll_backoff_min * poll_backoff_base ** poll_idx

            with upper bound set to ``poll_backoff_max``.

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

        defaults (dict, optional):
            Defaults for the client instance that override the class
            :attr:`.Client.DEFAULTS`.

    Note:
        Default values of all constructor arguments listed above are kept in
        a class variable :attr:`.Client.DEFAULTS`.

        Instance-level defaults can be specified via ``defaults`` argument.

    .. deprecated:: 0.10.0

        Positional arguments in :class:`.Client` constructor are deprecated and
        will be removed in 0.12.0.

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
    DEFAULT_API_ENDPOINT = api.constants.DEFAULT_SOLVER_API_ENDPOINT
    DEFAULT_API_REGION = api.constants.DEFAULT_REGION

    # Class-level defaults for all constructor and factory arguments
    DEFAULTS = {
        # factory only
        'config_file': None,
        'profile': None,
        'client': 'base',
        # constructor (and factory)
        'metadata_api_endpoint': api.constants.DEFAULT_METADATA_API_ENDPOINT,
        'region': DEFAULT_API_REGION,
        # NOTE: should we rename endpoint to solver_api_endpoint for clarity?
        'endpoint': None,       # defined via region, resolved on client init
        'token': None,
        'solver': None,
        'proxy': None,
        'permissive_ssl': False,
        'request_timeout': 60,
        'polling_timeout': None,
        'connection_close': False,
        'headers': None,
        'client_cert': None,
        'client_cert_key': None,
        # poll back-off schedule defaults [sec]
        'poll_backoff_min': 0.05,
        'poll_backoff_max': 60,
        'poll_backoff_base': 1.3,
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
    _POLL_THREAD_COUNT = 2
    _LOAD_THREAD_COUNT = 5

    # Poll grouping time frame; two scheduled polls are grouped if closer than [sec]:
    _POLL_GROUP_TIMEFRAME = 2

    # Downloaded solver definition cache maxage [sec]
    _SOLVERS_CACHE_MAXAGE = 300     # 5 min

    # Downloaded region metadata cache maxage [sec]
    _REGIONS_CACHE_MAXAGE = 86400   # 1 day

    # Multipart upload parameters
    _UPLOAD_PART_SIZE_BYTES = 5 * 1024 * 1024
    _UPLOAD_PART_RETRIES = 2
    _UPLOAD_REQUEST_RETRIES = 2
    _UPLOAD_RETRIES_BACKOFF = lambda retry: 2 ** retry

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
            :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
                Config file specified or detected could not be opened or read.

            :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
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

    def _resolve_region_endpoint(self, *,
                                 region: Optional[str] = None,
                                 endpoint: Optional[str] = None) -> Tuple[str, str]:
        """For a region/endpoint pair from config, return the Solver API
        endpoint to use (and the matching region).

        Explicit endpoint will override the region (i.e. region extension is
        backwards-compatible).

        Regional endpoint is fetched from Metadata API. If Metadata API is not
        available, default global endpoint is used.
        """
        if endpoint:
            return (region, endpoint)

        if not region:
            return (self.DEFAULT_API_REGION, self.DEFAULT_API_ENDPOINT)

        try:
            regions = self.get_regions()
        except (api.exceptions.RequestError, ValueError) as exc:
            logger.warning("Failed to fetch available regions: %r. "
                           "Using the default Solver API endpoint.", exc)
            return (self.DEFAULT_API_REGION, self.DEFAULT_API_ENDPOINT)

        if region not in regions:
            raise ValueError(f"Region {region!r} unknown. "
                             f"Try one of {list(regions.keys())!r}.")
        return (region, regions[region]['endpoint'])

    @dispatches_events('client_init')
    def __init__(self, *args, **kwargs):
        # for (reasonable) backwards compatibility, accept only the first few
        # positional args.
        if len(args) > 3:
            raise TypeError(
                "Client constructor takes up to 3 positional "
                f"arguments, but {len(args)} were given")

        if len(args) > 0:
            warnings.warn(
                "Positional arguments in Client constructor are deprecated "
                "since dwave-cloud-client 0.10.0, and will be removed in 0.12.0. "
                "Use keyword arguments instead.",
                DeprecationWarning, stacklevel=3)

            argsdict = dict(zip(('endpoint', 'token', 'solver'), args))
            intersection = argsdict.keys() & kwargs
            if intersection:
                raise TypeError(f"Client() got multiple values for {intersection}")
            kwargs.update(argsdict)

        logger.debug("Client init called with: %r", kwargs)

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

        logger.debug("Client options with defaults: %r", options)

        # configure MetadataAPI access -- needed by Client.get_regions()
        self.metadata_api_endpoint = options['metadata_api_endpoint']

        # parse headers as they might be needed by Client.get_regions()
        headers = options['headers']
        if not headers:
            headers_dict = {}
        elif isinstance(headers, abc.Mapping):
            headers_dict = headers
        elif isinstance(headers, str):
            try:
                # valid  headers = "Field-1: value-1\nField-2: value-2"
                headers_dict = {key.strip(): val.strip()
                                for key, val in [line.split(':')
                                                 for line in headers.strip().split('\n')]}
            except Exception as e:
                logger.debug("Invalid headers: %r", headers)
                headers_dict = {}
        else:
            raise ValueError("HTTP headers expected in a dict, or a string")
        logger.debug("Parsed headers=%r", headers_dict)
        self.headers = headers_dict

        # resolve endpoint using region
        region, endpoint = self._resolve_region_endpoint(
            region=options.get('region'), endpoint=options.get('endpoint'))

        # sanity check
        if not endpoint:
            raise ValueError("API endpoint not defined")

        token = options['token']
        if not token:
            raise ValueError("API token not defined")

        # parse optional client certificate
        client_cert = options['client_cert']
        client_cert_key = options['client_cert_key']
        if client_cert_key is not None:
            if client_cert is not None:
                client_cert = (client_cert, client_cert_key)
            else:
                raise ValueError(
                    "Client certificate key given, but the cert is missing")

        # parse solver
        solver = options['solver']
        if not solver:
            solver_def = {}
        elif isinstance(solver, abc.Mapping):
            solver_def = solver
        elif isinstance(solver, str):
            # support features dict encoded as JSON in our config INI file
            # TODO: push this decoding to the config module, once we switch to a
            #       richer config format (JSON or YAML)
            try:
                solver_def = json.loads(solver)
            except Exception:
                # unparseable json, assume string name for solver
                # we'll deprecate this eventually, but for now just convert it to
                # features dict (equality constraint on full solver name)
                logger.debug("Invalid solver JSON, assuming string name: %r", solver)
                solver_def = dict(name__eq=solver)
        else:
            raise ValueError("Expecting a features dictionary or a string name for 'solver'")
        logger.debug("Parsed solver=%r", solver_def)

        # Store connection/session parameters
        # TODO: consolidate all options under Client.options or similar
        self.region = region    # for record only
        self.endpoint = endpoint
        self.token = token
        self.default_solver = solver_def

        self.client_cert = client_cert
        self.request_timeout = parse_float(options['request_timeout'])
        self.polling_timeout = parse_float(options['polling_timeout'])

        self.proxy = options['proxy']
        self.permissive_ssl = parse_boolean(options['permissive_ssl'])
        self.connection_close = parse_boolean(options['connection_close'])

        self.poll_backoff_min = parse_float(options['poll_backoff_min'])
        self.poll_backoff_max = parse_float(options['poll_backoff_max'])
        self.poll_backoff_base = parse_float(options['poll_backoff_base'])

        self.http_retry_total = parse_int(options['http_retry_total'])
        self.http_retry_connect = parse_int(options['http_retry_connect'])
        self.http_retry_read = parse_int(options['http_retry_read'])
        self.http_retry_redirect = parse_int(options['http_retry_redirect'])
        self.http_retry_status = parse_int(options['http_retry_status'])
        self.http_retry_backoff_factor = parse_float(options['http_retry_backoff_factor'])
        self.http_retry_backoff_max = parse_float(options['http_retry_backoff_max'])

        opts = (
            'region', 'endpoint', 'token', 'default_solver',
            'client_cert', 'request_timeout', 'polling_timeout',
            'proxy', 'headers', 'permissive_ssl', 'connection_close',
            'poll_backoff_min', 'poll_backoff_max', 'poll_backoff_base',
            'http_retry_total', 'http_retry_connect', 'http_retry_read',
            'http_retry_redirect', 'http_retry_status',
            'http_retry_backoff_factor', 'http_retry_backoff_max')
        logger.debug(
            "Client initialized with (%s)",
            ", ".join("{}={!r}".format(o, getattr(self, o)) for o in opts))

        # Create session for main thread only
        self.session = self.create_session()

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

    # note: @cached_property available only in py38+
    @property
    @lru_cache(maxsize=None)
    def _user_agent(self):
        """User-Agent string for this client instance, as returned by
        :meth:`~dwave.cloud.utils.user_agent`, computed on first access and
        cached for the lifespan of the client.

        Note:
            The only tags that might change are platform tags, as returned by
            ``dwave.common.platform.tags`` entry points, and `platform.platform()`
            (like linux kernel version). Assuming OS/machine won't change during
            client's lifespan, and typical platform tags defined via entry points
            depend on process environments (which rarely change), it's pretty safe
            to always use the per-instance cached user agent.
        """
        return user_agent(__packagename__, __version__)

    def create_session(self):
        """Create a new requests session based on client's (self) params.

        Note: since `requests.Session` is NOT thread-safe, every thread should
        create and use an isolated session.
        """

        # allow endpoint path to not end with /
        endpoint = self.endpoint
        if not endpoint.endswith('/'):
            endpoint += '/'

        # create http idempotent Retry config
        def get_retry_conf():
            # need a subclass to override the backoff_max
            class Retry(urllib3.Retry):
                BACKOFF_MAX = self.http_retry_backoff_max

            return Retry(
                total=self.http_retry_total,
                connect=self.http_retry_connect,
                read=self.http_retry_read,
                redirect=self.http_retry_redirect,
                status=self.http_retry_status,
                backoff_factor=self.http_retry_backoff_factor,
                raise_on_redirect=True,
                raise_on_status=True,
                respect_retry_after_header=True)

        session = BaseUrlSession(base_url=endpoint)
        session.mount('http://',
            TimeoutingHTTPAdapter(timeout=self.request_timeout,
                                  max_retries=get_retry_conf()))
        session.mount('https://',
            TimeoutingHTTPAdapter(timeout=self.request_timeout,
                                  max_retries=get_retry_conf()))

        session.headers.update({'User-Agent': self._user_agent})
        if self.headers:
            session.headers.update(self.headers)
        if self.token:
            session.headers.update({'X-Auth-Token': self.token})
        if self.client_cert:
            session.cert = self.client_cert

        session.proxies = {'http': self.proxy, 'https': self.proxy}
        if self.permissive_ssl:
            session.verify = False
        if self.connection_close:
            session.headers.update({'Connection': 'close'})

        # Debug-log headers
        logger.trace("create_session(session.headers=%r)", session.headers)

        return session

    def close(self):
        """Perform a clean shutdown.

        Waits for all the currently scheduled work to finish, kills the workers,
        and closes the connection pool.

        .. note:: Ensure your code does not submit new work while the connection is closing.

        Where possible, it is recommended you use a context manager (a :code:`with Client.from_config(...) as`
        construct) to ensure your code properly closes all resources.

        Examples:
            This example creates a client (based on an auto-detected configuration file), executes
            some code (represented by a placeholder comment), and then closes the client.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()    # doctest: +SKIP
            >>> # code that uses client
            >>> client.close()    # doctest: +SKIP

        """
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
        self._upload_problem_executor.shutdown()
        logger.debug("Shutting down problem part upload executor")
        self._upload_part_executor.shutdown()
        logger.debug("Shutting down problem encoder executor")
        self._encode_problem_executor.shutdown()

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
        for worker in chain(self._submission_workers, self._cancel_workers,
                            self._poll_workers, self._load_workers):
            worker.join()

        # Close the main thread's session
        self.session.close()

    def __enter__(self):
        """Let connections be used in with blocks."""
        return self

    def __exit__(self, *args):
        """At the end of a with block perform a clean shutdown of the connection."""
        self.close()
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
                    return solver and solver.id.startswith('My_Solver_')

        """
        return True

    @staticmethod
    @cached.ondisk(maxage=_REGIONS_CACHE_MAXAGE)
    def _fetch_available_regions(metadata_api_endpoint, **config):
        logger.info("Fetching available regions from the Metadata API at %r",
                    metadata_api_endpoint)

        with api.Regions(endpoint=metadata_api_endpoint, **config) as regions:
            data = regions.list_regions()

        logger.debug("Received region metadata: %r", data)

        return data

    def get_regions(self, refresh: bool = False) -> Dict[str, Dict[str, str]]:
        """Retrieve available API regions.

        Args:
            refresh:
                Force cache refresh.

        Returns:
            Mapping of region details (name and endpoint) over region codes.
        """
        try:
            rs = Client._fetch_available_regions(
                metadata_api_endpoint=self.metadata_api_endpoint,
                headers=self.headers,
                refresh_=refresh)
        except api.exceptions.RequestError as exc:
            logger.debug("Metadata API unavailable", exc_info=True)
            raise ValueError(
                f"Metadata API unavailable at {self.metadata_api_endpoint!r}")

        logger.info("Using region metadata: %r", rs)

        return {r.code: {"name": r.name, "endpoint": r.endpoint} for r in rs}

    @cached(maxage=_SOLVERS_CACHE_MAXAGE)
    def _fetch_solvers(self, name=None):
        if name is not None:
            logger.info("Fetching definition of a solver with name=%r", name)
            url = 'solvers/remote/{}/'.format(name)
        else:
            logger.info("Fetching definitions of all available solvers")
            url = 'solvers/remote/'

        try:
            data = Client._sapi_request(self.session.get, url)
        except SAPIError as exc:
            if name is not None and exc.error_code == 404:
                raise SolverNotFoundError("No solver with name={!r} available".format(name))
            else:
                raise

        if name is not None:
            data = [data]

        logger.info("Received solver data for %d solver(s).", len(data))
        logger.trace("Solver data received for solver name=%r: %r", name, data)

        solvers = []
        for solver_desc in data:
            for solver_class in available_solvers:
                try:
                    logger.debug("Trying to instantiate %r", solver_class.__name__)
                    solver = solver_class(self, solver_desc)
                    if self.is_solver_handled(solver):
                        solvers.append(solver)
                        logger.info("Adding solver %r", solver)
                        break
                    else:
                        logger.debug("Skipping solver %r (not handled by this client)", solver)

                except UnsupportedSolverError as e:
                    logger.debug("Skipping solver due to %r", e)

            # propagate all other/decoding errors, like InvalidAPIResponseError, etc.

        return solvers

    def retrieve_answer(self, id_):
        """Retrieve a problem by id.

        Args:
            id_ (str):
                As returned by :attr:`Future.id`.

        Returns:
            :class:`Future`

        """
        future = Future(None, id_)
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

        * `name` (str): Solver name/id.
        * `qpu` (bool): Solver is a QPU?
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
                num_qubits__gt=2000,                # we need more than 2000 qubits
                num_qubits__lt=4000,                # ... but fewer than 4000 qubits
                num_qubits__within=(2000, 4000),    # an alternative to the previous two lines
                num_active_qubits=1089,             # we want a particular number of active qubits
                vfyc=True,                          # we require a fully yielded Chimera
                vfyc__in=[False, None],             # inverse of the previous filter
                vfyc__available=False,              # we want solvers that do not advertize the vfyc property
                anneal_schedule=True,               # we need support for custom anneal schedule
                max_anneal_schedule_points__gte=4,  # we need at least 4 points for our anneal schedule
                num_reads_range__covers=1000,       # our solver must support returning 1000 reads
                extended_j_range__covers=[-2, 2],   # we need extended J range to contain subrange [-2,2]
                couplers__contains=[0, 128],        # coupler (edge between) qubits (0,128) must exist
                couplers__issuperset=[[0,128], [0,4]],
                                                    # two couplers required: (0,128) and (0,4)
                qubits__issuperset={0, 4, 215},     # qubits 0, 4 and 215 must exist
                supported_problem_types__issubset={'ising', 'qubo'},
                                                    # require Ising, QUBO or both to be supported
                name='DW_2000Q_5',                  # full solver name/ID match
                name__regex='.*2000.*',             # partial/regex-based solver name match
                chip_id__regex='DW_.*',             # chip ID prefix must be DW_
                topology__type__eq="chimera"        # topology.type must be chimera
                topology__type="chimera"            # same as above, `eq` implied even for nested properties
            )
        """

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
                return op(None, val)

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

        # filter
        solvers = self._fetch_solvers(**query)
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
            uses the default solver, the second explicitly selects another solver.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()    # doctest: +SKIP
            >>> client.get_solvers()   # doctest: +SKIP
            [Solver(id='2000Q_ONLINE_SOLVER1'), Solver(id='2000Q_ONLINE_SOLVER2')]
            >>> solver1 = client.get_solver()    # doctest: +SKIP
            >>> solver2 = client.get_solver(name='2000Q_ONLINE_SOLVER2')    # doctest: +SKIP
            >>> solver1.id  # doctest: +SKIP
            '2000Q_ONLINE_SOLVER1'
            >>> solver2.id   # doctest: +SKIP
            '2000Q_ONLINE_SOLVER2'
            >>> # code that uses client
            >>> client.close() # doctest: +SKIP

        """
        logger.info("Requested a solver that best matches feature filters=%r", filters)

        # backward compatibility: name as the first feature
        if name is not None:
            filters.setdefault('name', name)

        # allow `order_by` to be specified as part of solver features dict
        order_by = filters.pop('order_by', None)

        # in absence of other filters, config/env solver filters/name are used
        if not filters and self.default_solver:
            filters = copy.deepcopy(self.default_solver)

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

    def _submit(self, body, future):
        """Enqueue a problem for submission to the server.

        This method is thread safe.
        """
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
                    body = '[' + ','.join(msg.body.result() for msg in ready_problems) + ']'
                    logger.debug('Size of POST body = %d', len(body))
                    message = Client._sapi_request(session.post, 'problems/', body)
                    logger.debug("Finished submitting %d problems", len(ready_problems))

                except Exception as exc:
                    logger.debug("Submit failed for %d problems with %r",
                                 len(ready_problems), exc)
                    for msg in ready_problems:
                        msg.future._set_exception(exc)
                        task_done()
                    continue

                # Pass on the information
                for submission, msg in zip_longest(ready_problems, message):
                    try:
                        self._handle_problem_status(msg, submission.future)
                    except Exception as exc:
                        submission.future._set_exception(exc)
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
                        future.solver = self.get_solver(name=message['solver'])

                    future._set_message(message)
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
                    ids = [item[0] for item in item_list]
                    Client._sapi_request(session.delete, 'problems/', json=ids)

                except Exception as exc:
                    for _, future in item_list:
                        if future is not None:
                            future._set_exception(exc)

                # Mark all the ids as processed regardless of success or failure.
                for _ in item_list:
                    self._cancel_queue.task_done()

        except Exception as err:
            logger.exception(err)

        finally:
            session.close()

    def _poll(self, future):
        """Enqueue a problem to poll the server for status."""

        if future._poll_backoff is None:
            # on first poll, start with minimal back-off
            future._poll_backoff = self.poll_backoff_min
        else:
            # on subsequent polls, do exponential back-off, clipped to a range
            future._poll_backoff = \
                max(self.poll_backoff_min,
                    min(future._poll_backoff * self.poll_backoff_base,
                        self.poll_backoff_max))

        # for poll priority we use timestamp of next scheduled poll
        at = time.time() + future._poll_backoff

        now = utcnow()
        future_age = (now - future.time_created).total_seconds()
        logger.debug("Polling scheduled at %.2f with %.2f sec new back-off for: %s (future's age: %.2f sec)",
                     at, future._poll_backoff, future.id, future_age)

        # don't enqueue for next poll if polling_timeout is exceeded by then
        future_age_on_next_poll = future_age + (at - datetime_to_timestamp(now))
        if self.polling_timeout is not None and future_age_on_next_poll > self.polling_timeout:
            logger.debug("Polling timeout exceeded before next poll: %.2f sec > %.2f sec, aborting polling!",
                         future_age_on_next_poll, self.polling_timeout)
            raise PollingTimeout

        self._poll_queue.put((at, future))

    def _do_poll_problems(self):
        """Poll the server for the status of a set of problems.

        Note:
            This method is always run inside of a daemon thread.
        """
        session = self.create_session()
        try:
            # grouped futures (all scheduled within _POLL_GROUP_TIMEFRAME)
            frame_futures = {}

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
                while len(frame_futures) < self._STATUS_QUERY_SIZE:
                    try:
                        task = self._poll_queue.get_nowait()
                    except queue.Empty:
                        break

                    at, future = task
                    if at - frame_earliest <= self._POLL_GROUP_TIMEFRAME:
                        if not add(future):
                            return
                    else:
                        task_done()
                        self._poll_queue.put(task)
                        break

                # build a query string with ids of all futures in this frame
                ids = [future.id for future in frame_futures.values()]
                logger.debug("Polling for status of futures: %s", ids)
                query_string = 'problems/?id=' + ','.join(ids)

                # if futures were cancelled while `add`ing, skip empty frame
                if not ids:
                    continue

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
                        if 500 <= exc.error_code < 600:
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

                for id_ in frame_futures.keys():
                    task_done()

        except Exception as err:
            logger.exception(err)

        finally:
            session.close()

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
                    self._load_queue.task_done()
                    continue

                # Dispatch the results, mark the task complete
                self._handle_problem_status(message, future)
                self._load_queue.task_done()

        except Exception as err:
            logger.error('Load result error: ' + str(err))

        finally:
            session.close()

    def upload_problem_encoded(self, problem, problem_id=None):
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

        caller = inspect.stack()[1].function
        verb = meth.__name__
        logger.trace("[%s] request: session.%s(*%r, **%r)", caller, verb, args, kwargs)

        # execute request
        try:
            response = meth(*args, **kwargs)
        except Exception as exc:
            if is_caused_by(exc, (requests.exceptions.Timeout,
                                  urllib3.exceptions.TimeoutError)):
                raise RequestTimeout
            else:
                raise

        # parse response
        logger.trace("[%s] response: (code=%r, body=%r)",
                     caller, response.status_code, response.text)

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
                return response.json()
            except:
                raise InvalidAPIResponseError("JSON response expected")

        else:
            if response.status_code == 401:
                raise SolverAuthenticationError(error_code=401)

            try:
                msg = response.json()
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
        # checksums: Dict[int, str] => hex(md5(cat(digests))): str
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

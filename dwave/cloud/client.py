"""
Base client for all D-Wave API clients.

Used by QPU and software sampler classes.

"""

from __future__ import division, absolute_import

import sys
import time
import logging
import threading
import requests
import posixpath
import collections
from itertools import chain

from dateutil.parser import parse as parse_datetime
from six.moves import queue, range

from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.exceptions import *
from dwave.cloud.config import load_config, legacy_load_config
from dwave.cloud.solver import Solver
from dwave.cloud.utils import datetime_to_timestamp


_LOGGER = logging.getLogger(__name__)


class Client(object):
    """
    Base client class for all D-Wave API clients.

    Implements workers (and handles thread pools) for problem submittal, task
    cancellation, problem status polling and results downloading.

    Args:
        endpoint (str):
            D-Wave API endpoint URL.

        token (str):
            Authentication token for the D-Wave API.

        solver (str):
            Default solver.

        proxy (str):
            Proxy URL to be used for accessing the D-Wave API.

        permissive_ssl (bool, default=False):
            Disables SSL verification.

    Examples:
        This example show direct :class:`~dwave.cloud.client.Client` initializiation.

        The most basic initialization of a new :class:`~dwave.cloud.client.Client`
        instance (like :class:`dwave.cloud.qpu.Client` or
        :class:`dwave.cloud.sw.Client`) is via class constructor arguments. You should
        specify values for at least ``endpoint`` and ``token``::

        >>> from dwave.cloud.qpu import Client
        >>> client = Client(endpoint='https://cloud.dwavesys.com/sapi', token='secret')

        This example shows unrecognized configuration file keys being passed through.

        All (unrecognized) config file keys are passed-through to the respective
        ``Client`` class constructor as string keyword arguments.

        This makes config-parsing future-proof and enables passing of (undocumented)
        variables like ``permissive_ssl``. With::

            [defaults]
            ...

            [testing]
            endpoint = <testing-endpoint>
            permissive_ssl = 1

        The ``Client`` is constructed with ``Client(endpoint=..., permissive_ssl='1')``.

    """

    # The status flags that a problem can have
    STATUS_IN_PROGRESS = 'IN_PROGRESS'
    STATUS_PENDING = 'PENDING'
    STATUS_COMPLETE = 'COMPLETED'
    STATUS_FAILED = 'FAILED'
    STATUS_CANCELLED = 'CANCELLED'

    # Identify as something like `dwave-cloud-client/0.4` in all requests
    USER_AGENT = '{}/{}'.format(__packagename__, __version__)

    # Cases when multiple status flags qualify
    ANY_STATUS_ONGOING = [STATUS_IN_PROGRESS, STATUS_PENDING]
    ANY_STATUS_NO_RESULT = [STATUS_FAILED, STATUS_CANCELLED]

    # Number of problems to include in a status query
    _STATUS_QUERY_SIZE = 100

    # Number of worker threads for each problem processing task
    _SUBMISSION_THREAD_COUNT = 5
    _CANCEL_THREAD_COUNT = 1
    _POLL_THREAD_COUNT = 2
    _LOAD_THREAD_COUNT = 5

    # Poll back-off interval [sec]
    _POLL_BACKOFF_MIN = 1
    _POLL_BACKOFF_MAX = 60

    # Poll grouping time frame; two scheduled polls are grouped if closer than [sec]:
    _POLL_GROUP_TIMEFRAME = 2

    @classmethod
    def from_config(cls, config_file=None, profile=None, client=None,
                    endpoint=None, token=None, solver=None, proxy=None,
                    legacy_config_fallback=True, **kwargs):
        """Client factory method which loads configuration from file(s),
        process environment variables and explicitly provided values, creating
        and returning the appropriate client instance
        (:class:`dwave.cloud.qpu.Client` or :class:`dwave.cloud.sw.Client`).

        Note:
            For details on config loading from files and environment, please
            see :func:`~dwave.cloud.config.load_config`.

        Args:
            config_file (str/None/False/True, default=None):
                Path to config file. ``None`` for auto-detect, ``False`` to
                skip loading from any file (including auto-detection), and
                ``True`` to force auto-detection, disregarding environment value
                for config file.

            profile (str, default=None):
                Profile name (config file section name). If undefined it is
                taken from ``DWAVE_PROFILE`` environment variable, or config
                file, or first section, or defaults. For details, see
                :func:`~dwave.cloud.config.load_config`.

            client (str, default=None):
                Client class (selected by name) to use for accessing the API.
                Use ``qpu`` to specify the :class:`dwave.cloud.qpu.Client` and
                ``sw`` for :class:`dwave.cloud.sw.Client`.

            endpoint (str, default=None):
                API endpoint URL.

            token (str, default=None):
                API authorization token.

            solver (str, default=None):
                Default solver to use in :meth:`~dwave.cloud.client.Client.get_solver`.
                If undefined, you'll have to explicitly specify the solver name/id
                in all calls to :meth:`~dwave.cloud.client.Client.get_solver`.

            proxy (str, default=None):
                URL for proxy to use in connections to D-Wave API. Can include
                username/password, port, scheme, etc. If undefined, client will
                connect directly to the API (unless you use a system-level proxy).

            legacy_config_fallback (bool, default=True):
                If loading from a ``dwave.conf`` config file fails, try
                loading the ``.dwrc`` legacy config.

            **kwargs:
                All remaining keyword arguments are passed-through as-is to the
                chosen `Client` constructor method.

                A notable custom argument is `permissive_ssl`.

                Note: all user-defined keys from config files are propagated to
                the `Client` constructor too, and can be overridden with these
                keyword arguments.

        Examples:
            (1) This example initializes :class:`~dwave.cloud.client.Client` from an
            explicitly given configuration file.

            Configuration parameters for ``Client`` classes can be read from a config file
            (and process environment variables) via ``Client`` factory method
            :func:`~dwave.cloud.client.Client.from_config`. Configuration loading is
            delegated to :func:`~dwave.cloud.config.load_config`.

            Assuming ``/path/to/config`` file contains::

                [prod]
                endpoint = https://cloud.dwavesys.com/sapi
                token = secret
                client = qpu
                solver = DW_2000Q_1

            >>> from dwave.cloud import Client
            >>> client = Client.from_config(config_file='/path/to/config')

            The above creates a client object which will connect to D-Wave production QPU,
            using :class:`dwave.cloud.qpu.Client` and ``DW_2000Q_1`` as a default solver.

            Note: in case the config file specified does not exist, or the file is
            unreadable (e.g. no read permission), or format is invalid,
            :exc:`~dwave.cloud.exceptions.ConfigFileReadError` or
            :exc:`~dwave.cloud.exceptions.ConfigFileParseError` will be raised.

            (2) This example demonstrates auto-detection of a configuration file.

            If ``config_file`` parameter to :func:`~dwave.cloud.client.Client.from_config`
            factory method is not specified (or is explicitly set to ``None``), config file
            location is auto-detected. Lookup order of paths examined is described in
            :func:`~dwave.cloud.config.load_config_from_files`.

            Assuming (on Linux) the file ``~/.config/dwave/dwave.conf`` contains::

                [prod]
                endpoint = https://cloud.dwavesys.com/sapi
                token = secret

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()

            Note: config file read/parse exceptions are not raised in the auto-detect case
            if no suitable file is found. If a file is found, but it's unreadable or
            unparseable, exception are still raised.

            (3) This example demonstrates defaults and profiles.
            One config file can contain multiple profiles, each defining a separate
            (endpoint, token, solver, etc.) combination. Since config file conforms to a
            standard Windows INI-style format, profiles are defined by sections like:
            ``[profile-a]`` and ``[profile-b]``.

            Default values for undefined profile keys are taken from the ``[defaults]``
            section.

            For example, assuming ``~/.config/dwave/dwave.conf`` contains::

                [defaults]
                endpoint = https://cloud.dwavesys.com/sapi
                client = qpu

                [dw2000]
                solver = DW_2000Q_1
                token = ...

                [software]
                client = sw
                solver = c4-sw_sample
                token = ...

                [alpha]
                endpoint = https://url.to.alpha/api
                proxy = http://user:pass@myproxy.com:8080/
                token = ...

            We can instantiate a client for D-Wave 2000Q QPU endpoint with

            >>> from dwave.cloud import Client
            >>> client = Client.from_config(profile='dw2000')

            and a client for remote software solver with::

            >>> client = Client.from_config(profile='software')

            ``alpha`` profile will connect to a pre-release API endpoint via defined HTTP
            proxy server.

            (4) This example demonstrates progressive config file override

            System-wide configuration files are overridden (on a key-by-key,
            section-by-section basis) with user-local config files, and then by current
            directory config files.

            For example, if ``/usr/local/share/dwave/dwave.conf`` has::

                [defaults]
                endpoint = <production>
                client = qpu

                [prod]
                token = <token>

            and ``~/.config/dwave/dwave.conf`` has::

                [alpha]
                endpoint = <alpha>
                token = <token>

            and ``./dwave.conf`` has::

                [alpha]
                proxy = <proxy>

            then both profiles can be loaded::

                from dwave.cloud import Client

                production_client = Client.from_config(profile='prod')

                alpha_client = Client.from_config(profile='alpha')

            Note: ``alpha`` profile will use alpha endpoint, but also a proxy and a QPU
            client.

            (5) This example demonstrates Environment variables and explicit argument override.

            All of the basic ``Client`` configuration variables can be specified **via
            environment variables** (see :func:`~dwave.cloud.config.load_config`). Values
            from the environment will **override** values read from config file(s).

            In addition, location of the very config file and profile can also be specified
            within process environment.

            For example, with ``/tests/test.conf`` containing::

                [defaults]
                endpoint = <production>

                [prod]
                token = <token-1>

                [test]
                token = <token-2>

            and environment variables set as::

                DWAVE_CONFIG_FILE=/tests/test.conf
                DWAVE_PROFILE=test
                DWAVE_API_SOLVER=DW_2000Q_1

            then the standard::

                client = Client.from_config()

            will construct a client to connect with ``test`` profile from ``test.conf``
            file, and in addition, to use the ``DW_2000Q_1`` solver.

            However, **explicit keyword argument** values in
            :func:`~dwave.cloud.client.Client.from_config` factory will override both file
            and environment values::

                client = Client.from_config(token='token-3', proxy='...')


            (6) For another example, create ``dwave.conf`` in your current directory or
            ``~/.config/dwave/dwave.conf``::

                [prod]
                endpoint = https://cloud.dwavesys.com/sapi
                token = DW-123123-secret
                solver = DW_2000Q_1

            Run::

                from dwave.cloud import Client
                with Client.from_config(profile='prod') as client:
                    solver = client.get_solver()
                    computation = solver.sample_ising({}, {})
                    samples = computation.result()

        Raises:
            :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
                Config file specified or detected could not be opened or read.

            :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
                Config file parse failed.
        """

        # try loading configuration from a preferred new config subsystem
        # (`./dwave.conf`, `~/.config/dwave/dwave.conf`, etc)
        config = load_config(
            config_file=config_file, profile=profile, client=client,
            endpoint=endpoint, token=token, solver=solver, proxy=proxy)

        # fallback to legacy `.dwrc` if key variables missing
        if legacy_config_fallback and (
                not config.get('token') or not config.get('endpoint')):
            config = legacy_load_config(
                profile=profile, client=client,
                endpoint=endpoint, token=token, solver=solver, proxy=proxy)

        # manual override of other (client-custom) arguments
        config.update(kwargs)

        from dwave.cloud import qpu, sw
        _clients = {'qpu': qpu.Client, 'sw': sw.Client}
        _client = config.pop('client', None) or 'qpu'
        return _clients[_client](**config)

    def __init__(self, endpoint=None, token=None, solver=None, proxy=None,
                 permissive_ssl=False, **kwargs):
        """To setup the connection a pipeline of queues/workers is constructed.

        There are five interactions with the server the connection manages:
        1. Downloading solver information.
        2. Submitting problem data.
        3. Polling problem status.
        4. Downloading problem results.
        5. Canceling problems

        Loading solver information is done synchronously. The other four tasks
        are performed by asynchronously workers. For 2, 3, and 5 the workers
        gather tasks in batches.
        """
        if not endpoint or not token:
            raise ValueError("Endpoint URL and/or token not defined")

        _LOGGER.debug("Creating a client for endpoint: %r", endpoint)

        self.endpoint = endpoint
        self.token = token
        self.default_solver = solver

        # Create a :mod:`requests` session. `requests` will manage our url parsing, https, etc.
        self.session = requests.Session()
        self.session.headers.update({'X-Auth-Token': self.token,
                                     'User-Agent': self.USER_AGENT})
        self.session.proxies = {'http': proxy, 'https': proxy}
        if permissive_ssl:
            self.session.verify = False

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

        # Prepare an empty set of solvers
        self._solvers = {}
        self._solvers_lock = threading.RLock()
        self._all_solvers_ready = False

        # Set the parameters for requests; disable SSL verification if needed
        self._request_parameters = {}
        if permissive_ssl:
            self._request_parameters['verify'] = False

    def close(self):
        """Perform a clean shutdown.

        Wait for all the currently scheduled work to finish, kill the workers,
        and close the connection pool. Assumes no one is submitting more work
        while the connection is closing.
        """
        # Finish all the work that requires the connection
        _LOGGER.debug("Joining submission queue")
        self._submission_queue.join()
        _LOGGER.debug("Joining cancel queue")
        self._cancel_queue.join()
        _LOGGER.debug("Joining poll queue")
        self._poll_queue.join()
        _LOGGER.debug("Joining load queue")
        self._load_queue.join()

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

        # Close the requests session
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
        """Predicate function that determines if the given solver should be
        handled by this client.

        Can be overridden in a subclass to specialize the client for a
        particular type of solvers.

        Default implementation accepts all solvers.
        """
        return True

    def get_solvers(self, refresh=False):
        """List all the solvers this client can provide, and load the data
        about the solvers.

        This is a blocking web call to `{endpoint}/solvers/remote/`` that
        caches the result and populates a list of available solvers described
        through :class:`.Solver` instances.

        To submit a sampling problem to the D-Wave API, filter the list returned
        and execute a ``sampling_*`` method on the solver of interest.
        Alternatively, if you know the solver name (or it's defined in config),
        use the :meth:`.get_solver` method.

        Args:
            refresh (bool, default=False):
                By default, ``get_solvers`` caches the list of solvers it
                receives from the API. Use this parameter to force refresh.

        Returns:
            dict[id, solver]: a mapping of solver name/id to :class:`.Solver`
        """
        with self._solvers_lock:
            if self._all_solvers_ready and not refresh:
                return self._solvers

            _LOGGER.debug("Requesting list of all solver data.")
            response = self.session.get(
                posixpath.join(self.endpoint, 'solvers/remote/'))

            if response.status_code == 401:
                raise SolverAuthenticationError
            response.raise_for_status()

            _LOGGER.debug("Received list of all solver data.")
            data = response.json()

            for solver_desc in data:
                try:
                    solver = Solver(self, solver_desc)
                    if self.is_solver_handled(solver):
                        self._solvers[solver.id] = solver
                        _LOGGER.debug("Adding solver %r", solver)
                    else:
                        _LOGGER.debug("Skipping solver %r inappropriate for client", solver)

                except UnsupportedSolverError as e:
                    _LOGGER.debug("Skipping solver due to %r", e)

            self._all_solvers_ready = True
            return self._solvers

    def get_solver(self, name=None, refresh=False):
        """Load the configuration for a single solver, as publicized by the API
        on ``{endpoint}/solvers/remote/{solver_name}/``.

        This is a blocking web call that returns a :class:`.Solver` instance,
        which in turn can be used to submit sampling problems to the D-Wave API
        and fetch the results.

        Args:
            name (str):
                Id of the requested solver. ``None`` will return the default solver.

            refresh (bool):
                Return solver from cache (if cached with ``get_solvers()``),
                unless set to ``True``.

        Returns:
            :class:`.Solver`
        """
        _LOGGER.debug("Looking for solver: %s", name)
        if name is None:
            if self.default_solver:
                name = self.default_solver
            else:
                raise ValueError("No name or default name provided when loading solver.")

        with self._solvers_lock:
            if refresh or name not in self._solvers:
                response = self.session.get(
                    posixpath.join(self.endpoint, 'solvers/remote/{}/'.format(name)))

                if response.status_code == 401:
                    raise SolverAuthenticationError

                if response.status_code == 404:
                    raise KeyError("No solver with the name {} was available".format(name))

                response.raise_for_status()

                solver = Solver(self, data=response.json())
                if solver.id != name:
                    raise InvalidAPIResponseError(
                        "Asked for solver named {!r}, got {!r}".format(name, solver.id))
                self._solvers[name] = solver

            return self._solvers[name]

    def _submit(self, body, future):
        """Enqueue a problem for submission to the server.

        This method is thread safe.
        """
        self._submission_queue.put(self._submit.Message(body, future))
    _submit.Message = collections.namedtuple('Message', ['body', 'future'])

    def _do_submit_problems(self):
        """Pull problems from the submission queue and submit them.

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            while True:
                # Pull as many problems as we can, block on the first one,
                # but once we have one problem, switch to non-blocking then
                # submit without blocking again.

                # `None` task is used to signal thread termination
                item = self._submission_queue.get()
                if item is None:
                    break

                ready_problems = [item]
                while True:
                    try:
                        ready_problems.append(self._submission_queue.get_nowait())
                    except queue.Empty:
                        break

                # Submit the problems
                _LOGGER.debug("Submitting %d problems", len(ready_problems))
                body = '[' + ','.join(mess.body for mess in ready_problems) + ']'
                try:
                    response = self.session.post(posixpath.join(self.endpoint, 'problems/'), body)

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    message = response.json()
                    _LOGGER.debug("Finished submitting %d problems", len(ready_problems))
                except BaseException as exception:
                    _LOGGER.debug("Submit failed for %d problems", len(ready_problems))
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    for mess in ready_problems:
                        mess.future._set_error(exception, sys.exc_info())
                        self._submission_queue.task_done()
                    continue

                # Pass on the information
                for submission, res in zip(ready_problems, message):
                    self._handle_problem_status(res, submission.future)
                    self._submission_queue.task_done()

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except BaseException as err:
            _LOGGER.exception(err)

    def _handle_problem_status(self, message, future):
        """Handle the results of a problem submission or results request.

        This method checks the status of the problem and puts it in the correct queue.

        Args:
            message (dict): Update message from the SAPI server wrt. this problem.
            future `Future`: future corresponding to the problem

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            status = message['status']
            _LOGGER.debug("Handling response for %s with status %s", message['id'], status)

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

            # Set the id field in the future
            future.id = message['id']
            future.remote_status = status

            if not future.time_received and message.get('submitted_on'):
                future.time_received = parse_datetime(message['submitted_on'])

            if not future.time_solved and message.get('solved_on'):
                future.time_solved = parse_datetime(message['solved_on'])

            if not future.eta_min and message.get('earliest_completion_time'):
                future.eta_min = parse_datetime(message['earliest_completion_time'])

            if not future.eta_max and message.get('latest_completion_time'):
                future.eta_max = parse_datetime(message['latest_completion_time'])

            if status == self.STATUS_COMPLETE:
                # TODO: find a better way to differentiate between
                # `completed-on-submit` and `completed-on-poll`.
                # Loading should happen only once, not every time when response
                # doesn't contain 'answer'.

                # If the message is complete, forward it to the future object
                if 'answer' in message:
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
                future._set_error(CanceledFutureError())
            else:
                # Return an error to the future object
                future._set_error(SolverFailureError(message.get('error_message', 'An unknown error has occurred.')))

        except Exception as error:
            # If there were any unhandled errors we need to release the
            # lock in the future, otherwise deadlock occurs.
            future._set_error(error, sys.exc_info())

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
                    body = [item[0] for item in item_list]
                    self.session.delete(posixpath.join(self.endpoint, 'problems/'), json=body)
                except Exception as err:
                    for _, future in item_list:
                        if future is not None:
                            future._set_error(err, sys.exc_info())

                # Mark all the ids as processed regardless of success or failure.
                [self._cancel_queue.task_done() for _ in item_list]

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except Exception as err:
            _LOGGER.exception(err)

    def _poll(self, future):
        """Enqueue a problem to poll the server for status."""

        if future._poll_backoff is None:
            # on first poll, start with minimal back-off
            future._poll_backoff = self._POLL_BACKOFF_MIN

            # if we have ETA of results, schedule the first poll for then
            if future.eta_min:
                at = datetime_to_timestamp(future.eta_min)
            else:
                at = time.time() + future._poll_backoff

        else:
            # update exponential poll back-off, clipped to a range
            future._poll_backoff = \
                max(self._POLL_BACKOFF_MIN,
                    min(future._poll_backoff * 2, self._POLL_BACKOFF_MAX))

            # for poll priority we use timestamp of next scheduled poll
            at = time.time() + future._poll_backoff

        _LOGGER.debug("Polling scheduled at %.2f with %.2f sec back-off for: %s",
                      at, future._poll_backoff, future.id)

        self._poll_queue.put((at, future))

    def _do_poll_problems(self):
        """Poll the server for the status of a set of problems.

        Note:
            This method is always run inside of a daemon thread.
        """
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
                _LOGGER.debug("Polling for status of futures: %s", ids)
                query_string = 'problems/?id=' + ','.join(ids)

                # if futures were cancelled while `add`ing, skip empty frame
                if not ids:
                    continue

                # wait until `frame_earliest` before polling
                delay = frame_earliest - time.time()
                if delay > 0:
                    _LOGGER.debug("Pausing polling %.2f sec for futures: %s", delay, ids)
                    time.sleep(delay)

                try:
                    response = self.session.get(posixpath.join(self.endpoint, query_string))

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    statuses = response.json()
                    for status in statuses:
                        self._handle_problem_status(status, frame_futures[status['id']])

                except BaseException as exception:
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    for id_ in frame_futures.keys():
                        frame_futures[id_]._set_error(IOError(exception), sys.exc_info())

                for id_ in frame_futures.keys():
                    task_done()

                time.sleep(0)

        except Exception as err:
            _LOGGER.exception(err)

    def _load(self, future):
        """Enqueue a problem to download results from the server.

        Args:
            future: Future` object corresponding to the query

        This method is threadsafe.
        """
        self._load_queue.put(future)

    def _do_load_results(self):
        """Submit a query asking for the results for a particular problem.

        To request the results of a problem: ``GET /problems/{problem_id}/``

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            while True:
                # Select a problem
                future = self._load_queue.get()
                # `None` task signifies thread termination
                if future is None:
                    break
                _LOGGER.debug("Loading results of: %s", future.id)

                # Submit the query
                query_string = 'problems/{}/'.format(future.id)
                try:
                    response = self.session.get(posixpath.join(self.endpoint, query_string))

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    message = response.json()
                except BaseException as exception:
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    future._set_error(IOError(exception), sys.exc_info())
                    continue

                # Dispatch the results, mark the task complete
                self._handle_problem_status(message, future)
                self._load_queue.task_done()

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except Exception as err:
            _LOGGER.error('Load result error: ' + str(err))

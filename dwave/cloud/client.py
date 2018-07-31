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

from __future__ import division, absolute_import

import sys
import time
import logging
import threading
import requests
import posixpath
import collections
from itertools import chain
from operator import attrgetter

from dateutil.parser import parse as parse_datetime
from six.moves import queue, range

from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.exceptions import *
from dwave.cloud.config import load_config, legacy_load_config, parse_float
from dwave.cloud.solver import Solver
from dwave.cloud.utils import datetime_to_timestamp, utcnow, TimeoutingHTTPAdapter

__all__ = ['Client']

_LOGGER = logging.getLogger(__name__)


class Client(object):
    """
    Base client class for all D-Wave API clients. Used by QPU and software :term:`sampler`
    classes.

    Manages workers and handles thread pools for submitting problems, cancelling tasks,
    polling problem status, and retrieving results.

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

    Other Parameters:
        Unrecognized keys (str):
            All unrecognized keys are passed through to the appropriate client class constructor
            as string keyword arguments.

            An explicit key value overrides an identical user-defined key value loaded from a
            configuration file.

    Examples:
        This example directly initializes a :class:`~dwave.cloud.client.Client`.
        Direct initialization uses class constructor arguments, the minimum being
        values for `endpoint` and `token`.

        >>> from dwave.cloud import Client
        >>> client = Client(endpoint='https://cloud.dwavesys.com/sapi', token='secret')
        >>> # code that uses client
        >>> client.close()


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

    # Number of problems to include in a submit/status query
    _SUBMIT_BATCH_SIZE = 20
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
        """Client factory method to instantiate a client instance from configuration.

        Configuration files comply with standard Windows INI-like format,
        parsable with Python's :mod:`configparser`. An optional ``defaults`` section
        provides default key-value pairs for all other sections. User-defined key-value
        pairs (unrecognized keys) are passed through to the client.

        Configuration values can be specified in multiple ways, ranked in the following
        order (with 1 the highest ranked):

        1. Values specified as keyword arguments in :func:`from_config()`
        2. Values specified as environment variables
        3. Values specified in the configuration file

        If the location of the configuration file is not specified, auto-detection
        searches for existing configuration files in the standard directories
        of :func:`get_configfile_paths`.

        If a configuration file explicitly specified, via an argument or
        environment variable, does not exist or is unreadable, loading fails with
        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`. Loading fails
        with :exc:`~dwave.cloud.exceptions.ConfigFileParseError` if the file is
        readable but invalid as a configuration file.

        Similarly, if a profile explicitly specified, via an argument or
        environment variable, is not present in the loaded configuration, loading fails
        with :exc:`ValueError`. Explicit profile selection also fails if the configuration
        file is not explicitly specified, detected on the system, or defined via
        an environment variable.

        Environment variables:

            ``DWAVE_CONFIG_FILE``:
                Configuration file path used if no configuration file is specified.

            ``DWAVE_PROFILE``:
                Name of profile (section) to use if no profile is specified.

            ``DWAVE_API_CLIENT``:
                API client class used if no client is specified. Supported values are
                ``qpu`` or ``sw``.

            ``DWAVE_API_ENDPOINT``:
                API endpoint URL used if no endpoint is specified.

            ``DWAVE_API_TOKEN``:
                API authorization token used if no token is specified.

            ``DWAVE_API_SOLVER``:
                Default solver used if no solver is specified.

            ``DWAVE_API_PROXY``:
                URL for proxy connections to D-Wave API used if no proxy is specified.

        Args:
            config_file (str/[str]/None/False/True, default=None):
                Path to configuration file.

                If ``None``, the value is taken from ``DWAVE_CONFIG_FILE`` environment
                variable if defined. If the environment variable is undefined or empty,
                auto-detection searches for existing configuration files in the standard
                directories of :func:`get_configfile_paths`.

                If ``False``, loading from file is skipped.

                If ``True``, forces auto-detection (regardless of the ``DWAVE_CONFIG_FILE``
                environment variable).

            profile (str, default=None):
                Profile name (name of the profile section in the configuration file).

                If undefined, inferred from ``DWAVE_PROFILE`` environment variable if
                defined. If the environment variable is undefined or empty, a profile is
                selected in the following order:

                1. From the default section if it includes a profile key.
                2. The first section (after the default section).
                3. If no other section is defined besides ``[defaults]``, the defaults
                   section is promoted and selected.

            client (str, default=None):
                Client type used for accessing the API. Supported values are ``qpu``
                for :class:`dwave.cloud.qpu.Client` and ``sw`` for
                :class:`dwave.cloud.sw.Client`.

            endpoint (str, default=None):
                API endpoint URL.

            token (str, default=None):
                API authorization token.

            solver (str, default=None):
                Default :term:`solver` to use in :meth:`~dwave.cloud.client.Client.get_solver`.
                If undefined, :meth:`~dwave.cloud.client.Client.get_solver` will return the
                first solver available.

            proxy (str, default=None):
                URL for proxy to use in connections to D-Wave API. Can include
                username/password, port, scheme, etc. If undefined, client
                uses the system-level proxy, if defined, or connects directly to the API.

            legacy_config_fallback (bool, default=True):
                If True (the default) and loading from a standard D-Wave Cloud Client configuration
                file (``dwave.conf``) fails, tries loading a legacy configuration file (``~/.dwrc``).

        Other Parameters:
            Unrecognized keys (str):
                All unrecognized keys are passed through to the appropriate client class constructor
                as string keyword arguments.

                An explicit key value overrides an identical user-defined key value loaded from a
                configuration file.

        Returns:
            :class:`~dwave.cloud.client.Client` (:class:`dwave.cloud.qpu.Client` or :class:`dwave.cloud.sw.Client`, default=:class:`dwave.cloud.qpu.Client`):
                Appropriate instance of a QPU or software client.

        Raises:
            :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
                Config file specified or detected could not be opened or read.

            :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
                Config file parse failed.

        Examples:
            This first example initializes :class:`~dwave.cloud.client.Client` from an
            explicitly specified configuration file, "~/jane/my_path_to_config/my_cloud_conf.conf"::

                [defaults]
                endpoint = https://url.of.some.dwavesystem.com/sapi
                client = qpu
                token = ABC-123456789123456789123456789

                [dw2000]
                solver = EXAMPLE_2000Q_SYSTEM
                token = DEF-987654321987654321987654321

            The example code below creates a client object that connects to a D-Wave QPU,
            using :class:`dwave.cloud.qpu.Client` and ``EXAMPLE_2000Q_SYSTEM`` as a default solver.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config(config_file='~/jane/my_path_to_config/my_cloud_conf.conf')  # doctest: +SKIP
            >>> # code that uses client
            >>> client.close()

            This second example auto-detects a configuration file on the local system following the
            user/system configuration paths of :func:`get_configfile_paths`. It passes through
            to the instantiated client an unrecognized key-value pair my_param=`my_value`.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config(my_param=`my_value`)
            >>> # code that uses client
            >>> client.close()

            This third example instantiates two clients, for managing both QPU and software
            solvers. Common key-value pairs are taken from the defaults section of a shared
            configuration file::

                [defaults]
                endpoint = https://url.of.some.dwavesystem.com/sapi
                client = qpu

                [dw2000A]
                solver = EXAMPLE_2000Q_SYSTEM_A
                token = ABC-123456789123456789123456789

                [sw_solver]
                client = sw
                solver = c4-sw_sample
                endpoint = https://url.of.some.software.resource.com/my_if
                token = DEF-987654321987654321987654321

                [dw2000B]
                solver = EXAMPLE_2000Q_SYSTEM_B
                proxy = http://user:pass@myproxy.com:8080/
                token = XYZ-0101010100112341234123412341234

            The example code below creates client objects for two QPU solvers (at the
            same URL but each with its own solver ID and token) and one software solver.

            >>> from dwave.cloud import Client
            >>> client_qpu1 = Client.from_config(profile='dw2000A')    # doctest: +SKIP
            >>> client_qpu1 = Client.from_config(profile='dw2000B')    # doctest: +SKIP
            >>> client_sw1 = Client.from_config(profile='sw_solver')   # doctest: +SKIP
            >>> client_qpu1.default_solver   # doctest: +SKIP
            u'EXAMPLE_2000Q_SYSTEM_A'
            >>> client_qpu2.endpoint   # doctest: +SKIP
            u'https://url.of.some.dwavesystem.com/sapi'
            >>> # code that uses client
            >>> client_qpu1.close() # doctest: +SKIP
            >>> client_qpu2.close() # doctest: +SKIP
            >>> client_sw1.close() # doctest: +SKIP

            This fourth example loads configurations auto-detected in more than one configuration
            file, with the higher priority file (in the current working directory) supplementing
            and overriding values from the lower priority user-local file. After instantiation,
            an endpoint from the default section and client from the profile section is provided
            from the user-local ``/usr/local/share/dwave/dwave.conf`` file::

                [defaults]
                endpoint = https://int.se.dwavesystems.com/sapi

                [dw2000]
                client = qpu
                token = ABC-123456789123456789123456789

            A solver is supplemented from the file in the current working directory, which also
            overrides the token value. ``./dwave.conf`` is the file in the current directory::

                [dw2000]
                solver = EXAMPLE_2000Q_SYSTEM_A
                token = DEF-987654321987654321987654321

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> client.default_solver   # doctest: +SKIP
            u'EXAMPLE_2000Q_SYSTEM_A'
            >>> client.endpoint  # doctest: +SKIP
            u'https://int.se.dwavesystems.com/sapi'
            >>> client.token  # doctest: +SKIP
            u'DEF-987654321987654321987654321'
            >>> # code that uses client
            >>> client.close() # doctest: +SKIP

        """

        # try loading configuration from a preferred new config subsystem
        # (`./dwave.conf`, `~/.config/dwave/dwave.conf`, etc)
        config = load_config(
            config_file=config_file, profile=profile, client=client,
            endpoint=endpoint, token=token, solver=solver, proxy=proxy)
        _LOGGER.debug("Config loaded: %r", config)

        # fallback to legacy `.dwrc` if key variables missing
        if legacy_config_fallback and (
                not config.get('token') or not config.get('endpoint')):
            config = legacy_load_config(
                profile=profile, client=client,
                endpoint=endpoint, token=token, solver=solver, proxy=proxy)
            _LOGGER.debug("Legacy config loaded: %r", config)

        # manual override of other (client-custom) arguments
        config.update(kwargs)

        from dwave.cloud import qpu, sw
        _clients = {'qpu': qpu.Client, 'sw': sw.Client, 'base': cls}
        _client = config.pop('client', None) or 'base'

        _LOGGER.debug("Final config used for %s.Client(): %r", _client, config)
        return _clients[_client](**config)

    def __init__(self, endpoint=None, token=None, solver=None, proxy=None,
                 permissive_ssl=False, request_timeout=60, polling_timeout=None, **kwargs):
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
        self.request_timeout = parse_float(request_timeout)
        self.polling_timeout = parse_float(polling_timeout)

        # Create a :mod:`requests` session. `requests` will manage our url parsing, https, etc.
        self.session = requests.Session()
        self.session.mount('http://', TimeoutingHTTPAdapter(timeout=self.request_timeout))
        self.session.mount('https://', TimeoutingHTTPAdapter(timeout=self.request_timeout))
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

        Waits for all the currently scheduled work to finish, kills the workers,
        and closes the connection pool.

        .. note:: Ensure your code does not submit new work while the connection is closing.

        Where possible, it is recommended you use a context manager (a :code:`with Client.from_config(...) as`
        construct) to ensure your code properly closes all resources.

        Examples:
            This example creates a client (based on an auto-detected configuration file), executes
            some code (represented by a placeholder comment), and then closes the client.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> # code that uses client
            >>> client.close()

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

    def get_solvers(self, refresh=False):
        """List all solvers this client can provide and load solvers' data.

        Makes a blocking web call to `{endpoint}/solvers/remote/``, where `{endpoint}`
        is a URL configured for the client, caches the result,
        and populates a list of available :term:`solver`s described through :class:`.Solver`
        instances.

        To submit a sampling problem to the D-Wave API, select a solver from the returned list,
        and execute a ``sampling_*`` method on it. Alternatively, use the :meth:`.get_solver` method
        if you know the solver ID (name), have it defined in your configuration file, or are just
        interested in fetching any/first solver.

        Args:
            refresh (bool, default=False):
                By default, ``get_solvers`` caches the list of solvers it
                receives from the API. Set to True to force a cache refresh.

        Returns:
            dict[id, solver]: Mapping of solver name/id to :class:`.Solver`

        Examples:
            This example lists all solvers available to a client instantiated from
            a local system's auto-detected default configuration file, which configures
            a connection to a D-Wave resource that provides two solvers.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> client.get_solvers()   # doctest: +SKIP
            {u'2000Q_ONLINE_SOLVER1': <dwave.cloud.solver.Solver at 0x7e84fd0>,
             u'2000Q_ONLINE_SOLVER2': <dwave.cloud.solver.Solver at 0x7e84828>}
            >>> # code that uses client
            >>> client.close() # doctest: +SKIP
        """

        with self._solvers_lock:
            if self._all_solvers_ready and not refresh:
                return self._solvers

            _LOGGER.debug("Requesting list of all solver data.")
            try:
                response = self.session.get(posixpath.join(self.endpoint, 'solvers/remote/'))
            except requests.exceptions.Timeout:
                raise RequestTimeout

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

    def solvers(self, qpu=None, software=None, vfyc=None,
                flux_biases=None, num_qubits=None, refresh=False):
        """Returns a filtered list of solvers handled by this client.

        Args:
            qpu (bool, default=None):
                Should solver be QPU based?

            software (bool, default=None):
                Should solver be software based?

            vfyc (bool, default=None):
                Should solver work on "virtual full-yield chip"?

            flux_biases (bool, default=None):
                Should solver accept flux biases?

            num_qubits (int/[int,int], default=None):
                What's the range (or exact number) of qubits solver should handle?

        Returns:
            list[Solver]: List of all solvers that satisfy the above conditions.

        Note:
            Client subclasses (e.g. :class:`dwave.cloud.qpu.Client` or
            :class:`dwave.cloud.sw.Client`) already filter solvers by resource
            type, so for ``qpu`` and ``software`` filters to have effect, you
            need to call :meth:`.solvers` on the base :class:`~dwave.cloud.client.Client`
            class.

        Examples:
            Get a solver not based on VFYC, with 2048 qubits:

                solver = client.solvers(vfyc=False, num_qubits=2048)[0]

            Get all solvers that have between 1024 and 2048 qubits:

                solvers = client.solvers(num_qubits=[1024, 2048])

            Get all solvers that have at least 2000 qubits:

                solvers = client.solvers(num_qubits=[2000, None])

            Get the first QPU solver that accepts flux biases:

                solver = client.solvers(qpu=True, flux_biases=True)[0]
        """

        def predicate(solver):
            if qpu is not None and solver.is_qpu != qpu:
                return False
            if software is not None and solver.is_software != software:
                return False
            if vfyc is not None and solver.is_vfyc != vfyc:
                return False
            if flux_biases is not None and solver.has_flux_biases != flux_biases:
                return False
            if num_qubits is not None:
                if isinstance(num_qubits, list) and len(num_qubits) == 2:
                    min_qubits, max_qubits = num_qubits
                else:
                    min_qubits, max_qubits = num_qubits, num_qubits
                if min_qubits is not None and solver.num_qubits < min_qubits:
                    return False
                if max_qubits is not None and max_qubits < solver.num_qubits:
                    return False
            return True

        solvers = list(filter(predicate, self.get_solvers(refresh=refresh).values()))
        solvers.sort(key=attrgetter('id'))
        return solvers

    def get_solver(self, name=None, refresh=False):
        """Load the configuration for a single solver.

        Makes a blocking web call to `{endpoint}/solvers/remote/{solver_name}/`, where `{endpoint}`
        is a URL configured for the client, and returns a :class:`.Solver` instance
        that can be used to submit sampling problems to the D-Wave API and retrieve results.

        Args:
            name (str):
                ID of the requested solver. ``None`` returns the default solver.
                If default solver is not configured, ``None`` returns the first available
                solver in ``Client``'s class (QPU/software/base).

            refresh (bool):
                Return solver from cache (if cached with ``get_solvers()``),
                unless set to ``True``.

        Returns:
            :class:`.Solver`

        Examples:
            This example creates two solvers for a client instantiated from
            a local system's auto-detected default configuration file, which configures
            a connection to a D-Wave resource that provides two solvers. The first
            uses the default solver, the second explicitly selects another solver.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> client.get_solvers()   # doctest: +SKIP
            {u'2000Q_ONLINE_SOLVER1': <dwave.cloud.solver.Solver at 0x7e84fd0>,
             u'2000Q_ONLINE_SOLVER2': <dwave.cloud.solver.Solver at 0x7e84828>}
            >>> solver1 = client.get_solver()    # doctest: +SKIP
            >>> solver2 = client.get_solver('2000Q_ONLINE_SOLVER2')    # doctest: +SKIP
            >>> solver1.id  # doctest: +SKIP
            u'2000Q_ONLINE_SOLVER1'
            >>> solver2.id   # doctest: +SKIP
            u'2000Q_ONLINE_SOLVER2'
            >>> # code that uses client
            >>> client.close() # doctest: +SKIP

        """
        _LOGGER.debug("Looking for solver: %s", name)
        if name is None:
            if self.default_solver:
                name = self.default_solver
            else:
                # get the first appropriate solver
                try:
                    return self.solvers()[0]
                except IndexError:
                    raise SolverError("No solvers available this client can handle")

        with self._solvers_lock:
            if refresh or name not in self._solvers:
                try:
                    response = self.session.get(
                        posixpath.join(self.endpoint, 'solvers/remote/{}/'.format(name)))
                except requests.exceptions.Timeout:
                    raise RequestTimeout

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
                while len(ready_problems) < self._SUBMIT_BATCH_SIZE:
                    try:
                        ready_problems.append(self._submission_queue.get_nowait())
                    except queue.Empty:
                        break

                # Submit the problems
                _LOGGER.debug("Submitting %d problems", len(ready_problems))
                body = '[' + ','.join(mess.body for mess in ready_problems) + ']'
                try:
                    try:
                        response = self.session.post(posixpath.join(self.endpoint, 'problems/'), body)
                    except requests.exceptions.Timeout:
                        raise RequestTimeout

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
            _LOGGER.trace("Handling response: %r", message)

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

            if not future.eta_min and message.get('earliest_estimated_completion'):
                future.eta_min = parse_datetime(message['earliest_estimated_completion'])

            if not future.eta_max and message.get('latest_estimated_completion'):
                future.eta_max = parse_datetime(message['latest_estimated_completion'])

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

                    try:
                        self.session.delete(posixpath.join(self.endpoint, 'problems/'), json=body)
                    except requests.exceptions.Timeout:
                        raise RequestTimeout

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

        now = utcnow()
        future_age = (now - future.time_created).total_seconds()
        _LOGGER.debug("Polling scheduled at %.2f with %.2f sec new back-off for: %s (age: %.2f sec)",
                      at, future._poll_backoff, future.id, future_age)

        # don't enqueue for next poll if polling_timeout is exceeded by then
        future_age_on_next_poll = future_age + (at - datetime_to_timestamp(now))
        if self.polling_timeout is not None and future_age_on_next_poll > self.polling_timeout:
            _LOGGER.debug("Polling timeout exceeded before next poll: %.2f sec > %.2f sec, aborting polling!",
                          future_age_on_next_poll, self.polling_timeout)
            raise PollingTimeout

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
                else:
                    _LOGGER.trace("Skipping non-positive delay of %.2f sec", delay)

                try:
                    _LOGGER.trace("Executing poll API request")

                    try:
                        response = self.session.get(posixpath.join(self.endpoint, query_string))
                    except requests.exceptions.Timeout:
                        raise RequestTimeout

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
                    try:
                        response = self.session.get(posixpath.join(self.endpoint, query_string))
                    except requests.exceptions.Timeout:
                        raise RequestTimeout

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

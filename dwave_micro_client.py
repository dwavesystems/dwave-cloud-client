"""
An implementation of the REST API exposed by D-Wave Solver API (SAPI) servers.

This API lets you submit an Ising model and receive samples from a distribution over the model
as defined by the solver you have selected.

 - The SAPI servers provide authentication, queuing, and scheduling services, and
   provide a network interface to the solvers.
 - A solver is a resource that can sample from a discrete quadratic model.
 - This package implements the REST interface these servers provide.

An example using the client:

.. code-block:: python
    :linenos:

    import dwave_micro_client
    import random

    # Connect using explicit connection information
    conn = dwave_micro_client.Connection('https://sapi-url', 'token-string')

    # Load a solver by name
    solver = conn.get_solver('test-solver')

    # Build a random Ising model on +1, -1. Build it to exactly fit the graph the solver provides
    linear = {index: random.choice([-1, 1]) for index in solver.nodes}
    quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

    # Send the problem for sampling, include a solver specific parameter 'num_reads'
    results = solver.sample_ising(linear, quad, num_reads=100)

    # Print out the first sample
    print(results.samples[0])

Rough workflow within the SAPI server:
 1. Submitted problems enter an input queue. Each user has an input queue per solver.
 2. Drawing from all input queues for a solver, problems are scheduled.
 3. Results of the server are cached for retrieval by the client.

By default all sampling requests will be processed asynchronously. Reading results from
any future object is a blocking operation.

.. code-block:: python
    :linenos:

    # We can submit several sample requests without blocking
    # (In this specific case we could accomplish the same thing by increasing 'num_reads')
    futures = [solver.sample_ising(linear, quad, num_reads=100) for _ in range(10)]

    # We can check if a set of samples are ready without blocking
    print(futures[0].done())

    # We can wait on a single future
    futures[0].wait()

    # Or we can wait on several futures
    dwave_micro_client.Future.wait_multiple(futures)

"""

# TODOS:
#  - More testing for sample_qubo

from __future__ import division, absolute_import

import json
import threading
import base64
import struct
import time
import sys
import os
import types
import logging
import requests
import collections
import datetime

import six
import six.moves.queue as queue
import six.moves
range = six.moves.range

# Get the logger using the recommended name
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
# log.addHandler(logging.StreamHandler(sys.stdout))

# Use numpy if available for fast decoding
try:
    import numpy as np
    _numpy = True
except ImportError:
    # If numpy isn't available we can do the encoding slower in native python
    _numpy = False


class SolverFailureError(Exception):
    """An exception raised when there is a remote failure calling a solver."""
    pass


class SolverAuthenticationError(Exception):
    """An exception raised when there is an authentication error."""

    def __init__(self):
        super(SolverAuthenticationError, self).__init__("Token not accepted for that action.")


class CanceledFutureError(Exception):
    """An exception raised when code tries to read from a canceled future."""

    def __init__(self):
        super(CanceledFutureError, self).__init__("An error occured reading results from a canceled request")


class Connection:
    """
    Connect to a SAPI server to expose the solvers that the server advertises.

    Args:
        url (str): URL of the SAPI server.
        token (str): Authentication token from the SAPI server.
        proxies (dict): Mapping from the connection scheme (http[s]) to the proxy server address.
        permissive_ssl (boolean; false by default): Disables SSL verification.
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

    # Number of problems to include in a status query
    _STATUS_QUERY_SIZE = 100

    # Number of worker threads for each problem processing task
    _SUBMISSION_THREAD_COUNT = 5
    _CANCEL_THREAD_COUNT = 1
    _POLL_THREAD_COUNT = 2
    _LOAD_THREAD_COUNT = 5

    def __init__(self, url=None, token=None, proxies=None, permissive_ssl=False):
        """To setup the connection a pipeline of queues/workers is costructed.

        There are five interations with the server the connection manages:
        1. Downloading solver information.
        2. Submitting problem data.
        3. Polling problem status.
        4. Downloading problem results.
        5. Canceling problems

        Loading solver information is done syncronously. The other four tasks are
        performed by asyncronous workers. For 2, 3, and 5 the workers gather
        togeather tasks into in batches.
        """
        # Use configuration from parameters passed, if parts are
        # missing, try the configuration function
        self.default_solver = None
        if token is None:
            url, token, proxies, self.default_solver = load_configuration(url)
        log.debug("Creating a connection to SAPI server: %s", url)

        self.base_url = url
        self.token = token

        # Create a :mod:`requests` session. `requests` will manage our url parsing, https, etc.
        self.session = requests.Session()
        self.session.headers.update({'X-Auth-Token': self.token})
        self.session.proxies = proxies
        if permissive_ssl:
            self.session.verify = False

        # Build the problem submission queue, start its workers
        self._submission_queue = queue.Queue()
        self._submission_workers = []
        for _ in range(self._SUBMISSION_THREAD_COUNT):
            worker = threading.Thread(target=self._do_submit_problems)
            worker.daemon = True
            worker.start()

        # Build the cancel problem queue, start its workers
        self._cancel_queue = queue.Queue()
        self._cancel_workers = []
        for _ in range(self._CANCEL_THREAD_COUNT):
            worker = threading.Thread(target=self._do_cancel_problems)
            worker.daemon = True
            worker.start()

        # Build the problem status polling queue, start its workers
        self._poll_queue = queue.Queue()
        self._poll_workers = []
        for _ in range(self._POLL_THREAD_COUNT):
            worker = threading.Thread(target=self._do_poll_problems)
            worker.daemon = True
            worker.start()

        # Build the result loading queue, start its workers
        self._load_queue = queue.Queue()
        self._load_workers = []
        for _ in range(self._LOAD_THREAD_COUNT):
            worker = threading.Thread(target=self._do_load_results)
            worker.daemon = True
            worker.start()

        # Prepare an empty set of solvers
        self.solvers = {}
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
        log.debug("Joining submission queue")
        self._submission_queue.join()
        log.debug("Joining cancel queue")
        self._cancel_queue.join()
        log.debug("Joining poll queue")
        self._poll_queue.join()
        log.debug("Joining load queue")
        self._load_queue.join()

        # Kill off the worker threads, (which should now be blocking on the empty)
        [worker.kill() for worker in self._submission_workers]
        [worker.kill() for worker in self._cancel_workers]
        [worker.kill() for worker in self._poll_workers]
        [worker.kill() for worker in self._load_workers]

        # Close the connection pool
        self.session.close()

    def __enter__(self):
        """Let connections be used in with blocks."""
        return self

    def __exit__(self, *args):
        """At the end of a with block perform a clean shutdown of the connection."""
        self.close()
        return False

    def solver_names(self):
        """List all the solvers this connection can provide, and load the data about the solvers.

        To get all solver data: ``GET /solvers/remote/``

        Returns:
            list of str
        """
        with self._solvers_lock:
            if self._all_solvers_ready:
                return self.solvers.keys()

            log.debug("Requesting list of all solver data.")
            response = self.session.get(os.path.join(self.base_url, 'solvers/remote/'))

            if response.status_code == 401:
                raise SolverAuthenticationError()
            response.raise_for_status()

            log.debug("Received list of all solver data.")

            data = response.json()

            for solver in data:
                log.debug("Found solver: %s", solver['id'])
                self.solvers[solver['id']] = Solver(self, solver)
            self._all_solvers_ready = True
            return self.solvers.keys()

    def get_solver(self, name=None):
        """Load the configuration for a single solver.

        To get specific solver data: ``GET /solvers/remote/{solver_name}/``

        Args:
            name (str): Id of the requested solver. None will return the default solver.
        Returns:
            :obj:`Solver`
        """
        log.debug("Looking for solver: %s", name)
        if name is None:
            if self.default_solver is not None:
                name = self.default_solver
            else:
                raise ValueError("No name or default name provided when loading solver.")

        with self._solvers_lock:
            if name not in self.solvers:
                if self._all_solvers_ready:
                    raise KeyError(name)

                response = self.session.get(os.path.join(self.base_url, 'solvers/remote/{}/'.format(name)))

                if response.status_code == 401:
                    raise SolverAuthenticationError()

                if response.status_code == 404:
                    raise KeyError("No solver with the name {} was available".format(name))

                response.raise_for_status()

                data = json.loads(response.text)
                self.solvers[data['id']] = Solver(self, data)

            return self.solvers[name]

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
                ready_problems = [self._submission_queue.get()]
                while True:
                    try:
                        ready_problems.append(self._submission_queue.get_nowait())
                    except queue.Empty:
                        break

                # Submit the problems
                log.debug("submitting {} problems".format(len(ready_problems)))
                body = '[' + ','.join(mess.body for mess in ready_problems) + ']'
                try:
                    response = self.session.post(os.path.join(self.base_url, 'problems/'), body)

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    message = response.json()
                    log.debug("Finished submitting {} problems".format(len(ready_problems)))
                except BaseException as exception:
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    for mess in ready_problems:
                        mess.future._set_error(exception, sys.exc_info())
                        self._submission_queue.task_done()
                    continue

                # Pass on the information
                for submission, res in zip(ready_problems, message):
                    self._handle_problem_status(res, submission.future, False)
                    self._submission_queue.task_done()

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except BaseException as err:
            log.exception(err)

    def _handle_problem_status(self, message, future, in_poll):
        """Handle the results of a problem submission or results request.

        This method checks the status of the problem and puts it in the correct queue.

        Args:
            message (dict): Update message from the SAPI server wrt. this problem.
            future `Future`: future corresponding to the problem
            in_poll (bool): Flag set to true if the problem is in the poll loop already.

        Returns:
            true if the problem has been processed out of the status poll loop

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            status = message['status']
            log.debug("Status: %s %s", message['id'], status)

            # The future may not have the ID set yet
            with future._single_cancel_lock:
                # This handles the case where cancel has been called on a future
                # before that future recived the problem id
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

            if future.time_received is not None and 'submitted_on' in message and message['submitted_on'] is not None:
                future.time_received = datetime.strptime(message['submitted_on'])

            if future.time_solved is not None and 'solved_on' in message and message['solved_on'] is not None:
                future.time_solved = datetime.strptime(message['solved_on'])

            if status == self.STATUS_COMPLETE:
                # If the message is complete, forward it to the future object
                if 'answer' in message:
                    future._set_message(message)
                # If the problem is complete, but we don't have the result data
                # put the problem in the queue for loading results.
                else:
                    self._load(future)
            elif status in self.ANY_STATUS_ONGOING:
                # If the response is pending add it to the queue.
                if not in_poll:
                    self._poll(future)
                return False
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
        return True

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
                # Pull as many problems as we can, block when none are avaialble.
                item_list = [self._cancel_queue.get()]
                while True:
                    try:
                        item_list.append(self._cancel_queue.get_nowait())
                    except queue.Empty:
                        break

                # Submit the problems, attach the ids as a json list in the
                # body of the delete query.
                try:
                    body = [item[0] for item in item_list]
                    self.session.delete(os.path.join(self.base_url, 'problems/'), json=body)
                except Exception as err:
                    for _, future in item_list:
                        if future is not None:
                            future._set_error(err, sys.exc_info())

                # Mark all the ids as processed regardless of success or failure.
                [self._cancel_queue.task_done() for _ in item_list]

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except Exception as err:
            log.exception(err)

    def _poll(self, future):
        """Enqueue a problem to poll the server for status.

        This method is threadsafe.
        """
        self._poll_queue.put(future)

    def _do_poll_problems(self):
        """Poll the server for the status of a set of problems.

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            # Maintain an active group of queries
            futures = {}
            active_queries = set()

            # Add a query to the active queries
            def add(ftr):
                if ftr.id not in futures and not ftr.done():
                    active_queries.add(ftr.id)
                    futures[ftr.id] = ftr
                else:
                    self._poll_queue.task_done()

            # Remve a query from the active set
            def remove(id_):
                del futures[id_]
                active_queries.remove(id_)
                self._poll_queue.task_done()

            while True:
                try:
                    # If we have no active queries, wait on the status queue
                    while len(active_queries) == 0:
                        add(self._poll_queue.get())
                    # Once there is any active queries try to fill up the set and move on
                    while len(active_queries) < self._STATUS_QUERY_SIZE:
                        add(self._poll_queue.get_nowait())
                except queue.Empty:
                    pass

                # Build a query string with block of ids
                log.debug("Query on futures: %s", ', '.join(active_queries))
                query_string = 'problems/?id=' + ','.join(active_queries)

                try:
                    response = self.session.get(os.path.join(self.base_url, query_string))

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    message = response.json()
                except BaseException as exception:
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    for id_ in list(active_queries):
                        futures[id_]._set_error(IOError(exception), sys.exc_info())
                        remove(id_)
                    continue

                # If problems are removed from the polling by _handle_problem_status
                # remove them from the active set
                for single_message in message:
                    if self._handle_problem_status(single_message, futures[single_message['id']], True):
                        remove(single_message['id'])

                # Remove the finished queries
                for id_ in list(active_queries):
                    if futures[id_].done():
                        remove(id_)

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except Exception as err:
            log.exception(err)

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
                log.debug("Query for results: %s", future.id)

                # Submit the query
                query_string = 'problems/{}/'.format(future.id)
                try:
                    response = self.session.get(os.path.join(self.base_url, query_string))

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
                self._handle_problem_status(message, future, False)
                self._load_queue.task_done()

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except Exception as err:
            log.error('Load result error: ' + str(err))


class Solver:
    """
    A solver enables sampling from an Ising model.

    Get solver objects by calling get_solver(name) on a connection object.

    The solver has responsibilty for:
    - Encoding problems submitted
    - Checking the submitted parameters
    - Add problems to the Connection's submission queue

    Args:
        connection (`Connection`): Connection through which the solver is accessed.
        data: Data from the server describing this solver.
    """

    # Special flag to notify the system a solver needs access to special hardware
    _PARAMETER_ENABLE_HARDWARE = 'use_hardware'

    def __init__(self, connection, data):
        self.connection = connection
        self.id = data['id']
        self.data = data

        #: When True the solution data will be returned as numpy matrices: False
        self.return_matrix = False

        # The exact sequence of nodes/edges is used in encoding problems and must be preserved
        self._encoding_qubits = data['properties']['qubits']
        self._encoding_couplers = [tuple(edge) for edge in data['properties']['couplers']]

        #: The nodes in this solver's graph: set(int)
        self.nodes = self.variables = set(self._encoding_qubits)

        #: The edges in this solver's graph, every edge will be present as (a, b) and (b, a): set(tuple(int, int))
        self.edges = self.couplers = set(tuple(edge) for edge in self._encoding_couplers) | \
            set((edge[1], edge[0]) for edge in self._encoding_couplers)

        #: The edges in this solver's graph, each edge will only be represented once: set(tuple(int, int))
        self.undirected_edges = {edge for edge in self.edges if edge[0] < edge[1]}

        #: Properties of this solver the server presents: dict
        self.properties = data['properties']

        #: The set of extra parameters this solver will accept in sample_ising or sample_qubo: dict
        self.parameters = self.properties['parameters']

        # Create a set of default parameters for the queries
        self._params = {}

        # As a heuristic to guess if this is a hardware sampler check if
        # the 'annealing_time_range' property is set.
        if 'annealing_time_range' in data['properties']:
            self._params[self._PARAMETER_ENABLE_HARDWARE] = True

    def sample_ising(self, linear, quadratic, **params):
        """Draw samples from the provided Ising model.

        To submit a problem: ``POST /problems/``

        Args:
            linear (list/dict): Linear terms of the model (h).
            quadratic (dict of (int, int):float): Quadratic terms of the model (J).
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj:`Future`
        """
        # Our linear and quadratic objective terms are already separated in an
        # ising model so we can just directly call `_sample`.
        return self._sample('ising', linear, quadratic, params)

    def sample_qubo(self, qubo, **params):
        """Draw samples from the provided QUBO.

        To submit a problem: ``POST /problems/``

        Args:
            qubo (dict of (int, int):float): Terms of the model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj:`Future`
        """
        # In a QUBO the linear and quadratic terms in the objective are mixed into
        # a matrix. For the sake of encoding, we will separate them before calling `_sample`
        linear = {i1: v for (i1, i2), v in _uniform_iterator(qubo) if i1 == i2}
        quadratic = {(i1, i2): v for (i1, i2), v in _uniform_iterator(qubo) if i1 != i2}
        return self._sample('qubo', linear, quadratic, params)

    def _sample(self, type_, linear, quadratic, params, reuse_future=None):
        """Internal method for both sample_ising and sample_qubo.

        Args:
            linear (list/dict): Linear terms of the model.
            quadratic (dict of (int, int):float): Quadratic terms of the model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj: `Future`
        """
        # Check the problem
        if not self.check_problem(linear, quadratic):
            raise ValueError("Problem graph incompatible with solver.")

        # Mix the new parameters with the default parameters
        combined_params = dict(self._params)
        combined_params.update(params)

        # Check the parameters before submitting
        for key in combined_params:
            if key not in self.parameters and key != self._PARAMETER_ENABLE_HARDWARE:
                raise KeyError("{} is not a parameter of this solver.".format(key))

        # Encode the problem, use the newer format
        data = self._base64_format(self, linear, quadratic)
        # data = self._text_format(solver, lin, quad)

        body = json.dumps({
            'solver': self.id,
            'data': data,
            'type': type_,
            'params': params
        })

        # Construct where we will put the result when we finish, submit the query
        if reuse_future is not None:
            future = reuse_future
            future.__init__(self, None, self.return_matrix, (type_, linear, quadratic, params))
        else:
            future = Future(self, None, self.return_matrix, (type_, linear, quadratic, params))

        log.debug("Submitting new problem to: %s", self.id)
        self.connection._submit(body, future)
        return future

    def check_problem(self, linear, quadratic):
        """Test if an Ising model matches the graph provided by the solver.

        Args:
            linear (list/dict): Linear terms of the model (h).
            quadratic (dict of (int, int):float): Quadratic terms of the model (J).

        Returns:
            boolean
        """
        for key, value in _uniform_iterator(linear):
            if value != 0 and key not in self.nodes:
                return False
        for key, value in _uniform_iterator(quadratic):
            if value != 0 and tuple(key) not in self.edges:
                return False
        return True

    def retrieve_problem(self, id_):
        """Resume polling for a problem previously submitted.

        Args:
            id_: Identification of the query.

        Returns:
            :obj: `Future`
        """
        future = Future(self, id_, self.return_matrix, None)
        self.connection._poll(future)
        return future

    def _text_format(self, solver, lin, quad):
        """Perform the legacy problem encoding.

        Deprecated encoding method; included only for reference.

        Args:
            solver: solver requested.
            lin: linear terms of the model.
            quad: Quadratic terms of the model.

        Returns:
            data: text formatted problem

        """
        data = ''
        counter = 0
        for index, value in _uniform_iterator(lin):
            if value != 0:
                data = data + '{} {} {}\n'.format(index, index, value)
                counter += 1

        for (index1, index2), value in six.iteritems(quad):
            if value != 0:
                data = data + '{} {} {}\n'.format(index1, index2, value)
                counter += 1

        data = '{} {}\n'.format(max(solver.nodes) + 1, counter) + data
        return data

    def _base64_format(self, solver, lin, quad):
        """Encode the problem for submission to a given solver.

        Args:
            solver: solver requested.
            lin: linear terms of the model.
            quad: Quadratic terms of the model.

        Returns:
            encoded submission dictionary

        """
        # Encode linear terms. The coefficients of the linear terms of the objective
        # are encoded as an array of little endian 64 bit doubles.
        # This array is then base64 encoded into a string safe for json.
        # The order of the terms is determined by the _encoding_qubits property
        # specified by the server.
        lin = [_uniform_get(lin, qubit, 0) for qubit in solver._encoding_qubits]
        lin = base64.b64encode(struct.pack('<' + ('d' * len(lin)), *lin))

        # Encode the coefficients of the quadratic terms of the objective
        # in the same manner as the linear terms, in the order given by the
        # _encoding_couplers property
        quad = [quad.get(edge, 0) + quad.get((edge[1], edge[0]), 0)
                for edge in solver._encoding_couplers]
        quad = base64.b64encode(struct.pack('<' + ('d' * len(quad)), *quad))

        # The name for this encoding is 'qp' and is explicitly included in the
        # message for easier extension in the future.
        return {
            'format': 'qp',
            'lin': lin.decode('utf-8'),
            'quad': quad.decode('utf-8')
        }


class Future:
    """An object for a pending SAPI call.

    Waits for a request to complete and parses the message returned.
    The future will be block to resolve when any data value is accessed.
    The method :meth:`done` can be used to query for resolution without blocking.
    :meth:`wait`, and :meth:`wait_multiple` can be used to block for a variable
    number of jobs for a given ammount of time.

    Note:
        Only constructed by :obj:`Solver` objects.

    Args:
        solver: The solver that is going to fulfil this future.
        id_: Identification of the query we are waiting for. (May be None and filled in later.)
        return_matrix: Request return values as numpy matrices.
    """

    def __init__(self, solver, id_, return_matrix, submission_data):
        self.solver = solver

        # Store the query data in case the problem needs to be resubmitted
        self._submission_data = submission_data

        # Has the client tried to cancel this job
        self._cancel_requested = False
        self._cancel_sent = False
        self._single_cancel_lock = threading.Lock()  # Make sure we only call cancel once

        # Should the results be decoded as python lists or numpy matrices
        if return_matrix and not _numpy:
            raise ValueError("Matrix result requested without numpy.")
        self.return_matrix = return_matrix

        #: The id the server will use to identify this problem, None until the id is actually known
        self.id = id_

        #: `datetime` corriesponding to the time when the problem was accepted by the server (None before then)
        self.time_received = None

        #: `datetime` corriesponding to the time when the problem was completed by the server (None before then)
        self.time_solved = None

        #: `datetime` corriesponding to the time when the problem was completed by the server (None before then)
        self.time_solved = None

        # Track how long it took us to parse the data
        self.parse_time = None

        # Data from the server before it is parsed
        self._message = None

        #: Status flag most recently returned by the server
        self.remote_status = None

        # Data from the server after it is parsed (either data or an error)
        self._result = None
        self.error = None

        # Event(s) to signal when the results are ready
        self._results_ready_event = threading.Event()
        self._other_events = []

    def _set_message(self, message):
        """Complete the future with a message from the server.

        The message from the server may actually be an error.

        Args:
            message (dict): Data from the server from trying to complete query.
        """
        self._message = message
        self._signal_ready()

    def _set_error(self, error, exc_info=None):
        """Complete the future with an error.

        Args:
            error: An error string or exception object.
            exc_info: Stack trace info from sys module for reraising exceptions nicely.
        """
        self.error = error
        self._exc_info = exc_info
        self._signal_ready()

    def _signal_ready(self):
        """Signal all the events waiting on this future."""
        self._results_ready_event.set()
        [ev.set() for ev in self._other_events]

    def _add_event(self, event):
        """Add an event to be signaled after this event completes."""
        self._other_events.append(event)
        if self.done():
            event.set()

    def _remove_event(self, event):
        """Remove a completion event from this future."""
        self._other_events.remove(event)

    @staticmethod
    def wait_multiple(futures, min_done=None, timeout=float('inf')):
        """Wait for multiple Future objects to complete.

        Python doesn't provide a multi-wait, but we can jury rig something reasonably
        efficent using an event object.

        Args:
            futures (list of Future): list of objects to wait on
            min_done (int): Stop waiting when this many results are ready
            timeout (float): Maximum number of seconds to wait

        Returns:
            boolean: True if the minimum number of results have been reached.
        """
        if min_done is None:
            min_done = len(futures)

        # Track the exit conditions
        finish = time.time() + timeout
        done = 0

        # Keep track of what futures havn't finished
        remaining = list(futures)

        # Insert our event into all the futures
        event = threading.Event()
        [f._add_event(event) for f in remaining]

        # Check the exit conditions
        while done < min_done and finish > time.time():
            # Prepare to wait on any of the jobs finishing
            event.clear()

            # Check if any of the jobs have finished. After the clear just in
            # case one finished and we erased the signal it by calling clear above
            finished_futures = {f for f in remaining if f.done()}
            if len(finished_futures) > 0:
                # If we did make a mistake reseting the event, undo that now
                # so that we double check the finished list before a wait blocks
                event.set()

                # Update our exit conditions
                done += len(finished_futures)
                remaining = [f for f in remaining if f not in finished_futures]
                continue

            # Block on any of the jobs finishing
            wait_time = finish - time.time() if abs(finish) != float('inf') else None
            event.wait(wait_time)

        # Clean up after ourselves
        [f._remove_event(event) for f in futures]
        return done >= min_done

    def wait(self, timeout=None):
        """Wait for the results to be available.

        Args:
            timeout (float): Maximum number of seconds to wait
        """
        return self._results_ready_event.wait(timeout)

    def done(self):
        """Test whether a response has arrived."""
        return self._message is not None or self.error is not None

    def cancel(self):
        """Try to cancel the problem corresponding to this result.

        An effort will be made to prevent the execution of the corresponding problem
        but there are no guarantees.
        """
        # Don't need to cancel something already finished
        if self.done():
            return

        with self._single_cancel_lock:
            # Already done
            if self._cancel_requested:
                return

            # Set the cancel flag
            self._cancel_requested = True

            # The cancel request will be sent here, or by the solver when it
            # gets a status update for this problem (in the case where the id hasn't been set yet)
            if self.id is not None and not self._cancel_sent:
                self._cancel_sent = True
                self.solver.connection._cancel(self.id, self)

    @property
    def energies(self):
        """The energy buffer, blocks if needed.

        Returns:
            list or numpy matrix of doubles.
        """
        result = self._load_result()
        return result['energies']

    @property
    def samples(self):
        """The state buffer, blocks if needed.

        Returns:
            list of lists or numpy matrix.
        """
        result = self._load_result()
        return result['solutions']

    @property
    def occurrences(self):
        """The occurrences buffer, blocks if needed.

        Returns:
            list or numpy matrix of doubles.
        """
        result = self._load_result()
        if 'num_occurrences' in result:
            return result['num_occurrences']
        elif self.return_matrix:
            return np.ones((len(result['solutions']),))
        else:
            return [1] * len(result['solutions'])

    @property
    def timing(self):
        """Information about the time the solver took in operation.

        The response is a mapping from string keys to numeric values.
        The exact keys used depend on the solver.

        Returns:
            dict
        """
        result = self._load_result()
        return result['timing']

    def __getitem__(self, key):
        """Provide dwave_sapi2 compatible access to results.

        Args:
            key: keywords for result fields.
        """
        if key == 'energies':
            return self.energies
        elif key in ['solutions', 'samples']:
            return self.samples
        elif key in ['occurrences', 'num_occurrences']:
            return self.occurrences
        elif key == 'timing':
            return self.timing
        else:
            raise KeyError('{} is not a property of response object'.format(key))

    def _load_result(self):
        """Get the result, waiting and decoding as needed."""
        if self._result is None:
            # Wait for the query response
            self._results_ready_event.wait()

            # Check for other error conditions
            if self.error is not None:
                if self._exc_info is not None:
                    six.reraise(*self._exc_info)
                if isinstance(self.error, Exception):
                    raise self.error
                raise RuntimeError(self.error)

            # If someone else took care of this while we were waiting
            if self._result is not None:
                return self._result
            self._decode()

        return self._result

    def _decode(self):
        """Choose the right decoding method based on format and environment."""
        start = time.time()
        try:
            if self._message['type'] not in ['qubo', 'ising']:
                raise ValueError('Unknown problem format used.')

            # If no format is set we fall back to legacy encoding
            if 'format' not in self._message['answer']:
                if _numpy:
                    return self._decode_legacy_numpy()
                return self._decode_legacy()

            # If format is set, it must be qp
            if self._message['answer']['format'] != 'qp':
                raise ValueError('Data format returned by server not understood.')
            if _numpy:
                return self._decode_qp_numpy()
            return self._decode_qp()
        finally:
            self.parse_time = time.time() - start

    def _decode_legacy(self):
        """Decode old format, without numpy.

        The legacy format, included mostly for information and contrast, used
        pure json for most of the data, with a dense encoding used for the
        samples themselves.
        """
        # Most of the data can be used as is
        self._result = self._message['answer']

        # Measure the shape of the binary data returned
        num_solutions = len(self._result['energies'])
        active_variables = self._result['active_variables']
        total_variables = self._result['num_variable']

        # Decode the solutions, which will be a continuous run of bits.
        # It was treated as a raw byte string and base64 encoded.
        binary = base64.b64decode(self._result['solutions'])  # Undo the base64 encoding
        byte_buffer = struct.unpack('B' * len(binary), binary)  # Read out the byte array
        bits = []
        for byte in byte_buffer:
            bits.extend(reversed(self._decode_byte(byte)))  # Turn the bytes back into bits

        # Figure out the null value for output
        default = 3 if self._message['type'] == 'qubo' else 0

        # Pull out a bit for each active variable, keep our spot in the
        # bit array between solutions using `index`
        index = 0
        solutions = []
        for solution_index in range(num_solutions):
            # Use None for any values not active
            solution = [default] * total_variables
            for i in active_variables:
                solution[i] = bits[index]
                index += 1

            # Make sure we are in the right variable space
            if self._message['type'] == 'ising':
                values = {0: -1, 1: 1}
                solution = [values.get(v, None) for v in solution]
            solutions.append(solution)

        self._result['solutions'] = solutions

    def _decode_legacy_numpy(self):
        """Decode old format, using numpy.

        Decodes the same format as _decode_legacy, but gains some speed using numpy.
        """
        # Load number lists into numpy buffers
        res = self._result = self._message['answer']
        if self.return_matrix:
            res['energies'] = np.array(res['energies'], dtype=float)
            if 'num_occurrences' in res:
                res['num_occurrences'] = np.array(res['num_occurrences'], dtype=int)
            res['active_variables'] = np.array(res['active_variables'], dtype=int)

        # Measure the shape of the data
        num_solutions = len(res['energies'])
        active_variables = res['active_variables']
        num_variables = len(active_variables)

        # Decode the solutions, which will be a continuous run of bits
        byte_type = np.dtype(np.uint8)
        byte_type = byte_type.newbyteorder('<')
        bits = np.unpackbits(np.frombuffer(base64.b64decode(res['solutions']), dtype=byte_type))

        # Clip off the extra bits from encoding
        bits = np.delete(bits, range(num_solutions * num_variables, bits.size))
        bits = np.reshape(bits, (num_solutions, num_variables))

        # Switch from bits to spins
        default = 3
        if self._message['type'] == 'ising':
            bits = bits.astype(np.int8)
            bits *= 2
            bits -= 1
            default = 0

        # Fill in the missing variables
        solutions = np.full((num_solutions, res['num_variables']), default, dtype=np.int8)
        solutions[:, active_variables] = bits
        res['solutions'] = solutions
        if not res['solutions']:
            res['solutions'] = res['solutions'].tolist()

    def _decode_qp(self):
        """Decode qp format, without numpy.

        The 'qp' format is the current encoding used for problems and samples.
        In this encoding the reply is generally json, but the samples, energy,
        and histogram data (the occurrence count of each solution), are all
        base64 encoded arrays.
        """
        # Decode the simple buffers
        res = self._result = self._message['answer']
        res['active_variables'] = self._decode_ints(res['active_variables'])
        active_variables = res['active_variables']
        if 'num_occurrences' in res:
            res['num_occurrences'] = self._decode_ints(res['num_occurrences'])
        res['energies'] = self._decode_doubles(res['energies'])

        # Measure out the size of the binary solution data
        num_solutions = len(res['energies'])
        num_variables = len(res['active_variables'])
        solution_bytes = -(-num_variables // 8)  # equivalent to int(math.ceil(num_variables / 8.))
        total_variables = res['num_variables']

        # Figure out the null value for output
        default = 3 if self._message['type'] == 'qubo' else 0

        # Decode the solutions, which will be byte aligned in binary format
        binary = base64.b64decode(res['solutions'])
        solutions = []
        for solution_index in range(num_solutions):
            # Grab the section of the buffer related to the current
            buffer_index = solution_index * solution_bytes
            solution_buffer = binary[buffer_index:buffer_index + solution_bytes]
            bytes = struct.unpack('B' * solution_bytes, solution_buffer)

            # Assume None values
            solution = [default] * total_variables
            index = 0
            for byte in bytes:
                # Parse each byte and read how ever many bits can be
                values = self._decode_byte(byte)
                for _ in range(min(8, len(active_variables) - index)):
                    i = active_variables[index]
                    index += 1
                    solution[i] = values.pop()

            # Switch to the right variable space
            if self._message['type'] == 'ising':
                values = {0: -1, 1: 1}
                solution = [values.get(v, default) for v in solution]
            solutions.append(solution)

        res['solutions'] = solutions

    def _decode_byte(self, byte):
        """Helper for _decode_qp, turns a single byte into a list of bits.

        Args:
            byte: byte to be decoded

        Returns:
            list of bits corresponding to byte
        """
        bits = []
        for _ in range(8):
            bits.append(byte & 1)
            byte >>= 1
        return bits

    def _decode_ints(self, message):
        """Helper for _decode_qp, decodes an int array.

        The int array is stored as little endian 32 bit integers.
        The array has then been base64 encoded. Since we are decoding we do these
        steps in reverse.
        """
        binary = base64.b64decode(message)
        return struct.unpack('<' + ('i' * (len(binary) // 4)), binary)

    def _decode_doubles(self, message):
        """Helper for _decode_qp, decodes a double array.

        The double array is stored as little endian 64 bit doubles.
        The array has then been base64 encoded. Since we are decoding we do these
        steps in reverse.
        Args:
            message: the double array

        Returns:
            decoded double array
        """
        binary = base64.b64decode(message)
        return struct.unpack('<' + ('d' * (len(binary) // 8)), binary)

    def _decode_qp_numpy(self):
        """Decode qp format, with numpy."""
        res = self._result = self._message['answer']

        # Build some little endian type encodings
        double_type = np.dtype(np.double)
        double_type = double_type.newbyteorder('<')
        int_type = np.dtype(np.int32)
        int_type = int_type.newbyteorder('<')

        # Decode the simple buffers
        res['energies'] = np.frombuffer(base64.b64decode(res['energies']), dtype=double_type)
        if 'num_occurrences' in res:
            res['num_occurrences'] = np.frombuffer(base64.b64decode(res['num_occurrences']), dtype=int_type)
        res['active_variables'] = np.frombuffer(base64.b64decode(res['active_variables']), dtype=int_type)

        # Measure out the binary data size
        num_solutions = len(res['energies'])
        active_variables = res['active_variables']
        num_variables = len(active_variables)
        total_variables = res['num_variables']

        # Decode the solutions, which will be a continuous run of bits
        byte_type = np.dtype(np.uint8)
        byte_type = byte_type.newbyteorder('<')
        bits = np.unpackbits(np.frombuffer(base64.b64decode(res['solutions']), dtype=byte_type))

        # Clip off the extra bits from encoding
        bits = np.reshape(bits, (num_solutions, bits.size // num_solutions))
        bits = np.delete(bits, range(num_variables, bits.shape[1]), 1)

        # Switch from bits to spins
        default = 3
        if self._message['type'] == 'ising':
            bits = bits.astype(np.int8)
            bits *= 2
            bits -= 1
            default = 0

        # Fill in the missing variables
        solutions = np.full((num_solutions, total_variables), default, dtype=np.int8)
        solutions[:, active_variables] = bits
        res['solutions'] = solutions

        # If the final result shouldn't be numpy formats switch back to python objects
        if not self.return_matrix:
            res['energies'] = res['energies'].tolist()
            if 'num_occurrences' in res:
                res['num_occurrences'] = res['num_occurrences'].tolist()
            res['active_variables'] = res['active_variables'].tolist()
            res['solutions'] = res['solutions'].tolist()


def load_configuration(key=None):
    """Load the configured URLs and token for the SAPI server.

    First, this method tries to read from environment variables.
    If these are not set, it tries to load a configuration file from ``~/.dwrc``.

    The environment variables searched are:

     - ``DW_INTERNAL__HTTPLINK``
     - ``DW_INTERNAL__TOKEN``
     - ``DW_INTERNAL__HTTPPROXY`` (optional)
     - ``DW_INTERNAL__SOLVER`` (optional)


    The configuration file is text file where each line encodes a connection as:

        {connection_name}|{sapi_url},{authentication_token},{proxy_url},{default_solver_name}

    An example configuration file::

        prod|https://qubist.dwavesys.com/sapi,prodtokenstring,,DW2X
        alpha|https://alpha.server.url/sapi,alphatokenstring,https://alpha.proxy.url,dummy-solver

    When there are multiple connections in a file, the first one is taken to be
    the default. Any commas in the urls are percent encoded.

    Args:
        key: The name or URL of the SAPI connection.

    Returns:
        A tuple of SAPI info, as (url, token, proxy, default_solver_name)
    """
    # Try to load environment variables
    url = os.environ.get('DW_INTERNAL__HTTPLINK')
    token = os.environ.get('DW_INTERNAL__TOKEN')
    proxy = os.environ.get('DW_INTERNAL__HTTPPROXY')
    solver = os.environ.get('DW_INTERNAL__SOLVER')

    if url is not None and token is not None:
        return url, token, proxy, solver

    # Load the configuration file
    user_path = os.path.expanduser('~')
    file_path = os.path.join(user_path, '.dwrc')

    # Parse the config file
    try:
        with open(file_path, 'r') as handle:
            lines = handle.readlines()
    except (IOError, OSError):
        # Make sure python 2 and 3 raise the same error
        raise IOError("Could not load configuration from {}".format(file_path))

    # Clean whitespace and empty lines
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

    # Go through the connections and select entry matching the key
    for line in lines:
        try:
            label, data = line.split('|', 1)
            data = {index: value for index, value in enumerate(data.split(','))}

            if label == key or data[0] == key or key is None:
                return data[0], data[1], data.get(2, ''), data.get(3, '')
        except:
            pass  # Just ignore any malformed lines
            # TODO issue a warning

    raise ValueError("No configuration for the connection could be discovered.")


def _evaluate_ising(linear, quad, state):
    """Calculate the energy of a state given the Hamiltonian.

    This is used to debug energy decoding.

    Args:
        linear: Linear Hamiltonian terms.
        quad: Quadratic Hamiltonian terms.
        state: Vector of spins describing the system state.

    Returns:
        Energy of the state evaluated by the given energy function.
    """
    # If we were given a numpy array cast to list
    if _numpy and isinstance(state, np.ndarray):
        return _evaluate_ising(linear, quad, state.tolist())

    # Accumulate the linear and quadratic values
    energy = 0.0
    for index, value in _uniform_iterator(linear):
        energy += state[index] * value
    for (index_a, index_b), value in six.iteritems(quad):
        energy += value * state[index_a] * state[index_b]
    return energy


def _uniform_iterator(sequence):
    """Key, value iteration on a dict or list."""
    if isinstance(sequence, dict):
        return six.iteritems(sequence)
    else:
        return enumerate(sequence)


def _uniform_get(sequence, index, default=None):
    """Get by key with default value for dict or list."""
    if isinstance(sequence, dict):
        return sequence.get(index, default)
    else:
        return sequence[index] if index < len(sequence) else default


#
# Export names to be compatible with full dwave_sapi2 package when possible
# If your code uses dwave_sapi2, but only for things this client provides
# `import dwave_micro_client as dwave_sapi2` should work for simple cases.
#

class AsyncInterfaceWrapper:
    """An incomplete compatibility layer between the async interface of this module and dwave_sapi2."""

    def __init__(self, handle):
        self.handle = handle

    def done(self):
        """Determine whether the submitted problem has finished or not."""
        return self.handle.done()

    def status(self):
        """Get a dictionary containing information about the progress of the problem.

        This is a fairly minimal wrapper that doesn't reach all the possible states
        of the original method, but is compatable.

        See the dwave_sapi2 documentation for details.
        """
        state = 'SUBMITTING'

        if self.handle.time_received is not None:
            state = 'SUBMITTED'

        if self.handle.error is not None or self.handle._message is not None:
            state = 'DONE'

        data = {
            'state': state,
            'remote_status': self.handle.remote_status,
            'problem_id': self.handle.id,
            'last_good_state': state,
        }

        if self.handle.error is not None:
            data['error_type'] = 'INTERNAL'
            if isinstance(self.handle.error, IOError):
                data['error_type'] = 'NETWORK'
            if isinstance(self.handle.error, SolverAuthenticationError):
                data['error_type'] = 'AUTH'
            if isinstance(self.handle.error, SolverFailureError):
                data['error_type'] = 'SOLVE'
            data['error_message'] = str(self.handle.error)

        if self.handle.time_received is not None:
            data['time_received'] = self.handle.time_received

        if self.handle.time_solved is not None:
            data['time_solved'] = self.handle.time_solved
        return data

    def result(self):
        """Return the results of the request."""
        return self.handle

    def retry(self):
        """Attempt to retry the problem.

        In dwave_sapi2 this only retries if particular kinds of errors had
        occurred. Here retry always results in a resubmission regardless of state.
        """
        if self.handle._submission_data is None:
            raise ValueError("Cannot retry on a future that wasn't created by submitting a problem.")

        data = self.handle._submission_data
        self.handle.solver._sample(*data, reuse_future=self.handle)

    def cancel(self):
        """Cancel a submitted problem."""
        self.handle.cancel()


class remote(types.ModuleType):
    """Try to reproduce the interface of the dwave_sapi2.remote module."""

    RemoteConnection = Connection


class core(types.ModuleType):
    """Try to reproduce the interface of the dwave_sapi2.core module."""

    @staticmethod
    def solve_ising(solver, h, J, **kwargs):
        """Forward a sampling problem to solver.sample_ising."""
        future = solver.sample_ising(h, J, **kwargs)
        future.wait()
        return future

    @staticmethod
    def solve_qubo(solver, Q, **kwargs):
        """Forward a sampling problem to solver.sample_qubo."""
        future = solver.sample_qubo(Q, **kwargs)
        future.wait()
        return future

    @staticmethod
    def async_solve_ising(solver, h, J, **kwargs):
        """Forward a sampling problem to solver.solve_ising and wrap the result with a different interface."""
        return AsyncInterfaceWrapper(solver.sample_ising(h, J, **kwargs))

    @staticmethod
    def async_solve_qubo(solver, Q, **kwargs):
        """Forward a sampling problem to solver.solve_qubo and wrap the result with a different interface."""
        return AsyncInterfaceWrapper(solver.sample_qubo(Q, **kwargs))

    @staticmethod
    def await_completion(submitted_problems, min_done, timeout):
        """Wait for a collection of problem handles to finish."""
        return Future.wait_multiple([prob.handle for prob in submitted_problems], min_done, timeout)


# Export the modules
def _add_submodule(submod):
    name = submod.__name__
    sys.modules[__name__ + "." + name] = submod(name)


_add_submodule(remote)
_add_submodule(core)

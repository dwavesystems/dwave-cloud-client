from __future__ import division, absolute_import

import threading
import time
import sys
import posixpath
import logging
import requests
import collections
import datetime
from itertools import chain
from six.moves import queue, range

from dwave.cloud.exceptions import *
from dwave.cloud.config import load_configuration
from dwave.cloud.qpu.solver import Solver

_LOGGER = logging.getLogger(__name__)
# _LOGGER.setLevel(logging.DEBUG)
# _LOGGER.addHandler(logging.StreamHandler(sys.stdout))


class Client(object):
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
        """To setup the connection a pipeline of queues/workers is constructed.

        There are five interations with the server the connection manages:
        1. Downloading solver information.
        2. Submitting problem data.
        3. Polling problem status.
        4. Downloading problem results.
        5. Canceling problems

        Loading solver information is done synchronously. The other four tasks are
        performed by asynchronously workers. For 2, 3, and 5 the workers gather
        tasks in batches.
        """
        # Use configuration from parameters passed, if parts are
        # missing, try the configuration function
        self.default_solver = None
        if token is None:
            url, token, proxies, self.default_solver = load_configuration(url)
        _LOGGER.debug("Creating a connection to SAPI server: %s", url)

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
        self._poll_queue = queue.Queue()
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
            self._poll_queue.put(None)
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

    def solver_names(self):
        """List all the solvers this connection can provide, and load the data about the solvers.

        To get all solver data: ``GET /solvers/remote/``

        Returns:
            list of str
        """
        with self._solvers_lock:
            if self._all_solvers_ready:
                return self.solvers.keys()

            _LOGGER.debug("Requesting list of all solver data.")
            response = self.session.get(posixpath.join(self.base_url, 'solvers/remote/'))

            if response.status_code == 401:
                raise SolverAuthenticationError()
            response.raise_for_status()

            _LOGGER.debug("Received list of all solver data.")

            data = response.json()

            for solver_desc in data:
                try:
                    solver = Solver(self, solver_desc)
                    self.solvers[solver.id] = solver
                    _LOGGER.debug("Adding solver: %s", solver)
                except UnsupportedSolverError as e:
                    _LOGGER.debug("Skipping solver due to %r", e)

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
        _LOGGER.debug("Looking for solver: %s", name)
        if name is None:
            if self.default_solver is not None:
                name = self.default_solver
            else:
                raise ValueError("No name or default name provided when loading solver.")

        with self._solvers_lock:
            if name not in self.solvers:
                if self._all_solvers_ready:
                    raise KeyError(name)

                response = self.session.get(posixpath.join(self.base_url, 'solvers/remote/{}/'.format(name)))

                if response.status_code == 401:
                    raise SolverAuthenticationError()

                if response.status_code == 404:
                    raise KeyError("No solver with the name {} was available".format(name))

                response.raise_for_status()

                solver = Solver(self, data=response.json())
                if solver.id != name:
                    raise InvalidAPIResponseError(
                        "Asked for solver named {!r}, got {!r}".format(name, solver.id))
                self.solvers[name] = solver

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
                _LOGGER.debug("submitting {} problems".format(len(ready_problems)))
                body = '[' + ','.join(mess.body for mess in ready_problems) + ']'
                try:
                    response = self.session.post(posixpath.join(self.base_url, 'problems/'), body)

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    message = response.json()
                    _LOGGER.debug("Finished submitting {} problems".format(len(ready_problems)))
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
            _LOGGER.exception(err)

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
            _LOGGER.debug("Status: %s %s", message['id'], status)

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
                    self.session.delete(posixpath.join(self.base_url, 'problems/'), json=body)
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
                # `None` task signifies thread termination
                if ftr is None:
                    return False
                if ftr.id not in futures and not ftr.done():
                    active_queries.add(ftr.id)
                    futures[ftr.id] = ftr
                else:
                    self._poll_queue.task_done()
                return True

            # Remove a query from the active set
            def remove(id_):
                del futures[id_]
                active_queries.remove(id_)
                self._poll_queue.task_done()

            while True:
                try:
                    # If we have no active queries, wait on the status queue
                    while len(active_queries) == 0:
                        if not add(self._poll_queue.get()):
                            return
                    # Once there is any active queries try to fill up the set and move on
                    while len(active_queries) < self._STATUS_QUERY_SIZE:
                        if not add(self._poll_queue.get_nowait()):
                            return
                except queue.Empty:
                    pass

                # Build a query string with block of ids
                _LOGGER.debug("Query on futures: %s", ', '.join(active_queries))
                query_string = 'problems/?id=' + ','.join(active_queries)

                try:
                    response = self.session.get(posixpath.join(self.base_url, query_string))

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
                _LOGGER.debug("Query for results: %s", future.id)

                # Submit the query
                query_string = 'problems/{}/'.format(future.id)
                try:
                    response = self.session.get(posixpath.join(self.base_url, query_string))

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
            _LOGGER.error('Load result error: ' + str(err))

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
Computation manages the interactions between your code and a :term:`solver`, which
manages interactions between the remote resource and your submitted problems.

Your solver instantiates a :class:`Future` object for its calls, via D-Wave Sampler API
(SAPI) servers, to the remote resource.

You can interact through the :class:`Future` object with pending (running) or completed
computation---sampling on a QPU or software solver---executed remotely, monitoring problem status,
waiting for and retrieving results, cancelling enqueued jobs, etc.

Some :class:`Future` methods are blocking.

"""

from __future__ import division, absolute_import

import threading
import time
import six
import functools
from concurrent.futures import TimeoutError

from dateutil.parser import parse

from dwave.cloud.coders import decode_qp, decode_qp_numpy
from dwave.cloud.utils import utcnow, datetime_to_timestamp

# Use numpy if available for fast decoding
try:
    import numpy as np
    _numpy = True
except ImportError:
    # If numpy isn't available we can do the encoding slower in native python
    _numpy = False

__all__ = ['Future']


@functools.total_ordering
class Future(object):
    """Class for interacting with jobs submitted to SAPI.

    :class:`~dwave.cloud.solver.Solver` uses :class:`Future` to construct objects for pending SAPI calls that can wait for
    requests to complete and parse returned messages.

    Objects are blocked for the duration of any data accessed on the remote resource.

    .. warning:: :class:`Future` objects are not intended to be directly created. Problem submittal is initiated by :class:`~dwave.cloud.solver.Solver` and executed by the client.

    Args:
        solver: Solver responsible for this :class:`Future` object.
        id_: Identification for a query submitted by a solver to SAPI.
            May be None following submission until an identification number is set.
        return_matrix: Return values for this :class:`Future` object are NumPy matrices.

    Examples:
        This example creates a solver using the local system's default D-Wave Cloud Client
        configuration file, submits a simple QUBO problem to a remote D-Wave resource for
        100 samples, and checks a couple of times whether the sampling is completed.

        >>> from dwave.cloud import Client
        >>> client = Client.from_config()
        >>> solver = client.get_solver()
        >>> u, v = next(iter(solver.edges))
        >>> Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
        >>> computation = solver.sample_qubo(Q, num_reads=100)   # doctest: +SKIP
        >>> computation.done()  # doctest: +SKIP
        False
        >>> computation.id   # doctest: +SKIP
        u'1cefeb6d-ebd5-4592-87c0-4cc43ec03e27'
        >>> computation.done()   # doctest: +SKIP
        True
        >>> client.close()
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

        #: `datetime` the Future was created (immediately before enqueued in Client's submit queue)
        self.time_created = utcnow()

        #: `datetime` corresponding to the time when the problem was accepted by the server (None before then)
        self.time_received = None

        #: `datetime` corresponding to the time when the problem was completed by the server (None before then)
        self.time_solved = None

        #: `datetime` the Future was resolved (marked as done; succeeded or failed), or None before then
        self.time_resolved = None

        # estimated `earliest_completion_time` as returned on problem submit
        self.eta_min = None

        # estimated `latest_completion_time` as returned on problem submit
        self.eta_max = None

        # Track how long it took us to parse the data
        self.parse_time = None

        # approx. server-client clocks difference in seconds
        self.clock_diff = None

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

        # current poll back-off interval, in seconds
        self._poll_backoff = None

    def __lt__(self, other):
        return id(self) < id(other)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

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
            exc_info: Stack trace info from sys module for re-raising exceptions nicely.
        """
        self.error = error
        self._exc_info = exc_info
        self._signal_ready()

    def _signal_ready(self):
        """Signal all the events waiting on this future."""
        self.time_resolved = utcnow()
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

    def _set_clock_diff(self, server_response, localtime_of_response):
        """Calculate and set the `.clock_diff`, based on headers from a server
        response, and the local time of response received.

        Based on `clock_diff`, `eta_min`/`eta_max` may or may not make sense.
        """
        try:
            server_time = datetime_to_timestamp(parse(server_response.headers['date']))
        except:
            server_time = 0
        self.clock_diff = abs(server_time - localtime_of_response)

    @staticmethod
    def wait_multiple(futures, min_done=None, timeout=None):
        """Wait for multiple :class:`Future` objects to complete.

        Blocking call that uses an event object to emulate multi-wait for Python.

        Args:
            futures (list of Futures): List of :class:`Future` objects to await.
            min_done (int, optional, default=None): Minimum required completions to end
                the waiting. The wait is terminated when this number of results are ready.
                If None, waits for all the :class:`Future` objects to complete.
            timeout (float, optional, default=None): Maximum number of seconds to await completion.
                If None, waits indefinitely.

        Returns:
            Two-tuple of :class:`Future` objects: completed and not completed submitted tasks. Similar to `concurrent.futures.wait()` method's returned two-tuple of `done` and `not_done` sets.

        See Also:
            :func:`as_completed` for a blocking iterable of resolved futures similar to `concurrent.futures.as_completed()` method.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple QUBO problem to a remote D-Wave resource 3 times
            for differing numers of samples, and waits for sampling to complete on any
            two of the submissions. The wait ends with the completion of two submissions while
            the third is still in progress. (A more typical approach would use something
            like :code:`first = next(Future.as_completed(computation))` instead.)

            >>> import dwave.cloud as dc
            >>> client = dc.Client.from_config()
            >>> solver = client.get_solver()
            >>> u, v = next(iter(solver.edges))
            >>> Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            >>> computation = [solver.sample_qubo(Q, num_reads=1000),
            ...                solver.sample_qubo(Q, num_reads=50),
            ...                solver.sample_qubo(Q, num_reads=10)]   # doctest: +SKIP
            >>> dc.computation.Future.wait_multiple(computation, min_done=1)    # doctest: +SKIP
            ([<dwave.cloud.computation.Future at 0x17dde518>,
              <dwave.cloud.computation.Future at 0x17ddee80>],
             [<dwave.cloud.computation.Future at 0x15078080>])
            >>> print(computation[0].done())   # doctest: +SKIP
            False
            >>> print(computation[1].done())  # doctest: +SKIP
            True
            >>> print(computation[2].done())   # doctest: +SKIP
            True
            >>> client.close()

        """
        if min_done is None:
            min_done = len(futures)

        if timeout is None:
            timeout = float('inf')

        # Track the exit conditions
        finish = time.time() + timeout
        done = []

        # Keep track of what futures haven't finished
        remaining = list(futures)

        # Insert our event into all the futures
        event = threading.Event()
        [f._add_event(event) for f in remaining]

        # Check the exit conditions
        while len(done) < min_done and finish > time.time():
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
                done.extend(finished_futures)
                remaining = [f for f in remaining if f not in finished_futures]
                continue

            # Block on any of the jobs finishing
            wait_time = finish - time.time() if abs(finish) != float('inf') else None
            event.wait(wait_time)

        # Clean up after ourselves
        [f._remove_event(event) for f in futures]
        return done, remaining

    @staticmethod
    def as_completed(fs, timeout=None):
        """Yield Futures objects as they complete.

        Returns an iterator over the specified list of :class:`Future` objects that
        yields those objects as they complete. Completion occurs
        when the submitted job is finished or cancelled.

        Emulates the behavior of the `concurrent.futures.as_completed()` function.

        Args:
            fs (list): List of :class:`Future` objects to iterate over.
            timeout (float, optional, default=None): Maximum number of seconds to await completion.
                If None, awaits indefinitely.

        Returns:
            Generator (:class:`Future` objects):
                Listed :class:`Future` objects as they complete.

        Raises:
            `concurrent.futures.TimeoutError` is raised if per-future timeout is exceeded.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple QUBO problem to a remote D-Wave resource 3 times
            for differing numers of samples, and yields timing information for each job
            as it completes.

            >>> import dwave.cloud as dc
            >>> client = dc.Client.from_config()
            >>> solver = client.get_solver()
            >>> u, v = next(iter(solver.edges))
            >>> Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            >>> computation = [solver.sample_qubo(Q, num_reads=1000),
            ...                solver.sample_qubo(Q, num_reads=50),
            ...                solver.sample_qubo(Q, num_reads=10)]   # doctest: +SKIP
            >>> for tasks in dc.computation.Future.as_completed(computation, timeout=10)
            ...     print(tasks.timing)   # doctest: +SKIP
            ...
            {u'total_real_time': 17318, ... u'qpu_readout_time_per_sample': 123}
            {u'total_real_time': 10816, ... u'qpu_readout_time_per_sample': 123}
            {u'total_real_time': 26285, ... u'qpu_readout_time_per_sample': 123}
            >>> # Snipped above response for brevity
            >>> client.close()

        """
        not_done = fs
        while not_done:
            done, not_done = Future.wait_multiple(not_done, min_done=1, timeout=timeout)
            if not done:
                raise TimeoutError
            for f in done:
                yield f

    def wait(self, timeout=None):
        """Wait for the solver to receive a response for a submitted problem.

        Blocking call that waits for a :class:`Future` object to complete.

        Args:
            timeout (float, optional, default=None): Maximum number of seconds to await completion.
                If None, waits indefinitely.

        Returns:
            Boolean: True if solver received a response.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple QUBO problem to a remote D-Wave resource for
            100 samples, and tries waiting for 10 seconds for sampling to complete.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> solver = client.get_solver()
            >>> u, v = next(iter(solver.edges))
            >>> Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            >>> computation = solver.sample_qubo(Q, num_reads=100)   # doctest: +SKIP
            >>> computation.wait(timeout=10)    # doctest: +SKIP
            False
            >>> computation.remote_status
            u'IN_PROGRESS'
            >>> computation.wait(timeout=10)  # doctest: +SKIP
            True
            >>> computation.remote_status   # doctest: +SKIP
            u'COMPLETED'
            >>> client.close()
        """
        return self._results_ready_event.wait(timeout)

    def done(self):
        """Check whether the solver received a response for a submitted problem.

        Non-blocking call that checks whether the solver has received a response from
        the remote resource.

        Returns:
            Boolean: True if solver received a response.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple QUBO problem to a remote D-Wave resource for
            100 samples, and checks a couple of times whether sampling is completed.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> solver = client.get_solver()
            >>> u, v = next(iter(solver.edges))
            >>> Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            >>> computation = solver.sample_qubo(Q, num_reads=100)   # doctest: +SKIP
            >>> computation.done()  # doctest: +SKIP
            False
            >>> computation.done()   # doctest: +SKIP
            True
            >>> client.close()
        """
        return self._results_ready_event.is_set()

    def cancel(self):
        """Try to cancel the problem corresponding to this result.

        Non-blocking call to the remote resource in a best-effort attempt to prevent execution of a problem.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple QUBO problem to a remote D-Wave resource for
            100 samples, and tries (and in this case succeeds) to cancel it.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> solver = client.get_solver()
            >>> u, v = next(iter(solver.edges))
            >>> Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            >>> computation = solver.sample_qubo(Q, num_reads=100)   # doctest: +SKIP
            >>> computation.cancel()  # doctest: +SKIP
            >>> computation.done()   # doctest: +SKIP
            True
            >>> computation.remote_status    # doctest: +SKIP
            u'CANCELLED'
            >>> client.close()

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
                self.solver.client._cancel(self.id, self)

    def result(self):
        """Results for a submitted job.

        Retrives result data in a :class:`Future` object that the solver submitted to a remote resource.
        First calls to access this data are blocking.

        Returns:
            dict: Results of the submitted job. Should be considered read-only.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple QUBO problem (representing a Boolean NOT gate
            by a penalty function) to a remote D-Wave resource for 5 samples, and prints part
            of the returned result (the relevant samples).

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            ...     computation = solver.sample_qubo(Q, num_reads=5)
            ...     for i in range(5):
            ...         print(computation.result()['samples'][i][u], computation.result()['samples'][i][v])
            ...
            ...
            (0, 1)
            (1, 0)
            (1, 0)
            (0, 1)
            (0, 1)

        """
        self._load_result()
        return self._result

    @property
    def energies(self):
        """Energy buffer for the submitted job.

        First calls to access data of a :class:`Future` object are blocking; subsequent access
        to this property is non-blocking.

        Returns:
            list or NumPy matrix of doubles: Energies for each set of samples.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a random Ising problem (+1 or -1 values of linear and
            quadratic biases on all nodes and edges, respectively, of the solver's garph) to
            a remote D-Wave resource for 10 samples, and prints the returned energies.

            >>> import random
            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            ...     quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}
            ...     computation = solver.sample_ising(linear, quad, num_reads=10)
            ...     print(computation.energies)
            ...
            [-3976.0, -3974.0, -3972.0, -3970.0, -3968.0, -3968.0, -3966.0,
             -3964.0, -3964.0, -3960.0]
        """
        return self.result()['energies']

    @property
    def samples(self):
        """State buffer for the submitted job.

        First calls to access data of a :class:`Future` object are blocking; subsequent access
        to this property is non-blocking.

        Returns:
            list of lists or NumPy matrix: Samples on the nodes of solver's graph.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple QUBO problem (representing a Boolean NOT gate
            by a penalty function) to a remote D-Wave resource for 5 samples, and prints part
            of the returned result (the relevant samples).

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            ...     computation = solver.sample_qubo(Q, num_reads=5)
            ...     for i in range(5):
            ...         print(computation.samples[i][u], computation.samples[i][v])
            ...
            ...
            (1, 0)
            (0, 1)
            (0, 1)
            (1, 0)
            (0, 1)
        """
        return self.result()['samples']

    @property
    def occurrences(self):
        """Occurrences buffer for the submitted job.

        First calls to access data of a :class:`Future` object are blocking; subsequent access
        to this property is non-blocking.

        Returns:
            list or NumPy matrix of doubles: Occurrences. When returned results are
            ordered in a histogram, `occurrences` indicates the number of times a particular
            solution recurred.

        Examples:
            This example creates a solver using the local system's default D-Wave Cloud Client
            configuration file, submits a simple Ising problem with several ground states to a
            remote D-Wave resource for 20 samples, and prints the returned results, which
            are ordered as a histogram. The problem's ground states tend to recur frequently,
            and so those solutions have `occurrences` greater than 1.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     quad = {(16, 20): -1, (17, 20): 1, (16, 21): 1, (17, 21): 1}
            ...     computation = solver.sample_ising({}, quad, num_reads=500, answer_mode='histogram')
            ...     for i in range(len(computation.occurrences)):
            ...         print(computation.samples[i][16],computation.samples[i][17], computation.samples[i][20], computation.samples[i][21], ' --> ', computation.energies[i], computation.occurrences[i])
            ...
            (-1, 1, -1, -1, ' --> ', -2.0, 41)
            (-1, -1, -1, 1, ' --> ', -2.0, 53)
            (1, -1, 1, 1, ' --> ', -2.0, 55)
            (1, 1, -1, -1, ' --> ', -2.0, 52)
            (1, 1, 1, -1, ' --> ', -2.0, 60)
            (1, -1, 1, -1, ' --> ', -2.0, 196)
            (-1, 1, -1, 1, ' --> ', -2.0, 15)
            (-1, -1, 1, 1, ' --> ', -2.0, 28)

        """
        self._load_result()
        if 'occurrences' in self._result:
            return self._result['occurrences']
        elif self.return_matrix:
            return np.ones((len(self._result['samples']),))
        else:
            return [1] * len(self._result['samples'])

    @property
    def timing(self):
        """Timing information about a solver operation.

        Mapping from string keys to numeric values representing timing details
        for a submitted job as returned from the remote resource. Keys are dependant on
        the particular solver.

        First calls to access data of a :class:`Future` object are blocking; subsequent access
        to this property is non-blocking.

        Returns:
            dict: Mapping from string keys to numeric values representing timing information.

        Examples:
            This example creates a client using the local system's default D-Wave Cloud Client
            configuration file, which is configured to access a D-Wave 2000Q QPU, submits a
            simple :term:`Ising` problem (opposite linear biases on two coupled qubits) for
            5 samples, and prints timing information for the job.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     computation = solver.sample_ising({u: -1, v: 1},{}, num_reads=5)   # doctest: +SKIP
            ...     print(computation.timing)
            ...
            {u'total_real_time': 10961, u'anneal_time_per_run': 20,
            >>> # Snipped above response for brevity

        """
        return self.result()['timing']

    def __getitem__(self, key):
        """Provide a simple results item getter. Blocks if future is unresolved.

        Args:
            key: keywords for result fields.
        """
        self._load_result()
        if key not in self._result:
            raise KeyError('{} is not a property of response object'.format(key))
        return self._result[key]

    def _load_result(self):
        """Get the result, waiting and decoding as needed."""
        if self._result is None:
            # Wait for the query response
            self.wait(timeout=None)

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

        if self._message['type'] not in ['qubo', 'ising']:
            raise ValueError('Unknown problem format used.')

        # If format is set, it must be qp
        if self._message.get('answer', {}).get('format') != 'qp':
            raise ValueError('Data format returned by server not understood.')

        # prefer numpy decoding, but fallback to python
        # TODO: we should really be explicit about numpy usage
        start = time.time()
        if _numpy:
            self._result = decode_qp_numpy(self._message,
                                           return_matrix=self.return_matrix)
        else:
            self._result = decode_qp(self._message)
        self.parse_time = time.time() - start

        self._alias_result()
        return self._result

    def _alias_result(self):
        """Create aliases for some of the keys in the results dict. Eventually,
        those will be renamed on the server side.
        """
        if not self._result:
            return

        aliases = {'samples': 'solutions',
                   'occurrences': 'num_occurrences'}
        for alias, original in aliases.items():
            if original in self._result and alias not in self._result:
                self._result[alias] = self._result[original]

        return self._result

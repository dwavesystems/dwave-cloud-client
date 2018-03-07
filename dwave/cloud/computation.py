from __future__ import division, absolute_import

import threading
import time
import six
from concurrent.futures import TimeoutError

from dwave.cloud.coders import decode_qp, decode_qp_numpy


# Use numpy if available for fast decoding
try:
    import numpy as np
    _numpy = True
except ImportError:
    # If numpy isn't available we can do the encoding slower in native python
    _numpy = False


class Future(object):
    """An object for a pending SAPI call.

    Waits for a request to complete and parses the message returned.
    The future will be block to resolve when any data value is accessed.
    The method :meth:`done` can be used to query for resolution without blocking.
    :meth:`wait`, and :meth:`wait_multiple` can be used to block for a variable
    number of jobs for a given amount of time.

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

        #: `datetime` corresponding to the time when the problem was accepted by the server (None before then)
        self.time_received = None

        #: `datetime` corresponding to the time when the problem was completed by the server (None before then)
        self.time_solved = None

        #: `datetime` corresponding to the time when the problem was completed by the server (None before then)
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
            exc_info: Stack trace info from sys module for re-raising exceptions nicely.
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
    def wait_multiple(futures, min_done=None, timeout=None):
        """Wait for multiple Future objects to complete.

        Python doesn't provide a multi-wait, but we can jury rig something reasonably
        efficient using an event object.

        Args:
            futures (list of Future): list of objects to wait on
            min_done (int): Stop waiting when this many results are ready
            timeout (float): Maximum number of seconds to wait, `None` for indefinite wait.

        Returns:
            2-tuple of futures done and not_done, similar to `concurrent.futures.wait()`
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
        """Emulate `concurrent.futures.as_completed()` behavior.

        Returns an iterator over the list of out `Future` instances given by
        `fs` that yields futures as they complete (finished or were cancelled).

        The `concurrent.futures.TimeoutError` is raised if per-future timeout is
        exceeded at any point.
        """
        not_done = fs
        while not_done:
            done, not_done = Future.wait_multiple(not_done, min_done=1, timeout=timeout)
            if not done:
                raise TimeoutError
            for f in done:
                yield f

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
                self.solver.client._cancel(self.id, self)

    def result(self):
        """Blocking call to retrieve the result.

        Returns:
            dict with the results. Should be considered read-only.
        """
        self._load_result()
        return self._result

    @property
    def energies(self):
        """The energy buffer, blocks if needed.

        Returns:
            list or numpy matrix of doubles.
        """
        return self.result()['energies']

    @property
    def samples(self):
        """The state buffer, blocks if needed.

        Returns:
            list of lists or numpy matrix.
        """
        return self.result()['samples']

    @property
    def occurrences(self):
        """The occurrences buffer, blocks if needed.

        Returns:
            list or numpy matrix of doubles.
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
        """Information about the time the solver took in operation.

        The response is a mapping from string keys to numeric values.
        The exact keys used depend on the solver.

        Returns:
            dict
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

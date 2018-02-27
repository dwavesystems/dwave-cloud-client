from __future__ import division, absolute_import

import json
import threading
import base64
import struct
import time
import sys
import os
import posixpath
import types
import logging
import requests
import collections
import datetime
import six

# Use numpy if available for fast decoding
try:
    import numpy as np
    _numpy = True
except ImportError:
    # If numpy isn't available we can do the encoding slower in native python
    _numpy = False


class Future:
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
    def wait_multiple(futures, min_done=None, timeout=float('inf')):
        """Wait for multiple Future objects to complete.

        Python doesn't provide a multi-wait, but we can jury rig something reasonably
        efficient using an event object.

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

        # Keep track of what futures haven't finished
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
        return self.result()['solutions']

    @property
    def occurrences(self):
        """The occurrences buffer, blocks if needed.

        Returns:
            list or numpy matrix of doubles.
        """
        self._load_result()
        if 'num_occurrences' in self._result:
            return self._result['num_occurrences']
        elif self.return_matrix:
            return np.ones((len(self._result['solutions']),))
        else:
            return [1] * len(self._result['solutions'])

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
        start = time.time()
        try:
            if self._message['type'] not in ['qubo', 'ising']:
                raise ValueError('Unknown problem format used.')

            # If format is set, it must be qp
            if self._message.get('answer', {}).get('format') != 'qp':
                raise ValueError('Data format returned by server not understood.')
            if _numpy:
                return self._decode_qp_numpy()
            return self._decode_qp()
        finally:
            self.parse_time = time.time() - start

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

"""Test problem submission against hard-coded replies with unittest.mock."""
from __future__ import division, absolute_import, print_function, unicode_literals

import time
import json
import unittest
import itertools
import threading

from datetime import datetime, timedelta
from dateutil.tz import UTC
from dateutil.parser import parse as parse_datetime

from dwave.cloud.utils import evaluate_ising
from dwave.cloud.qpu import Client, Solver
from dwave.cloud.exceptions import SolverFailureError, CanceledFutureError
from dwave.cloud.testing import mock


def solver_data(id_, incomplete=False):
    """Return data for a solver."""
    obj = {
        "properties": {
            "supported_problem_types": ["qubo", "ising"],
            "qubits": [0, 1, 2, 3, 4],
            "couplers": list(itertools.combinations(range(5), 2)),
            "num_qubits": 3,
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "id": id_,
        "description": "A test solver"
    }

    if incomplete:
        del obj['properties']['parameters']

    return obj


def complete_reply(id_, solver_name):
    """Reply with solutions for the test problem."""
    return json.dumps({
        "status": "COMPLETED",
        "solved_on": "2013-01-18T10:26:00.020954",
        "solver": solver_name,
        "submitted_on": "2013-01-18T10:25:59.941674",
        "answer": {
            'format': 'qp',
            "num_variables": 5,
            "energies": 'AAAAAAAALsA=',
            "num_occurrences": 'ZAAAAA==',
            "active_variables": 'AAAAAAEAAAACAAAAAwAAAAQAAAA=',
            "solutions": 'AAAAAA==',
            "timing": {}
        },
        "type": "ising",
        "id": id_
    })


def complete_no_answer_reply(id_, solver_name):
    """A reply saying a problem is finished without providing the results."""
    return json.dumps({
        "status": "COMPLETED",
        "solved_on": "2012-12-05T19:15:07+00:00",
        "solver": solver_name,
        "submitted_on": "2012-12-05T19:06:57+00:00",
        "type": "ising",
        "id": id_
    })


def error_reply(id_, solver_name, error):
    """A reply saying an error has occurred."""
    return json.dumps({
        "status": "FAILED",
        "solved_on": "2013-01-18T10:26:00.020954",
        "solver": solver_name,
        "submitted_on": "2013-01-18T10:25:59.941674",
        "type": "ising",
        "id": id_,
        "error_message": error
    })


def cancel_reply(id_, solver_name):
    """A reply saying a problem was canceled."""
    return json.dumps({
        "status": "CANCELLED",
        "solved_on": "2013-01-18T10:26:00.020954",
        "solver": solver_name,
        "submitted_on": "2013-01-18T10:25:59.941674",
        "type": "ising",
        "id": id_
    })


now = datetime.utcnow().replace(tzinfo=UTC)
continue_reply_eta_min = now + timedelta(seconds=10)
continue_reply_eta_max = now + timedelta(seconds=30)

def continue_reply(id_, solver_name):
    """A reply saying a problem is still in the queue."""
    return json.dumps({
        "status": "PENDING",
        "solved_on": None,
        "solver": solver_name,
        "submitted_on": now.isoformat(),
        "earliest_completion_time": continue_reply_eta_min.isoformat(),
        "latest_completion_time": continue_reply_eta_max.isoformat(),
        "type": "ising",
        "id": id_
    })


def choose_reply(path, replies):
    """Choose the right response based on the path and make a mock response."""
    if path in replies:
        response = mock.Mock(['json', 'raise_for_status'])
        response.status_code = 200
        response.json.side_effect = lambda: json.loads(replies[path])
        return response
    else:
        raise NotImplementedError(path)


class _QueryTest(unittest.TestCase):
    def _check(self, results, linear, quad, num):
        # Did we get the right number of samples?
        self.assertTrue(100 == sum(results.occurrences))

        # Make sure the number of occurrences and energies are all correct
        for energy, state in zip(results.energies, results.samples):
            self.assertTrue(energy == evaluate_ising(linear, quad, state))


@mock.patch('time.sleep', lambda *x: None)
class MockSubmission(_QueryTest):
    """Test connecting and some related failure modes."""

    def test_submit_null_reply(self):
        """Get an error when the server's response is incomplete."""
        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            client.session.post = lambda a, _: choose_reply(a, {'endpoint/problems/': ''})
            solver = Solver(client, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            with self.assertRaises(ValueError):
                results.samples

    def test_submit_ok_reply(self):
        """Handle a normal query and response."""
        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            client.session.post = lambda a, _: choose_reply(a, {
                'endpoint/problems/': '[%s]' % complete_no_answer_reply('123', 'abc123')})
            client.session.get = lambda a: choose_reply(a, {'endpoint/problems/123/': complete_reply('123', 'abc123')})
            solver = Solver(client, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            self._check(results, linear, quad, 100)

    def test_submit_error_reply(self):
        """Handle an error on problem submission."""
        error_body = 'An error message'
        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            client.session.post = lambda a, _: choose_reply(a, {
                'endpoint/problems/': '[%s]' % error_reply('123', 'abc123', error_body)})
            solver = Solver(client, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            with self.assertRaises(SolverFailureError):
                results.samples

    def test_submit_cancel_reply(self):
        """Handle a response for a canceled job."""
        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            client.session.post = lambda a, _: choose_reply(a, {'endpoint/problems/': '[%s]' % cancel_reply('123', 'abc123')})
            solver = Solver(client, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            with self.assertRaises(CanceledFutureError):
                results.samples

    def test_submit_continue_then_ok_reply(self):
        """Handle polling for a complete problem."""
        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            client.session.post = lambda a, _: choose_reply(a, {'endpoint/problems/': '[%s]' % continue_reply('123', 'abc123')})
            client.session.get = lambda a: choose_reply(a, {
                'endpoint/problems/?id=123': '[%s]' % complete_no_answer_reply('123', 'abc123'),
                'endpoint/problems/123/': complete_reply('123', 'abc123')
            })
            solver = Solver(client, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            self._check(results, linear, quad, 100)

            # test future has eta_min and eta_max parsed correctly
            self.assertEqual(results.eta_min, continue_reply_eta_min)
            self.assertEqual(results.eta_max, continue_reply_eta_max)

    def test_submit_continue_then_error_reply(self):
        """Handle polling for an error message."""
        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            client.session.post = lambda a, _: choose_reply(a, {'endpoint/problems/': '[%s]' % continue_reply('123', 'abc123')})
            client.session.get = lambda a: choose_reply(a, {
                'endpoint/problems/?id=123': '[%s]' % error_reply('123', 'abc123', "error message")})
            solver = Solver(client, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            with self.assertRaises(SolverFailureError):
                self._check(results, linear, quad, 100)

    # Reduce the number of poll and submission threads so that the system can be tested
    @mock.patch.object(Client, "_POLL_THREAD_COUNT", 1)
    @mock.patch.object(Client, "_SUBMISSION_THREAD_COUNT", 1)
    def test_submit_continue_then_ok_and_error_reply(self):
        """Handle polling for the status of multiple problems."""

        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()

            # on first status poll, return pending for both problems
            # on second status poll, return error for first problem and complete for second
            def continue_then_complete(path, state={'count': 0}):
                state['count'] += 1
                if state['count'] < 2:
                    return choose_reply(path, {
                        'endpoint/problems/?id=1': '[{}]'.format(continue_reply('1', 'abc123')),
                        'endpoint/problems/?id=2': '[{}]'.format(continue_reply('2', 'abc123')),
                        'endpoint/problems/1/': continue_reply('1', 'abc123'),
                        'endpoint/problems/2/': continue_reply('2', 'abc123'),
                        'endpoint/problems/?id=1,2': '[{},{}]'.format(continue_reply('1', 'abc123'),
                                                                      continue_reply('2', 'abc123')),
                        'endpoint/problems/?id=2,1': '[{},{}]'.format(continue_reply('2', 'abc123'),
                                                                      continue_reply('1', 'abc123'))
                    })
                else:
                    return choose_reply(path, {
                        'endpoint/problems/?id=1': '[{}]'.format(error_reply('1', 'abc123', 'error')),
                        'endpoint/problems/?id=2': '[{}]'.format(complete_reply('2', 'abc123')),
                        'endpoint/problems/1/': error_reply('1', 'abc123', 'error'),
                        'endpoint/problems/2/': complete_reply('2', 'abc123'),
                        'endpoint/problems/?id=1,2': '[{},{}]'.format(error_reply('1', 'abc123', 'error'),
                                                                      complete_reply('2', 'abc123')),
                        'endpoint/problems/?id=2,1': '[{},{}]'.format(complete_reply('2', 'abc123'),
                                                                      error_reply('1', 'abc123', 'error'))
                    })

            client.session.get = continue_then_complete

            def accept_problems_with_continue_reply(path, body, ids=iter('12')):
                problems = json.loads(body)
                return choose_reply(path, {
                    'endpoint/problems/': json.dumps(
                        [json.loads(continue_reply(next(ids), 'abc123')) for _ in problems])
                })

            client.session.post = accept_problems_with_continue_reply

            solver = Solver(client, solver_data('abc123'))
            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}

            results1 = solver.sample_ising(linear, quad, num_reads=100)
            results2 = solver.sample_ising(linear, quad, num_reads=100)

            with self.assertRaises(SolverFailureError):
                self._check(results1, linear, quad, 100)
            self._check(results2, linear, quad, 100)

    # Reduce the number of poll and submission threads so that the system can be tested
    @mock.patch.object(Client, "_POLL_THREAD_COUNT", 1)
    @mock.patch.object(Client, "_SUBMISSION_THREAD_COUNT", 1)
    def test_exponential_backoff_polling(self):
        "After each poll, back-off should double"

        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            # on submit, return status pending
            client.session.post = lambda a, _: choose_reply(a, {
                'endpoint/problems/': '[%s]' % continue_reply('123', 'abc123')
            })
            # on first and second status poll, return pending
            # on third status poll, return completed
            def continue_then_complete(path, state={'count': 0}):
                state['count'] += 1
                if state['count'] < 3:
                    return choose_reply(path, {
                        'endpoint/problems/?id=123': '[%s]' % continue_reply('123', 'abc123'),
                        'endpoint/problems/123/': continue_reply('123', 'abc123')
                    })
                else:
                    return choose_reply(path, {
                        'endpoint/problems/?id=123': '[%s]' % complete_no_answer_reply('123', 'abc123'),
                        'endpoint/problems/123/': complete_reply('123', 'abc123')
                    })

            client.session.get = continue_then_complete

            solver = Solver(client, solver_data('abc123'))

            future = solver.sample_qubo({})
            future.result()

            # after third poll, back-off interval should be 4
            self.assertEqual(future._poll_backoff, 4)


class DeleteEvent(Exception):
    """Throws exception when mocked client submits an HTTP DELETE request."""

    def __init__(self, url, body):
        """Return the URL of the request with the exception for test verification."""
        self.url = url
        self.body = body

    @staticmethod
    def handle(path, **kwargs):
        """Callback useable to mock a delete request."""
        raise DeleteEvent(path, json.dumps(kwargs['json']))


class MockCancel(unittest.TestCase):
    """Make sure cancel works at the two points in the process where it should."""

    def test_cancel_with_id(self):
        """Make sure the cancel method submits to the right endpoint.

        When cancel is called after the submission is finished.
        """
        submission_id = 'test-id'
        reply_body = '[%s]' % continue_reply(submission_id, 'solver')

        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()

            client.session.get = lambda a: choose_reply(a, {'endpoint/problems/?id={}'.format(submission_id): reply_body})
            client.session.delete = DeleteEvent.handle

            solver = Solver(client, solver_data('abc123'))
            future = solver.retrieve_problem(submission_id)
            future.cancel()

            try:
                self.assertTrue(future.id is not None)
                future.samples
                self.fail()
            except DeleteEvent as event:
                if event.url == 'endpoint/problems/':
                    self.assertEqual(event.body, '["{}"]'.format(submission_id))
                else:
                    self.assertEqual(event.url, 'endpoint/problems/{}/'.format(submission_id))

    def test_cancel_without_id(self):
        """Make sure the cancel method submits to the right endpoint.

        When cancel is called before the submission has returned the problem id.
        """
        submission_id = 'test-id'
        reply_body = '[%s]' % continue_reply(submission_id, 'solver')

        release_reply = threading.Event()

        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            client.session.get = lambda a: choose_reply(a, {'endpoint/problems/?id={}'.format(submission_id): reply_body})

            def post(a, _):
                release_reply.wait()
                return choose_reply(a, {'endpoint/problems/': reply_body})
            client.session.post = post
            client.session.delete = DeleteEvent.handle

            solver = Solver(client, solver_data('abc123'))
            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            future = solver.sample_ising(linear, quad)
            future.cancel()

            try:
                release_reply.set()
                future.samples
                self.fail()
            except DeleteEvent as event:
                if event.url == 'endpoint/problems/':
                    self.assertEqual(event.body, '["{}"]'.format(submission_id))
                else:
                    self.assertEqual(event.url, 'endpoint/problems/{}/'.format(submission_id))

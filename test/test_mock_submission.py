"""Test problem submission against hard-coded replies with unittest.mock."""
from __future__ import division, absolute_import, print_function, unicode_literals

try:
    import unittest.mock as mock
except ImportError:
    import mock
import json
import unittest
import itertools
import dwave_micro_client
import threading


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


def continue_reply(id_, solver_name):
    """A reply saying a problem is still in the queue."""
    return json.dumps({
        "status": "PENDING",
        "solved_on": None,
        "solver": solver_name,
        "submitted_on": "2013-01-18T10:06:10.025064",
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
            self.assertTrue(energy == dwave_micro_client._evaluate_ising(linear, quad, state))


class MockSubmission(_QueryTest):
    """Test connecting and some related failure modes."""

    def test_submit_null_reply(self):
        """Get an error when the server's response is incomplete."""
        # con = mock.Mock()
        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.post = lambda a, _: choose_reply(a, {'problems/': ''})
            solver = dwave_micro_client.Solver(con, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            #
            with self.assertRaises(ValueError):
                results.samples

    def test_submit_ok_reply(self):
        """Handle a normal query and response."""
        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.post = lambda a, _: choose_reply(a, {
                'problems/': '[%s]' % complete_no_answer_reply('123', 'abc123')})
            con.session.get = lambda a: choose_reply(a, {'problems/123/': complete_reply('123', 'abc123')})
            solver = dwave_micro_client.Solver(con, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            #
            self._check(results, linear, quad, 100)

    def test_submit_error_reply(self):
        """Handle an error on problem submission."""
        error_body = 'An error message'
        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.post = lambda a, _: choose_reply(a, {
                'problems/': '[%s]' % error_reply('123', 'abc123', error_body)})
            solver = dwave_micro_client.Solver(con, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            #
            with self.assertRaises(dwave_micro_client.SolverFailureError):
                results.samples

    def test_submit_cancel_reply(self):
        """Handle a response for a canceled job."""
        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.post = lambda a, _: choose_reply(a, {'problems/': '[%s]' % cancel_reply('123', 'abc123')})
            solver = dwave_micro_client.Solver(con, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            #
            with self.assertRaises(dwave_micro_client.CanceledFutureError):
                results.samples

    def test_submit_continue_then_ok_reply(self):
        """Handle polling for a complete problem."""
        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.post = lambda a, _: choose_reply(a, {'problems/': '[%s]' % continue_reply('123', 'abc123')})
            con.session.get = lambda a: choose_reply(a, {
                'problems/?id=123': '[%s]' % complete_no_answer_reply('123', 'abc123'),
                'problems/123/': complete_reply('123', 'abc123')
            })
            solver = dwave_micro_client.Solver(con, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            #
            self._check(results, linear, quad, 100)

    def test_submit_continue_then_error_reply(self):
        """Handle polling for an error message."""
        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.post = lambda a, _: choose_reply(a, {'problems/': '[%s]' % continue_reply('123', 'abc123')})
            con.session.get = lambda a: choose_reply(a, {
                'problems/?id=123': '[%s]' % error_reply('123', 'abc123', "error message")})
            solver = dwave_micro_client.Solver(con, solver_data('abc123'))

            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}
            results = solver.sample_ising(linear, quad, num_reads=100)

            #
            with self.assertRaises(dwave_micro_client.SolverFailureError):
                self._check(results, linear, quad, 100)

    def test_submit_continue_then_ok_and_error_reply(self):
        """Handle polling for the status of multiple problems."""
        # Reduce the number of poll threads to 1 so that the system can be tested
        old_value = dwave_micro_client.Connection._POLL_THREAD_COUNT
        dwave_micro_client.Connection._POLL_THREAD_COUNT = 1
        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.get = lambda a: choose_reply(a, {
                # Wait until both problems are
                'problems/?id=1': '[%s]' % continue_reply('1', 'abc123'),
                'problems/?id=2': '[%s]' % continue_reply('2', 'abc123'),
                'problems/?id=2,1': '[' + complete_no_answer_reply('2', 'abc123') + ',' +
                                     error_reply('1', 'abc123', 'error') + ']',
                'problems/?id=1,2': '[' + error_reply('1', 'abc123', 'error') + ',' +
                                     complete_no_answer_reply('2', 'abc123') + ']',
                'problems/1/': error_reply('1', 'abc123', 'Error message'),
                'problems/2/': complete_reply('2', 'abc123')
            })

            def switch_post_reply(path, body):
                message = json.loads(body)
                if len(message) == 1:
                    con.session.post = lambda a, _: choose_reply(a, {
                        'problems/': '[%s]' % continue_reply('2', 'abc123')})
                    return choose_reply('', {'': '[%s]' % continue_reply('1', 'abc123')})
                else:
                    con.session.post = None
                    return choose_reply('', {
                        '': '[%s, %s]' % (continue_reply('1', 'abc123'), continue_reply('2', 'abc123'))
                    })

            con.session.post = lambda a, body: switch_post_reply(a, body)

            solver = dwave_micro_client.Solver(con, solver_data('abc123'))
            # Build a problem
            linear = {index: 1 for index in solver.nodes}
            quad = {key: -1 for key in solver.undirected_edges}

            results1 = solver.sample_ising(linear, quad, num_reads=100)
            results2 = solver.sample_ising(linear, quad, num_reads=100)

            #
            with self.assertRaises(dwave_micro_client.SolverFailureError):
                self._check(results1, linear, quad, 100)
            self._check(results2, linear, quad, 100)
        dwave_micro_client.Connection._POLL_THREAD_COUNT = old_value


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

        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()

            con.session.get = lambda a: choose_reply(a, {'problems/?id={}'.format(submission_id): reply_body})
            con.session.delete = DeleteEvent.handle

            solver = dwave_micro_client.Solver(con, solver_data('abc123'))
            future = solver.retrieve_problem(submission_id)
            future.cancel()

            try:
                self.assertTrue(future.id is not None)
                future.samples
                self.fail()
            except DeleteEvent as event:
                if event.url == 'problems/':
                    self.assertEqual(event.body, '["{}"]'.format(submission_id))
                else:
                    self.assertEqual(event.url, 'problems/{}/'.format(submission_id))

    def test_cancel_without_id(self):
        """Make sure the cancel method submits to the right endpoint.

        When cancel is called before the submission has returned the problem id.
        """
        submission_id = 'test-id'
        reply_body = '[%s]' % continue_reply(submission_id, 'solver')

        release_reply = threading.Event()

        with dwave_micro_client.Connection('', '') as con:
            con.session = mock.Mock()
            con.session.get = lambda a: choose_reply(a, {'problems/?id={}'.format(submission_id): reply_body})

            def post(a, _):
                release_reply.wait()
                return choose_reply(a, {'problems/'.format(submission_id): reply_body})
            con.session.post = post
            con.session.delete = DeleteEvent.handle

            solver = dwave_micro_client.Solver(con, solver_data('abc123'))
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
                if event.url == 'problems/':
                    self.assertEqual(event.body, '["{}"]'.format(submission_id))
                else:
                    self.assertEqual(event.url, 'problems/{}/'.format(submission_id))

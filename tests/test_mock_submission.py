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

"""Test problem submission against hard-coded replies with unittest.mock."""

import time
import json
import unittest
import threading
import warnings
import collections

from unittest import mock
from requests.structures import CaseInsensitiveDict
from requests.exceptions import HTTPError
from concurrent.futures import TimeoutError
from parameterized import parameterized

try:
    import dimod
except ImportError:
    dimod = None

from dwave.cloud.client import Client
from dwave.cloud.solver import Solver
from dwave.cloud.computation import Future
from dwave.cloud.utils import evaluate_ising, utcrel
from dwave.cloud.exceptions import (
    SolverFailureError, CanceledFutureError, SolverError,
    InvalidAPIResponseError)

from tests.api.mocks import StructuredSapiMockResponses


def choose_reply(path, replies, statuses=None, date=None):
    """Choose the right response based on the path and make a mock response."""

    if statuses is None:
        statuses = collections.defaultdict(lambda: iter([200]))

    if date is None:
        date = utcrel(0)

    if path in replies:
        response = mock.Mock(['text', 'json', 'raise_for_status', 'headers'])
        response.status_code = next(statuses[path])
        text = replies[path]
        if not isinstance(text, str):
            text = json.dumps(text)
        response.text = text
        response.json.side_effect = lambda: replies[path]
        response.headers = CaseInsensitiveDict({'Date': date.isoformat()})

        def raise_for_status():
            if not 200 <= response.status_code < 400:
                raise HTTPError(response.status_code)
        response.raise_for_status = raise_for_status

        def ok():
            try:
                response.raise_for_status()
            except HTTPError:
                return False
            return True
        ok_property = mock.PropertyMock(side_effect=ok)
        type(response).ok = ok_property

        return response
    else:
        raise NotImplementedError(path)


class _QueryTest(unittest.TestCase):
    def _check(self, results, linear, quad, offset=0, num_reads=1):
        # Did we get the right number of samples?
        self.assertEqual(num_reads, sum(results.num_occurrences))

        # verify .occurrences property still works, although is deprecated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(100, sum(results.occurrences))

        # Make sure energies are correct in raw results
        for energy, state in zip(results.energies, results.samples):
            self.assertEqual(energy, evaluate_ising(linear, quad, state, offset=offset))

        # skip sampleset test if dimod is not installed
        if not dimod:
            return

        # Make sure the sampleset matches raw results
        for record, energy, num_occurrences, state in \
                zip(results.sampleset.record,
                    results['energies'],
                    results['num_occurrences'],
                    results['solutions']):
            recalc_energy = evaluate_ising(linear, quad, state, offset=offset)
            self.assertEqual(energy, recalc_energy)
            self.assertEqual(energy, float(record.energy))
            self.assertEqual(num_occurrences, int(record.num_occurrences))
            self.assertEqual(state, list(record.sample))


@mock.patch('time.sleep', lambda *x: None)
class MockSubmission(_QueryTest):
    """Test connecting and some related failure modes."""

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    def test_submit_null_reply(self):
        """Get an error when the server's response is incomplete."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': ''})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                results = solver.sample_ising(linear, quadratic)

                with self.assertRaises(InvalidAPIResponseError):
                    results.samples

    def test_submit_ising_ok_reply(self):
        """Handle a normal query and response."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, **params)

                self._check(results, linear, quadratic, **params)

    @unittest.skipUnless(dimod, "dimod required for 'Solver.sample_bqm'")
    def test_submit_bqm_ising_ok_reply(self):
        """Handle a normal query and response."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                h, J = self.sapi.problem
                bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

                params = dict(num_reads=100)
                results = solver.sample_bqm(bqm, **params)

                self._check(results, h, J, **params)

    def test_submit_qubo_ok_reply(self):
        """Handle a normal query and response."""

        qubo_msg_diff = dict(type="qubo")
        qubo_answer_diff = {
            'energies': 'AAAAAAAAAAA=',
            'solutions': 'AA==',
            'active_variables': 'AAAAAAQAAAA='
        }

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=qubo_answer_diff, **qubo_msg_diff)})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                qubo = {(0, 0): 4.0, (0, 4): -4, (4, 4): 4.0}
                offset = -2.0
                params = dict(num_reads=100)

                results = solver.sample_qubo(qubo, offset, **params)

                # make sure energies are correct in raw results
                for energy, sample in zip(results.energies, results.samples):
                    self.assertEqual(energy, evaluate_ising({}, qubo, sample, offset=offset))

    @unittest.skipUnless(dimod, "dimod required for 'Solver.sample_bqm'")
    def test_submit_bqm_qubo_ok_reply(self):
        """Handle a normal query and response."""

        qubo_msg_diff = dict(type="qubo")
        qubo_answer_diff = {
            'energies': 'AAAAAAAAAAA=',
            'solutions': 'AA==',
            'active_variables': 'AAAAAAQAAAA='
        }

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=qubo_answer_diff, **qubo_msg_diff)})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                qubo = {(0, 0): 4.0, (0, 4): -4, (4, 4): 4.0}
                offset = -2.0
                params = dict(num_reads=100)

                results = solver.sample_qubo(qubo, offset, **params)

                # make sure energies are correct in raw results
                for energy, sample in zip(results.energies, results.samples):
                    self.assertEqual(energy, evaluate_ising({}, qubo, sample, offset=offset))

    def test_submit_error_reply(self):
        """Handle an error on problem submission."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.error_reply(error_message='An error message')]})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                results = solver.sample_ising(linear, quadratic)

                with self.assertRaises(SolverFailureError):
                    results.samples

    def test_submit_immediate_error_reply(self):
        """Handle an (obvious) error on problem submission."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.immediate_error_reply(
                    code=400, msg="Missing parameter 'num_reads' in problem JSON")]})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                results = solver.sample_ising(linear, quadratic)

                # waiting should be canceled on exception (i.e. NOT timeout)
                self.assertTrue(results.wait(timeout=1))
                self.assertIsNone(results.wait_id(timeout=1))

                # resolving result should raise exception
                with self.assertRaises(SolverFailureError):
                    results.samples

    def test_submit_cancel_reply(self):
        """Handle a response for a canceled job."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.cancel_reply()]})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                results = solver.sample_ising(linear, quadratic)

                with self.assertRaises(CanceledFutureError):
                    results.samples

    def test_answer_load_error(self):
        """Answer load error is propagated as exception."""

        error_code = 404
        error_message = 'Problem not found'

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda path, _: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path: choose_reply(
                path, replies={
                    'problems/123/': error_message
                }, statuses={
                    'problems/123/': iter([error_code])
                })
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                future = solver.sample_ising(linear, quadratic)

                with self.assertRaises(SolverError) as exc:
                    future.result()

                self.assertEqual(str(exc.exception), error_message)

    def test_submit_continue_then_ok_reply(self):
        """Handle polling for a complete problem."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.continue_reply(id='123')]
            })
            session.get = lambda a: choose_reply(a, {
                'problems/?id=123': [self.sapi.complete_no_answer_reply(id='123')],
                'problems/123/': self.sapi.complete_reply(id='123')
            })
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, **params)

                self._check(results, linear, quadratic, **params)

    def test_submit_continue_then_error_reply(self):
        """Handle polling for an error message."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.continue_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/?id=123': [self.sapi.error_reply(id='123')]})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, **params)

                with self.assertRaises(SolverFailureError):
                    self._check(results, linear, quadratic, **params)

    def test_submit_continue_then_ok_and_error_reply(self):
        """Handle polling for the status of multiple problems."""

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.Mock()

            # on first status poll, return pending for both problems
            # on second status poll, return error for first problem and complete for second
            def continue_then_complete(path, state={'count': 0}):
                state['count'] += 1
                if state['count'] < 2:
                    return choose_reply(path, {
                        'problems/?id=1': [self.sapi.continue_reply(id='1')],
                        'problems/?id=2': [self.sapi.continue_reply(id='2')],
                        'problems/1/': self.sapi.continue_reply(id='1'),
                        'problems/2/': self.sapi.continue_reply(id='2'),
                        'problems/?id=1,2': [self.sapi.continue_reply(id='1'),
                                             self.sapi.continue_reply(id='2')],
                        'problems/?id=2,1': [self.sapi.continue_reply(id='2'),
                                             self.sapi.continue_reply(id='1')]
                    })
                else:
                    return choose_reply(path, {
                        'problems/?id=1': [self.sapi.error_reply(id='1')],
                        'problems/?id=2': [self.sapi.complete_no_answer_reply(id='2')],
                        'problems/1/': self.sapi.error_reply(id='1'),
                        'problems/2/': self.sapi.complete_reply(id='2'),
                        'problems/?id=1,2': [self.sapi.error_reply(id='1'),
                                             self.sapi.complete_no_answer_reply(id='2')],
                        'problems/?id=2,1': [self.sapi.complete_no_answer_reply(id='2'),
                                             self.sapi.error_reply(id='1')]
                    })

            def accept_problems_with_continue_reply(path, body, ids=iter('12')):
                problems = json.loads(body)
                return choose_reply(path, {
                    'problems/': [self.sapi.continue_reply(id=next(ids)) for _ in problems]
                })

            session.get = continue_then_complete
            session.post = accept_problems_with_continue_reply

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)

                results1 = solver.sample_ising(linear, quadratic, **params)
                results2 = solver.sample_ising(linear, quadratic, **params)

                with self.assertRaises(SolverFailureError):
                    self._check(results1, linear, quadratic, **params)
                self._check(results2, linear, quadratic, **params)

    def test_exponential_backoff_polling(self):
        "After each poll, back-off should double"

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.Mock()

            # on submit, return status pending
            session.post = lambda path, _: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='123')]
            })

            # on first and second status poll, return pending
            # on third status poll, return completed
            def continue_then_complete(path, state={'count': 0}):
                state['count'] += 1
                if state['count'] < 3:
                    return choose_reply(path, {
                        'problems/?id=123': [self.sapi.continue_reply(id='123')],
                        'problems/123/': self.sapi.continue_reply(id='123')
                    })
                else:
                    return choose_reply(path, {
                        'problems/?id=123': [self.sapi.complete_no_answer_reply(id='123')],
                        'problems/123/': self.sapi.complete_reply(id='123')
                    })

            session.get = continue_then_complete

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                future = solver.sample_qubo({})
                future.result()

                # after third poll, back-off interval should be 4 x initial back-off
                self.assertAlmostEqual(
                    future._poll_backoff,
                    client.poll_backoff_min * client.poll_backoff_base**2)

    def test_immediate_polling(self):
        "First poll happens with minimal delay"

        # each thread can have its instance of a session because
        # responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda path, _: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='1')]
            })
            session.get = lambda path: choose_reply(path, {
                'problems/?id=1': [self.sapi.complete_no_answer_reply(id='1')],
                'problems/1/': self.sapi.complete_reply(id='1')
            })
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                def assert_no_delay(s):
                    s and self.assertTrue(
                        abs(s - client.poll_backoff_min) < client.DEFAULTS['poll_backoff_min'])

                with mock.patch('time.sleep', assert_no_delay):
                    future = solver.sample_qubo({})
                    future.result()

    def test_immediate_polling_with_local_clock_unsynced(self):
        """First poll happens with minimal delay if local clock is way off from
        the remote/server clock."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            badnow = utcrel(100)
            session = mock.Mock()
            session.post = lambda path, _: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='1')]
            }, date=badnow)
            session.get = lambda path: choose_reply(path, {
                'problems/?id=1': [self.sapi.complete_no_answer_reply(id='1')],
                'problems/1/': self.sapi.complete_reply(id='1')
            }, date=badnow)
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                def assert_no_delay(s):
                    s and self.assertTrue(
                        abs(s - client.poll_backoff_min) < client.DEFAULTS['poll_backoff_min'])

                with mock.patch('time.sleep', assert_no_delay):
                    future = solver.sample_qubo({})
                    future.result()

    def test_polling_recovery_after_5xx(self):
        "Polling shouldn't be aborted on 5xx responses."

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.Mock()

            # on submit, return status pending
            session.post = lambda path, _: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='123')]
            })

            # on first and second status poll, fail with 503 and 504
            # on third status poll, return completed
            statuses = iter([503, 504])
            def continue_then_complete(path, state={'count': 0}):
                state['count'] += 1
                if state['count'] < 3:
                    return choose_reply(path, replies={
                        'problems/?id=123': [self.sapi.continue_reply(id='123')],
                        'problems/123/': self.sapi.continue_reply(id='123')
                    }, statuses={
                        'problems/?id=123': statuses,
                        'problems/123/': statuses
                    })
                else:
                    return choose_reply(path, {
                        'problems/?id=123': [self.sapi.complete_no_answer_reply(id='123')],
                        'problems/123/': self.sapi.complete_reply(id='123')
                    })

            session.get = continue_then_complete

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                future = solver.sample_qubo({})
                future.result()

                # after third poll, back-off interval should be 4 x initial back-off
                self.assertAlmostEqual(
                    future._poll_backoff,
                    client.poll_backoff_min * client.poll_backoff_base**2)


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


@mock.patch('time.sleep', lambda *x: None)
class MockCancel(unittest.TestCase):
    """Make sure cancel works at the two points in the process where it should."""

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    def test_cancel_with_id(self):
        """Make sure the cancel method submits to the right endpoint.

        When cancel is called after the submission is finished.
        """
        submission_id = 'test-id'

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            reply_body = [self.sapi.continue_reply(id=submission_id, solver='solver')]
            session.get = lambda a: choose_reply(a, {
                'problems/?id={}'.format(submission_id): reply_body})
            session.delete = DeleteEvent.handle
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)
                future = solver._retrieve_problem(submission_id)
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
        release_reply = threading.Event()

        # each thread can have its instance of a session because
        # we use a global lock (event) in the mocked responses
        def create_mock_session(client):
            reply_body = [self.sapi.continue_reply(id=submission_id)]

            session = mock.Mock()
            session.get = lambda a: choose_reply(a, {
                'problems/?id={}'.format(submission_id): reply_body})

            def post(a, _):
                release_reply.wait()
                return choose_reply(a, {'problems/': reply_body})

            session.post = post
            session.delete = DeleteEvent.handle

            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem

                future = solver.sample_ising(linear, quadratic)
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


class TestComputationID(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    def test_id_getter_setter(self):
        """Future.get_id/get_id works in isolation as expected."""

        f = Future(solver=None, id_=None)

        # f.id should be None
        self.assertIsNone(f.id)
        with self.assertRaises(TimeoutError):
            f.wait_id(timeout=1)

        # set it
        submission_id = 'test-id'
        f.id = submission_id

        # validate it's available
        self.assertEqual(f.wait_id(), submission_id)
        self.assertEqual(f.wait_id(timeout=1), submission_id)
        self.assertEqual(f.id, submission_id)

    def test_id_integration(self):
        """Problem ID getter blocks correctly when ID set by the client."""

        submission_id = 'test-id'
        solver_name = 'solver-id'
        release_reply = threading.Event()

        # each thread can have its instance of a session because
        # we use a global lock (event) in the mocked responses
        def create_mock_session(client):
            session = mock.Mock()

            # delayed submit; emulates waiting in queue
            def post(path, _):
                release_reply.wait()
                reply_body = self.sapi.complete_reply(id=submission_id, solver=solver_name)
                return choose_reply(path, {'problems/': [reply_body]})

            session.post = post

            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem

                future = solver.sample_ising(linear, quadratic)

                # initially, the id is not available
                with self.assertRaises(TimeoutError):
                    future.wait_id(timeout=1)

                # release the mocked sapi reply with the id
                release_reply.set()

                # verify the id is now available
                self.assertEqual(future.wait_id(), submission_id)

    @unittest.skipUnless(dimod, "dimod required for 'Solver.sample_bqm'")
    def test_sampleset_id(self):
        f = Future(solver=None, id_=None)

        # f.id should be None
        self.assertIsNone(f.id)
        with self.assertRaises(TimeoutError):
            f.sampleset.wait_id(timeout=1)

        # set it
        submission_id = 'test-id'
        f.id = submission_id

        # validate it's available
        self.assertEqual(f.sampleset.wait_id(), submission_id)
        self.assertEqual(f.sampleset.wait_id(timeout=1), submission_id)

@mock.patch('time.sleep', lambda *x: None)
class TestOffsetHandling(_QueryTest):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    def test_submit_offset_answer_includes_it(self):
        """Handle a normal query with offset and response that includes it."""

        # ising problem energy offset
        offset = 3

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(
                    id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=dict(offset=offset))})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, offset, **params)

                self._check(results, linear, quadratic, offset=offset, **params)

    def test_submit_offset_answer_does_not_include_it(self):
        """Handle a normal query with offset and response that doesn't include it."""

        # ising problem energy offset
        offset = 3

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, offset, **params)

                # although SAPI response doesn't include offset, Future should patch it on-the-fly
                self._check(results, linear, quadratic, offset=offset, **params)

    def test_submit_offset_wrong_offset_in_answer(self):
        """Energy levels don't match because offset in answer is respected, even if wrong"""

        # ising problem energy offset
        offset = 3
        answer_offset = 2 * offset      # make it wrong

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=dict(offset=answer_offset))})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, offset, **params)

                # since SAPI response includes offset, Future shouldn't patch it;
                # but because the offset in answer is wrong, energies are off
                with self.assertRaises(AssertionError):
                    self._check(results, linear, quadratic, offset=offset, **params)


@mock.patch('time.sleep', lambda *x: None)
class TestComputationDeprecations(_QueryTest):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    def test_deprecations(self):
        """Proper deprecation warnings are raised."""

        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda a, _: choose_reply(a, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda a: choose_reply(a, {
                'problems/123/': self.sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, **params)

                # aliased keys are deprecated in 0.8.0
                with self.assertWarns(DeprecationWarning):
                    results['samples']
                with self.assertWarns(DeprecationWarning):
                    results['occurrences']

                # .occurrences is deprecated in 0.8.0, scheduled for removal in 0.10.0+
                with self.assertWarns(DeprecationWarning):
                    results.occurrences


class TestProblemLabel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    class PrimaryAssertionSatisfied(Exception):
        """Raised by `on_submit_label_verifier` to signal correct label."""

    def on_submit_label_verifier(self, expected_label):
        """Factory for mock Client._submit() that will verify existence, and
        optionally validate label value.
        """

        # replacement for Client._submit()
        def _submit(client, body_data, computation):
            body = json.loads(body_data.result())

            if 'label' not in body:
                if expected_label is None:
                    raise TestProblemLabel.PrimaryAssertionSatisfied
                else:
                    raise AssertionError("label field missing")

            label = body['label']
            if label != expected_label:
                raise AssertionError(
                    "unexpected label value: {!r} != {!r}".format(label, expected_label))

            raise TestProblemLabel.PrimaryAssertionSatisfied

        return _submit

    def generate_sample_problems(self, solver):
        linear, quadratic = self.sapi.problem

        # test sample_{ising,qubo,bqm}
        problems = [("sample_ising", (linear, quadratic)),
                    ("sample_qubo", (quadratic,))]
        if dimod:
            bqm = dimod.BQM.from_ising(linear, quadratic)
            problems.append(("sample_bqm", (bqm,)))

        return problems

    @parameterized.expand([
        ("undefined", None),
        ("empty", ""),
        ("string", "text label")
    ])
    @mock.patch.object(Client, 'create_session', lambda client: mock.Mock())
    def test_label_is_sent(self, name, label):
        """Problem label is set on problem submit."""

        with Client('endpoint', 'token') as client:
            solver = Solver(client, self.sapi.solver.data)
            problems = self.generate_sample_problems(solver)

            for method_name, problem_args in problems:
                with self.subTest(method_name=method_name):
                    sample = getattr(solver, method_name)

                    with mock.patch.object(Client, '_submit', self.on_submit_label_verifier(label)):

                        with self.assertRaises(self.PrimaryAssertionSatisfied):
                            sample(*problem_args, label=label).result()

    @parameterized.expand([
        ("undefined", None),
        ("empty", ""),
        ("string", "text label")
    ])
    def test_label_is_received(self, name, label):
        """Problem label is set from response in result/sampleset."""

        def make_session_generator(label):
            def create_mock_session(client):
                session = mock.Mock()
                session.post = lambda a, _: choose_reply(a, {
                    'problems/': [self.sapi.complete_no_answer_reply(id='123', label=None)]})
                session.get = lambda a: choose_reply(a, {
                    'problems/123/': self.sapi.complete_reply(id='123', label=label)})
                return session
            return create_mock_session

        with mock.patch.object(Client, 'create_session', make_session_generator(label)):
            with Client('endpoint', 'token') as client:
                solver = Solver(client, self.sapi.solver.data)
                problems = self.generate_sample_problems(solver)

                for method_name, problem_args in problems:
                    with self.subTest(method_name=method_name):
                        sample = getattr(solver, method_name)

                        future = sample(*problem_args, label=label)
                        future.result()     # ensure future is resolved

                        self.assertEqual(future.label, label)

                        # sampleset will only be available if dimod is installed
                        if dimod:
                            info = future.sampleset.info
                            self.assertEqual(info.get('problem_label'), label)

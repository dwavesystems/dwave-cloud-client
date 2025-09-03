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

import threading
import time
import unittest
import weakref
import zlib
from concurrent.futures import TimeoutError, ThreadPoolExecutor
from typing import Any
from unittest import mock
from urllib.parse import urlencode

import numpy
import orjson
import requests_mock
from parameterized import parameterized

try:
    import dimod
except ImportError:
    dimod = None

from dwave.cloud.client import Client
from dwave.cloud.computation import Future
from dwave.cloud.exceptions import (
    SolverFailureError, CanceledFutureError, SolverError,
    InvalidAPIResponseError, UseAfterCloseError)
from dwave.cloud.solver import Solver
from dwave.cloud.utils.qubo import evaluate_ising
from dwave.cloud.utils.time import utcrel

from tests.api.mocks import StructuredSapiMockResponses, choose_reply


def fix_status_paths(replies: dict[str, Any], **params) -> dict[str, Any]:
    """Append URL params to problem status paths."""

    def fix(path, **params):
        query = urlencode(params)
        if 'problems/?id=' in path and query:
            return f"{path}&{query}"
        return path

    return {fix(path, **params): reply for path, reply in replies.items()}


class _QueryTest:
    def _check(self, results, linear, quad, offset=0, num_reads=1):
        # Did we get the right number of samples?
        self.assertEqual(num_reads, sum(results.num_occurrences))

        # verify num_occurrences sum corresponds to num_reads
        self.assertEqual(100, sum(results.num_occurrences))

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


class MockSubmissionBase(_QueryTest):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()
        cls.config = dict(
            endpoint='endpoint',
            token='token',
        )
        if hasattr(cls, 'poll_strategy'):
            cls.config['poll_strategy'] = cls.poll_strategy


@mock.patch('time.sleep', lambda *x: None)
class MockSubmissionBaseTests(MockSubmissionBase):
    """Test connecting and some related failure modes."""

    def test_submit_null_reply(self):
        """Get an error when the server's response is incomplete."""

        # each thread can have its instance of a session because
        # the mocked responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': ''})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=qubo_answer_diff, **qubo_msg_diff)})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=qubo_answer_diff, **qubo_msg_diff)})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.error_reply(error_message='An error message')]})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.immediate_error_reply(
                    code=400, msg="Missing parameter 'num_reads' in problem JSON")]})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.cancel_reply()]})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path: choose_reply(
                path, replies={
                    'problems/123/': error_message
                }, statuses={
                    'problems/123/': iter([error_code])
                })
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='123')]
            })
            session.get = lambda a: choose_reply(a, fix_status_paths({
                'problems/?id=123': [self.sapi.complete_no_answer_reply(id='123')],
                'problems/123/': self.sapi.complete_reply(id='123')
            }, **self.poll_params))
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='123')]})
            session.get = lambda a: choose_reply(a, fix_status_paths({
                'problems/?id=123': [self.sapi.error_reply(id='123')]
            }, **self.poll_params))
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
                    return choose_reply(path, fix_status_paths({
                        'problems/?id=1': [self.sapi.continue_reply(id='1')],
                        'problems/?id=2': [self.sapi.continue_reply(id='2')],
                        'problems/1/': self.sapi.continue_reply(id='1'),
                        'problems/2/': self.sapi.continue_reply(id='2'),
                        'problems/?id=1,2': [self.sapi.continue_reply(id='1'),
                                             self.sapi.continue_reply(id='2')],
                        'problems/?id=2,1': [self.sapi.continue_reply(id='2'),
                                             self.sapi.continue_reply(id='1')]
                    }, **self.poll_params))
                else:
                    return choose_reply(path, fix_status_paths({
                        'problems/?id=1': [self.sapi.error_reply(id='1')],
                        'problems/?id=2': [self.sapi.complete_no_answer_reply(id='2')],
                        'problems/1/': self.sapi.error_reply(id='1'),
                        'problems/2/': self.sapi.complete_reply(id='2'),
                        'problems/?id=1,2': [self.sapi.error_reply(id='1'),
                                             self.sapi.complete_no_answer_reply(id='2')],
                        'problems/?id=2,1': [self.sapi.complete_no_answer_reply(id='2'),
                                             self.sapi.error_reply(id='1')]
                    }, **self.poll_params))

            def accept_problems_with_continue_reply(path, data=None, headers=None, ids=iter('12')):
                if headers is None:
                    headers = {}
                encoding = headers.get('Content-Encoding', 'identity').lower()
                if encoding == 'deflate':
                    data = zlib.decompress(data)
                problems = orjson.loads(data)
                return choose_reply(path, {
                    'problems/': [self.sapi.continue_reply(id=next(ids)) for _ in problems]
                })

            session.get = continue_then_complete
            session.post = accept_problems_with_continue_reply

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)

                results1 = solver.sample_ising(linear, quadratic, **params)
                results2 = solver.sample_ising(linear, quadratic, **params)

                with self.assertRaises(SolverFailureError):
                    self._check(results1, linear, quadratic, **params)
                self._check(results2, linear, quadratic, **params)


class TestAPIVersionCheck(MockSubmissionBase, unittest.TestCase):

    def setUp(self):
        self.endpoint = 'http://test.com/path'
        self.problem_type = 'application/vnd.dwave.sapi.problem+json'
        self.problems_type = 'application/vnd.dwave.sapi.problems+json'
        self.right_version = '2.5.0'
        self.wrong_version = '1.0.0'

        self.pid = '123'
        self.sapi = StructuredSapiMockResponses(problem_id=self.pid)
        self.config = dict(
            endpoint=self.endpoint,
            token='token',
        )

        self.mocker = requests_mock.Mocker()
        self.mocker.get(requests_mock.ANY, status_code=404)
        self.mocker.start()

    def tearDown(self):
        self.mocker.stop()

    def test_submit(self):
        # problem submit endpoint returns unsupported version
        self.mocker.post(f"{self.endpoint}/problems/", json=[self.sapi.complete_no_answer_reply(id=self.pid)],
                         headers={'Content-Type': f'{self.problems_type}; version={self.wrong_version}'})

        with Client(**self.config) as client:
            solver = Solver(client, self.sapi.solver.data)

            with self.assertRaisesRegex(InvalidAPIResponseError, "Try upgrading"):
                response = solver.sample_ising(*self.sapi.problem)
                response.result()

    def test_cancel(self):
        # problem cancel endpoint returns unsupported version
        self.mocker.post(f"{self.endpoint}/problems/", json=[self.sapi.continue_reply()],
                         headers={'Content-Type': f'{self.problems_type}; version={self.right_version}'})
        self.mocker.get(f"{self.endpoint}/problems/?id={self.pid}", json=[self.sapi.continue_reply()],
                        headers={'Content-Type': f'{self.problems_type}; version={self.right_version}'})
        self.mocker.delete(f"{self.endpoint}/problems/", json=[self.sapi.cancel_reply()],
                           headers={'Content-Type': f'{self.problems_type}; version={self.wrong_version}'})

        with Client(**self.config) as client:
            solver = Solver(client, self.sapi.solver.data)

            with self.assertRaisesRegex(InvalidAPIResponseError, "Try upgrading"):
                response = solver.sample_ising(*self.sapi.problem)
                response.wait_id()      # without id, sapi cancel is not called
                response.cancel()
                response.result()

    def test_poll(self):
        # problem status endpoint returns unsupported version
        self.mocker.post(f"{self.endpoint}/problems/", json=[self.sapi.continue_reply(id=self.pid)],
                         headers={'Content-Type': f'{self.problems_type}; version={self.right_version}'})
        self.mocker.get(f"{self.endpoint}/problems/?id={self.pid}", json=[self.sapi.complete_no_answer_reply(id=self.pid)],
                        headers={'Content-Type': f'{self.problems_type}; version={self.wrong_version}'})

        with Client(**self.config) as client:
            solver = Solver(client, self.sapi.solver.data)

            with self.assertRaisesRegex(InvalidAPIResponseError, "Try upgrading"):
                response = solver.sample_ising(*self.sapi.problem)
                response.result()

    def test_answer_load(self):
        # problem answer endpoint returns unsupported version
        self.mocker.post(f"{self.endpoint}/problems/", json=[self.sapi.complete_no_answer_reply(id=self.pid)],
                         headers={'Content-Type': f'{self.problems_type}; version={self.right_version}'})
        self.mocker.get(f"{self.endpoint}/problems/{self.pid}/", json=[self.sapi.complete_reply(id=self.pid)],
                        headers={'Content-Type': f'{self.problem_type}; version={self.wrong_version}'})

        with Client(**self.config) as client:
            solver = Solver(client, self.sapi.solver.data)

            with self.assertRaisesRegex(InvalidAPIResponseError, "Try upgrading"):
                response = solver.sample_ising(*self.sapi.problem)
                response.result()


class TestUploadCompression(MockSubmissionBase, unittest.TestCase):
    """Verify `compress_qpu_problem_data` config directive is respected, i.e.
    QPU problem data is compressed on upload, and Content-Encoding is correctly
    set."""

    @parameterized.expand([
        (True, ),
        (False, ),
    ])
    def test_compression_on_upload(self, compress):
        # make sure POST data is compressed if `compress_qpu_problem_data` config
        # option is set

        class Invalid(Exception):
            pass

        def create_mock_session(client):
            session = mock.Mock()

            def post(path, **kwargs):
                data = kwargs.pop('data')
                headers = kwargs.pop('headers', {})
                encoding = headers.get('Content-Encoding', 'identity')

                if compress and encoding == 'deflate':
                    data = zlib.decompress(data)

                if compress and encoding != 'deflate':
                    raise Invalid

                problem = orjson.loads(data)
                if problem[0]['data'] != self.sapi.problem_data():
                    raise Invalid

                return choose_reply(path, {
                    'problems/': [self.sapi.complete_no_answer_reply(id='123')]})

            session.post = post
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(id='123')})

            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(compress_qpu_problem_data=compress, **self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, **params)

                self._check(results, linear, quadratic, **params)


class MockSubmissionWithShortPolling(MockSubmissionBaseTests,
                                     unittest.TestCase):

    poll_strategy = "backoff"
    poll_params = {}

    def test_exponential_backoff_polling(self):
        "After each poll, back-off should double"

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.Mock()

            # on submit, return status pending
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='123')]
            })

            # on first and second status poll, return pending
            # on third status poll, return completed
            def continue_then_complete(path, state={'count': 0}):
                state['count'] += 1
                if state['count'] < 3:
                    return choose_reply(path, fix_status_paths({
                        'problems/?id=123': [self.sapi.continue_reply(id='123')],
                        'problems/123/': self.sapi.continue_reply(id='123')
                    }, **self.poll_params))
                else:
                    return choose_reply(path, fix_status_paths({
                        'problems/?id=123': [self.sapi.complete_no_answer_reply(id='123')],
                        'problems/123/': self.sapi.complete_reply(id='123')
                    }, **self.poll_params))

            session.get = continue_then_complete

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                future = solver.sample_qubo({})
                future.result()

                # after third poll, back-off interval should be 4 x initial back-off
                schedule = client.config.polling_schedule
                self.assertAlmostEqual(
                    future._poll_backoff,
                    schedule.backoff_min * schedule.backoff_base**2)

    def test_immediate_polling(self):
        "First poll happens with minimal delay"

        # each thread can have its instance of a session because
        # responses are stateless
        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='1')]
            })
            session.get = lambda path: choose_reply(path, fix_status_paths({
                'problems/?id=1': [self.sapi.complete_no_answer_reply(id='1')],
                'problems/1/': self.sapi.complete_reply(id='1')
            }, **self.poll_params))
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                def assert_no_delay(s):
                    s = s or 0
                    delay = abs(s - client.config.polling_schedule.backoff_min)
                    self.assertLess(delay, client.DEFAULTS['poll_backoff_min'])

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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='1')]
            }, date=badnow)
            session.get = lambda path: choose_reply(path, fix_status_paths({
                'problems/?id=1': [self.sapi.complete_no_answer_reply(id='1')],
                'problems/1/': self.sapi.complete_reply(id='1')
            }, **self.poll_params), date=badnow)
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                def assert_no_delay(s):
                    s = s or 0
                    delay = abs(s - client.config.polling_schedule.backoff_min)
                    self.assertLess(delay, client.DEFAULTS['poll_backoff_min'])

                with mock.patch('time.sleep', assert_no_delay):
                    future = solver.sample_qubo({})
                    future.result()

    def test_polling_recovery_after_5xx(self):
        "Polling shouldn't be aborted on 5xx responses."

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.Mock()

            # on submit, return status pending
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id='123')]
            })

            # on first and second status poll, fail with 503 and 504
            # on third status poll, return completed
            statuses = iter([503, 504])
            def continue_then_complete(path, state={'count': 0}):
                state['count'] += 1
                if state['count'] < 3:
                    return choose_reply(path, replies=fix_status_paths({
                        'problems/?id=123': [self.sapi.continue_reply(id='123')],
                        'problems/123/': self.sapi.continue_reply(id='123')
                    }, **self.poll_params), statuses=fix_status_paths({
                        'problems/?id=123': statuses,
                        'problems/123/': statuses
                    }, **self.poll_params))
                else:
                    return choose_reply(path, fix_status_paths({
                        'problems/?id=123': [self.sapi.complete_no_answer_reply(id='123')],
                        'problems/123/': self.sapi.complete_reply(id='123')
                    }, **self.poll_params))

            session.get = continue_then_complete

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                future = solver.sample_qubo({})
                future.result()

                # after third poll, back-off interval should be 4 x initial back-off
                schedule = client.config.polling_schedule
                self.assertAlmostEqual(
                    future._poll_backoff,
                    schedule.backoff_min * schedule.backoff_base**2)


class MockSubmissionWithLongPolling(MockSubmissionBaseTests,
                                    unittest.TestCase):

    poll_strategy = "long-polling"
    poll_params = dict(timeout=Client.DEFAULTS['poll_wait_time'])


class DeleteEvent(Exception):
    """Throws exception when mocked client submits an HTTP DELETE request."""

    def __init__(self, url, body):
        """Return the URL of the request with the exception for test verification."""
        self.url = url
        self.body = body

    @staticmethod
    def handle(path, **kwargs):
        """Callback useable to mock a delete request."""
        data = kwargs.get('data') or orjson.dumps(kwargs.get('json'))
        if isinstance(data, bytes):
            data = data.decode('utf8')
        raise DeleteEvent(path, data)


@mock.patch('time.sleep', lambda *x: None)
class MockCancel(MockSubmissionBase, unittest.TestCase):
    """Make sure cancel works at the two points in the process where it should."""

    poll_strategy = "long-polling"
    poll_params = dict(timeout=Client.DEFAULTS['poll_wait_time'])

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
            session.get = lambda path, **kwargs: choose_reply(path, fix_status_paths({
                f'problems/?id={submission_id}': reply_body
            }, **self.poll_params))
            session.delete = DeleteEvent.handle
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)
                future = Future(solver, submission_id)
                client._poll(future)
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
            session.get = lambda path, **kwargs: choose_reply(path, fix_status_paths({
                f'problems/?id={submission_id}': reply_body
            }, **self.poll_params))

            def post(a, **kwargs):
                release_reply.wait()
                return choose_reply(a, {'problems/': reply_body})

            session.post = post
            session.delete = DeleteEvent.handle

            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
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
            def post(path, **kwargs):
                release_reply.wait()
                reply_body = self.sapi.complete_reply(id=submission_id, solver=solver_name)
                return choose_reply(path, {'problems/': [reply_body]})

            session.post = post

            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(endpoint='endpoint', token='token') as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(
                    id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=dict(offset=offset))})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(endpoint='endpoint', token='token') as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(endpoint='endpoint', token='token') as client:
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
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': self.sapi.complete_reply(
                    id='123', answer_patch=dict(offset=answer_offset))})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(endpoint='endpoint', token='token') as client:
                solver = Solver(client, self.sapi.solver.data)

                linear, quadratic = self.sapi.problem
                params = dict(num_reads=100)
                results = solver.sample_ising(linear, quadratic, offset, **params)

                # since SAPI response includes offset, Future shouldn't patch it;
                # but because the offset in answer is wrong, energies are off
                with self.assertRaises(AssertionError):
                    self._check(results, linear, quadratic, offset=offset, **params)


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
            body = orjson.loads(body_data.result())

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

        with Client(endpoint='endpoint', token='token') as client:
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
                session.post = lambda path, **kwargs: choose_reply(path, {
                    'problems/': [self.sapi.complete_no_answer_reply(id='123', label=None)]})
                session.get = lambda path, **kwargs: choose_reply(path, {
                    'problems/123/': self.sapi.complete_reply(id='123', label=label)})
                return session
            return create_mock_session

        with mock.patch.object(Client, 'create_session', make_session_generator(label)):
            with Client(endpoint='endpoint', token='token') as client:
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


@unittest.skipUnless(dimod, "dimod required for sampleset tests")
class TestComputationSamplesetCaching(unittest.TestCase):

    def test_sampleset_ref(self):
        # sampleset is constructed and cached from .wait_sampleset()

        sapi = StructuredSapiMockResponses()

        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': sapi.complete_reply(id='123')})
            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(endpoint='endpoint', token='token') as client:
                solver = Solver(client, sapi.solver.data)
                response = solver.sample_ising(*sapi.problem)

                # sampleset is initially unset
                self.assertIsNone(response._sampleset)

                sampleset = response.wait_sampleset()

                # sampleset constructed and available via weakref
                self.assertIsInstance(response._sampleset(), dimod.SampleSet)
                self.assertEqual(response._sampleset(), sampleset)

                # sampleset available as property
                self.assertEqual(response.sampleset, sampleset)

                # other shorthand properties are using sampleset
                numpy.testing.assert_array_equal(response.energies, sampleset.record.energy)
                numpy.testing.assert_array_equal(response.num_occurrences, sampleset.record.num_occurrences)
                numpy.testing.assert_array_equal(response.samples, sampleset.record.sample)
                numpy.testing.assert_array_equal(response.variables, sampleset.variables)

                # verify gc
                del sampleset
                self.assertIsNone(response._sampleset())


@mock.patch.object(Client, 'create_session', lambda _: mock.Mock())
class TestClientClose(MockSubmissionBase, unittest.TestCase):
    # exception is raised when client is used after it's been closed

    def test_submit_fails(self):
        with Client(**self.config) as client:
            solver = Solver(client, self.sapi.solver.data)

        with self.subTest('problem submit fails'):
            with self.assertRaises(UseAfterCloseError):
                solver.sample_ising(*self.sapi.problem)

        with self.subTest('problem data multipart upload fails'):
            with self.assertRaises(UseAfterCloseError):
                client.upload_problem_encoded('mock')

    def test_retrieve_fails(self):
        with Client(**self.config) as client:
            solver = Solver(client, self.sapi.solver.data)

        with self.subTest('problem status poll fails'):
            with self.assertRaises(UseAfterCloseError):
                future = Future(solver, id_='mock-id')
                client._poll(future)

        with self.subTest('qpu answer load fails'):
            with self.assertRaises(UseAfterCloseError):
                client.retrieve_answer('problem-id')

        with self.subTest('binary answer download fails'):
            with self.assertRaises(UseAfterCloseError):
                solver._download_binary_ref(auth_method='mock', url='mock')

    def test_cancel_fails(self):
        with Client(**self.config) as client:
            ...

        with self.assertRaises(UseAfterCloseError):
            client._cancel('future-id', 'future')

    def test_ref_cleanup(self):
        with Client(**self.config) as client:
            ref = weakref.ref(client)
            solver = Solver(client, self.sapi.solver.data)

        del client

        self.assertIsNone(ref())

    def test_solvers_session_access_fails(self):
        with Client(**self.config) as client:
            self.assertIsNotNone(client.solvers_session)

        # session is closed and disabled
        with self.assertRaises(UseAfterCloseError):
            client.solvers_session

        # hence, get_solver fails as well
        with self.assertRaises(UseAfterCloseError):
            client.get_solver()


@mock.patch('time.sleep', lambda *x: None)
class TestClientUseWhileClosing(MockSubmissionBase, unittest.TestCase):
    # make sure client can't be used even while closing (not just after close)

    poll_strategy = "long-polling"
    poll_params = dict(timeout=Client.DEFAULTS['poll_wait_time'])

    def test_submit_while_closing(self):
        submission_id = 'test-id'
        submission_status = 'PENDING'
        release_status = threading.Event()

        def create_mock_session(client):
            reply_body = [self.sapi.continue_reply(id=submission_id, status=submission_status)]

            session = mock.Mock()

            # make sure polling for status stalls until we set `release_status`
            def get(path, **kwargs):
                release_status.wait()
                return choose_reply(path, fix_status_paths({
                    f'problems/?id={submission_id}': reply_body}, **self.poll_params))

            session.get = get
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': reply_body})

            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                future = solver.sample_ising(*self.sapi.problem)

                # check the first job is in `pending`
                future.wait_id()
                self.assertEqual(future.id, submission_id)
                self.assertEqual(future.remote_status, submission_status)

                # now close the client, but since client.close() is blocking,
                # do it in a background thread
                t = threading.Thread(target=lambda: client.close(wait=False))
                t.start()

                # try submitting another job while the client is closing
                with self.assertRaises(UseAfterCloseError):
                    solver.sample_ising(*self.sapi.problem)

                # verify client.close is still running
                t.join(timeout=0.1)
                self.assertTrue(t.is_alive())

                # release the first job
                release_status.set()

                # verify client is closed
                t.join()
                self.assertFalse(t.is_alive())

    def test_results_loaded_while_closing_with_wait(self):
        submission_id = 'test-id'
        ev_poll = threading.Event()
        ev_load = threading.Event()

        def create_mock_session(client):
            session = mock.Mock()

            def _blocking_poll():
                ev_poll.wait()
                return [self.sapi.complete_no_answer_reply(id=submission_id)]

            def _blocking_load():
                ev_load.wait()
                return self.sapi.complete_reply(id=submission_id)

            # make sure polling for status stalls until we set `ev_poll`,
            # and loading results stalls until we set `ev_load`
            def get(path, **kwargs):
                return choose_reply(path, fix_status_paths({
                    f'problems/?id={submission_id}': _blocking_poll,
                    f'problems/{submission_id}/': _blocking_load,
                }, **self.poll_params))

            session.get = get
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [self.sapi.continue_reply(id=submission_id)]})

            return session

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with Client(**self.config) as client:
                solver = Solver(client, self.sapi.solver.data)

                future = solver.sample_ising(*self.sapi.problem)

                # check the first job is in `pending`
                future.wait_id()
                self.assertEqual(future.id, submission_id)
                self.assertEqual(future.remote_status, 'PENDING')

                # now close the client, but since client.close() is blocking,
                # do it in a background thread
                t = threading.Thread(target=lambda: client.close(wait=True))
                t.start()

                # try submitting another job while the client is closing
                with self.assertRaises(UseAfterCloseError):
                    solver.sample_ising(*self.sapi.problem)

                # verify client.close is still running
                t.join(timeout=0.1)
                self.assertTrue(t.is_alive())

                # release poll result
                ev_poll.set()

                # verify the job is now `completed`, but not yet resolved
                t.join(timeout=0.1)
                self.assertEqual(future.id, submission_id)
                self.assertEqual(future.remote_status, 'COMPLETED')
                self.assertIsNone(future._result)
                self.assertFalse(future.done())

                # verify client.close is still running
                t.join(timeout=0.1)
                self.assertTrue(t.is_alive())

                # release load result
                ev_load.set()

                # verify the job is now resolved
                t.join(timeout=0.1)
                self.assertTrue(future.done())
                self.assertIsNotNone(future._result)

                # verify client is closed
                t.join()
                self.assertFalse(t.is_alive())


class TestThreadSafety(unittest.TestCase):

    def test_result_loading(self):
        sapi = StructuredSapiMockResponses()
        executor = ThreadPoolExecutor(max_workers=2)
        gate = threading.Event()
        limit = threading.BoundedSemaphore(value=1)

        def create_mock_session(client):
            session = mock.Mock()
            session.post = lambda path, **kwargs: choose_reply(path, {
                'problems/': [sapi.complete_no_answer_reply(id='123')]})
            session.get = lambda path, **kwargs: choose_reply(path, {
                'problems/123/': sapi.complete_reply(id='123')})
            return session

        def gated_decoder(self, msg, **kwargs):
            gate.wait()
            with limit:
                time.sleep(0.1)
                return self._decode_qp(msg)

        with mock.patch.object(Client, 'create_session', create_mock_session):
            with mock.patch.object(Solver, 'decode_response', gated_decoder):
                with Client(endpoint='endpoint', token='token') as client:
                    solver = Solver(client, sapi.solver.data)
                    response = solver.sample_ising(*sapi.problem)

                    # run `response.result()` in parallel:
                    timing = executor.submit(lambda: response.timing)
                    problem_type = executor.submit(lambda: response.problem_type)

                    gate.set()

                    try:
                        timing.result()
                        problem_type.result()
                    except:
                        # might fail with `ValueError` raised by the `BoundedSemaphore`,
                        # or some other decoder-specific error due to repeated decoding
                        # of a mutated structure (message)
                        self.fail('parallel resolve failed')


class TestSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    class AssertionSatisfied(Exception):
        """Raised by `on_submit_data_verifier` to signal correct serialization."""

    def on_submit_data_verifier(self, expected_params):
        """Factory for mock Client._submit() that will validate parameter values."""

        # replacement for Client._submit(), called with exact network request data
        def _submit(client, body_data, computation):
            body = orjson.loads(body_data.result())

            params = body.get('params')
            if params != expected_params:
                raise AssertionError("params don't match")

            raise TestSerialization.AssertionSatisfied

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
        (numpy.bool_(1), True),
        (numpy.byte(1), 1), (numpy.int8(1), 1),
        (numpy.ubyte(1), 1), (numpy.uint8(1), 1),
        (numpy.short(1), 1), (numpy.int16(1), 1),
        (numpy.ushort(1), 1), (numpy.uint16(1), 1),
        (numpy.int32(1), 1),    # numpy.intc
        (numpy.uint32(1), 1),   # numpy.uintc
        (numpy.int_(1), 1), (numpy.int32(1), 1),
        (numpy.uint(1), 1), (numpy.uint32(1), 1),
        (numpy.int64(1), 1),    # numpy.longlong
        (numpy.uint64(1), 1),   # numpy.ulonglong
        (numpy.half(1.0), 1.0), (numpy.float16(1.0), 1.0),
        (numpy.single(1.0), 1.0), (numpy.float32(1.0), 1.0),
        (numpy.double(1.0), 1.0), (numpy.float64(1.0), 1.0),
        # note: orjson does not currently support:
        #       longlong, ulonglong, longdouble/float128, intc, uintc
        # see: https://github.com/ijl/orjson/issues/469
    ])
    @mock.patch.object(Client, 'create_session', lambda client: mock.Mock())
    def test_params_are_serialized(self, np_val, py_val):
        """Parameters supplied as NumPy types are correctly serialized."""

        user_params = dict(num_reads=np_val)
        expected_params = dict(num_reads=py_val)

        with Client(endpoint='endpoint', token='token') as client:
            solver = Solver(client, self.sapi.solver.data)
            problems = self.generate_sample_problems(solver)

            for method_name, problem_args in problems:
                with self.subTest(method_name=method_name, np_val=np_val, py_val=py_val):
                    sample = getattr(solver, method_name)

                    with mock.patch.object(
                            Client, '_submit', self.on_submit_data_verifier(expected_params)):

                        with self.assertRaises(self.AssertionSatisfied):
                            sample(*problem_args, **user_params).result()

# Copyright 2020 D-Wave Systems Inc.
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

import unittest

try:
    import dimod
except ImportError:
    dimod = None

from dwave.cloud.client import Client
from dwave.cloud.solver import StructuredSolver, UnstructuredSolver
from dwave.cloud.events import add_handler, dispatches_events
from dwave.cloud.concurrency import Present
from dwave.cloud.testing import mocks


class TestEventDispatch(unittest.TestCase):

    def setUp(self):
        # mock client
        self.client = Client(endpoint='e', token='t', solver=dict(name__contains='test', order_by='-name'))
        self.client._fetch_solvers = lambda **kw: self.solvers
        self.client._submit = lambda *pa, **kw: None
        self.client.upload_problem_encoded = lambda *pa, **kw: Present(result='mock_problem_id')

        # mock solvers
        self.structured_solver = StructuredSolver(
            client=self.client, data=mocks.qpu_clique_solver_data(name="test_qpu", size=3))
        self.unstructured_solver = UnstructuredSolver(
            client=self.client, data=mocks.hybrid_bqm_solver_data(name="test_hss"))

        self.solvers = [self.structured_solver]
        # we can't use unstructured solvers without dimod installed,
        # so don't even try testing it
        if dimod:
            self.solvers.append(self.unstructured_solver)

        # reset all event handlers
        from dwave.cloud.events import _client_event_hooks_registry as reg
        reg.update({k: [] for k in reg})

    def test_validation(self):
        """Event name and handler are validated."""

        with self.assertRaises(ValueError):
            add_handler('invalid_event_name', lambda: None)
        with self.assertRaises(TypeError):
            add_handler('before_client_init', None)

    def test_client_init(self):
        """Before/After client init events are dispatched with correct signatures."""

        # setup event handlers
        memo = {}
        def handler(event, **data):
            memo[event] = data

        add_handler('before_client_init', handler)
        add_handler('after_client_init', handler)

        # client init
        client = Client(endpoint='endpoint', token='token', unknown='unknown')

        # test entry values
        before = memo['before_client_init']
        self.assertEqual(before['obj'], client)
        self.assertIn('kwargs', before['args'])
        self.assertIn('endpoint', before['args']['kwargs'])
        self.assertIn('token', before['args']['kwargs'])
        self.assertEqual(before['args']['kwargs']['token'], 'token')
        self.assertEqual(before['args']['kwargs']['unknown'], 'unknown')

        # test exit values
        after = memo['after_client_init']
        self.assertEqual(after['obj'], client)
        self.assertEqual(after['args']['kwargs']['token'], 'token')
        self.assertEqual(after['args']['kwargs']['unknown'], 'unknown')
        self.assertEqual(after['args']['kwargs']['endpoint'], 'endpoint')
        self.assertEqual(after['return_value'], None)

    def test_get_solvers(self):
        """Before/After get_solvers events are dispatched with correct signatures."""

        # setup event handlers
        memo = {}
        def handler(event, **data):
            memo[event] = data

        add_handler('before_get_solvers', handler)
        add_handler('after_get_solvers', handler)

        # get solver(s)
        self.client.get_solver()

        # test entry values
        before = memo['before_get_solvers']
        self.assertEqual(before['obj'], self.client)
        self.assertIn('refresh', before['args'])
        self.assertIn('filters', before['args'])
        self.assertIn('name__contains', before['args']['filters'])

        # test exit values
        after = memo['after_get_solvers']
        self.assertEqual(after['obj'], self.client)
        self.assertIn('name__contains', after['args']['filters'])
        self.assertEqual(after['return_value'], self.solvers)

    def subtest_sample(self, solver):
        # setup event handlers
        memo = {}
        def handler(event, **data):
            memo[event] = data

        add_handler('before_sample', handler)
        add_handler('after_sample', handler)

        # sample
        lin = {0: 1}
        quad = {(0, 1): 1}
        offset = 2
        sample_params = dict(num_reads=100)
        upload_params = dict(encoding='good')

        # test entry values
        if solver.qpu:
            future = solver.sample_ising(lin, quad, offset, **sample_params)
            args = dict(type_='ising', linear=lin, quadratic=quad,
                        offset=offset, params=sample_params,
                        undirected_biases=False, label=None)
        elif solver.hybrid:
            if not dimod:
                self.skipTest("dimod not installed")
            future = solver.sample_ising(lin, quad, offset,
                                         upload_params=upload_params, **sample_params)
            bqm = dimod.BQM.from_ising(lin, quad, offset)
            args = dict(problem=bqm, problem_type=None, label=None,
                        sample_params=sample_params, upload_params=upload_params)

        before = memo['before_sample']
        self.assertEqual(before['obj'], solver)
        self.assertDictEqual(before['args'], args)

        # test exit values
        after = memo['after_sample']
        self.assertEqual(after['obj'], solver)
        self.assertDictEqual(after['args'], args)
        self.assertEqual(after['return_value'], future)

    def test_sample(self):
        """Before/After solver sample events are dispatched with correct signatures."""

        for solver in self.solvers:
            with self.subTest("solver=%r" % solver):
                self.subtest_sample(solver)


class TestEventDispatchDecorator(unittest.TestCase):

    def setUp(self):
        # reset all event handlers
        from dwave.cloud.events import _client_event_hooks_registry as reg
        reg.update({k: [] for k in reg})

    def test_decorator(self):
        """Decorator adds on-entry and on-exit event calls, with correct args."""

        class MockSampler:
            @dispatches_events('sample')
            def mock_sample(self, h, J, offset=0, fail=False, **kwargs):
                if fail:
                    raise ValueError
                return offset + 1

        mock_object = MockSampler()
        h = [1, 1]
        J = {(0, 1): 1}

        def before(name, obj, args):
            self.assertEqual(obj, mock_object)
            args.pop('fail')
            self.assertEqual(args, dict(h=h, J=J, offset=0, kwargs={}))

        def after(name, obj, args, return_value=None, exception=None):
            self.assertEqual(obj, mock_object)
            fail = args.pop('fail')
            self.assertEqual(args, dict(h=h, J=J, offset=0, kwargs={}))
            if fail:
                self.assertIsInstance(exception, ValueError)
            else:
                self.assertEqual(return_value, 1)

        add_handler('before_sample', before)
        add_handler('before_sample', after)

        mock_object.mock_sample(h, J)
        with self.assertRaises(ValueError):
            mock_object.mock_sample(h, J, fail=True)

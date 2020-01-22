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
from functools import partial

from dwave.cloud.testing import mock
from dwave.cloud.client import Client
from dwave.cloud.solver import Solver
from dwave.cloud.events import add_handler


class TestEventDispatch(unittest.TestCase):

    def setUp(self):
        # mock client
        self.client = Client(token='token', solver={'qpu': True})
        self.client._fetch_solvers = lambda **kw: self.solvers
        self.client._submit = lambda *pa, **kw: None

        # mock solvers
        self.solver = Solver(client=self.client, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1, 2],
                "couplers": [[0, 1], [0, 2], [1, 2]],
                "num_qubits": 3,
                "num_reads_range": [0, 100],
                "parameters": {
                    "num_reads": "Number of samples to return.",
                    "postprocess": "either 'sampling' or 'optimization'"
                },
                "topology": {
                    "type": "chimera",
                    "shape": [16, 16, 4]
                },
                "tags": ["lower_noise"]
            },
            "id": "solver1",
            "description": "A test solver 1",
            "status": "online"
        })
        self.solvers = [self.solver]

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
        def handler(cat, data):
            memo[cat] = data

        add_handler('before_client_init', partial(handler, 'before'))
        add_handler('after_client_init', partial(handler, 'after'))

        # client init
        client = Client(token='token', unknown='unknown')

        # test entry values
        before = memo['before']
        self.assertEqual(before['self'], client)
        self.assertEqual(before['endpoint'], None)
        self.assertEqual(before['token'], 'token')
        self.assertEqual(before['kwargs']['unknown'], 'unknown')

        # test exit values
        after = memo['after']
        self.assertEqual(after['self'], client)
        self.assertEqual(after['args']['token'], 'token')
        self.assertEqual(after['args']['kwargs']['unknown'], 'unknown')
        self.assertEqual(after['return_value'], None)

    def test_get_solvers(self):
        """Before/After get_solvers events are dispatched with correct signatures."""

        # setup event handlers
        memo = {}
        def handler(cat, data):
            memo[cat] = data

        add_handler('before_get_solvers', partial(handler, 'before'))
        add_handler('after_get_solvers', partial(handler, 'after'))

        # get solver(s)
        self.client.get_solver()

        # test entry values
        before = memo['before']
        self.assertEqual(before['self'], self.client)
        self.assertIn('refresh', before)
        self.assertIn('qpu', before['filters'])

        # test exit values
        after = memo['after']
        self.assertEqual(after['self'], self.client)
        self.assertIn('qpu', after['args']['filters'])
        self.assertEqual(after['return_value'], self.solvers)

    def test_sample(self):
        """Before/After solver sample events are dispatched with correct signatures."""

        # setup event handlers
        memo = {}
        def handler(cat, data):
            memo[cat] = data

        add_handler('before_sample', partial(handler, 'before'))
        add_handler('after_sample', partial(handler, 'after'))

        # sample
        lin = {0: 1}
        quad = {(0, 1): 1}
        params = dict(num_reads=100)
        future = self.solver.sample_ising(lin, quad, **params)

        # test entry values
        before = memo['before']
        args = dict(self=self.solver, type_='ising', linear=lin, quadratic=quad, params=params)
        self.assertDictEqual(before, args)

        # test exit values
        after = memo['after']
        self.assertEqual(after['self'], self.solver)
        self.assertDictEqual(after['args'], args)
        self.assertEqual(after['return_value'], future)

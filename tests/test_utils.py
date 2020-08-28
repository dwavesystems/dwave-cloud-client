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

import logging
import unittest
from unittest import mock
from collections import OrderedDict
from itertools import count
from datetime import datetime

from dwave.cloud.utils import (
    uniform_iterator, uniform_get, strip_head, strip_tail,
    active_qubits, generate_random_ising_problem,
    default_text_input, utcnow, cached, retried, parse_loglevel,
    user_agent)


class TestSimpleUtils(unittest.TestCase):

    def test_uniform_iterator(self):
        items = [('a', 1), ('b', 2)]
        self.assertEqual(list(uniform_iterator(OrderedDict(items))), items)
        self.assertEqual(list(uniform_iterator('ab')), list(enumerate('ab')))

    def test_uniform_get(self):
        d = {0: 0, 1: 1}
        self.assertEqual(uniform_get(d, 0), 0)
        self.assertEqual(uniform_get(d, 2), None)
        self.assertEqual(uniform_get(d, 2, default=0), 0)
        l = [0, 1]
        self.assertEqual(uniform_get(l, 0), 0)
        self.assertEqual(uniform_get(l, 2), None)
        self.assertEqual(uniform_get(l, 2, default=0), 0)

    def test_strip_head(self):
        self.assertEqual(strip_head([0, 0, 1, 2], [0]), [1, 2])
        self.assertEqual(strip_head([1], [0]), [1])
        self.assertEqual(strip_head([1], []), [1])
        self.assertEqual(strip_head([0, 0, 1, 2], [0, 1, 2]), [])

    def test_strip_tail(self):
        self.assertEqual(strip_tail([1, 2, 0, 0], [0]), [1, 2])
        self.assertEqual(strip_tail([1], [0]), [1])
        self.assertEqual(strip_tail([1], []), [1])
        self.assertEqual(strip_tail([0, 0, 1, 2], [0, 1, 2]), [])

    def test_active_qubits_dict(self):
        self.assertEqual(active_qubits({}, {}), set())
        self.assertEqual(active_qubits({0: 0}, {}), {0})
        self.assertEqual(active_qubits({}, {(0, 1): 0}), {0, 1})
        self.assertEqual(active_qubits({2: 0}, {(0, 1): 0}), {0, 1, 2})

    def test_active_qubits_list(self):
        self.assertEqual(active_qubits([], {}), set())
        self.assertEqual(active_qubits([2], {}), {0})
        self.assertEqual(active_qubits([2, 2, 0], {}), {0, 1, 2})
        self.assertEqual(active_qubits([], {(0, 1): 0}), {0, 1})
        self.assertEqual(active_qubits([0, 0], {(0, 2): 0}), {0, 1, 2})

    def test_default_text_input(self):
        val = "value"
        with mock.patch("dwave.cloud.utils.input", side_effect=[val]):
            self.assertEqual(default_text_input("prompt", val), val)
        with mock.patch("dwave.cloud.utils.input", side_effect=[val]):
            self.assertEqual(default_text_input("prompt", val+val), val)

    def test_generate_random_ising_problem(self):
        class MockSolver(object):
            nodes = [0, 1, 3]
            undirected_edges = {(0, 1), (1, 3), (0, 4)}
            properties = dict(h_range=[2, 2], j_range=[-1, -1])
        mock_solver = MockSolver()

        lin, quad = generate_random_ising_problem(mock_solver)

        self.assertDictEqual(lin, {0: 2.0, 1: 2.0, 3: 2.0})
        self.assertDictEqual(quad, {(0, 1): -1.0, (1, 3): -1.0, (0, 4): -1.0})

    def test_generate_random_ising_problem_default_solver_ranges(self):
        class MockSolver(object):
            nodes = [0, 1, 3]
            undirected_edges = {(0, 1), (1, 3), (0, 4)}
            properties = {}
        mock_solver = MockSolver()

        lin, quad = generate_random_ising_problem(mock_solver)

        for q, v in lin.items():
            self.assertTrue(v >= -1 and v <= 1)
        for e, v in quad.items():
            self.assertTrue(v >= -1 and v <= 1)

    def test_generate_random_ising_problem_with_user_constrained_ranges(self):
        class MockSolver(object):
            nodes = [0, 1, 3]
            undirected_edges = {(0, 1), (1, 3), (0, 4)}
            properties = dict(h_range=[2, 2], j_range=[-1, -1])
        mock_solver = MockSolver()

        lin, quad = generate_random_ising_problem(mock_solver, h_range=[0,0], j_range=[1,1])

        self.assertDictEqual(lin, {0: 0.0, 1: 0.0, 3: 0.0})
        self.assertDictEqual(quad, {(0, 1): 1.0, (1, 3): 1.0, (0, 4): 1.0})

    def test_utcnow(self):
        t = utcnow()
        now = datetime.utcnow()
        self.assertEqual(t.utcoffset().total_seconds(), 0.0)
        unaware = t.replace(tzinfo=None)
        self.assertLess((now - unaware).total_seconds(), 1.0)

    def test_parse_loglevel_invalid(self):
        """Parsing invalid log levels returns NOTSET."""
        notset = logging.NOTSET

        self.assertEqual(parse_loglevel(''), notset)
        self.assertEqual(parse_loglevel('  '), notset)
        self.assertEqual(parse_loglevel(None), notset)
        self.assertEqual(parse_loglevel(notset), notset)
        self.assertEqual(parse_loglevel('nonexisting'), notset)
        self.assertEqual(parse_loglevel({'a': 1}), notset)
        self.assertIsNone(parse_loglevel('nonexisting', default=None))

    def test_parse_loglevel_numeric_and_symbolic(self):
        self.assertEqual(parse_loglevel('info'), logging.INFO)
        self.assertEqual(parse_loglevel('INFO'), logging.INFO)
        self.assertEqual(parse_loglevel(logging.INFO), logging.INFO)
        self.assertEqual(parse_loglevel(str(logging.INFO)), logging.INFO)
        self.assertEqual(parse_loglevel('  %d  ' % logging.INFO), logging.INFO)

    def test_user_agent(self):
        from dwave.cloud.package_info import __packagename__, __version__
        ua = user_agent(__packagename__, __version__)

        required = [__packagename__, 'python', 'machine', 'system', 'platform']
        for key in required:
            self.assertIn(key, ua)


class TestCachedDecorator(unittest.TestCase):

    def test_args_hashing(self):
        counter = count()

        @cached(maxage=300)
        def f(*a, **b):
            return next(counter)

        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            self.assertEqual(f(), 0)
            self.assertEqual(f(1), 1)
            self.assertEqual(f(1, 2), 2)
            self.assertEqual(f(1), 1)
            self.assertEqual(f(1, refresh_=True), 3)
            self.assertEqual(f(1, 2), 2)

            self.assertEqual(f(a=1, b=2), 4)
            self.assertEqual(f(b=2, a=1), 4)
            self.assertEqual(f(b=2, a=1, refresh_=1), 5)
            self.assertEqual(f(), 0)

            self.assertEqual(f(2), 6)
            self.assertEqual(f(1), 3)

    def test_args_collision(self):
        counter = count()

        @cached(maxage=300)
        def f(*a, **b):
            return next(counter)

        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            # NB: in python2, without hash seed randomization,
            # hash('\0B') == hash('\0\0C')
            self.assertEqual(f(x='\0B'), 0)
            self.assertEqual(f(x='\0\0C'), 1)

    def test_expiry(self):
        counter = count()

        @cached(maxage=300)
        def f(*a, **b):
            return next(counter)

        # populate
        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            self.assertEqual(f(), 0)
            self.assertEqual(f(1), 1)
            self.assertEqual(f(a=1, b=2), 2)

        # verify expiry
        with mock.patch('dwave.cloud.utils.epochnow', lambda: 301):
            self.assertEqual(f(), 3)
            self.assertEqual(f(1), 4)
            self.assertEqual(f(a=1, b=2), 5)

        # verify maxage
        with mock.patch('dwave.cloud.utils.epochnow', lambda: 299):
            self.assertEqual(f(), 3)
            self.assertEqual(f(1), 4)
            self.assertEqual(f(a=1, b=2), 5)

    def test_default_maxage(self):
        counter = count()

        @cached()
        def f(*a, **b):
            return next(counter)

        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            self.assertEqual(f(), 0)
            self.assertEqual(f(), 1)
            self.assertEqual(f(), 2)

    def test_exceptions(self):
        counter = count(0)

        @cached()
        def f():
            # raises ZeroDivisionError only on first call
            # we do not want to cache that!
            return 1.0 / next(counter)

        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            self.assertRaises(ZeroDivisionError, f)
            self.assertEqual(f(), 1)
            self.assertEqual(f(), 0.5)


class TestRetriedDecorator(unittest.TestCase):

    def test_func_called(self):
        """Wrapped function is called with correct arguments."""

        @retried()
        def f(a, b):
            return a, b

        self.assertEqual(f(1, b=2), (1, 2))

    def test_exc_raised(self):
        """Correct exception is raised after number of retries exceeded."""

        @retried()
        def f():
            raise ValueError

        with self.assertRaises(ValueError):
            f()

    def test_func_called_only_until_succeeds(self):
        """Wrapped function is called no more times then it takes to succeed."""

        err = ValueError
        val = mock.sentinel
        attrs = dict(__name__='f')

        # f succeeds on 3rd try
        f = mock.Mock(side_effect=[err, err, val.a, val.b], **attrs)
        ret = retried(3)(f)()
        self.assertEqual(ret, val.a)
        self.assertEqual(f.call_count, 3)

        # fail with only on retry
        f = mock.Mock(side_effect=[err, err, val.a, val.b], **attrs)
        with self.assertRaises(err):
            ret = retried(1)(f)()

        # no errors, return without retries
        f = mock.Mock(side_effect=[val.a, val.b, val.c], **attrs)
        ret = retried(3)(f)()
        self.assertEqual(ret, val.a)
        self.assertEqual(f.call_count, 1)

    def test_decorator(self):
        with self.assertRaises(TypeError):
            retried()("not-a-function")

    def test_backoff_constant(self):
        """Constant retry back-off."""

        # 1s delay before a retry
        backoff = 1

        with mock.patch('time.sleep') as sleep:

            @retried(retries=2, backoff=backoff)
            def f():
                raise ValueError

            with self.assertRaises(ValueError):
                f()

            calls = [mock.call(backoff), mock.call(backoff)]
            sleep.assert_has_calls(calls)

    def test_backoff_seq(self):
        """Retry back-off defined via list."""

        # progressive delay
        backoff = [1, 2, 3]

        with mock.patch('time.sleep') as sleep:

            @retried(retries=3, backoff=backoff)
            def f():
                raise ValueError

            with self.assertRaises(ValueError):
                f()

            calls = [mock.call(b) for b in backoff]
            sleep.assert_has_calls(calls)

    def test_backoff_func(self):
        """Retry back-off defined via callable."""

        def backoff(retry):
            return 2 ** retry

        with mock.patch('time.sleep') as sleep:

            @retried(retries=3, backoff=backoff)
            def f():
                raise ValueError

            with self.assertRaises(ValueError):
                f()

            calls = [mock.call(backoff(1)), mock.call(backoff(2)), mock.call(backoff(3))]
            sleep.assert_has_calls(calls)


if __name__ == '__main__':
    unittest.main()

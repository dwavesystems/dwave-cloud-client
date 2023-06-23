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

import copy
import json
import uuid
import logging
import unittest
import warnings
import tempfile
from unittest import mock
from collections import OrderedDict
from itertools import count
from datetime import datetime
from functools import partial
from typing import Tuple, Union

import numpy
from parameterized import parameterized

from dwave.cloud import FilteredSecretsFormatter
from dwave.cloud.utils import (
    uniform_iterator, uniform_get, strip_head, strip_tail,
    active_qubits, generate_random_ising_problem, NumpyEncoder,
    default_text_input, utcnow, cached, retried, deprecated, aliasdict,
    parse_loglevel, user_agent, hasinstance, exception_chain, is_caused_by)


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
        with mock.patch("click.termui.visible_prompt_func", side_effect=[val]):
            self.assertEqual(default_text_input("prompt", val), val)
        with mock.patch("click.termui.visible_prompt_func", side_effect=[val]):
            self.assertEqual(default_text_input("prompt", val+val), val)

    def test_optional_text_input(self):
        with mock.patch("click.termui.visible_prompt_func", side_effect=[""]):
            self.assertEqual(default_text_input("prompt", optional=True), None)

    def test_optional_choices_text_input(self):
        with mock.patch("click.termui.visible_prompt_func", side_effect=[""]):
            self.assertEqual(
                default_text_input("prompt", choices='abc', optional=True), None)
        with mock.patch("click.termui.visible_prompt_func", side_effect=["d", "skip"]):
            self.assertEqual(
                default_text_input("prompt", choices='abc', optional=True), None)
        with mock.patch("click.termui.visible_prompt_func", side_effect=["e", "a"]):
            self.assertEqual(
                default_text_input("prompt", choices='abc', optional=True), 'a')

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


# copied from dwave-hybrid utils
# TODO: remove these tests when we/if switch to `dwave.common`
class TestNumpyJSONEncoder(unittest.TestCase):

    @parameterized.expand([
        (numpy.bool_(1), True), (numpy.bool8(1), True),
        (numpy.byte(1), 1), (numpy.int8(1), 1),
        (numpy.ubyte(1), 1), (numpy.uint8(1), 1),
        (numpy.short(1), 1), (numpy.int16(1), 1),
        (numpy.ushort(1), 1), (numpy.uint16(1), 1),
        (numpy.intc(1), 1), (numpy.int32(1), 1),
        (numpy.uintc(1), 1), (numpy.uint32(1), 1),
        (numpy.int_(1), 1), (numpy.int32(1), 1),
        (numpy.uint(1), 1), (numpy.uint32(1), 1),
        (numpy.longlong(1), 1), (numpy.int64(1), 1),
        (numpy.ulonglong(1), 1), (numpy.uint64(1), 1),
        (numpy.half(1.0), 1.0), (numpy.float16(1.0), 1.0),
        (numpy.single(1.0), 1.0), (numpy.float32(1.0), 1.0),
        (numpy.double(1.0), 1.0), (numpy.float64(1.0), 1.0),
        (numpy.longdouble(1.0), 1.0)
    ] + ([
        (numpy.float128(1.0), 1.0)      # unavailable on windows
    ] if hasattr(numpy, 'float128') else [
    ]))
    def test_numpy_primary_type_encode(self, np_val, py_val):
        self.assertEqual(
            json.dumps(py_val),
            json.dumps(np_val, cls=NumpyEncoder)
        )

    @parameterized.expand([
        (numpy.array([1, 2, 3], dtype=int), [1, 2, 3]),
        (numpy.array([[1], [2], [3]], dtype=float), [[1.0], [2.0], [3.0]]),
        (numpy.zeros((2, 2), dtype=bool), [[False, False], [False, False]]),
        (numpy.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
                     dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]),
         [['Rex', 9, 81.0], ['Fido', 3, 27.0]]),
        (numpy.rec.array([(1, 2., 'Hello'), (2, 3., "World")],
                         dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'U10')]),
         [[1, 2.0, "Hello"], [2, 3.0, "World"]])
    ])
    def test_numpy_array_encode(self, np_val, py_val):
        self.assertEqual(
            json.dumps(py_val),
            json.dumps(np_val, cls=NumpyEncoder)
        )


class TestCachedInMemoryDecorator(unittest.TestCase):
    """Test @cached using in-memory store."""

    def setUp(self):
        self.cached = cached

    def test_args_hashing(self):
        counter = count()

        @self.cached(maxage=300)
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

        @self.cached(maxage=300)
        def f(*a, **b):
            return next(counter)

        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            # NB: in python2, without hash seed randomization,
            # hash('\0B') == hash('\0\0C')
            self.assertEqual(f(x='\0B'), 0)
            self.assertEqual(f(x='\0\0C'), 1)

    def test_expiry(self):
        counter = count()

        @self.cached(maxage=300)
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

    def test_default_zero_maxage(self):
        counter = count()

        @self.cached()
        def f(*a, **b):
            return next(counter)

        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            self.assertEqual(f(), 0)
            self.assertEqual(f(), 1)
            self.assertEqual(f(), 2)

    def test_exceptions(self):
        counter = count(0)

        @self.cached()
        def f():
            # raises ZeroDivisionError only on first call
            # we do not want to cache that!
            return 1.0 / next(counter)

        with mock.patch('dwave.cloud.utils.epochnow', lambda: 0):
            self.assertRaises(ZeroDivisionError, f)
            self.assertEqual(f(), 1)
            self.assertEqual(f(), 0.5)


class TestCachedOnDiskDecorator(TestCachedInMemoryDecorator):
    """Test @cached using on-disk store (via @cached.ondisk)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cached = partial(cached.ondisk, directory=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    @mock.patch('dwave.cloud.utils.epochnow', lambda: 0)
    def test_persistency(self):
        counter = count()
        def f():
            return next(counter)

        f1 = self.cached(maxage=1)(f)
        self.assertEqual(f1(), 0)

        f2 = self.cached(maxage=1)(f)
        self.assertEqual(f2(), 0)


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


class TestDeprecatedDecorator(unittest.TestCase):

    def test_func_called(self):
        """Wrapped function is called with correct arguments."""

        @deprecated()
        def f(a, b):
            return a, b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.assertEqual(f(1, b=2), (1, 2))

    def test_warning_raised(self):
        """Correct deprecation message is raised."""

        msg = "deprecation message"

        @deprecated(msg)
        def f():
            return

        with self.assertWarns(DeprecationWarning, msg=msg):
            f()

    def test_warning_raised_automsg(self):
        """Deprecation message is auto-generated and raised."""

        @deprecated()
        def f():
            return

        automsg_regex = r'f\(\) has been deprecated'

        with self.assertWarnsRegex(DeprecationWarning, automsg_regex):
            f()

    def test_decorator(self):
        with self.assertRaises(TypeError):
            deprecated()("not-a-function")


class TestAliasdict(unittest.TestCase):

    def assert_dict_interface(self, aliased, origin):
        "Assert `aliased` behaves exactly as `origin` dict."

        self.assertIsInstance(aliased, dict)
        self.assertDictEqual(aliased, origin)

        self.assertEqual(len(aliased), len(origin))
        self.assertSetEqual(set(aliased), set(origin))
        self.assertSetEqual(set(aliased.keys()), set(origin.keys()))
        self.assertSetEqual(set(aliased.items()), set(origin.items()))
        self.assertListEqual(list(aliased.values()), list(origin.values()))
        for k in origin:
            self.assertIn(k, aliased)
        stranger = "unique-{}".format(''.join(origin))
        self.assertNotIn(stranger, aliased)

        self.assertSetEqual(set(iter(aliased)), set(origin))

        # copy
        new = aliased.copy()
        self.assertIsNot(new, aliased)
        self.assertDictEqual(new, aliased)

        # dict copy constructor on aliasdict
        new = dict(aliased)
        self.assertDictEqual(new, origin)

        # pop on copy
        key = next(iter(origin))
        new.pop(key)
        self.assertSetEqual(set(new), set(origin).difference(key))
        self.assertDictEqual(aliased, origin)

        # get
        self.assertEqual(aliased[key], origin[key])
        self.assertEqual(aliased.get(key), origin.get(key))

        # set
        new = aliased.copy()
        ref = origin.copy()
        new[stranger] = 4
        ref[stranger] = 4
        self.assertDictEqual(new, ref)

        # del
        del new[stranger]
        self.assertDictEqual(new, origin)

    def test_construction(self):
        # aliasdict can be constructed from a mapping/dict
        src = dict(a=1)
        ad = aliasdict(src)
        self.assert_dict_interface(ad, src)

        # aliasdict can be updated, without affecting the source dict
        ad.update(b=2)
        self.assertDictEqual(ad, dict(a=1, b=2))
        self.assertDictEqual(src, dict(a=1))

        # source dict can be updated without affecting the aliased dict
        src.update(c=3)
        self.assertDictEqual(ad, dict(a=1, b=2))
        self.assertDictEqual(src, dict(a=1, c=3))

    def test_dict_interface(self):
        src = dict(a=1, b=2)
        ad = aliasdict(**src)
        self.assert_dict_interface(ad, src)

    def test_alias_concrete(self):
        src = dict(a=1)

        ad = aliasdict(src)
        ad.alias(b=2)

        self.assert_dict_interface(ad, src)
        self.assertEqual(ad['b'], 2)

    def test_alias_callable(self):
        src = dict(a=1)

        ad = aliasdict(src)
        ad.alias(b=lambda d: d.get('a'))

        self.assert_dict_interface(ad, src)

        # 'b' equals 'a'
        self.assertEqual(ad['b'], ad['a'])
        self.assertEqual(ad['b'], src['a'])
        self.assertEqual(ad['b'], 1)

        # it's dynamic, works also when 'a' changes
        ad['a'] = 2
        self.assertEqual(ad['b'], ad['a'])
        self.assertNotEqual(ad['b'], src['a'])
        self.assertEqual(ad['b'], 2)

    def test_alias_get_set_del(self):
        src = dict(a=1)
        ad = aliasdict(src)

        # getitem and get on alias
        ad.alias(b=2)
        self.assertEqual(ad['b'], 2)
        self.assertEqual(ad.get('b'), 2)
        self.assertEqual(ad.aliases['b'], 2)

        # get with a default
        randomkey = str(uuid.uuid4())
        self.assertEqual(ad.get(randomkey), None)
        self.assertEqual(ad.get(randomkey, 1), 1)

        # set alias
        ad['b'] = 3
        self.assertEqual(ad['b'], 3)
        self.assertEqual(ad.aliases['b'], 3)

        # update alias
        ad.alias(b=4)
        self.assertEqual(ad['b'], 4)
        self.assertEqual(ad.aliases['b'], 4)

        # delete alias
        del ad['b']
        self.assertEqual(ad.get('b'), None)

    def test_shadowing(self):
        "Alias keys take precedence."

        ad = aliasdict(a=1)
        ad.alias(a=2)

        self.assertEqual(ad['a'], 2)

        self.assertEqual(dict.__getitem__(ad, 'a'), 1)

    def test_copy(self):
        src = dict(a=1)
        aliases = dict(b=2)

        ad = aliasdict(src)
        ad.alias(**aliases)

        self.assertDictEqual(ad, src)
        self.assertDictEqual(ad.aliases, aliases)

        new = ad.copy()
        self.assertIsInstance(new, aliasdict)
        self.assertIsNot(new, ad)
        self.assertDictEqual(new, src)
        self.assertDictEqual(new.aliases, aliases)

        new = copy.deepcopy(ad)
        self.assertIsInstance(new, aliasdict)
        self.assertIsNot(new, ad)
        self.assertDictEqual(new, src)
        self.assertDictEqual(new.aliases, aliases)


class TestExceptionUtils(unittest.TestCase):

    def raise_implicit():
        try:
            1/0
        except:
            raise ValueError

    def raise_explicit():
        try:
            1/0
        except Exception as e:
            raise ValueError from e

    def raise_mixed():
        try:
            TestExceptionUtils.raise_explicit()
        except:
            raise TypeError

    def test_hasinstance(self):
        # not contained
        self.assertFalse(hasinstance([], ValueError))
        self.assertFalse(hasinstance([TypeError], ValueError))
        self.assertFalse(hasinstance([TypeError()], ValueError))
        self.assertFalse(hasinstance([ValueError], ValueError))

        # contained in unit list
        self.assertTrue(hasinstance([ValueError()], ValueError))
        self.assertTrue(hasinstance([ValueError('msg')], ValueError))

        # contained in a list
        self.assertTrue(hasinstance([TypeError(), ValueError()], ValueError))
        self.assertTrue(hasinstance([ValueError(), ValueError()], ValueError))

        # contained in a tuple
        self.assertTrue(hasinstance((TypeError(), ValueError()), ValueError))

        # base class also contained
        self.assertTrue(hasinstance([ValueError()], Exception))

        # multiple types can be checked
        self.assertFalse(hasinstance([ValueError()], (dict, list)))
        self.assertTrue(hasinstance([ValueError()], (dict, Exception)))
        self.assertTrue(hasinstance([ValueError()], (ValueError, TypeError)))
        self.assertTrue(hasinstance([TypeError()], (ValueError, TypeError)))
        self.assertTrue(hasinstance([TypeError(), ValueError], (ValueError, TypeError)))

    @parameterized.expand([
        (raise_implicit, (ValueError, ZeroDivisionError)),
        (raise_explicit, (ValueError, ZeroDivisionError)),
        (raise_mixed, (TypeError, ValueError, ZeroDivisionError))
    ])
    def test_exception_chain(self, raise_exc, chained_types):
        try:
            raise_exc()
        except Exception as e:
            exc = e
        else:
            self.fail('Exception not raised')

        chain = list(exception_chain(exc))
        self.assertEqual(len(chain), len(chained_types))

        for idx, typ in enumerate(chained_types):
            self.assertIsInstance(chain[idx], typ)

        for typ in chained_types:
            self.assertTrue(hasinstance(exception_chain(exc), typ))

    @parameterized.expand([
        (raise_implicit, (ValueError, ZeroDivisionError)),
        (raise_explicit, (ValueError, ZeroDivisionError)),
        (raise_mixed, (TypeError, ValueError, ZeroDivisionError))
    ])
    def test_is_caused_by(self, raise_exc, exception_types):
        try:
            raise_exc()
        except Exception as exc:
            self.assertTrue(is_caused_by(exc, exception_types))

            for typ in exception_types:
                self.assertTrue(is_caused_by(exc, typ))

            self.assertFalse(is_caused_by(exc, KeyError))


class TestFilteredSecretsFormatter(unittest.TestCase):

    @staticmethod
    def _filtered(msg: str) -> str:
        """Filter `msg` using `FilteredSecretsFormatter`."""
        fmt = FilteredSecretsFormatter('%(msg)s')
        rec = logging.makeLogRecord(dict(msg=msg))
        return fmt.format(rec)

    # dev note: define as unbound local function, to be used in class def context only
    # in py310+ we can declare it as @staticmethod and call it as regular function
    def _snipped(inp: Union[str, uuid.UUID]) -> Tuple[str, str]:
        """Naively assuming `inp` is a secret/token, scrub the middle part."""
        return (lambda x: (x, f"{x[:3]}...{x[-3:]}"))(str(inp))

    @parameterized.expand([
        # prefixed 160-bit+ hex tokens (SAPI token format)
        ('AB-0123456789012345678901234567890123456789', 'AB-012...789'),
        ('ABC-0123456789012345678901234567890123456789', 'ABC-012...789'),
        ('abc-c0f3456789012345678901234567890123456fee', 'abc-c0f...fee'),
        ('ABC-c0f3456789012345678901234567890123456beeffee', 'ABC-c0f...fee'),
        ('ABC-c0f3456789012345678901234567890123456beefxfee', None),
        ('ABCD-0123456789012345678901234567890123456789', 'ABCD-012...789'),
        # 128-bit+ hex tokens
        ('01234567890123456789012345678901', '012...901'),
        ('0123456789012345678901234567890123456789', '012...789'),
        ('0123456789012345_not_hex_567890123456789', None),
        # sapi-like tokens filtered as partial hex
        ('A-0123456789012345678901234567890123456789', 'A-012...789'),
        ('ABCDE-0123456789012345678901234567890123456789', 'ABCDE-012...789'),
        _snipped(uuid.uuid4().hex),
        # 128-bit uuid tokens
        _snipped(uuid.uuid1()),
        _snipped(uuid.uuid3(uuid.NAMESPACE_DNS, 'example.com')),
        _snipped(uuid.uuid4()),
        _snipped(uuid.uuid5(uuid.NAMESPACE_DNS, 'example.com')),
        (f"{uuid.uuid1()}1", None),
        (f"{str(uuid.uuid1())[-3:]}xyz", None),
    ])
    def test_token_filtration(self, inp, out=None):
        if out is None:
            out = inp

        self.assertEqual(self._filtered(inp), out)

        ctx = '{} suffix'
        self.assertEqual(self._filtered(ctx.format(inp)), ctx.format(out))

        ctx = 'prefix {}'
        self.assertEqual(self._filtered(ctx.format(inp)), ctx.format(out))

        ctx = 'a{}word'
        self.assertEqual(self._filtered(ctx.format(inp)), ctx.format(inp))


if __name__ == '__main__':
    unittest.main()

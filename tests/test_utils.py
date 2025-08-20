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

import contextlib
import copy
import inspect
import io
import json
import logging
import orjson
import os
import sqlite3
import subprocess
import tempfile
import textwrap
import time
import unittest
import uuid
import warnings
from collections import OrderedDict
from datetime import datetime
from dateutil.tz import UTC
from functools import partial, wraps
from itertools import count
from typing import Union
from unittest import mock
from urllib.parse import urljoin

import numpy
import requests
import requests_mock
from parameterized import parameterized

from dwave.cloud.utils.coders import coerce_numpy_to_python
from dwave.cloud.utils.cli import default_text_input
from dwave.cloud.utils.decorators import aliasdict, cached, deprecated, retried
from dwave.cloud.utils.dist import (
    get_contrib_packages, get_distribution, PackageNotFoundError, VersionNotFoundError)
from dwave.cloud.utils.exception import hasinstance, exception_chain, is_caused_by
from dwave.cloud.utils.http import (
    user_agent, default_user_agent, platform_tags, BaseUrlSessionMixin)
from dwave.cloud.utils.logging import (
    FilteredSecretsFormatter, configure_logging, parse_loglevel,
    fast_stack, get_caller_name)
from dwave.cloud.utils.qubo import (
    uniform_iterator, uniform_get,
    active_qubits, generate_random_ising_problem)
from dwave.cloud.utils.time import utcnow

logger = logging.getLogger(__name__)


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
        now = datetime.fromtimestamp(time.time(), tz=UTC)
        self.assertEqual(t.utcoffset().total_seconds(), 0.0)
        self.assertEqual(now.utcoffset().total_seconds(), 0.0)
        self.assertLess((now - t).total_seconds(), 1.0)

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

    def test_platform_tags_smoke(self):
        pt = platform_tags()
        if pt:
            self.assertNotEqual(
                user_agent(include_platform_tags=False),
                user_agent(include_platform_tags=True))

    def test_default_user_agent(self):
        from dwave.cloud.package_info import __packagename__, __version__
        ua = default_user_agent()
        ref = user_agent(
            name=__packagename__, version=__version__, include_platform_tags=False)
        self.assertEqual(ua, ref)

    @requests_mock.Mocker()
    def test_base_url_session_mixin(self, m):
        m.get(requests_mock.ANY, status_code=200)

        class BaseSession(requests.Session):
            def __init__(self, *args, **kwargs):
                self._init_kwargs = kwargs.copy()
                passed = {k: v for k, v in kwargs.items() if not k.startswith('test_')}
                return super().__init__(*args, **passed)

            def request(self, *args, **kwargs):
                self._request_kwargs = kwargs.copy()
                passed = {k: v for k, v in kwargs.items() if not k.startswith('test_')}
                return super().request(*args, **passed)

        class Session(BaseUrlSessionMixin, BaseSession):
            pass

        base_url = "http://mock"
        path = 'test/path'

        with self.subTest('URL composition'):
            s = Session(base_url=base_url)
            resp = s.get(path)
            self.assertEqual(resp.request.url, urljoin(base_url, path))

        with self.subTest('additional kwargs passed through'):
            s = Session(base_url=base_url, test_init='extra')
            resp = s.get(path, test_request='extra')
            self.assertEqual(resp.request.url, urljoin(base_url, path))
            self.assertEqual(s._init_kwargs.get('test_init'), 'extra')
            self.assertEqual(s._request_kwargs.get('test_request'), 'extra')


# initially copied from dwave-hybrid/NumpyEncoder tests, but expanded to cover
# `coerce_numpy_to_python`
class TestNumpyTypesEncoding(unittest.TestCase):

    NUMPY_SCALARS = [
        (numpy.bool_(1), True),
        (numpy.byte(1), 1), (numpy.int8(1), 1),
        (numpy.ubyte(1), 1), (numpy.uint8(1), 1),
        (numpy.short(1), 1), (numpy.int16(1), 1),
        (numpy.ushort(1), 1), (numpy.uint16(1), 1),
        (numpy.int_(1), 1), (numpy.int32(1), 1),
        (numpy.uint(1), 1), (numpy.uint32(1), 1),
        (numpy.int64(1), 1), (numpy.uint64(1), 1),
        (numpy.half(1.0), 1.0), (numpy.float16(1.0), 1.0),
        (numpy.single(1.0), 1.0), (numpy.float32(1.0), 1.0),
        (numpy.double(1.0), 1.0), (numpy.float64(1.0), 1.0),
    ]

    # unsupported by orjson
    NUMPY_SCALARS_EXTRA = [
        (numpy.longlong(1), 1), (numpy.ulonglong(1), 1),
        (numpy.longdouble(1.0), 1.0),
    ] + ([
        (numpy.intc(1), 1), (numpy.uintc(1), 1),    # unsupported by orjson on windows
        (numpy.float128(1.0), 1.0),                 # unavailable in numpy on windows
    ] if hasattr(numpy, 'float128') else [
    ])

    NUMPY_ARRAYS = [
        (numpy.array([1, 2, 3], dtype=int), [1, 2, 3]),
        (numpy.array([[1], [2], [3]], dtype=float), [[1.0], [2.0], [3.0]]),
        (numpy.zeros((2, 2), dtype=bool), [[False, False], [False, False]]),
    ]

    NUMPY_RECARRAYS = [
        (numpy.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
                     dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]),
         [('Rex', 9, 81.0), ('Fido', 3, 27.0)]),
        (numpy.rec.array([(1, 2., 'Hello'), (2, 3., "World")],
                         dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'U10')]),
         [(1, 2.0, "Hello"), (2, 3.0, "World")]),
    ]

    NUMPY_STRUCTURES = [
        ([numpy.int8(1), numpy.array([2], dtype=numpy.int64), 'three'], [1, [2], 'three']),
        ({'one': numpy.float64(1), 'two': numpy.array([2.0])}, {'one': 1.0, 'two': [2.0]}),
        ({(numpy.int32(1), numpy.int32(2)): {'v': [1, 2]}}, {(1, 2): {'v': [1, 2]}}),
    ]

    @parameterized.expand(NUMPY_SCALARS + NUMPY_ARRAYS)
    def test_numpy_dump(self, np_val, py_val):
        self.assertEqual(
            json.dumps(py_val, separators=(',', ':')).encode('utf8'),
            orjson.dumps(np_val, option=orjson.OPT_SERIALIZE_NUMPY)
        )

    @parameterized.expand(NUMPY_SCALARS + NUMPY_SCALARS_EXTRA
                          + NUMPY_ARRAYS + NUMPY_RECARRAYS + NUMPY_STRUCTURES)
    def test_numpy_to_python_coercion(self, np_val, py_val):
        self.assertEqual(py_val, coerce_numpy_to_python(np_val))


class TestCachedCommon(unittest.TestCase):
    """@cached tests common to in-memory and on-disk caches."""

    def setUp(self):
        self.cached = cached

    def test_all_args_hashing(self):
        counter = count()

        @self.cached()
        def f(*a, **b):
            return next(counter)

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

    def test_single_arg_hashing(self):
        counter = count()

        @self.cached(key='key')
        def f(*a, key, **b):
            return next(counter)

        self.assertEqual(f(key=None), 0)
        self.assertEqual(f(1, key=None), 0)
        self.assertEqual(f(1, 2, key=None), 0)
        self.assertEqual(f(key=1), 1)
        self.assertEqual(f(key=1, b=2), 1)
        self.assertEqual(f(key=1, refresh_=True), 2)

    def test_args_collision(self):
        counter = count()

        @self.cached()
        def f(*a, **b):
            return next(counter)

        # NB: in python2, without hash seed randomization,
        # hash('\0B') == hash('\0\0C')
        self.assertEqual(f(x='\0B'), 0)
        self.assertEqual(f(x='\0\0C'), 1)

    @parameterized.expand([
        (1, ),
        ('abc', ),
        ([1, 'b', None], ),
        ({"a": 1, "b": {"c": 2}}, ),
    ])
    def test_value_serialization(self, val):
        counter = count()

        @self.cached()
        def f(*a, **b):
            i = next(counter)
            if i == 0:
                return val

        self.assertEqual(f(), val)
        self.assertEqual(f(), val)

    def test_expiry(self):
        counter = count()

        @self.cached(maxage=10)
        def f(*a, **b):
            return next(counter)

        # populate
        with mock.patch('dwave.cloud.utils.decorators.epochnow', lambda: 0):
            self.assertEqual(f(), 0)
            self.assertEqual(f(1), 1)
            self.assertEqual(f(a=1, b=2), 2)

        # cache miss, expired
        with mock.patch('dwave.cloud.utils.decorators.epochnow', lambda: 15):
            self.assertEqual(f(), 3)
            self.assertEqual(f(1), 4)
            self.assertEqual(f(a=1, b=2), 5)

        # cache hit, still hot
        with mock.patch('dwave.cloud.utils.decorators.epochnow', lambda: 5):
            self.assertEqual(f(), 3)
            self.assertEqual(f(1), 4)
            self.assertEqual(f(a=1, b=2), 5)

    def test_per_call_maxage(self):
        counter = count()

        @self.cached(maxage=10)
        def f(*a, **b):
            return next(counter)

        # populate
        with mock.patch('dwave.cloud.utils.decorators.epochnow', lambda: 0):
            self.assertEqual(f(), 0)

        # expired for default maxage
        with mock.patch('dwave.cloud.utils.decorators.epochnow', lambda: 15):
            self.assertEqual(f(maxage_=20), 0)
            self.assertEqual(f(maxage_=15.01), 0)
            self.assertEqual(f(maxage_=15), 1)

        # cache hot for default maxage
        with mock.patch('dwave.cloud.utils.decorators.epochnow', lambda: 25):
            self.assertEqual(f(maxage_=11), 1)
            self.assertEqual(f(), 2)
            self.assertEqual(f(maxage_=1), 2)

    def test_default_zero_maxage(self):
        counter = count()

        @self.cached(maxage=0)
        def f(*a, **b):
            return next(counter)

        self.assertEqual(f(), 0)
        self.assertEqual(f(), 1)
        self.assertEqual(f(), 2)

    def test_exceptions(self):
        counter = count(0)

        @self.cached(maxage=0)
        def f():
            # raises ZeroDivisionError only on first call
            # we do not want to cache that!
            return 1.0 / next(counter)

        self.assertRaises(ZeroDivisionError, f)
        self.assertEqual(f(), 1)
        self.assertEqual(f(), 0.5)

    def test_disable(self):
        counter = count()

        @self.cached()
        def f():
            return next(counter)

        self.assertEqual(f(), 0)

        f.cached.disable()
        self.assertEqual(f(), 1)

        f.cached.enable()
        self.assertEqual(f(), 0)

    def test_disable_is_isolated(self):
        counter = count()

        @self.cached()
        def f():
            return next(counter)

        @self.cached()
        def g():
            return next(counter)

        self.assertEqual(f(), 0)
        self.assertEqual(g(), 1)

        f.cached.disable()
        self.assertEqual(f(), 2)
        self.assertEqual(g(), 1)

    def test_disabled_context(self):
        counter = count()

        @self.cached()
        def f():
            return next(counter)

        self.assertEqual(f(), 0)

        with cached.disabled():
            self.assertEqual(f(), 1)

        self.assertEqual(f(), 0)

    def test_disabled_decorator(self):
        counter = count()

        @self.cached()
        def f():
            return next(counter)

        self.assertEqual(f(), 0)

        @cached.disabled()
        def no_cache():
            return f()

        self.assertEqual(no_cache(), 1)
        self.assertEqual(f(), 0)


class TestCachedInMemoryDecorator(TestCachedCommon):

    def test_cached_isolation(self):
        counter = count()
        store = {}

        @self.cached(store=store)
        def f():
            return next(counter)

        @self.cached(store=store)
        def g():
            return next(counter)

        self.assertEqual(f(), 0)
        self.assertEqual(g(), 1)
        self.assertEqual(f(), 0)
        self.assertEqual(g(), 1)

    def test_shared_cache_bucket(self):
        counter = count()
        store = {}

        @self.cached(store=store, bucket='f')
        def f():
            return next(counter)

        @self.cached(store=store, bucket='f')
        def g():
            return next(counter)

        self.assertEqual(f(), 0)
        self.assertEqual(g(), 0)


class TestCachedOnDiskDecorator(TestCachedCommon):
    """Test @cached using on-disk store (via @cached.ondisk)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cached = partial(cached.ondisk, directory=self.tmpdir.name)

    def tearDown(self):
        # suppress tmp dir cleanup failures on windows when files are still in use
        # note: the temp dir is cleaned up automatically when `self.tmpdir` object is gc'ed
        with contextlib.suppress(OSError):
            self.tmpdir.cleanup()

    def test_persistency(self):
        counter = count()
        def f():
            return next(counter)

        f1 = self.cached(bucket='fixed')(f)
        self.assertEqual(f1(), 0)

        f2 = self.cached(bucket='fixed')(f)
        self.assertEqual(f2(), 0)


@unittest.skipUnless(hasattr(os, 'fork'), "os.fork() not available on this platform")
class TestCachedForking(unittest.TestCase):

    @unittest.skipUnless(sqlite3.threadsafety == 3,
                         "sqlite3 is not compiled with serialized threading mode support")
    def test_cache_connection_safety_post_forking(self):
        # note: this convoluted subprocess call approach seems to be simpler
        # than trying to kill the forked unittest suite after a forking test case.
        program = textwrap.dedent("""
            import os
            from dwave.cloud.utils.decorators import cached

            @cached.ondisk()
            def f():
                return 42

            # make sure db connection is opened
            f()

            os.fork()

            # make sure cache works in both parent and child
            if f() != 42:
                print("FAIL")
        """)

        run = subprocess.run(f"echo '{program}' | python", shell=True, capture_output=True)

        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.strip(), b'')
        self.assertEqual(run.stderr.strip(), b'')

    def test_import_only_survives_forking(self):
        # note: this convoluted subprocess call approach seems to be simpler
        # than trying to kill the forked unittest suite after a forking test case.
        program = textwrap.dedent("""
            import os
            import tempfile
            from dwave.cloud.utils.decorators import cached

            # isolate cache to avoid https://github.com/grantjenks/python-diskcache/issues/325
            @cached.ondisk(directory=tempfile.mkdtemp())
            def f():
                return 42

            os.fork()

            # make sure cache works in both parent and child
            if f() != 42:
                print("FAIL")
        """)

        run = subprocess.run(f"echo '{program}' | python", shell=True, capture_output=True)

        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout.strip(), b'')
        self.assertEqual(run.stderr.strip(), b'')


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
    def _snipped(inp: Union[str, uuid.UUID]) -> tuple[str, str]:
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


class TestDistUtils(unittest.TestCase):

    def test_get_distribution(self):
        # for simplicity, use a package we know for sure is installed in (test) env

        # baseline
        dist = get_distribution("pip")
        self.assertEqual(dist.name, "pip")
        ver = tuple(map(int, dist.version.split('.')))

        # version matching
        dist = get_distribution(f"pip~={ver[0]}.0")
        self.assertEqual(dist.name, "pip")

        with self.assertRaises(VersionNotFoundError):
            get_distribution(f"pip<{ver[0]}")
        with self.assertRaises(VersionNotFoundError):
            get_distribution(f"pip>{ver[0]+1}")
        with self.assertRaises(VersionNotFoundError):
            get_distribution(f"pip=={ver[0]-1}")

        # package matching
        with self.assertRaises(PackageNotFoundError):
            get_distribution("package-that-is-not-installed")

    def test_contrib_packages_smoke(self):
        # requires installed contrib packages, or the entry_points patcher we
        # plan to port from dwave-inspector
        try:
            get_contrib_packages()
        except:
            self.fail("can't enumerate dwave contrib packages")


def capture_stderr(fn):
    """Decorator that captures stderr and provides access via `output` argument."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        stream = io.TextIOWrapper(io.BytesIO())
        with contextlib.redirect_stderr(stream) as output:
            return fn(*args, output=output, **kwargs)
    return wrapper


class TestLogging(unittest.TestCase):

    def tearDown(self):
        configure_logging(logger)

    @capture_stderr
    def test_default_configure(self, output):
        configure_logging(logger)

        logger.info('test')
        logger.warning('secret: beefcafe-aaaa-bbbb-cccc-0123456789ab')

        # level is warning+, and secrets are filtered
        output.seek(0)
        lines = output.readlines()
        self.assertEqual(len(lines), 1)
        self.assertIn('bee...9ab', lines[0])

    @capture_stderr
    def test_structured_logging(self, output):
        configure_logging(logger, level=logging.INFO, structured_output=True)

        logger.info('test', extra=dict(key='value'))

        # level is info+, json output contains the extra dict
        output.seek(0)
        lines = output.readlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record.get('message'), 'test')
        self.assertEqual(record.get('key'), 'value')

    @capture_stderr
    def test_multiple_handlers(self, output):
        structured_stream = io.StringIO()

        configure_logging(
            logger, handler_level=logging.ERROR)
        configure_logging(
            logger, level=logging.DEBUG, additive=True,
            output_stream=structured_stream, structured_output=True)

        logger.debug('debug')
        logger.error('error')

        # structured_stream has all as json, stderr only errors in text
        output.seek(0)
        error = output.readlines()
        self.assertEqual(len(error), 1)
        self.assertIn('error', error[0])

        debug = list(map(json.loads, structured_stream.getvalue().splitlines()))
        self.assertEqual(len(debug), 2)
        self.assertEqual(debug[0].get('message'), 'debug')
        self.assertEqual(debug[1].get('message'), 'error')

    @capture_stderr
    def test_file_logging(self, output):
        with tempfile.TemporaryDirectory() as dir:
            logfile = os.path.join(dir, "out.log")

            configure_logging(logger, output_file=logfile)

            logger.error('test-msg')

            # log msg is in the log file
            with open(logfile, "r") as fp:
                self.assertIn('test-msg', fp.read())

            # there're no logs on stderr
            output.seek(0)
            self.assertEqual(output.read(), '')

            # reset log config to enable temp dir cleanup
            configure_logging(logger)


class TestLoggingHelpers(unittest.TestCase):

    def test_fast_stack(self):
        def f():
            return inspect.stack()

        def g():
            return fast_stack()

        stack = f()
        fast = g()

        self.assertEqual(len(stack), len(fast))
        self.assertTrue(all(s.filename == f.filename for s,f in zip(stack,fast)))

    def test_max_depth(self):
        self.assertEqual(len(fast_stack(max_depth=3)), 3)

    def test_get_caller_name(self):
        self.assertEqual(get_caller_name(), inspect.stack()[0].function)
        self.assertEqual(get_caller_name(1), inspect.stack()[1].function)


if __name__ == '__main__':
    unittest.main()

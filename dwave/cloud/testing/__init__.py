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

"""Testing utils."""

import os
import functools
from unittest import mock

__all__ = ['mock', 'iterable_mock_open', 'isolated_environ']


def iterable_mock_open(read_data):
    """Version of `mock.mock_open` that supports iteration
    (required when mocking `open` for `configparser.read`).

    Note the difference:

        1) iteration not working with `mock.mock_open`:

            >>> with mock.patch('builtins.open', mock.mock_open('1\n2\n3'), create=True):
            ...  for x in open('asd'):
            ...   print(x)
            ...
            Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "/usr/lib/python3.5/unittest/mock.py", line 2361, in mock_open
                mock.side_effect = reset_data
            AttributeError: 'str' object has no attribute 'side_effect'

        2) iteration working with our `iterable_mock_open`

            >>> with mock.patch('builtins.open', iterable_mock_open('1\n2\n3'), create=True):
            ...  for x in open('asd'):
            ...   print(x)
            ...
            1
            2
            3
    """
    m = mock.mock_open(read_data=read_data)
    m.return_value.__iter__ = lambda self: self
    m.return_value.__next__ = lambda self: next(iter(self.readline, ''))
    return m


class isolated_environ:
    """Context manager and a decorator for modified process environment
    isolation.

    Environment variables can be updated, added and removed. Complete
    environment can be cleared, or cleared only of a subset of variables
    that affect config loading (``DWAVE_*`` vars).

    On context clear, original `os.environ` is restored.

    Args:
        add (dict/Mapping):
            Values to add (or update) to the isolated `os.environ`.

        remove (dict/Mapping, or set/Iterable):
            Values to remove from the isolated `os.environ`.

        remove_dwave (bool, default=False):
            Remove dwave-cloud-client specific variables that affect config
            loading (prefixed with ``DWAVE_`` or ``DW_INTERNAL__``)

        empty (bool, default=False):
            Return empty environment.

    Context:
        Modified copy of global `os.environ`. Restored on context/decorator exit.

    Examples:
        Patch environment in a function scope::

            @isolated_environ(empty=True)
            def f():
                assert len(os.environ) == 0

            f()
            assert len(os.environ) > 0

        Patch environment in a context::

            with isolated_environ(empty=True) as env:
                assert len(os.environ) == 0

            assert len(os.environ) > 0
    """

    def start(self):
        self.patcher = mock.patch.dict(os.environ, values=self.add, clear=self.empty)
        self.patcher.start()

        for key in self.remove:
            os.environ.pop(key, None)

        for key in frozenset(os.environ.keys()):
            if self.remove_dwave and key.startswith("DWAVE_"):
                os.environ.pop(key, None)

        return self

    def stop(self):
        self.patcher.stop()

    def __init__(self, add=None, remove=None, remove_dwave=False, empty=False):
        if add is None:
            add = {}

        if remove is None:
            remove = {}

        self.add = add
        self.remove = remove
        self.remove_dwave = remove_dwave
        self.empty = empty

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            self.start()
            try:
                return fn(*args, **kwargs)
            finally:
                self.stop()

        return wrapper

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

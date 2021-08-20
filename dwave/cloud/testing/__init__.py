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
import contextlib
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


@contextlib.contextmanager
def isolated_environ(add=None, remove=None, remove_dwave=False, empty=False):
    """Context manager for modified process environment isolation.

    Environment variables can be updated, added and removed. Complete
    environment can be cleared, or cleared only only of a subset of variables
    that affect config loading (``DWAVE_*`` and ``DW_INTERNAL__*`` vars).

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
        Modified copy of global `os.environ`. Restored on context exit.
    """

    if add is None:
        add = {}

    if remove is None:
        remove = {}

    with mock.patch.dict(os.environ, values=add, clear=empty):
        for key in remove:
            os.environ.pop(key, None)

        for key in frozenset(os.environ.keys()):
            if remove_dwave and (key.startswith("DWAVE_") or key.startswith("DW_INTERNAL__")):
                os.environ.pop(key, None)

        yield os.environ

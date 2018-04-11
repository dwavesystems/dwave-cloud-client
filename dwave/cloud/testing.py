"""Testing utils."""

import os
import contextlib

try:
    from unittest import mock
except ImportError:
    import mock


@contextlib.contextmanager
def isolated_environ(add=None, remove=None, remove_dwave=False, empty=False):
    if add is None:
        add = {}
    with mock.patch.dict(os.environ, values=add, clear=empty):
        for key in frozenset(os.environ.keys()):
            if remove and key in remove:
                os.environ.pop(key, None)
            if remove_dwave and (key.startswith("DWAVE_") or key.startswith("DW_INTERNAL__")):
                os.environ.pop(key, None)
        yield os.environ

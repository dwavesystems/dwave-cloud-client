"""Testing utils."""

import os
import contextlib

try:
    from unittest import mock
except ImportError:
    import mock


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

    with mock.patch.dict(os.environ, values=add, clear=empty):
        for key in frozenset(os.environ.keys()):
            if remove and key in remove:
                os.environ.pop(key, None)
            if remove_dwave and (key.startswith("DWAVE_") or key.startswith("DW_INTERNAL__")):
                os.environ.pop(key, None)

        yield os.environ

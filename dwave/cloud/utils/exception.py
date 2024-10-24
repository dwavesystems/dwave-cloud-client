# Copyright 2024 D-Wave Inc.
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

"""Exception handling utilities for private and Ocean-internal use.

.. versionchanged:: 0.12.0
   These functions previously lived under ``dwave.cloud.utils``.
"""

from collections import abc
from typing import Any, Union

__all__ = ['hasinstance', 'exception_chain', 'is_caused_by']


def hasinstance(objs: abc.Iterable[Any], class_or_tuple: Union[Any, tuple[Any]]):
    """Extension of ``isinstance`` to iterables/sequences. Returns True iff the
    sequence contains at least one object which is instance of ``class_or_tuple``.
    """

    return any(isinstance(e, class_or_tuple) for e in objs)


def exception_chain(exception: Exception):
    """Traverse the chain of embedded exceptions, yielding one at the time.

    Args:
        exception (:class:`Exception`): Chained exception.

    Yields:
        :class:`Exception`: The next exception in the input exception's chain.

    Examples:
        def f():
            try:
                1/0
            except ZeroDivisionError:
                raise ValueError

        try:
            f()
        except Exception as e:
            assert(hasinstance(exception_chain(e), ZeroDivisionError))

    See: PEP-3134.
    """

    while exception:
        yield exception

        # explicit exception chaining, i.e `raise .. from ..`
        if exception.__cause__:
            exception = exception.__cause__

        # implicit exception chaining
        elif exception.__context__:
            exception = exception.__context__

        else:
            return


def is_caused_by(exception: Exception,
                 exception_types: Union[Exception, tuple[Exception]]
                 ) -> bool:
    """Check if any of ``exception_types`` is causing the ``exception``.
    Equivalently, check if any of ``exception_types`` is contained in the
    exception chain rooted at ``exception``.

    Args:
        exception (:class:`Exception`):
            Chained exception.

        exception_types (:class:`Exception` or tuple of :class:`Exception`):
            Exception type or a tuple of exception types to check for.

    Returns:
        bool:
            True when ``exception`` is caused by any of the exceptions in
            ``exception_types``.
    """

    return hasinstance(exception_chain(exception), exception_types)

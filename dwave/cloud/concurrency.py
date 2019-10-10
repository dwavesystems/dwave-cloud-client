# Copyright 2019 D-Wave Systems Inc.
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

"""Concurrency utilities."""

import sys
import functools
import concurrent.futures

try:
    import queue
except ImportError:     # pragma: no cover
    # python 2
    import Queue as queue

__all__ = ['PriorityThreadPoolExecutor']


@functools.total_ordering
class _PrioritizedWorkItem(concurrent.futures.thread._WorkItem):
    """Extension of :class:`concurrent.futures.thread._WorkItem` that handles
    execution priority passed in `kwargs`.
    """

    def __init__(self, item):
        if not isinstance(item, concurrent.futures.thread._WorkItem):
            raise TypeError("concurrent.futures.thread._WorkItem expected")

        # copy constructor
        kwargs = item.kwargs.copy()
        if isinstance(item, _PrioritizedWorkItem):
            kwargs.update(priority=item.priority)

        # init from _WorkItem
        super(_PrioritizedWorkItem, self).__init__(
            item.future, item.fn, item.args, kwargs)

        self.priority = self.kwargs.pop('priority', sys.maxsize)

    def __lt__(self, other):
        # None < _PrioritizedWorkItem(..)
        if other is None:
            return False

        # otherwise, compare only on priority
        return self.priority < other.priority


class _PrioritizingQueue(queue.PriorityQueue):
    """Re-pack :class:`concurrent.futures.thread._WorkItem` (on queue put) into
    :class:`_PrioritizedWorkItem` subclass that is ordered according to
    priority.
    """

    def put(self, item, *args, **kwargs):
        # unpack item, extract priority
        if isinstance(item, concurrent.futures.thread._WorkItem):
            item = _PrioritizedWorkItem(item)

        # in python 2, `queue.PriorityQueue` is an old-style class!
        queue.PriorityQueue.put(self, item, *args, **kwargs)


class PriorityThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """Add support for task priority to the standard
    :class:`concurrent.futures.ThreadPoolExecutor` FIFO-queue based
    :class:`concurrent.futures.Executor` implementation.

    Interface is identical to :class:`concurrent.futures.ThreadPoolExecutor`,
    except the `.submit()` which now accepts optional `priority` keyword
    argument::

        def submit(self, fn, *args, priority=sys.maxsize, **kwargs):
            ...

    Note: if `priority` is omitted, the behavior is identical to that of the
    superclass, `ThreadPoolExecutor`.
    """

    def __init__(self, *args, **kwargs):
        super(PriorityThreadPoolExecutor, self).__init__(*args, **kwargs)
        self._work_queue = _PrioritizingQueue()

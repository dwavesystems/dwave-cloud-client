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
import queue

__all__ = ['PriorityThreadPoolExecutor']


@functools.total_ordering
class _PriorityOrderedItem(object):
    """Generic priority queue item with ordering defined according to the
    priority attribute.
    """

    def __init__(self, item, priority=None):
        # by default, None < Any
        if priority is None:
            if item is None:
                priority = -sys.maxsize
            else:
                priority = sys.maxsize

        self.item = item
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __eq__(self, other):
        return self.priority == other.priority


class _PrioritizedWorkItem(_PriorityOrderedItem):
    """Extension of :class:`_PriorityOrderedItem` that handles execution
    priority passed in `kwargs` of :class:`concurrent.futures.thread._WorkItem`
    items.
    """

    def __init__(self, item):
        if not isinstance(item, concurrent.futures.thread._WorkItem):
            raise TypeError("concurrent.futures.thread._WorkItem expected")

        # copy constructor
        if isinstance(item, _PrioritizedWorkItem):
            priority = item.priority
        else:
            priority = item.kwargs.pop('priority', sys.maxsize)

        super().__init__(item, priority)


class _PrioritizingQueue(queue.PriorityQueue):
    """Re-pack :class:`concurrent.futures.thread._WorkItem` (on queue put) into
    :class:`_PrioritizedWorkItem` subclass that is ordered according to
    priority.
    """

    def put(self, item, *args, **kwargs):
        # unpack item, extract priority
        if isinstance(item, concurrent.futures.thread._WorkItem):
            item = _PrioritizedWorkItem(item)
        else:
            item = _PriorityOrderedItem(item)

        super().put(item, *args, **kwargs)

    def get(self, *args, **kwargs):
        prioritized_item = super().get(*args, **kwargs)
        return prioritized_item.item


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
        super().__init__(*args, **kwargs)
        self._work_queue = _PrioritizingQueue()


class Present(concurrent.futures.Future):
    """Already resolved :class:`~concurrent.futures.Future` object.

    Users should treat this class as just another
    :class:`~concurrent.futures.Future`, the difference being an implementation
    detail: :class:`Present` is "resolved" at construction time.
    """

    def __init__(self, result=None, exception=None):
        super().__init__()
        if result is not None:
            self.set_result(result)
        elif exception is not None:
            self.set_exception(exception)
        else:
            raise ValueError("can't provide both 'result' and 'exception'")

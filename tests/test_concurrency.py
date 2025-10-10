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

import sys
import unittest
import threading
import concurrent.futures

from dwave.cloud.concurrency import (
    _PriorityOrderedItem,
    _PrioritizedWorkItem,
    _PrioritizingQueue,
    PriorityThreadPoolExecutor,
)

# in python 3.14, _WorkItem aggregates (fn, args, kwargs) under task
try:
    # in py313 and earlier, _WorkItem construction with only 2 positional args fails with TypeError
    workitem_uses_task = hasattr(concurrent.futures.thread._WorkItem(None, None), 'task')
except TypeError:
    workitem_uses_task = False

def create_workitem(future, task):
    if workitem_uses_task:
        return concurrent.futures.thread._WorkItem(future, task)
    else:
        return concurrent.futures.thread._WorkItem(future, *task)

def get_workitem_task(w):
    if workitem_uses_task:
        return w.task
    else:
        return (w.fn, w.args, w.kwargs)


class Test_PrioritizedWorkItem(unittest.TestCase):

    def test_construction(self):
        priority = 2
        future = concurrent.futures.Future()
        fn = lambda: None
        args = (1,)
        kwargs = {'a': 1}
        task = (fn, args, kwargs)

        # without priority
        w = create_workitem(future, task)
        pw = _PrioritizedWorkItem(w)
        self.assertEqual(pw.priority, sys.maxsize)
        self.assertEqual(pw.item.future, future)
        self.assertEqual(get_workitem_task(pw.item), task)

        # with priority
        kwargs_pri = kwargs | dict(priority=priority)
        task = (fn, args, kwargs_pri)
        w = create_workitem(future, task)
        pw = _PrioritizedWorkItem(w)
        self.assertEqual(pw.priority, priority)
        self.assertEqual(pw.item.future, future)
        self.assertEqual(get_workitem_task(pw.item), task)

    def test_priority_ordering(self):
        w1 = _PrioritizedWorkItem(create_workitem(None, (None, (), dict(priority=1))))
        w2 = _PrioritizedWorkItem(create_workitem(None, (None, (), dict(priority=2))))

        # ordered
        self.assertLess(w1, w2)
        self.assertGreater(w2, w1)

        # always greater that None
        none = _PriorityOrderedItem(None)
        self.assertLess(none, w1)
        self.assertGreater(w1, none)
        self.assertNotEqual(w1, none)


class Test_PrioritizingQueue(unittest.TestCase):

    def test_prioritization(self):
        w1 = create_workitem(None, t1 := (None, (1,), dict(priority=1)))
        w2 = _PrioritizedWorkItem(create_workitem(None, t2 := (None, (2,), dict(priority=2))))
        w3 = None
        w4 = create_workitem(None, t4 := (None, (4,), {}))

        q = _PrioritizingQueue()

        # put different types of items in queue
        q.put(w2)
        q.put(w1)
        q.put(w3)
        q.put(w4)

        # verify order on get
        self.assertEqual(q.get(), None)     # w3
        self.assertEqual(get_workitem_task(q.get()), t1)
        self.assertEqual(get_workitem_task(q.get().item), t2)
        self.assertEqual(get_workitem_task(q.get()), t4)
        self.assertTrue(q.empty())

    def test_double_none_edgecase(self):
        # this can happen on interpreter exit (concurrent.futures.thread._python_exit)
        # thread count adjustment, manual calls to executor.shutdown, etc.
        q = _PrioritizingQueue()
        q.put(None)
        q.put(None)
        self.assertEqual(q.get(), None)
        self.assertEqual(q.get(), None)


class TestPriorityThreadPoolExecutor(unittest.TestCase):

    def test_fallback(self):
        """Without priority specified, it falls back to ThreadPoolExecutor mode."""

        counter = threading.BoundedSemaphore(value=3)

        def worker():
            counter.acquire(blocking=False)

        with PriorityThreadPoolExecutor(max_workers=3) as executor:
            fs = [executor.submit(worker) for _ in range(3)]
            concurrent.futures.wait(fs)

        self.assertFalse(counter.acquire(blocking=False))

        # verify executor shutdown (all threads stopped)
        self.assertFalse(any(t.is_alive() for t in executor._threads))

    def test_prioritization(self):

        # we need a dummy busy bee task to block the executor thread pool
        # until we queue actual jobs -- in order to test priority queue
        # (executor might running the first task as soon as it's enqueued)
        go = threading.Event()

        def bee():
            go.wait()

        # we'll collect results here, ensuring only one worker writes at the time
        results = []
        lock = threading.Lock()

        def worker(val):
            with lock:
                results.append(val)

        with PriorityThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(bee)
            fs = [executor.submit(worker, p, priority=p) for p in [2, 0, 3, 1]]
            go.set()
            concurrent.futures.wait(fs)

        self.assertListEqual(results, [0, 1, 2, 3])

        # verify executor shutdown (all threads stopped)
        self.assertFalse(any(t.is_alive() for t in executor._threads))

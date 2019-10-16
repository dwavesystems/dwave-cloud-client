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

import io
import os
import time
import unittest
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait

from dwave.cloud.utils import tictoc
from dwave.cloud.upload import RandomAccessIOBaseBuffer, FileBuffer, ChunkedData
from dwave.cloud.client import Client

from tests import config


class TestFileBufferABC(unittest.TestCase):

    def test_invalid(self):
        class InvalidFileBuffer(RandomAccessIOBaseBuffer):
            pass

        with self.assertRaises(TypeError):
            InvalidFileBuffer()

    def test_valid(self):
        class ValidFileBuffer(RandomAccessIOBaseBuffer):
            def __len__(self):
                return NotImplementedError
            def __getitem__(self, key):
                return NotImplementedError

        try:
            ValidFileBuffer()
        except:
            self.fail("unexpected interface of RandomAccessIOBaseBuffer")


class TestFileBuffer(unittest.TestCase):
    data = b'0123456789'

    def verify_getter(self, fb, data):
        n = len(data)

        # integer indexing
        self.assertEqual(fb[0], data[0:1])
        self.assertEqual(fb[n-1], data[n-1:n])

        # negative integer indexing
        self.assertEqual(fb[-1], data[-1:n])
        self.assertEqual(fb[-n], data[0:1])

        # out of bounds integer indexing
        self.assertEqual(fb[n], data[n:n+1])

        # non-integer key
        with self.assertRaises(TypeError):
            fb['a']

        # empty slices
        self.assertEqual(fb[1:0], b'')

        # slicing
        self.assertEqual(fb[:], data[:])
        self.assertEqual(fb[0:n//2], data[0:n//2])
        self.assertEqual(fb[-n//2:], data[-n//2:])

    def verify_getinto(self, fb, data):
        n = len(data)

        # integer indexing
        b = bytearray(n)
        self.assertEqual(fb.getinto(0, b), 1)
        self.assertEqual(b[0], data[0])

        self.assertEqual(fb.getinto(n-1, b), 1)
        self.assertEqual(b[0], data[n-1])

        # negative integer indexing
        self.assertEqual(fb.getinto(-1, b), 1)
        self.assertEqual(b[0], data[-1])

        self.assertEqual(fb.getinto(-n, b), 1)
        self.assertEqual(b[0], data[-n])

        # out of bounds integer indexing => nop
        self.assertEqual(fb.getinto(n, b), 0)

        # non-integer key
        with self.assertRaises(TypeError):
            fb.getinto('a', b)

        # empty slices
        self.assertEqual(fb.getinto(slice(1, 0), b), 0)

        # slicing
        b = bytearray(n)
        self.assertEqual(fb.getinto(slice(None), b), n)
        self.assertEqual(b, data)

        b = bytearray(n)
        self.assertEqual(fb.getinto(slice(0, n//2), b), n//2)
        self.assertEqual(b[0:n//2], data[0:n//2])
        self.assertEqual(b[n//2:], bytearray(n//2))

        b = bytearray(n)
        self.assertEqual(fb.getinto(slice(-n//2, None), b), n//2)
        self.assertEqual(b[:n//2], data[-n//2:])
        self.assertEqual(b[n//2:], bytearray(n//2))

    def test_view_from_memory_bytes(self):
        data = self.data
        fp = io.BytesIO(data)
        fb = FileBuffer(fp)

        self.assertEqual(len(fb), len(data))
        self.verify_getter(fb, data)
        self.verify_getinto(fb, data)

    def test_view_from_memory_string(self):
        data = self.data.decode()
        fp = io.StringIO(data)

        with self.assertRaises(TypeError):
            fb = FileBuffer(fp)

    def test_view_from_file_like(self):
        data = self.data

        # create file-like temporary object (on POSIX this is a tmp file)
        with tempfile.TemporaryFile() as fp:
            fp.write(data)
            fp.seek(0)
            fb = FileBuffer(fp, strict=False)

            self.assertEqual(len(fb), len(data))
            self.verify_getter(fb, data)
            self.verify_getinto(fb, data)

    def test_view_from_disk_file(self):
        data = self.data

        # create temporary file
        fd, path = tempfile.mkstemp()
        os.write(fd, data)
        os.close(fd)

        # test FileBuffer from file on disk (read access)
        with io.open(path, 'rb') as fp:
            fb = FileBuffer(fp)

            self.assertEqual(len(fb), len(data))
            self.verify_getter(fb, data)
            self.verify_getinto(fb, data)

        # works also for read+write access
        with io.open(path, 'r+b') as fp:
            fb = FileBuffer(fp)

            self.assertEqual(len(fb), len(data))
            self.verify_getter(fb, data)
            self.verify_getinto(fb, data)

        # fail without read access
        with io.open(path, 'wb') as fp:
            with self.assertRaises(TypeError):
                FileBuffer(fp)

        # remove temp file
        os.unlink(path)

    def test_critical_section_respected(self):
        # setup a shared file view
        data = self.data
        fp = io.BytesIO(data)
        fb = FileBuffer(fp)

        # file slices
        slice_a = slice(0, 7)
        slice_b = slice(3, 5)

        # add a noticeable sleep inside the critical section (on `file.seek`),
        # resulting in minimal runtime equal to (N runs * sleep in crit sect)
        sleep = 0.25
        def blocking_seek(start):
            time.sleep(sleep)
            return io.BytesIO.seek(fb._fp, start)
        fb._fp.seek = blocking_seek

        # define the worker
        def worker(slice_):
            return fb[slice_]

        # run the worker a few times in parallel
        executor = ThreadPoolExecutor(max_workers=3)
        slices = [slice_a, slice_b, slice_a]
        futures = [executor.submit(worker, s) for s in slices]
        with tictoc() as timer:
            wait(futures)

        # verify results
        results = [f.result() for f in futures]
        expected = [data[s] for s in slices]
        self.assertEqual(results, expected)

        # verify runtime is consistent with a blocking critical section
        self.assertGreaterEqual(timer.dt, 0.9 * len(results) * sleep)


class TestChunkedData(unittest.TestCase):
    data = b'0123456789'

    def verify_chunking(self, cd, chunks_expected):
        self.assertEqual(len(cd), len(chunks_expected))
        self.assertEqual(cd.num_chunks, len(chunks_expected))

        chunks_iter = [c.getvalue() for c in cd]
        chunks_explicit = []
        for idx in range(len(cd)):
            chunks_explicit.append(cd.chunk(idx).getvalue())

        self.assertListEqual(chunks_iter, chunks_expected)
        self.assertListEqual(chunks_explicit, chunks_iter)

    def test_chunks_from_bytes(self):
        cd = ChunkedData(self.data, chunk_size=3)
        chunks_expected = [b'012', b'345', b'678', b'9']
        self.verify_chunking(cd, chunks_expected)

    def test_chunks_from_str(self):
        cd = ChunkedData(self.data.decode('ascii'), chunk_size=3)
        chunks_expected = [b'012', b'345', b'678', b'9']
        self.verify_chunking(cd, chunks_expected)

    def test_chunks_from_memory_file(self):
        data = io.BytesIO(self.data)
        cd = ChunkedData(data, chunk_size=3)
        chunks_expected = [b'012', b'345', b'678', b'9']
        self.verify_chunking(cd, chunks_expected)

    def test_chunk_size_edges(self):
        with self.assertRaises(ValueError):
            cd = ChunkedData(self.data, chunk_size=0)

        cd = ChunkedData(self.data, chunk_size=1)
        chunks_expected = [self.data[i:i+1] for i in range(len(self.data))]
        self.verify_chunking(cd, chunks_expected)

        cd = ChunkedData(self.data, chunk_size=len(self.data))
        chunks_expected = [self.data]
        self.verify_chunking(cd, chunks_expected)


@unittest.skipUnless(config, "No live server configuration available.")
class TestMultipartUpload(unittest.TestCase):

    def test_smoke_test(self):
        with Client(**config) as client:
            data = b'123'
            future = client.upload_problem(data)
            try:
                future.result()
            except Exception as e:
                self.fail()

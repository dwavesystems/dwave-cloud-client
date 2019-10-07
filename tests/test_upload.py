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
import unittest
import tempfile

from dwave.cloud.upload import RandomAccessIOBaseView, FileView


class TestFileViewABC(unittest.TestCase):

    def test_invalid(self):
        class InvalidFileView(RandomAccessIOBaseView):
            pass

        with self.assertRaises(TypeError):
            InvalidFileView()

    def test_valid(self):
        class ValidFileView(RandomAccessIOBaseView):
            def __len__(self):
                return NotImplementedError
            def __getitem__(self, key):
                return NotImplementedError

        try:
            fv = ValidFileView()
        except:
            self.fail("unexpected interface of RandomAccessIOBaseView")


class TestFileView(unittest.TestCase):
    data = b'0123456789'

    def verify_getter(self, fv, data):
        n = len(data)

        # integer indexing
        self.assertEqual(fv[0], data[0:1])
        self.assertEqual(fv[n-1], data[n-1:n])

        # negative integer indexing
        self.assertEqual(fv[-1], data[-1:n])
        self.assertEqual(fv[-n], data[0:1])

        # out of bounds integer indexing
        self.assertEqual(fv[n], data[n:n+1])

        # slicing
        self.assertEqual(fv[:], data[:])
        self.assertEqual(fv[0:n//2], data[0:n//2])
        self.assertEqual(fv[-n//2:], data[-n//2:])

    def test_view_from_memory_bytes(self):
        data = self.data
        fp = io.BytesIO(data)
        fv = FileView(fp)

        self.assertEqual(len(fv), len(data))
        self.verify_getter(fv, data)

    def test_view_from_memory_string(self):
        data = self.data.decode()
        fp = io.StringIO(data)
        fv = FileView(fp)

        self.assertEqual(len(fv), len(data))
        self.verify_getter(fv, data)

    def test_view_from_file_like(self):
        data = self.data

        # create file-like temporary object (on POSIX this is a tmp file)
        with tempfile.TemporaryFile() as fp:
            fp.write(data)
            fp.seek(0)
            fv = FileView(fp)

            self.assertEqual(len(fv), len(data))
            self.verify_getter(fv, data)

    def test_view_from_disk_file(self):
        data = self.data

        # create temporary file
        fd, path = tempfile.mkstemp()
        os.write(fd, data)
        os.close(fd)

        # test FileView from file on disk
        with open(path, 'rb') as fp:
            fv = FileView(fp)

            self.assertEqual(len(fv), len(data))
            self.verify_getter(fv, data)

        # file has to be open for reading
        with open(path, 'wb') as fp:
            with self.assertRaises(ValueError):
                fv = FileView(fp)

        # remove temp file
        os.unlink(path)

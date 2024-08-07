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
import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from unittest import mock

from parameterized import parameterized

from dwave.cloud.client import Client
from dwave.cloud.exceptions import SAPIError, ProblemUploadError
from dwave.cloud.upload import (
    Gettable, GettableFile, GettableMemory, FileView, ChunkedData)
from dwave.cloud.utils.time import tictoc

from tests import config
from tests.api.mocks import choose_reply


class TestGettableABC(unittest.TestCase):

    def test_invalid(self):
        class InvalidGettable(Gettable):
            pass

        with self.assertRaises(TypeError):
            InvalidGettable()

    def test_valid(self):
        class ValidGettable(Gettable):
            def __len__(self):
                return NotImplementedError
            def __getitem__(self, key):
                return NotImplementedError
            def getinto(self, key):
                return NotImplementedError

        try:
            ValidGettable()
        except:
            self.fail("unexpected interface of Gettable")


class TestGettables(unittest.TestCase):
    data = b'0123456789'

    def verify_getitem(self, gettable, data):
        n = len(data)
        # python 2 fix: indexing of bytes returns a slice (not int)
        data = bytearray(data)

        # integer indexing
        self.assertEqual(gettable[0], data[0])
        self.assertEqual(gettable[n-1], data[n-1])

        # negative integer indexing
        self.assertEqual(gettable[-1], data[-1])
        self.assertEqual(gettable[-n], data[-n])

        # out of bounds integer indexing
        with self.assertRaises(IndexError):
            gettable[n]

        # non-integer key
        with self.assertRaises(TypeError):
            gettable['a']

        # empty slices
        self.assertEqual(gettable[1:0], b'')

        # slicing
        self.assertEqual(gettable[:], data[:])
        self.assertEqual(gettable[0:n//2], data[0:n//2])
        self.assertEqual(gettable[-n//2:], data[-n//2:])

    def verify_getinto(self, gettable, data):
        n = len(data)
        # python 2 fix: indexing of bytes returns a slice (not int)
        data = bytearray(data)

        # integer indexing
        b = bytearray(n)
        self.assertEqual(gettable.getinto(0, b), 1)
        self.assertEqual(b[0], data[0])

        self.assertEqual(gettable.getinto(n-1, b), 1)
        self.assertEqual(b[0], data[n-1])

        # negative integer indexing
        self.assertEqual(gettable.getinto(-1, b), 1)
        self.assertEqual(b[0], data[-1])

        self.assertEqual(gettable.getinto(-n, b), 1)
        self.assertEqual(b[0], data[-n])

        # out of bounds integer indexing => nop
        self.assertEqual(gettable.getinto(n, b), 0)

        # non-integer key
        with self.assertRaises(TypeError):
            gettable.getinto('a', b)

        # empty slices
        self.assertEqual(gettable.getinto(slice(1, 0), b), 0)

        # slicing
        b = bytearray(n)
        self.assertEqual(gettable.getinto(slice(None), b), n)
        self.assertEqual(b, data)

        b = bytearray(n)
        self.assertEqual(gettable.getinto(slice(0, n//2), b), n//2)
        self.assertEqual(b[0:n//2], data[0:n//2])
        self.assertEqual(b[n//2:], bytearray(n//2))

        b = bytearray(n)
        self.assertEqual(gettable.getinto(slice(-n//2, None), b), n//2)
        self.assertEqual(b[:n//2], data[-n//2:])
        self.assertEqual(b[n//2:], bytearray(n//2))

        # slicing into a buffer too small
        m = 3
        b = bytearray(m)
        self.assertEqual(gettable.getinto(slice(None), b), m)
        self.assertEqual(b, data[:m])

    def test_gettable_file_from_memory_bytes(self):
        data = self.data
        fp = io.BytesIO(data)
        gf = GettableFile(fp)

        self.assertEqual(len(gf), len(data))
        self.verify_getitem(gf, data)
        self.verify_getinto(gf, data)

    def test_gettable_file_from_memory_string(self):
        data = self.data.decode()
        fp = io.StringIO(data)

        with self.assertRaises(TypeError):
            GettableFile(fp)

    def test_gettable_file_from_file_like(self):
        data = self.data

        # create file-like temporary object (on POSIX this is a tmp file)
        with tempfile.TemporaryFile() as fp:
            fp.write(data)
            fp.seek(0)
            gf = GettableFile(fp, strict=False)

            self.assertEqual(len(gf), len(data))
            self.verify_getitem(gf, data)
            self.verify_getinto(gf, data)

    def test_gettable_file_from_disk_file(self):
        data = self.data

        # create temporary file
        fd, path = tempfile.mkstemp()
        os.write(fd, data)
        os.close(fd)

        # test GettableFile from file on disk (read access)
        with io.open(path, 'rb') as fp:
            gf = GettableFile(fp)

            self.assertEqual(len(gf), len(data))
            self.verify_getitem(gf, data)
            self.verify_getinto(gf, data)

        # works also for read+write access
        with io.open(path, 'r+b') as fp:
            gf = GettableFile(fp)

            self.assertEqual(len(gf), len(data))
            self.verify_getitem(gf, data)
            self.verify_getinto(gf, data)

        # fail without read access
        with io.open(path, 'wb') as fp:
            with self.assertRaises(TypeError):
                GettableFile(fp)

        # remove temp file
        os.unlink(path)

    def test_gettable_file_critical_section_respected(self):
        # setup a shared file view
        data = self.data
        fp = io.BytesIO(data)
        gf = GettableFile(fp)

        # file slices
        slice_a = slice(0, 7)
        slice_b = slice(3, 5)

        # add a noticeable sleep inside the critical section (on `file.seek`),
        # resulting in minimal runtime equal to (N runs * sleep in crit sect)
        sleep = 0.25
        def blocking_seek(start):
            time.sleep(sleep)
            return io.BytesIO.seek(gf._fp, start)
        gf._fp.seek = blocking_seek

        # define the worker
        def worker(slice_):
            return gf[slice_]

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

    def test_gettable_memory_from_bytes_like(self):
        data_objects = [
            bytes(self.data),
            bytearray(self.data),
            memoryview(self.data)
        ]

        for data in data_objects:
            gm = GettableMemory(data)

            self.assertEqual(len(gm), len(data))
            self.verify_getitem(gm, data)
            self.verify_getinto(gm, data)


class TestFileView(unittest.TestCase):
    # python 2 fix: indexing of bytes returns a slice (not int)
    data = bytearray(b'0123456789')

    def test_file_interface(self):
        data = self.data
        size = len(data)
        fp = io.BytesIO(data)
        gf = GettableFile(fp)
        fv = FileView(gf)

        # partial read
        self.assertEqual(fv.read(1), data[0:1])

        # read all, also check continuity
        self.assertEqual(fv.read(), data[1:])

        # seek and tell
        self.assertEqual(len(fv), size)

        self.assertEqual(fv.seek(2), 2)
        self.assertEqual(fv.tell(), 2)

        self.assertEqual(fv.seek(2, io.SEEK_CUR), 4)
        self.assertEqual(fv.tell(), 4)

        self.assertEqual(fv.seek(0, io.SEEK_END), size)
        self.assertEqual(fv.tell(), size)

        # IOBase derived methods
        fv.seek(0)
        self.assertEqual(fv.readlines(), [data])

    def test_view_interface(self):
        data = self.data
        size = len(data)
        fp = io.BytesIO(data)
        gf = GettableFile(fp)
        fv = FileView(gf)

        # view, slice index
        subfv = fv[1:-1]
        self.assertEqual(subfv.read(), data[1:-1])
        self.assertEqual(len(subfv), size - 2)

        # view, integer index
        self.assertEqual(fv[2], data[2])

        # view, out of bounds index
        with self.assertRaises(IndexError):
            fv[size]

        # view are independent
        self.assertEqual(fv[:2].read(), data[:2])
        self.assertEqual(fv[-2:].read(), data[-2:])


class TestChunkedData(unittest.TestCase):
    data = b'0123456789'

    def verify_chunking(self, cd, chunks_expected):
        self.assertEqual(len(cd), len(chunks_expected))
        self.assertEqual(cd.num_chunks, len(chunks_expected))
        self.assertEqual(cd.total_size, sum(map(len, chunks_expected)))

        chunks_iter = [c.read() for c in cd]
        chunks_explicit = []
        for idx in range(len(cd)):
            chunks_explicit.append(cd.chunk(idx).read())

        self.assertListEqual(chunks_iter, chunks_expected)
        self.assertListEqual(chunks_explicit, chunks_iter)

    def test_chunks_from_bytes(self):
        cd = ChunkedData(self.data, chunk_size=3)
        chunks_expected = [b'012', b'345', b'678', b'9']
        self.verify_chunking(cd, chunks_expected)

    def test_chunks_from_bytearray(self):
        cd = ChunkedData(bytearray(self.data), chunk_size=3)
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

    @parameterized.expand([
        ("BQM",),
        ("BinaryQuadraticModel",),
        ("AdjVectorBQM",),
    ])
    def test_chunks_from_bqm(self, bqm_cls_name):
        try:
            import dimod
            bqm_cls = getattr(dimod, bqm_cls_name)
            bqm_cls.to_file
        except (ImportError, AttributeError):
            self.skipTest(f"dimod.{bqm_cls_name}.to_file() unavailable")

        # serialize a BQM via .to_file
        bqm = bqm_cls.from_ising({'a': 1}, {})
        with bqm.to_file() as bqm_file:     # returns dimod's FileView
            chunk_size = 10
            cd = ChunkedData(bqm_file, chunk_size=chunk_size)

            # verify chunks
            bqm_file.seek(0)
            raw = bqm_file.read()
            chunks_expected = [raw[i:i+chunk_size] for i in range(0, len(raw), chunk_size)]
            self.verify_chunking(cd, chunks_expected)

    def test_chunks_from_dqm(self):
        try:
            from dimod import DQM
        except ImportError:
            self.skipTest("dimod.DQM unavailable")

        # serialize a DQM via .to_file
        dqm = DQM()
        dqm.add_variable(1)
        with dqm.to_file() as dqm_file:    # returns SpooledTemporaryFile subclass
            chunk_size = 100
            cd = ChunkedData(dqm_file, chunk_size=chunk_size)

            # verify chunks
            dqm_file.seek(0)
            raw = dqm_file.read()
            chunks_expected = [raw[i:i+chunk_size] for i in range(0, len(raw), chunk_size)]
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

    def test_chunk_generators(self):
        cd = ChunkedData(self.data, chunk_size=3)
        chunks_expected = [b'012', b'345', b'678', b'9']
        chunks_generated = [g().read() for g in cd.generators()]
        self.assertListEqual(chunks_expected, chunks_generated)


@mock.patch('time.sleep', lambda *args: None)
class TestMockedMultipartUpload(unittest.TestCase):

    @mock.patch.multiple(Client, _UPLOAD_PART_SIZE_BYTES=1)
    def test_single_problem_end_to_end(self):
        """Verify a fresh problem multipart upload works end to end."""

        upload_data = b'123'
        upload_problem_id = '84ef154c-28f9-46ed-9f22-aec0583499f2'

        parts = list(range(len(upload_data)))
        part_data = [upload_data[i:i+1] for i in parts]

        _md5 = Client._digest
        _hex = Client._checksum_hex
        _b64 = Client._checksum_b64
        part_digest = [_md5(part_data[i]) for i in parts]
        combine_checksum = _hex(_md5(b''.join(part_digest)))

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.MagicMock()
            session.__enter__ = lambda *args: session

            def get(path, seq=iter(range(2))):
                all_parts = [{"part_number": i+1,
                              "checksum": _hex(part_digest[i])} for i in parts]

                return choose_reply((path, next(seq)), {
                    # initial upload status
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 0):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": []}),

                    # final upload status
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 1):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": all_parts}),
                })

            def post(path, **kwargs):
                json_ = kwargs.pop('json')
                body = json.dumps(sorted(json_.items()))
                return choose_reply((path, body), {
                    # initiate upload
                    ('bqm/multipart',
                     json.dumps([('size', len(upload_data))])):
                        json.dumps({'id': upload_problem_id}),

                    # combine parts
                    ('bqm/multipart/{}/combine'.format(upload_problem_id),
                     json.dumps([('checksum', combine_checksum)])):
                        json.dumps({}),
                })

            def put(path, data, headers):
                body = data.read()
                headers = json.dumps(sorted(headers.items()))
                replies = {
                    (
                        'bqm/multipart/{}/part/{}'.format(upload_problem_id, i+1),
                        part_data[i],
                        json.dumps(sorted([
                            ('Content-MD5', _b64(part_digest[i])),
                            ('Content-Type', 'application/octet-stream')
                        ]))
                    ): json.dumps({})
                    for i in parts
                }
                return choose_reply((path, body, headers), replies)

            session.get = get
            session.put = put
            session.post = post

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(endpoint='endpoint', token='token') as client:

                future = client.upload_problem_encoded(upload_data)
                try:
                    returned_problem_id = future.result()
                except Exception as e:
                    self.fail(e)

                self.assertEqual(returned_problem_id, upload_problem_id)

    @mock.patch.multiple(Client, _UPLOAD_PART_SIZE_BYTES=1)
    def test_partial_upload(self):
        """Verify only missing parts are uploaded."""

        upload_data = b'123'
        upload_problem_id = '84ef154c-28f9-46ed-9f22-aec0583499f2'

        parts = list(range(len(upload_data)))
        part_data = [upload_data[i:i+1] for i in parts]

        _md5 = Client._digest
        _hex = Client._checksum_hex
        _b64 = Client._checksum_b64
        part_digest = [_md5(part_data[i]) for i in parts]
        combine_checksum = _hex(_md5(b''.join(part_digest)))

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.MagicMock()
            session.__enter__ = lambda *args: session

            def get(path, seq=iter(range(2))):
                all_parts = [{"part_number": i+1,
                              "checksum": _hex(part_digest[i])} for i in parts]

                return choose_reply((path, next(seq)), {
                    # initial upload status: all parts uploaded except the first one
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 0):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": all_parts[1:]}),

                    # final upload status
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 1):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": all_parts}),
                })

            def post(path, **kwargs):
                json_ = kwargs.pop('json')
                body = json.dumps(sorted(json_.items()))
                return choose_reply((path, body), {
                    # initiate upload
                    ('bqm/multipart',
                     json.dumps([('size', len(upload_data))])):
                        json.dumps({'id': upload_problem_id}),

                    # combine parts
                    ('bqm/multipart/{}/combine'.format(upload_problem_id),
                     json.dumps([('checksum', combine_checksum)])):
                        json.dumps({}),
                })

            def put(path, data, headers):
                body = data.read()
                headers = json.dumps(sorted(headers.items()))
                replies = {
                    # only the first part!
                    (
                        'bqm/multipart/{}/part/{}'.format(upload_problem_id, i+1),
                        part_data[i],
                        json.dumps(sorted([
                            ('Content-MD5', _b64(part_digest[i])),
                            ('Content-Type', 'application/octet-stream')
                        ]))
                    ): json.dumps({})
                    for i in parts[:1]
                }
                return choose_reply((path, body, headers), replies)

            session.get = get
            session.put = put
            session.post = post

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(endpoint='endpoint', token='token') as client:

                future = client.upload_problem_encoded(upload_data)
                try:
                    returned_problem_id = future.result()
                except Exception as e:
                    self.fail(e)

                self.assertEqual(returned_problem_id, upload_problem_id)

    def test_part_upload_retried(self):
        """Verify upload successful even if part upload fails a few times."""

        # using the default part size here (5MB), so we have only one part
        upload_data = b'123'
        upload_problem_id = '84ef154c-28f9-46ed-9f22-aec0583499f2'

        parts = [0]
        part_data = [upload_data]

        _md5 = Client._digest
        _hex = Client._checksum_hex
        _b64 = Client._checksum_b64
        part_digest = [_md5(part_data[i]) for i in parts]
        combine_checksum = _hex(_md5(b''.join(part_digest)))

        # we need a "global session", because mocked responses are stateful
        def global_mock_session(n_failures):
            session = mock.MagicMock()
            session.__enter__ = lambda *args: session

            def get(path, seq=iter(range(2))):
                all_parts = [{"part_number": i+1,
                              "checksum": _hex(part_digest[i])} for i in parts]

                return choose_reply((path, next(seq)), {
                    # initial upload status
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 0):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": []}),

                    # final upload status
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 1):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": all_parts}),
                })

            def post(path, **kwargs):
                json_ = kwargs.pop('json')
                body = json.dumps(sorted(json_.items()))
                return choose_reply((path, body), {
                    # initiate upload
                    ('bqm/multipart',
                     json.dumps([('size', len(upload_data))])):
                        json.dumps({'id': upload_problem_id}),

                    # combine parts
                    ('bqm/multipart/{}/combine'.format(upload_problem_id),
                     json.dumps([('checksum', combine_checksum)])):
                        json.dumps({}),
                })

            def put(path, data, headers, seq=iter(range(Client._UPLOAD_PART_RETRIES+1))):
                body = data.read()
                headers = json.dumps(sorted(headers.items()))
                keys = [
                    (
                        'bqm/multipart/{}/part/{}'.format(upload_problem_id, i+1),
                        part_data[i],
                        json.dumps(sorted([
                            ('Content-MD5', _b64(part_digest[i])),
                            ('Content-Type', 'application/octet-stream')
                        ]))
                    ) for i in parts
                ]
                attempt = next(seq)
                if attempt < n_failures:
                    return choose_reply((path, body, headers),
                                        replies={key: '{}' for key in keys},
                                        statuses={key: iter([500]) for key in keys})
                else:
                    return choose_reply((path, body, headers),
                                        replies={key: '{}' for key in keys})

            session.get = get
            session.put = put
            session.post = post

            return session

        # part upload fails exactly _UPLOAD_PART_RETRIES times;
        # problem upload must recover
        session = global_mock_session(n_failures=Client._UPLOAD_PART_RETRIES)
        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(endpoint='endpoint', token='token') as client:

                future = client.upload_problem_encoded(upload_data)
                try:
                    returned_problem_id = future.result()
                except Exception as e:
                    self.fail(e)

                self.assertEqual(returned_problem_id, upload_problem_id)

        # part upload fails exactly _UPLOAD_PART_RETRIES + 1 times;
        # problem upload will also fail
        session = global_mock_session(n_failures=Client._UPLOAD_PART_RETRIES + 1)
        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(endpoint='endpoint', token='token') as client:

                with self.assertRaises(ProblemUploadError):
                    client.upload_problem_encoded(upload_data).result()

    @mock.patch.multiple(Client, _UPLOAD_PART_SIZE_BYTES=1)
    def test_problem_reupload_end_to_end(self):
        """Verify problem multipart upload continued."""

        upload_data = b'123'
        upload_problem_id = '84ef154c-28f9-46ed-9f22-aec0583499f2'

        parts = list(range(len(upload_data)))
        part_data = [upload_data[i:i+1] for i in parts]

        _md5 = Client._digest
        _hex = Client._checksum_hex
        _b64 = Client._checksum_b64
        part_digest = [_md5(part_data[i]) for i in parts]
        combine_checksum = _hex(_md5(b''.join(part_digest)))

        # we need a "global session", because mocked responses are stateful
        def global_mock_session():
            session = mock.MagicMock()
            session.__enter__ = lambda *args: session

            def get(path, seq=iter(range(2))):
                all_parts = [{"part_number": i+1,
                              "checksum": _hex(part_digest[i])} for i in parts]

                return choose_reply((path, next(seq)), {
                    # initial upload status
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 0):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": all_parts[:2]}),

                    # final upload status
                    ('bqm/multipart/{}/status'.format(upload_problem_id), 1):
                        json.dumps({"status": "UPLOAD_IN_PROGRESS", "parts": all_parts}),
                })

            def post(path, **kwargs):
                json_ = kwargs.pop('json')
                body = json.dumps(sorted(json_.items()))
                return choose_reply((path, body), {
                    # combine parts
                    ('bqm/multipart/{}/combine'.format(upload_problem_id),
                     json.dumps([('checksum', combine_checksum)])):
                        json.dumps({}),
                })

            def put(path, data, headers):
                body = data.read()
                headers = json.dumps(sorted(headers.items()))
                replies = {
                    (
                        'bqm/multipart/{}/part/{}'.format(upload_problem_id, i+1),
                        part_data[i],
                        json.dumps(sorted([
                            ('Content-MD5', _b64(part_digest[i])),
                            ('Content-Type', 'application/octet-stream')
                        ]))
                    ): json.dumps({})
                    for i in parts[2:]
                }
                return choose_reply((path, body, headers), replies)

            session.get = get
            session.put = put
            session.post = post

            return session

        session = global_mock_session()

        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(endpoint='endpoint', token='token') as client:

                future = client.upload_problem_encoded(
                    upload_data, problem_id=upload_problem_id)

                try:
                    returned_problem_id = future.result()
                except Exception as e:
                    self.fail(e)

                self.assertEqual(returned_problem_id, upload_problem_id)


@unittest.skipUnless(config, "No live server configuration available.")
class TestMultipartUpload(unittest.TestCase):
    _100gb = 100 * 2**30

    def test_smoke_test(self):
        data = b'123'
        with Client(**config) as client:
            future = client.upload_problem_encoded(data)
            try:
                problem_id = future.result()
            except Exception as e:
                self.fail(e)

    def test_initiate_size_limit(self):
        size = self._100gb

        with Client(**config) as client:
            with client.create_session() as session:
                with self.assertRaisesRegex(SAPIError,
                                            'bigger than the maximum'):
                    client._initiate_multipart_upload(session, size)

    @mock.patch.object(ChunkedData, 'total_size', _100gb)
    def test_initiate_size_limit_end_to_end(self):

        with Client(**config) as client:
            future = client.upload_problem_encoded(b'')
            with self.assertRaisesRegex(ProblemUploadError,
                                        'bigger than the maximum'):
                problem_id = future.result()

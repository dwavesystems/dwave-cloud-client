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

"""Multipart upload helpers."""

from __future__ import division

import io
import os
import math
import logging
import threading

import six

__all__ = ['FileSlicer', 'ChunkedData']

logger = logging.getLogger(__name__)


class FileSlicer(object):
    """Thread-safe random access to a file-like object."""

    def __init__(self, fp):
        if not (isinstance(fp, io.IOBase) and fp.seekable() and fp.readable()):
            raise ValueError("expected file-like, seekable, readable object")

        # store file size, assuming it won't change
        fp.seek(0, os.SEEK_END)
        self._size = fp.tell()
        self.fp = fp

        # multiple threads will be accessing the underlying file 
        self._lock = threading.RLock()

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("slice key expected")

        start, stop, stride = key.indices(len(self))
        if stride != 1:
            raise NotImplementedError("stride of 1 required")

        with self._lock:
            self.fp.seek(start)
            return self.fp.read(stop - start)


class ChunkedData(object):
    """Unifying and performant streaming file-like interface to (large problem)
    data chunks.

    Handles streaming, in-file and in-memory input of problem data in Ising,
    QUBO and BQM form. Provides by-ref slice operation.

    Args:
        data (bytes/str/file-like):
            Encoded problem data, in-memory or in-file. This data is typically
            obtained from one of the encoders in the :mod:`dwave.cloud.coders`
            module (directly or indirectly)

        chunk_size (int):
            Problem part size in bytes.

    """

    def __init__(self, data, chunk_size):
        self.data = data
        self.chunk_size = chunk_size

        # convenience string handler
        if isinstance(data, six.string_types):
            data = bytes(data, encoding='ascii')

        if isinstance(data, bytes):
            self.view = io.BytesIO(data).getbuffer()

        elif isinstance(data, io.IOBase):
            if not data.seekable():
                raise ValueError("seekable file-like data object expected")
            if not data.readable():
                raise ValueError("readable file-like data object expected")
            self.view = FileSlicer(data)

    @property
    def num_chunks(self):
        """Total number of chunks."""

        total_size = len(self.view)
        return math.ceil(total_size / self.chunk_size)

    def chunk(self, idx):
        """Return :class:`io.BytesIO`-wrapped zero-indexed chunk data."""

        start = idx * self.chunk_size
        stop = start + self.chunk_size
        return io.BytesIO(self.view[start:stop])

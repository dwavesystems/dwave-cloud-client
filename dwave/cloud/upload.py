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

from abc import abstractmethod
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import six

__all__ = ['FileView', 'ChunkedData']

logger = logging.getLogger(__name__)


class RandomAccessIOBaseView(abc.Sized):
    """Abstract base class for random access file-like object views.

    Concrete subclasses must provide __len__ and __getitem__.
    """

    __slots__ = ()

    @abstractmethod
    def __getitem__(self, key):
        raise KeyError


class FileView(RandomAccessIOBaseView):
    """Provide thread-safe random access to a file-like object via item getter
    interface.

    Args:
        fp (:class:`io.IOBase`/file-like):
            A file-like object that supports seek, read and tell operations.
            Thread-safety of these operations is not assumed.

        strict (bool, default=True):
            Require file-like object to be a :class:`io.IOBase` subclass.

    Note:
        :class:`FileView` behavior is invariant to data encoding of the
        file-like object. For example, if the file is opened in binary mode
        (or file is :class:`io.BytesIO` instance), :class:`bytes` are returned
        as slices. If the file is opened in text mode (or it's
        :class:`io.StringIO`), slicing returns :class:`str` instances.

    Example:
        Access overlapping segments of a file from multiple threads::

            with open('/path/to/file', 'rb') as fp:   # binary mode, read access
                fv = FileView(fp):

                # in thread 1:
                seg = fv[0:10]

                # in thread 2:
                seg = fv[5:15]

    """

    def __init__(self, fp, strict=True):
        if strict:
            valid = lambda f: (
                isinstance(f, io.IOBase) and f.seekable() and f.readable())
        else:
            valid = lambda f: all([
                hasattr(f, 'read'), hasattr(f, 'seek'), hasattr(f, 'tell')])

        if not valid(fp):
            raise ValueError("expected file-like, seekable, readable object")

        # store file size, assuming it won't change
        fp.seek(0, os.SEEK_END)
        self._size = fp.tell()
        self._fp = fp

        # multiple threads will be accessing the underlying file 
        self._lock = threading.RLock()

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        """Fetch a slice of file's content.

        Returns:
            :class:`bytes`/:class:`str`
        """

        if isinstance(key, slice):
            start, stop, stride = key.indices(len(self))
            if stride != 1:
                raise NotImplementedError("stride of 1 required")
        else:
            try:
                start = int(key)
            except:
                raise TypeError("slice or integral key expected")

            # negative indices wrap around
            if start < 0:
                start %= len(self)

            stop = start + 1

        # empty slices
        if stop <= start:
            return bytes()

        # slice is an atomic "seek and read" operation
        with self._lock:
            self._fp.seek(start)
            return self._fp.read(stop - start)


class ChunkedData(object):
    """Unifying and performant streaming file-like interface to (large problem)
    data chunks.

    Handles streaming (not yet), in-file and in-memory data. Provides access to
    chunk data.

    Args:
        data (bytes/str/binary-file-like):
            Encoded problem data, in-memory or in-file.

        chunk_size (int):
            Chunk size in bytes.

    """

    def __init__(self, data, chunk_size):
        self.data = data
        self.chunk_size = int(chunk_size)

        if self.chunk_size <= 0:
            raise ValueError("positive integer required for chunk size")

        # convenience string handler
        if isinstance(data, six.string_types):
            data = data.encode('ascii')

        if isinstance(data, bytes):
            self.view = io.BytesIO(data).getbuffer()

        elif isinstance(data, io.IOBase):
            if not data.seekable():
                raise ValueError("seekable file-like data object expected")
            if not data.readable():
                raise ValueError("readable file-like data object expected")
            self.view = FileView(data)

    @property
    def num_chunks(self):
        """Total number of chunks."""

        total_size = len(self.view)
        return math.ceil(total_size / self.chunk_size)

    def __len__(self):
        return self.num_chunks

    def chunk(self, idx):
        """Return :class:`io.BytesIO`-wrapped zero-indexed chunk data."""

        start = idx * self.chunk_size
        stop = start + self.chunk_size
        return io.BytesIO(self.view[start:stop])

    def __iter__(self):
        for idx in range(len(self)):
            yield self.chunk(idx)

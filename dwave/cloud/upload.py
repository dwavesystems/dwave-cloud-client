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

import io
import os
import math
import logging
import threading

from abc import abstractmethod
from collections.abc import Sized
from functools import partial

__all__ = ['ChunkedData']

logger = logging.getLogger(__name__)


class Gettable(Sized):
    """Abstract base class for objects that implement standard and efficient
    item getters.

    Concrete subclasses must provide __len__ and __getitem__/getinto.
    """

    __slots__ = ()

    @abstractmethod
    def __getitem__(self, key):     # pragma: no cover
        """Standard item getter, integer and slice indices supported."""
        raise KeyError

    @abstractmethod
    def getinto(self, key, buf):    # pragma: no cover
        """Optimized item getter (without the temporary buffer)."""
        raise NotImplementedError


class GettableBase(Gettable):
    """Base class for sized data containers with a thread-safe efficient item
    getter.

    Subclasses must implement __len__ and getinto.
    """

    def __len__(self):
        raise NotImplementedError

    def _getkey_to_range(self, key):
        """Resolve slice/int key to start-stop range bounds.

        Returns: (start, stop, is_item?)
        """

        if isinstance(key, slice):
            start, stop, stride = key.indices(len(self))
            if stride != 1:
                raise NotImplementedError("stride of 1 required")
            is_item = False
        else:
            try:
                start = int(key)
            except:
                raise TypeError("slice or integral key expected")

            # negative indices wrap around
            if start < 0:
                start %= len(self)

            stop = start + 1
            is_item = True

        return start, stop, is_item

    def getinto(self, key, buf):
        raise NotImplementedError

    def __getitem__(self, key):
        """Fetch a slice of file's content as bytes. For integer index, return
        a single byte value as integer.

        Returns:
            int/:class:`bytes`

        Note:
            Behavior consistent with bytes/bytearray/memoryview.
        """

        start, stop, is_item = self._getkey_to_range(key)
        size = stop - start
        if size <= 0:
            return bytes()

        b = bytearray(size)
        n = self.getinto(key, b)
        del b[n:]

        if is_item:
            # note: for out-of-bounds access this will automatically raise an
            # `IndexError`, as expected, because `n` will be zero
            return b[0]
        else:
            return bytes(b)


class GettableFile(GettableBase):
    """Provide thread-safe memory buffer-like random read access to a file-like
    object via an efficient item getter interface.

    Args:
        fp (:class:`io.BufferedIOBase`/:class:`io.RawIOBase`/binary-file-like):
            A file-like object that supports seek and readinto operations.
            Thread-safety of these operations is not assumed.

        strict (bool, default=True):
            Require file-like object to be a :class:`io.BufferedIOBase` or
            :class:`io.RawIOBase` instance.

    Note:
        :class:`.GettableFile` implements :class:`.Gettable` interface over a
        file-like object that supports buffered binary read access.

    Example:
        Access overlapping segments of a file from multiple threads::

            with open('/path/to/file', 'rb') as fp:   # binary mode, read access
                gf = GettableFile(fp):

                # in thread 1:
                seg = gf[0:10]

                # in thread 2:
                seg = gf[5:15]

    """

    def __init__(self, fp, strict=True):
        if strict:
            valid = lambda f: (
                isinstance(f, (io.BufferedIOBase, io.RawIOBase))
                and f.seekable() and f.readable())
        else:
            valid = lambda f: all([
                hasattr(f, 'readinto'), hasattr(f, 'seek'), hasattr(f, 'tell')])

        if not valid(fp):
            raise TypeError("expected file-like, seekable, readable object")

        # store file size, assuming it won't change
        self._size = fp.seek(0, os.SEEK_END)
        if self._size is None:
            # handle non-python3 and/or non-standard file seek() impl.
            # (like tempfile.SpooledTemporaryFile)
            # note: not thread-safe!
            self._size = fp.tell()

        self._fp = fp

        # multiple threads will be accessing the underlying file 
        self._lock = threading.RLock()

    def __len__(self):
        return self._size

    def getinto(self, key, buf):
        """Copy a slice of file's content into a pre-allocated bytes-like
        object buf. For example, buf might be a `bytearray`.

        Args:
            key (slice/int):
                Source data address coded as a `slice` object or int position.
            buf (bytes-like):
                Target pre-allocated bytes-like object.

        Returns:
            int:
                The number of bytes read (0 for EOF).
        """

        start, stop, _ = self._getkey_to_range(key)

        # empty slice
        if stop <= start:
            return 0

        # copy source[start:stop] => target[0:stop-start]
        size = stop - start
        target = memoryview(buf)[:size]

        # slice is an atomic "seek and read" operation
        with self._lock:
            self._fp.seek(start)
            return self._fp.readinto(target)


class GettableMemory(GettableBase):
    """Provide the :class:`Gettable` interface to a bytes-like object.

    Args:
        buf (bytes/bytearray/memoryview/bytes-like):
            Bytes-like object.

    """

    def __init__(self, buf):
        self._buf = buf

    def __len__(self):
        return len(self._buf)

    def getinto(self, key, buf):
        start, stop, _ = self._getkey_to_range(key)

        # empty slice
        if stop <= start:
            return 0

        # copy source[start:stop] => target[0:stop-start]
        source_view = self._buf[start:stop]
        target_view = memoryview(buf)
        size = min(len(source_view), len(target_view))
        target_view[:size] = source_view[:size]

        return size


class FileView(io.RawIOBase):
    """A raw binary stream subclass with memoryview-like interface to a
    :class:`.GettableBase`-derived object.

    Args:
        raw (:class:`.GettableFile`/:class:`.GettableMemory`/:class:`.GettableBase`-derived):
            A :class:`.GettableBase`-derived object that (essentially) supports
            efficient and thread-safe `getinto` operation.

    Note:
        Although similar to :mod:`mmap` for files, :class:`.FileView` provides:
        - a unified interface to *both* files and memory objects
        - a way to ensure file "seek & read" operation is atomic (thread-safe)
          via the :class:`.Gettable` layer

    Note:
        Use the slice syntax (item getter) to retrieve an isolated file segment
        view (a new instance of :class:`.FileView` that references the same
        underlying data, but supports independent read operations).
    """

    def __init__(self, raw):
        super().__init__()
        self._raw = raw
        self._pos = 0
        self._offset = 0
        self._size = len(raw)

    def seek(self, pos, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self._pos = pos
        elif whence == os.SEEK_CUR:
            self._pos += pos
        elif whence == os.SEEK_END:
            self._pos = self._size + pos
        else:
            raise ValueError("whence must be one of 'io.SEEK_{SET,CUR,END}'")

        return self._pos

    def tell(self):
        return self._pos

    def readinto(self, b):
        """Read bytes into a pre-allocated bytes-like object b.

        Returns:
            int:
                The number of bytes read.
        """
        start = self._offset + self._pos
        stop = self._offset + self._size
        key = slice(start, stop)
        n = self._raw.getinto(key, b)
        self._pos += n
        return n

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        """Return memoryview-like slice: byte value for an integer index,
        sub-FileView for a slice index.
        """

        # slice key
        if isinstance(key, slice):
            start, stop, stride = key.indices(len(self))
            if stride != 1:
                raise NotImplementedError("stride of 1 required")

            view = FileView(self._raw)
            view._offset += start
            view._size = stop - start
            return view

        # integer key
        try:
            start = int(key)
        except:
            raise TypeError("slice or integer key expected")

        # negative indices wrap around
        if start < 0:
            start %= len(self)

        return self._raw[self._offset + start]


class ChunkedData(object):
    """Unifying and performant streaming file-like interface to (large problem)
    data chunks.

    Handles in-file and in-memory data. Non-seekable streams are not yet
    supported. Provides access to chunk data via file-like interface.

    Args:
        data (bytes-like/file-like):
            Encoded problem data, in-memory or in-file (opened for binary read).

        chunk_size (int):
            Chunk size in bytes.

    """

    def _thread_safe_data_view(self, data):
        # convenience string handler
        if isinstance(data, str):
            data = data.encode('ascii')

        if isinstance(data, (bytes, bytearray)):
            # note: `BytesIO` (in py3.5+) will use buffer protocol (reuse data
            # buffer) only for `bytes` objects, and make a copy otherwise!

            # use non-locking memory view over data buffer
            return FileView(GettableMemory(data))

        # use locking file view if possible
        try:
            return FileView(GettableFile(data, strict=True))
        except TypeError:
            logger.debug("data does not conform to strict file-like requirements")

        # TODO: use stream view if possible

        # fallback to a less strict check of file-like's capabilities,
        # accepting we might fail later
        try:
            return FileView(GettableFile(data, strict=False))
        except TypeError:
            logger.debug("data does not conform to loose file-like requirements")

        raise TypeError("bytes/str/file-like data required")

    def __init__(self, data, chunk_size):
        self.data = data
        self.view = self._thread_safe_data_view(data)

        if chunk_size <= 0:
            raise ValueError("positive integer required for chunk size")
        self.chunk_size = int(chunk_size)

    @property
    def total_size(self):
        """Total data size, in bytes."""
        return len(self.view)

    @property
    def num_chunks(self):
        """Total number of chunks."""
        return math.ceil(self.total_size / self.chunk_size)

    def __len__(self):
        return self.num_chunks

    def chunk(self, idx):
        """Return binary file-like object for a specified data chunk.

        Args:
            idx (int):
                Zero-based chunk index.

        Returns:
            :class:`.FileView`

        """

        start = idx * self.chunk_size
        stop = start + self.chunk_size
        return self.view[start:stop]

    def __iter__(self):
        for idx in range(len(self)):
            yield self.chunk(idx)

    def generators(self):
        """Iterator of (immutable) chunk generators."""

        for idx in range(len(self)):
            yield partial(self.chunk, idx=idx)

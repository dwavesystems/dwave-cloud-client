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

"""Date/time utilities for private and Ocean-internal use.

.. versionchanged:: 0.12.0
   These functions previously lived under ``dwave.cloud.utils``.
"""

import time
from datetime import datetime, timedelta
from dateutil.tz import UTC

__all__ = ['datetime_to_timestamp', 'utcnow', 'epochnow', 'utcrel', 'tictoc']


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert timezone-aware `datetime` to POSIX timestamp and
    return seconds since UNIX epoch.

    Note: similar to `datetime.timestamp()` in Python 3.3+.
    """

    epoch = datetime.fromtimestamp(0, tz=UTC)
    return (dt - epoch).total_seconds()


def utcnow() -> datetime:
    """Returns tz-aware now in UTC."""
    return datetime.now(tz=UTC)


def epochnow() -> float:
    """Returns now as UNIX timestamp.

    Invariant:
        epochnow() ~= datetime_to_timestamp(utcnow())

    """
    return time.time()


def utcrel(offset: int) -> datetime:
    """Return a timezone-aware `datetime` relative to now (UTC), shifted by
    `offset` seconds in to the future.

    Example:
        a_minute_from_now = utcrel(60)
    """
    return utcnow() + timedelta(seconds=offset)


class tictoc:
    """Timer as a context manager."""

    def __enter__(self):
        self.tick = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dt = time.perf_counter() - self.tick

# Copyright 2023 D-Wave Systems Inc.
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

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Type, Union

from dwave.cloud.auth._creds import _Cache
from dwave.cloud.config import get_configfile_paths, get_default_configfile_path

__all__ = ['Credentials']

logger = logging.getLogger(__name__)

CREDS_FILENAME = "credentials.db"

class AutoDetect:
    """Sentinel value for creds_file auto-detect."""


def _get_creds_paths(
        *, system: bool = True, user: bool = True, local: bool = True,
        only_existing: bool = True) -> list[str]:
    return get_configfile_paths(system=system, user=user, local=local,
                                only_existing=only_existing, filename=CREDS_FILENAME)


def _get_default_creds_path() -> str:
    return get_default_configfile_path(filename=CREDS_FILENAME)


class Credentials(_Cache):
    """Proxy to an on-disk credentials SQLite database.

    Use :class:`Credentials` dictionary interface for transparent reads and
    writes to the on-disk database.

    Args:
        creds_file:
            Credentials file path on disk. Special value
            :class:`~dwave.cloud.auth.creds.AutoDetect` initiates a search for
            credentials file in the expected system/user/local configuration
            directories, with a fallback to the default credentials location,
            a user configuration directory.
            Special value of ``None`` is a shortcut to auto-detect fallback:
            using the default location.

        create:
            Boolean flag used to disable creation of a new credentials file,
            if one does not exist. By default, :class:`Credentials` acts as
            a transparent proxy and file creation on-the-fly is enabled.
            By setting ``create=False``, :class:`Credentials` will
            effectively act as an in-memory credentials store.

        **kwargs:
            Arguments passed-thru to the underlying disk cache.

    Examples:
        This example searches for an existing credentials file, and if one is not
        found, it creates it in the default location, inside user-level config
        directory. It then, fetches a token, if one exists.

        >>> from dwave.cloud.auth.creds import Credentials
        >>> creds = Credentials()
        >>> token = creds.get('token')

    """

    # wrapper around on-disk cache; we keep this layer very thin for simplicity,
    # at least for now. in the future, we might want to hide internals, and
    # expose a token interface with transparent on-the-fly persistency.

    # note: no consensus yet on annotating sentinel value types
    # (https://github.com/python/typing/issues/689)
    def __init__(self, *,
                 creds_file: Optional[Union[str, Type[AutoDetect]]] = AutoDetect,
                 create: bool = True):

        if creds_file is AutoDetect:
            if paths := _get_creds_paths(only_existing=True):
                creds_file = paths[-1]
            else:
                creds_file = None

        if creds_file is None:
            creds_file = _get_default_creds_path()

        self.creds_file: Path = Path(creds_file).expanduser().resolve()

        # stage the ground for db creation; i.e. fail early in case of access limited
        if create and not self.creds_file.exists():
            self.creds_file.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"{type(self).__name__}() initialized with "
                     f"creds_file={self.creds_file!r} and create={create!r}.")

        directory = self.creds_file.parent if create else None
        super().__init__(directory=directory, dbname=CREDS_FILENAME)

        logger.debug(f"{type(self).__name__} db loaded from {self.directory!r}.")

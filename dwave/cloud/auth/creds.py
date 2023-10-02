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

import logging
from pathlib import Path
from typing import List, Optional, Union

import diskcache

from dwave.cloud.config import get_configfile_paths, get_default_configfile_path

__all__ = ['Credentials']

logger = logging.getLogger(__name__)

CREDS_FILENAME = diskcache.core.DBNAME = "credentials.db"

AutoDetect = object()


def _get_creds_paths(
        *, system: bool = True, user: bool = True, local: bool = True,
        only_existing: bool = True) -> List[str]:
    return get_configfile_paths(system=system, user=user, local=local,
                                only_existing=only_existing, filename=CREDS_FILENAME)


def _get_default_creds_path() -> str:
    return get_default_configfile_path(filename=CREDS_FILENAME)


class Credentials(diskcache.Cache):
    # wrapper around on-disk cache; we keep this layer very thin for simplicity,
    # at least for now. in the future, we might want to hide internals, and
    # expose a token interface with transparent on-the-fly persistency.

    # note: no consensus yet on annotating sentinel value types
    # (https://github.com/python/typing/issues/689)
    def __init__(self, *,
                 creds_file: Optional[Union[str, AutoDetect]] = AutoDetect,
                 create: bool = True,
                 **kwargs):

        if creds_file is AutoDetect:
            if paths := _get_creds_paths(only_existing=True):
                creds_file = paths[-1]
            else:
                creds_file = None

        if creds_file is None:
            creds_file = _get_default_creds_path()

        self._creds_file: Path = Path(creds_file).expanduser().resolve()
        self._create = create

        # stage the ground for db creation; i.e. fail early in case of access limited
        if create and not self._creds_file.exists():
            self._creds_file.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"{type(self).__name__}() initialized with "
                     f"creds_file={self._creds_file!r} and create={self._create!r}.")

        directory = directory=self._creds_file.parent if create else None
        super().__init__(directory=directory, **kwargs)

        logger.debug(f"{type(self).__name__} db loaded from {self.directory!r}.")

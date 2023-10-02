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
from typing import List

from dwave.cloud.config import get_configfile_paths, get_default_configfile_path

__all__ = []

logger = logging.getLogger(__name__)

CREDS_FILENAME = "credentials.db"


def _get_creds_paths(
        *, system: bool = True, user: bool = True, local: bool = True,
        only_existing: bool = True) -> List[str]:
    return get_configfile_paths(system=system, user=user, local=local,
                                only_existing=only_existing, filename=CREDS_FILENAME)


def _get_default_creds_path() -> str:
    return get_default_configfile_path(filename=CREDS_FILENAME)

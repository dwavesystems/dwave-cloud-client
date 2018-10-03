# Copyright 2017 D-Wave Systems Inc.
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

from __future__ import absolute_import

import os
import logging

from dwave.cloud.client import Client
from dwave.cloud.solver import Solver
from dwave.cloud.computation import Future

__all__ = ['Client', 'Solver', 'Future']


# configure logger `dwave.cloud` root logger, inherited in submodules
# (write level warning+ to stderr, include timestamp/module/level)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
root = logging.getLogger(__name__)
root.setLevel(logging.WARNING)
root.addHandler(handler)


# add TRACE log level and Logger.trace() method
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")

def _trace(logger, message, *args, **kwargs):
    if logger.isEnabledFor(logging.TRACE):
        logger._log(logging.TRACE, message, args, **kwargs)

logging.Logger.trace = _trace


# apply DWAVE_LOG_LEVEL
def _apply_loglevel_from_env(logger):
    name = os.getenv('DWAVE_LOG_LEVEL') or ''
    if not name:
        return
    levels = {'debug': logging.DEBUG, 'trace': logging.TRACE}
    requested_level = levels.get(name.lower())
    if requested_level:
        logger.setLevel(requested_level)

_apply_loglevel_from_env(root)

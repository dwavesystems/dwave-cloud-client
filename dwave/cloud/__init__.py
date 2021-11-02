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

import os
import re
import sys
import logging
import importlib

from dwave.cloud.client import Client
from dwave.cloud.solver import Solver
from dwave.cloud.computation import Future
from dwave.cloud.utils import set_loglevel

__all__ = ['Client', 'Solver', 'Future']


class FilteredSecretsFormatter(logging.Formatter):
    """Logging formatter that filters out secrets (like Solver API tokens).

    Note: we assume, for easier disambiguation, a secret/token is prefixed with
    a short alphanumeric string, and comprises 40 or more hex digits.
    """

    _SECRETS_PATTERN = re.compile(
        r'\b([0-9A-Za-z]{2,4})-([0-9A-Fa-f]{3})([0-9A-Fa-f]{34,})([0-9A-Fa-f]{3})\b')

    def format(self, record):
        output = super().format(record)
        filtered = re.sub(self._SECRETS_PATTERN, r'\1-\2...\4', output)
        return filtered

# configure logger `dwave.cloud` root logger, inherited in submodules
# (write level warning+ to stderr, include timestamp/module/level)
_formatter = FilteredSecretsFormatter('%(asctime)s %(name)s %(levelname)s %(threadName)s %(message)s')
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)

# expose the root logger to simplify access; for example:
# `dwave.cloud.logger.setLevel(logging.DEBUG)`
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(_handler)


# add TRACE log level and Logger.trace() method
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")

def _trace(logger, message, *args, **kwargs):
    if logger.isEnabledFor(logging.TRACE):
        logger._log(logging.TRACE, message, args, **kwargs)

logging.Logger.trace = _trace


# apply DWAVE_LOG_LEVEL
def _apply_loglevel_from_env(logger):
    set_loglevel(logger, os.getenv('DWAVE_LOG_LEVEL'))

_apply_loglevel_from_env(logger)


# alias dwave.cloud.client.{qpu,sw,hybrid} as dwave.cloud.*
def _alias_old_client_submodules():
    for name in ('qpu', 'sw', 'hybrid'):
        # note: create both module and local attribute
        globals()[name] = sys.modules['dwave.cloud.{}'.format(name)] = \
            importlib.import_module('dwave.cloud.client.{}'.format(name))

_alias_old_client_submodules()

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

import sys
import logging
import importlib
import types

from dwave.cloud.utils.logging import add_loglevel, configure_logging_from_env

__all__ = ['Client', 'Solver', 'Future']


# prevent log output (from library) when logging not configured by user/app
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# make sure TRACE level is available
add_loglevel('TRACE', 5)

# configure logger if DWAVE_LOG_LEVEL present in environment
configure_logging_from_env(logger)


# lazily import Client/Solver/Future -- only when actually asked for
def __getattr__(cls):
    modname = {'Client': 'client', 'Solver': 'solver', 'Future': 'computation'}
    if cls in modname:
        mod = importlib.import_module(f'dwave.cloud.{modname[cls]}')
        return getattr(mod, cls)

    raise AttributeError(f"module '{__name__}' has no attribute '{cls}'")


# lazily alias dwave.cloud.client.{qpu,sw,hybrid} as dwave.cloud.*
# (i.e. don't preemptively import Client from submodules)
def _alias_old_client_submodules():
    class _submod(types.ModuleType):
        def __getattr__(self, name):
            modname = self.__name__
            if name == 'Client':
                mod = importlib.import_module('dwave.cloud.client.{}'.format(modname))
                return getattr(mod, 'Client')

            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

    for name in ('qpu', 'sw', 'hybrid'):
        globals()[name] = sys.modules[f'dwave.cloud.{name}'] = _submod(name)

_alias_old_client_submodules()

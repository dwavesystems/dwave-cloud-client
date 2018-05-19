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

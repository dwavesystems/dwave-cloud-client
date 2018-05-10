from __future__ import absolute_import

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
TRACE_LOGLEVEL = 5

logging.addLevelName(TRACE_LOGLEVEL, "TRACE")
def _trace(logger, message, *args, **kws):
    if logger.isEnabledFor(TRACE_LOGLEVEL):
        logger._log(TRACE_LOGLEVEL, message, args, **kws)

logging.Logger.trace = _trace

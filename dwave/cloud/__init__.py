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

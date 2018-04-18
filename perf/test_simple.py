from __future__ import absolute_import

import sys
import logging
import unittest

from dwave.cloud.config import load_config
from dwave.cloud.qpu import Client
from dwave.cloud.exceptions import SolverAuthenticationError
from dwave.cloud.testing import mock

from tests import config


# increase logging verbosity for root logger
logging.getLogger('dwave.cloud').setLevel(logging.DEBUG)

# setup local logger
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


with Client.from_config(profile='prod') as client:
    solvers = client.get_solvers()
    logger.info("Solvers available: %r", solvers.keys())

    solver = client.get_solver()
    comp = solver.sample_qubo({})

    logger.info("Computation")
    logger.info(" - time received: %s", comp.time_received)
    logger.info(" - time solved: %s", comp.time_solved)
    logger.info(" - parse time: %s", comp.parse_time)
    logger.info(" - remote status: %s", comp.remote_status)

    result = comp.result()
    logger.info("Result received.")

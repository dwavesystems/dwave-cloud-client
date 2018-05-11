from __future__ import absolute_import

import logging

from dwave.cloud.qpu import Client
from dwave.cloud.computation import Future


# setup local logger
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


with Client.from_config(profile='prod') as client:
    solvers = client.get_solvers()
    logger.info("Solvers available: %r", solvers.keys())

    solver = client.get_solver()
    comps = [solver.sample_qubo({}, num_reads=1) for _ in range(20)]

    for comp in Future.as_completed(comps):
        try:
            result = comp.result()
        except Exception as e:
            logger.info("Computation %s failed: %r", comp.id, e)

        logger.info("Computation %s succeeded:", comp.id)
        logger.info(" - time received: %s", comp.time_received)
        logger.info(" - time solved: %s", comp.time_solved)
        logger.info(" - parse time: %s", comp.parse_time)
        logger.info(" - remote status: %s", comp.remote_status)

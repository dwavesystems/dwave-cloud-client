from __future__ import absolute_import

import logging

from dwave.cloud.qpu import Client


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
    comp = solver.sample_qubo({})

    logger.info("Problem submitted:")
    logger.info(" - time received: %s", comp.time_received)
    logger.info(" - time solved: %s", comp.time_solved)
    logger.info(" - parse time: %s", comp.parse_time)
    logger.info(" - remote status: %s", comp.remote_status)
    logger.info(" - min_eta: %s", comp.eta_min)

    result = comp.result()
    logger.info("Result received:")
    logger.info(" - time received: %s", comp.time_received)
    logger.info(" - time solved: %s", comp.time_solved)
    logger.info(" - parse time: %s", comp.parse_time)
    logger.info(" - remote status: %s", comp.remote_status)

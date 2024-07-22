# Copyright 2024 D-Wave Inc.
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

import json
import subprocess

from functools import partial
from pathlib import Path

from dwave.cloud import Client
from dwave.cloud.coders import encode_problem_as_qp
from dwave.cloud.config import ClientConfig
from dwave.cloud.solver import StructuredSolver, BQMSolver, CQMSolver, DQMSolver, NLSolver
from dwave.cloud.regions import resolve_endpoints, get_regions
from dwave.cloud.testing import isolated_environ, mocks
from dwave.cloud.utils.qubo import generate_random_ising_problem


# note: looks like timeraw benchmarks can't be parameterized
class ImportDeps:
    version = "1"

    def timeraw_import_numpy(self):
        return "import numpy"

    def timeraw_import_pydantic(self):
        return "import pydantic"

    def timeraw_import_requests(self):
        return "import requests"

    def timeraw_import_dimod(self):
        return "import dimod"

    def timeraw_import_dwave_optimization(self):
        return "import dwave.optimization"


class ImportPkg:
    version = "1"

    def timeraw_import_dwave_cloud(self):
        return "import dwave.cloud"

    def timeraw_import_dwave_cloud_api(self):
        return "from dwave.cloud import api"

    def timeraw_import_dwave_cloud_config(self):
        return "from dwave.cloud import config"

    def timeraw_import_dwave_cloud_client(self):
        return "from dwave.cloud import Client"


class CLI:
    version = "1"

    def time_dwave_help(self):
        subprocess.run('dwave --help', capture_output=True, shell=True)

    def time_dwave_ping_unauthenticated(self):
        with isolated_environ(add=dict(DWAVE_API_TOKEN='invalid')):
            ret = subprocess.run('dwave ping', capture_output=True, shell=True)
            assert ret.returncode == 2


class InitClient:
    version = "1"

    def time_config_init(self):
        defaults = ClientConfig()
        resolve_endpoints(defaults, shortcircuit=True)

    def time_client_init(self):
        Client(token='mock')


class InitQPUSolver:
    version = "1"
    params = ["C16", "P16", "Z6", "Z12"]
    param_names = ["topology"]

    generators = {
        "C16": partial(mocks.qpu_chimera_solver_data, m=16),
        "P16": partial(mocks.qpu_pegasus_solver_data, m=16),
        "Z6": partial(mocks.qpu_zephyr_solver_data, m=6),
        "Z12": partial(mocks.qpu_zephyr_solver_data, m=12),
    }

    def setup(self, key):
        self.solver_data = self.generators[key]()

    def time_solver_init(self, key):
        StructuredSolver(client=None, data=self.solver_data)


class InitHybridSolver:
    version = "1"
    params = ["bqm", "cqm", "dqm", "nl"]
    param_names = ["type"]

    def setup(self, key):
        solver_class = {"bqm": BQMSolver,
                        "cqm": CQMSolver,
                        "dqm": DQMSolver,
                        "nl": NLSolver}
        self.solver_class = solver_class[key]

        solver_data = {
            "bqm": partial(mocks.hybrid_bqm_solver_data),
            "cqm": partial(mocks.hybrid_cqm_solver_data),
            "dqm": partial(mocks.hybrid_dqm_solver_data),
            "nl": partial(mocks.hybrid_nl_solver_data),
        }
        self.solver_data = solver_data[key]()

    def time_solver_init(self, key):
        self.solver_class(client=None, data=self.solver_data)


class ProblemEncoding:
    version = "1"
    params = InitQPUSolver.params
    param_names = InitQPUSolver.param_names

    def setup(self, key):
        self.solver = StructuredSolver(client=None, data=InitQPUSolver.generators[key]())
        self.problem = generate_random_ising_problem(self.solver)

    def time_encode_qp(self, key):
        encode_problem_as_qp(self.solver, *self.problem)


# requires internet access
class RegionsMetadata:
    version = "1"

    def setup(self):
        # force region metadata cache refresh
        get_regions(refresh=True, maxage=0)

    def time_get_region_from_cache(self):
        get_regions()


class SolverMetadataJSONDecode:
    version = "1"

    def setup(self):
        self.data = (Path(__file__).parent / 'fixtures/solvers.json').read_bytes()

    def time_json_loads(self):
        # match the decoder used by requests (Response.json())
        json.loads(self.data)


class SolverSelection:
    version = "1"

    def setup(self):
        self.client = Client(token='mock')
        solvers_data = json.loads((Path(__file__).parent / 'fixtures/solvers.json').read_bytes())
        self._original_sapi_request = Client._sapi_request
        Client._sapi_request = lambda session, url: solvers_data

    def teardown(self):
        Client._sapi_request = self._original_sapi_request

    def time_get_solvers(self):
        self.client.get_solvers()

    def time_get_solver(self):
        self.client.get_solver()

    def time_get_solver_nl(self):
        self.client.get_solver(
            supported_problem_types__contains='nl',
            order_by='-properties.version')

    def time_get_solver_pegasus(self):
        self.client.get_solver(
            topology__type='pegasus',
            num_qubits__within=(5000, 6000),
            order_by='-num_active_qubits')

    def time_get_solver_zephyr_small(self):
        self.client.get_solver(
            topology__type='zephyr',
            num_qubits__within=(1000, 2000),
            order_by='-num_active_qubits')

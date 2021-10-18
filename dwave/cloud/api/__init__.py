# Copyright 2021 D-Wave Systems Inc.
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

# NOTE: dwave.cloud.api module and submodules considered private for now!


import dwave.cloud.api.client
import dwave.cloud.api.constants
import dwave.cloud.api.exceptions
import dwave.cloud.api.models
import dwave.cloud.api.resources

from dwave.cloud.api import client
from dwave.cloud.api import constants
from dwave.cloud.api import exceptions
from dwave.cloud.api import models
from dwave.cloud.api import resources

from dwave.cloud.api.client import DWaveAPIClient, SolverAPIClient, MetadataAPIClient
from dwave.cloud.api.resources import Solvers, Problems, Regions

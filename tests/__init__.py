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
import warnings

from dwave.cloud.config import load_config
from dwave.cloud.exceptions import CanceledFutureError, ConfigFileError


# try to load client config needed for live tests on SAPI web service
try:
    # by default, use `test` profile from `tests/dwave.conf`,
    # with secrets (token) read from env
    default_config_path = os.path.join(os.path.dirname(__file__), 'dwave.conf')
    default_config_profile = 'test'

    # allow manual override of config file and profile used for tests
    test_config_path = os.getenv('DWAVE_CONFIG_FILE', default_config_path)
    test_config_profile = os.getenv('DWAVE_PROFILE', default_config_profile)

    config = load_config(config_file=test_config_path,
                         profile=test_config_profile)

    # ensure config is complete
    for var in ['token']:
        if not config[var]:
            raise ValueError("Config incomplete, missing: {!r}".format(var))

except (ConfigFileError, ValueError) as e:
    config = None
    warnings.warn("Skipping live tests due to: {!s}".format(e))

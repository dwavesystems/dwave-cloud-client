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
    for var in 'endpoint token solver'.split():
        if not config[var]:
            raise ValueError("Config incomplete, missing: {!r}".format(var))

except (ConfigFileError, ValueError) as e:
    config = None
    warnings.warn("Skipping live tests due to: {!s}".format(e))

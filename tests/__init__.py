import os
import warnings

from dwave.cloud.config import load_config
from dwave.cloud.exceptions import CanceledFutureError, ConfigFileError


# try to load client config needed for live tests on SAPI webservice
try:
    # explicitly use tests/dwave.conf, with secrets (token) read from env
    test_config_path = os.path.join(os.path.dirname(__file__), 'dwave.conf')

    # use `sw` resource instead of QPU
    config = load_config(config_file=test_config_path, profile='sw')

    # ensure config is complete
    for var in 'endpoint token solver'.split():
        if not config[var]:
            raise ValueError("Config incomplete, missing: {!r}".format(var))

except (ConfigFileError, ValueError) as e:
    config = None
    warnings.warn("Skipping live tests due to: {!s}".format(e))

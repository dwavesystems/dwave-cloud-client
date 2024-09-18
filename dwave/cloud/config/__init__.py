# Copyright 2023 D-Wave Systems Inc.
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

"""
Configuration for communicating with a solver.

Communicating with a solver---submitting a problem, monitoring its progress, receiving
samples---requires configuration of several parameters such as the selected solver,
its URL, an API token, etc. D-Wave Cloud Client provides multiple options for configuring
those parameters:

* One or more locally saved configuration files.
* Environment variables
* Direct setting of key values in functions

These options can be flexibly used together. The standard use is through
the :func:`~dwave.cloud.client.Client.from_config` classmethod.

Configuration values can be specified in multiple ways, ranked in the following
order (with 1 the highest ranked):

1. Values specified as keyword arguments.
2. Values specified as environment variables.
3. Values specified in the configuration file.
4. Values specified in :class:`~dwave.cloud.client.Client` instance ``defaults``.
5. Values specified in :class:`~dwave.cloud.client.Client` class
   :attr:`~dwave.cloud.client.Client.DEFAULTS`.

Configuration files comply with standard Windows INI-like format,
parsable with Python's :mod:`configparser`. An optional ``defaults`` section
provides default key-value pairs for all other sections. User-defined key-value
pairs (unrecognized keys) are passed through to the client.

Typically configuration files are created, inspected, and changed using interactive
CLI commands from your system's console, such as :code:`dwave config create` and
:code:`dwave config inspect` (run :code:`dwave --help` for information on CLI options).

Environment variables:

*   ``DWAVE_API_CLIENT``: API client class. Supported values are ``qpu``, ``sw`` and ``hybrid``.
*   ``DWAVE_API_ENDPOINT``: Solver API endpoint URL.
*   ``DWAVE_API_HEADERS``: Optional additional HTTP headers.
*   ``DWAVE_API_PROXY``: URL for proxy connections to D-Wave API.
*   ``DWAVE_API_REGION``: API region code.
*   ``DWAVE_API_SOLVER``: Default solver.
*   ``DWAVE_API_TOKEN``: Solver API authorization token.
*   ``DWAVE_CONFIG_FILE``: Configuration file path.
*   ``DWAVE_LEAP_API_ENDPOINT``: Leap API endpoint URL.
*   ``DWAVE_LEAP_CLIENT_ID``: Leap OAuth client ID.
*   ``DWAVE_METADATA_API_ENDPOINT``: Metadata API endpoint URL.
*   ``DWAVE_PROFILE``: Name of profile (section).

Examples:
    The following are typical and advanced examples of using
    :func:`~dwave.cloud.client.Client.from_config` to create a configured client.

    *   **Example** Typical use for QPU and hybrid solvers

    This example uses a configuration file in one of the standard paths,
    :code:`~/.config/dwave/dwave.conf`, selected through auto-detection
    on the local system following the user/system configuration paths of
    :func:`get_configfile_paths`, to provide the following default
    configuration::

        [defaults]
        token = ABC-123456789123456789123456789

        [default-solver]
        solver = {"qpu": true, "num_qubits__gt": 5000}

        [bqm]
        client = hybrid
        solver = {"supported_problem_types__contains": "bqm"}

        [cqm]
        client = hybrid
        solver = {"supported_problem_types__contains": "cqm"}

        [europe]
        region = eu-central-1

    This configuration file sets a default API token used by all following
    profiles unless overridden, ensures that selecting a solver with the
    :func:`~dwave.cloud.client.Client.get_solver` method by default returns a
    QPU solver with at least 5000 qubits, and configures profiles for selecting
    a quantum-classical hybrid solver and solvers from Leap's European region.

    The code below instantiates clients for a QPU solver and a hybrid CQM solver.

    >>> with Client.from_config() as client:   # doctest: +SKIP
    ...     solver_qpu = client.get_solver()
    >>> with Client.from_config(profile="cqm") as client:   # doctest: +SKIP
    ...     solver_cqm = client.get_solver()

    *   **Example:** Explicitly specified configuration file, unrecognized parameter

    This example explicitly specifies the following configuration file,
    :code:`~/jane/my_path_to_config/my_cloud_conf.conf`::

        [defaults]
        token = ABC-123456789123456789123456789

        [first-qpu]
        solver = {"qpu": true}

    The code below creates a client that it later explicitly closes. It also
    passes through to the instantiated client an unrecognized key-value pair
    ``my_param="my_value"``.

    >>> from dwave.cloud import Client
    >>> client = Client.from_config(config_file='~/jane/my_path_to_config/my_cloud_conf.conf',
    ...                             my_param="my_value")  # doctest: +SKIP
    >>> # code that uses client
    >>> client.close()   # doctest: +SKIP

    *   **Advanced Example:** Multiple auto-detected configuration files

    This example uses two configuration files: (1) a user-local file,
    ``/usr/local/share/dwave/dwave.conf``::

        [defaults]
        token = ABC-123456789123456789123456789
        solver = {"qpu": true}

        [advantage]
        region = eu-central-1

    and (2), a ``./dwave.conf`` file in the current working directory::

        [advantage]
        token = DEF-987654321987654321987654321

    The code below supplements the API token from higher priority file (the
    ``./dwave.conf`` file in the current working directory), overriding the value
    from the ``[defaults]`` and first (``[advantage]``) sections of the
    lower-priority user-local file, ``/usr/local/share/dwave/dwave.conf``. Use
    of the :func:`~dwave.cloud.client.Client.get_solver` method would select
    an Advantage from Leap's European region using the ``DEF-987 ...`` token.

    >>> from dwave.cloud import Client
    >>> client = Client.from_config()  # doctest: +SKIP
    >>> print(client.config.endpoint)      # doctest: +SKIP
    https://eu-central-1.cloud.dwavesys.com/sapi/v2/
    >>> print(client.config.token)  # doctest: +SKIP
    DEF-987654321987654321987654321
    >>> # code that uses client
    >>> client.close() # doctest: +SKIP
"""

from .constants import *
from .exceptions import *
from .loaders import *
from .models import *

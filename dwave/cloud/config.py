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
parsable with Python's :mod:`configparser`. An optional `defaults` section
provides default key-value pairs for all other sections. User-defined key-value
pairs (unrecognized keys) are passed through to the client.

Typically configuration files are created, inspected, and changed using interactive
CLI commands from your system's console, such as :code:`dwave config create` and
:code:`dwave config inspect` (run :code:`dwave --help` for information on CLI options).

Environment variables:

    ``DWAVE_CONFIG_FILE``: Configuration file path.

    ``DWAVE_PROFILE``: Name of profile (section).

    ``DWAVE_API_CLIENT``: API client class. Supported values are ``qpu``, ``sw`` and ``hybrid``.

    ``DWAVE_API_REGION``: API region code.

    ``DWAVE_API_ENDPOINT``: API endpoint URL.

    ``DWAVE_METADATA_API_ENDPOINT``: Metadata API endpoint URL.

    ``DWAVE_API_TOKEN``: API authorization token.

    ``DWAVE_API_SOLVER``: Default solver.

    ``DWAVE_API_PROXY``: URL for proxy connections to D-Wave API.

    ``DWAVE_API_HEADERS``: Optional additional HTTP headers.

Examples:
    The following are typical examples of using :func:`~dwave.cloud.client.Client.from_config`
    to create a configured client.

    This first example initializes :class:`~dwave.cloud.client.Client` from an
    explicitly specified configuration file, "~/jane/my_path_to_config/my_cloud_conf.conf"::

        [defaults]
        token = ABC-123456789123456789123456789

        [first-qpu]
        solver = {"qpu": true}

        [feature]
        endpoint = https://url.of.some.dwavesystem.com/sapi
        token = DEF-987654321987654321987654321
        solver = {"num_qubits__gte": 2000, "max_anneal_schedule_points__gte": 4}

    The example code below creates a client object that connects to a D-Wave QPU,
    using :class:`dwave.cloud.qpu.Client` and the first available online D-Wave system
    at the default API endpoint URL (https://cloud.dwavesys.com/sapi).
    The ``feature`` profile specifies a solver selected based on available features,
    namely we're requesting the first solver that has at least 2000 qubits and the
    anneal schedule can be described with at least 4 points.

    >>> from dwave.cloud import Client
    >>> client = Client.from_config(config_file='~/jane/my_path_to_config/my_cloud_conf.conf')  # doctest: +SKIP
    >>> # code that uses client
    >>> client.close()   # doctest: +SKIP

    This second example auto-detects a configuration file on the local system following the
    user/system configuration paths of :func:`get_configfile_paths`. It passes through
    to the instantiated client an unrecognized key-value pair my_param=`my_value`.

    >>> from dwave.cloud import Client
    >>> client = Client.from_config(my_param="my_value")    # doctest: +SKIP
    >>> # code that uses client
    >>> client.close()      # doctest: +SKIP

    This third example instantiates two clients, for managing both QPU and software
    solvers. Common key-value pairs are taken from the defaults section of a shared
    configuration file::

        [defaults]
        token = ABC-123456789123456789123456789

        [primary-qpu]
        solver = {"qpu": true}

        [sw-solver]
        client = sw
        solver = c4-sw_sample
        endpoint = https://url.of.some.software.resource.com/my_if
        token = DEF-987654321987654321987654321

        [backup-qpu]
        solver = {"qpu": true, "num_qubits__gte": 2000}
        endpoint = https://url.of.some.dwavesystem.com/sapi
        proxy = http://user:pass@myproxy.com:8080/
        token = XYZ-0101010100112341234123412341234

    The example code below creates client objects for two QPU solvers (at the
    same URL but each with its own solver ID and token) and one software solver.

    >>> from dwave.cloud import Client
    >>> client_qpu1 = Client.from_config(profile='primary-qpu')    # doctest: +SKIP
    >>> client_qpu1 = Client.from_config(profile='backup-qpu')    # doctest: +SKIP
    >>> client_sw1 = Client.from_config(profile='sw-solver')   # doctest: +SKIP
    >>> client_qpu1.default_solver   # doctest: +SKIP
    'EXAMPLE_2000Q_SYSTEM_A'
    >>> client_qpu2.endpoint   # doctest: +SKIP
    'https://url.of.some.dwavesystem.com/sapi'
    >>> # code that uses client
    >>> client_qpu1.close() # doctest: +SKIP
    >>> client_qpu2.close() # doctest: +SKIP
    >>> client_sw1.close() # doctest: +SKIP

    This fourth example loads configurations auto-detected in more than one configuration
    file, with the higher priority file (in the current working directory) supplementing
    and overriding values from the lower priority user-local file. After instantiation,
    an endpoint from the default section and client from the profile section is provided
    from the user-local ``/usr/local/share/dwave/dwave.conf`` file::

        [defaults]
        solver = {"qpu": true}

        [dw2000]
        endpoint = https://int.se.dwavesystems.com/sapi
        token = ABC-123456789123456789123456789

    A solver is supplemented from the file in the current working directory, which also
    overrides the token value. ``./dwave.conf`` is the file in the current directory::

        [dw2000]
        token = DEF-987654321987654321987654321

    >>> from dwave.cloud import Client
    >>> client = Client.from_config()  # doctest: +SKIP
    >>> client.default_solver   # doctest: +SKIP
    'EXAMPLE_2000Q_SYSTEM_A'
    >>> client.endpoint  # doctest: +SKIP
    'https://int.se.dwavesystems.com/sapi'
    >>> client.token  # doctest: +SKIP
    'DEF-987654321987654321987654321'
    >>> # code that uses client
    >>> client.close() # doctest: +SKIP

    The next example uses :func:`~dwave.cloud.config.load_config` to load profile values.
    **Most users do not need to use this method.** It loads from the following configuration
    file, dwave_c.conf, located in the current working directory, and specified explicitly::

        [defaults]
        endpoint = https://url.of.some.dwavesystem.com/sapi
        solver = {"qpu": true}

        [dw2000a]
        solver = {"software": true, "name": "EXAMPLE_2000Q"}
        token = ABC-123456789123456789123456789

        [dw2000b]
        solver = {"qpu": true}
        token = DEF-987654321987654321987654321

    This configuration file contains two profiles in addition to the defaults section.
    In the following example code, first no profile is specified, and the first profile
    after the defaults section is loaded with the solver overridden by the environment
    variable. Next, the second profile is selected
    with the explicitly named solver overriding the environment variable setting.

    >>> import dwave.cloud
    >>> import os
    >>> os.environ['DWAVE_API_SOLVER'] = 'EXAMPLE_2000Q_SYSTEM'   # doctest: +SKIP
    >>> dwave.cloud.config.load_config("./dwave_c.conf")   # doctest: +SKIP
    {'client': 'sw',
     'endpoint': 'https://url.of.some.dwavesystem.com/sapi',
     'proxy': None,
     'headers': None,
     'solver': 'EXAMPLE_2000Q_SYSTEM',
     'token': 'ABC-123456789123456789123456789'}
    >>> dc.config.load_config("./dwave_c.conf", profile='dw2000b', solver='Solver3')   # doctest: +SKIP
    {'client': 'qpu',
     'endpoint': 'https://url.of.some.dwavesystem.com/sapi',
     'proxy': None,
     'headers': None,
     'solver': 'Solver3',
     'token': 'DEF-987654321987654321987654321'}

"""

import os
import ast
import logging
import configparser

import homebase

from dwave.cloud.exceptions import ConfigFileReadError, ConfigFileParseError
from dwave.cloud.package_info import __version__, __packagename__

__all__ = ['get_configfile_paths', 'get_configfile_path', 'get_default_configfile_path',
           'load_config_from_files', 'load_profile_from_files', 'load_config']

logger = logging.getLogger(__name__)

CONF_APP = "dwave"
CONF_AUTHOR = "dwavesystem"
CONF_FILENAME = "dwave.conf"

ENV_OPTION_MAP = {
    'DWAVE_API_CLIENT': 'client',
    'DWAVE_API_REGION': 'region',
    'DWAVE_API_ENDPOINT': 'endpoint',
    'DWAVE_API_TOKEN': 'token',
    'DWAVE_API_SOLVER': 'solver',
    'DWAVE_API_PROXY': 'proxy',
    'DWAVE_API_HEADERS': 'headers',
    'DWAVE_METADATA_API_ENDPOINT': 'metadata_api_endpoint',
}
"""Map of environment variable names to config options."""

MUTUALLY_EXCLUSIVE_OPTIONS = {
    'region-endpoint-group': {'endpoint', 'region'},
}
"""List of options to be cleared on *lower level*, per each option set on higher
level, in addition to the option value override (assumed).
"""


def parse_float(s, default=None):
    """Parse value as returned by ConfigParse as float.

    NB: we need this instead of ``ConfigParser.getfloat`` when we're parsing
    values downstream.
    """

    if s is None or s == '':
        return default
    return float(s)


def parse_int(s, default=None):
    """Parse value as returned by ConfigParse as int.

    NB: we need this instead of ``ConfigParser.getint`` when we're parsing
    values downstream.
    """

    if s is None or s == '':
        return default
    if float(s) != int(s):
        raise ValueError
    return int(s)


def parse_boolean(s, default=None):
    """Parse value as returned by ConfigParse as bool.

    Rules::

        0, false, off => False
        empty/None    => default
        otherwise     => True

    NB: we need this instead of ``ConfigParser.getboolean`` when we're parsing
    values downstream, out of ConfigParser context.
    """

    if s is None or s == '':
        return default

    if isinstance(s, str):
        s = s.lower()
        if s in ['false', 'off']:
            s = 'False'
        elif s in ['true', 'on']:
            s = 'True'
        try:
            s = ast.literal_eval(s)
        except:
            raise ValueError('invalid boolean value: {!r}'.format(s))

    return bool(s)


def get_configfile_paths(system=True, user=True, local=True, only_existing=True):
    """Return a list of local configuration file paths.

    Search paths for configuration files on the local system
    are based on homebase_ and depend on operating system; for example, for Linux systems
    these might include ``dwave.conf`` in the current working directory (CWD),
    user-local ``.config/dwave/``, and system-wide ``/etc/dwave/``.

    .. _homebase: https://github.com/dwavesystems/homebase

    Args:
        system (boolean, default=True):
            Search for system-wide configuration files.

        user (boolean, default=True):
            Search for user-local configuration files.

        local (boolean, default=True):
            Search for local configuration files (in CWD).

        only_existing (boolean, default=True):
            Return only paths for files that exist on the local system.

    Returns:
        list[str]:
            List of configuration file paths.

    Examples:
        This example displays all paths to configuration files on a Windows system
        running Python 2.7 and then finds the single existing configuration file.

        >>> from dwave.cloud.config import get_configfile_paths
        >>> # Display paths
        >>> get_configfile_paths(only_existing=False)   # doctest: +SKIP
        ['C:\\ProgramData\\dwavesystem\\dwave\\dwave.conf',
         'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf',
         '.\\dwave.conf']
        >>> # Find existing files
        >>> get_configfile_paths()   # doctest: +SKIP
        ['C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf']

    """

    candidates = []

    # system-wide has the lowest priority, `/etc/dwave/dwave.conf`
    if system:
        candidates.extend(homebase.site_config_dir_list(
            app_author=CONF_AUTHOR, app_name=CONF_APP,
            use_virtualenv=False, create=False))

    # user-local will override it, `~/.config/dwave/dwave.conf`
    if user:
        candidates.append(homebase.user_config_dir(
            app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
            use_virtualenv=False, create=False))

    # highest priority (overrides all): `./dwave.conf`
    if local:
        candidates.append(".")

    paths = [os.path.join(base, CONF_FILENAME) for base in candidates]
    if only_existing:
        paths = list(filter(os.path.exists, paths))

    return paths


def get_configfile_path():
    """Return the highest-priority local configuration file.

    Selects the top-ranked configuration file path from a list of candidates returned
    by :func:`get_configfile_paths()`, or ``None`` if no candidate path exists.

    Returns:
        str:
            Configuration file path.

    Examples:
        This example displays the highest-priority configuration file on a
        Windows system.

        >>> from dwave.cloud import config
        >>> # Display paths
        >>> config.get_configfile_paths(only_existing=False)   # doctest: +SKIP
        ['C:\\ProgramData\\dwavesystem\\dwave\\dwave.conf',
         'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf',
         '.\\dwave.conf']
        >>> # Find highest-priority local configuration file
        >>> config.get_configfile_path()   # doctest: +SKIP
        'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf'

    """
    paths = get_configfile_paths()
    return paths[-1] if paths else None


def get_default_configfile_path():
    """Return the default configuration-file path.

    Typically returns a user-local configuration file; e.g:
    ``~/.config/dwave/dwave.conf``.

    Returns:
        str:
            Configuration file path.

    Examples:
        This example displays the default configuration file on an Ubuntu Linux
        system.

        >>> from dwave.cloud import config
        >>> # Display paths
        >>> config.get_configfile_paths(only_existing=False)   # doctest: +SKIP
        ['/etc/xdg/xdg-ubuntu/dwave/dwave.conf',
         '/usr/share/upstart/xdg/dwave/dwave.conf',
         '/etc/xdg/dwave/dwave.conf',
         '/home/mary/.config/dwave/dwave.conf',
         './dwave.conf']
        >>> # Find default configuration path
        >>> config.get_default_configfile_path()   # doctest: +SKIP
        '/home/mary/.config/dwave/dwave.conf'

    """
    base = homebase.user_config_dir(
        app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
        use_virtualenv=False, create=False)
    path = os.path.join(base, CONF_FILENAME)
    return path


def get_cache_dir():
    """Return a directory path convenient for storing user-local,
    package-local and version-specific cache data.
    """
    return homebase.user_cache_dir(
        app_name=__packagename__, app_author=CONF_AUTHOR,
        version=__version__, use_virtualenv=False, create=True)


def load_config_from_files(filenames=None):
    """Load D-Wave Cloud Client configuration from a list of files.

    .. note:: This method is not standardly used to set up D-Wave Cloud Client configuration.
        It is recommended you use :meth:`.Client.from_config` or
        :meth:`.config.load_config` instead.

    Configuration files comply with standard Windows INI-like format,
    parsable with Python's :mod:`configparser`. A section called
    ``defaults`` contains default values inherited by other sections.

    Each filename in the list (each configuration file loaded) progressively upgrades
    the final configuration, on a key by key basis, per each section.

    Args:
        filenames (list[str], default=None):
            D-Wave Cloud Client configuration files (paths and names).

            If ``None``, searches for a configuration file named ``dwave.conf``
            in all system-wide configuration directories, in the user-local
            configuration directory, and in the current working directory,
            following the user/system configuration paths of :func:`get_configfile_paths`.

    Returns:
        :obj:`~configparser.ConfigParser`:
            :class:`dict`-like mapping of configuration sections (profiles) to
            mapping of per-profile keys holding values.

    Raises:
        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

    Examples:
        This example loads configurations from two files. One contains a default
        section with key/values that are overwritten by any profile section that
        contains that key/value; for example, profile dw2000b in file dwave_b.conf
        overwrites the default URL and client type, which profile dw2000a inherits
        from the defaults section, while profile dw2000a overwrites the API token that
        profile dw2000b inherits.

        The files, which are located in the current working directory, are
        (1) dwave_a.conf::

            [defaults]
            endpoint = https://url.of.some.dwavesystem.com/sapi
            client = qpu
            token = ABC-123456789123456789123456789

            [dw2000a]
            solver = EXAMPLE_2000Q_SYSTEM
            token = DEF-987654321987654321987654321

        and (2) dwave_b.conf::

            [dw2000b]
            endpoint = https://url.of.some.other.dwavesystem.com/sapi
            client = sw
            solver = EXAMPLE_2000Q_SYSTEM

        The following example code loads configuration from both these files, with
        the defined overrides and inheritance.

        .. code:: python

            >>> import dwave.cloud as dc
            >>> import sys
            >>> configuration = dc.config.load_config_from_files(["./dwave_a.conf", "./dwave_b.conf"])   # doctest: +SKIP
            >>> configuration.write(sys.stdout)    # doctest: +SKIP
            [defaults]
            endpoint = https://url.of.some.dwavesystem.com/sapi
            client = qpu
            token = ABC-123456789123456789123456789

            [dw2000a]
            solver = EXAMPLE_2000Q_SYSTEM
            token = DEF-987654321987654321987654321

            [dw2000b]
            endpoint = https://url.of.some.other.dwavesystem.com/sapi
            client = sw
            solver = EXAMPLE_2000Q_SYSTEM

    """
    if filenames is None:
        filenames = get_configfile_paths()

    config = configparser.ConfigParser(default_section="defaults")
    for filename in filenames:
        try:
            filename = os.path.expandvars(os.path.expanduser(filename))
            with open(filename, 'r') as f:
                config.read_file(f, filename)
        except (IOError, OSError):
            raise ConfigFileReadError("Failed to read {!r}".format(filename))
        except configparser.Error:
            raise ConfigFileParseError("Failed to parse {!r}".format(filename))
    return config


def load_profile_from_files(filenames=None, profile=None):
    """Load a profile from a list of D-Wave Cloud Client configuration files.

    .. note:: This method is not standardly used to set up D-Wave Cloud Client configuration.
        It is recommended you use :meth:`.Client.from_config` or
        :meth:`.config.load_config` instead.

    Configuration files comply with standard Windows INI-like format,
    parsable with Python's :mod:`configparser`.

    Each file in the list is progressively searched until the first profile is found.
    This function does not input profile information from environment variables.

    Args:
        filenames (list[str], default=None):
            D-Wave cloud client configuration files (path and name). If ``None``,
            searches for existing configuration files in the standard directories
            of :func:`get_configfile_paths`.

        profile (str, default=None):
            Name of profile to return from reading the configuration from the specified
            configuration file(s). If ``None``, progressively falls back in the
            following order:

            (1) ``profile`` key following ``[defaults]`` section.
            (2) First non-``[defaults]`` section.
            (3) ``[defaults]`` section.

    Returns:
        dict:
            Mapping of configuration keys to values. If no valid config/profile
            is found, returns an empty dict.

    Raises:
        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

        :exc:`ValueError`:
            Profile name not found.

    Examples:
        This example loads a profile based on configurations from two files. It
        finds the first profile, dw2000a, in the first file, dwave_a.conf, and adds to
        the values of the defaults section, overwriting the existing client value,
        while ignoring the profile in the second file, dwave_b.conf.

        The files, which are located in the current working directory, are
        (1) dwave_a.conf::

            [defaults]
            endpoint = https://url.of.some.dwavesystem.com/sapi
            client = qpu
            token = ABC-123456789123456789123456789

            [dw2000a]
            client = sw
            solver = EXAMPLE_2000Q_SYSTEM_A
            token = DEF-987654321987654321987654321

        and (2) dwave_b.conf::

            [dw2000b]
            endpoint = https://url.of.some.other.dwavesystem.com/sapi
            client = qpu
            solver = EXAMPLE_2000Q_SYSTEM_B

        The following example code loads profile values from parsing both these files,
        by default loading the first profile encountered or an explicitly specified profile.

        >>> from dwave.cloud import config
        >>> config.load_profile_from_files(["./dwave_a.conf", "./dwave_b.conf"])   # doctest: +SKIP
        {'client': 'sw',
         'endpoint': 'https://url.of.some.dwavesystem.com/sapi',
         'solver': 'EXAMPLE_2000Q_SYSTEM_A',
         'token': 'DEF-987654321987654321987654321'}
        >>> config.load_profile_from_files(["./dwave_a.conf", "./dwave_b.conf"],
        ...                                profile='dw2000b')   # doctest: +SKIP
        {'client': 'qpu',
        'endpoint': 'https://url.of.some.other.dwavesystem.com/sapi',
        'solver': 'EXAMPLE_2000Q_SYSTEM_B',
        'token': 'ABC-123456789123456789123456789'}

    """

    # progressively build config from a file, or a list of auto-detected files
    # raises ConfigFileReadError/ConfigFileParseError on error
    config = load_config_from_files(filenames)

    # determine profile name fallback:
    #  (1) profile key under [defaults],
    #  (2) first non-[defaults] section
    #  (3) [defaults] section
    first_section = next(iter(config.sections() + [None]))
    config_defaults = config.defaults()
    if not profile:
        profile = config_defaults.get('profile', first_section)

    if profile:
        try:
            section = dict(config[profile])
        except KeyError:
            raise ValueError("Config profile {!r} not found".format(profile))
    else:
        # as the very last resort (unspecified profile name and
        # no profiles defined in config), try to use [defaults]
        if config_defaults:
            section = config_defaults
        else:
            section = {}

    return section


def get_default_config():
    # Used only internally by the package
    config = configparser.ConfigParser(default_section="defaults")
    config.read_string(u"""
        [defaults]
        # This section provides default values for some variables.
        # Any of these can be overridden in a specific profile definition below,
        # with an environment variable, and in code.

        # D-Wave solver API endpoint URL defaults to a production endpoint
        #endpoint = https://cloud.dwavesys.com/sapi

        # Default client is the generic base client which works with all
        # available remote solvers/samplers. It can be specialized for the
        # QPU resource (QPU sampler), software, and hybrid samplers.
        # Possible values: `qpu`, `sw`, `hybrid`, `base` (equal to unspecified)
        #client = base

        # Profile name to use if otherwise unspecified.
        # If undefined, the first section below will be used as the default profile.
        #profile = prod

        # Feature-based definition of solver to be used for sampling. If defining
        # the solver in here, make sure that solver is provided on the endpoint used
        # and with the client used (`qpu` or `sw`; `base` can handle any).
        #solver = {"qpu": true, "online": true, "num_qubits__gte": 2000}

        # Proxy URL (including authentication credentials) that shall be used
        # for all requests to D-Wave API endpoint URL.
        #proxy = http://user:pass@proxy.org:8080/
    """)
    return config


def load_config(config_file=None, profile=None, **kwargs):
    """Load D-Wave Cloud Client configuration based on a configuration file.

    Configuration values can be specified in multiple ways, ranked in the following
    order (with 1 the highest ranked):

    1. Values specified as keyword arguments in :func:`load_config()`. These values replace
       values read from a configuration file, and therefore must be **strings**, including float
       values for timeouts, boolean flags, and solver feature
       constraints (a dictionary encoded as JSON).
    2. Values specified as environment variables.
    3. Values specified in the configuration file.
    4. Values specified as :class:`~dwave.cloud.client.Client` instance defaults.
    5. Values specified in :class:`~dwave.cloud.client.Client` class
       :attr:`~dwave.cloud.client.Client.DEFAULTS`.

    Configuration-file format is described in :mod:`dwave.cloud.config`.

    Available configuration-file options are identical to
    :class:`~dwave.cloud.client.Client` constructor argument names.

    If the location of the configuration file is not specified, auto-detection
    searches for existing configuration files in the standard directories
    of :func:`get_configfile_paths`.

    If a configuration file explicitly specified, via an argument or
    environment variable, does not exist or is unreadable, loading fails with
    :exc:`~dwave.cloud.exceptions.ConfigFileReadError`. Loading fails
    with :exc:`~dwave.cloud.exceptions.ConfigFileParseError` if the file is
    readable but invalid as a configuration file.

    Similarly, if a profile explicitly specified, via an argument or
    environment variable, is not present in the loaded configuration, loading fails
    with :exc:`ValueError`. Explicit profile selection also fails if the configuration
    file is not explicitly specified, detected on the system, or defined via
    an environment variable.

    Environment variables: ``DWAVE_CONFIG_FILE``, ``DWAVE_PROFILE``,
    ``DWAVE_API_CLIENT``, ``DWAVE_API_REGION``, ``DWAVE_API_ENDPOINT``,
    ``DWAVE_API_TOKEN``, ``DWAVE_API_SOLVER``, ``DWAVE_API_PROXY``,
    ``DWAVE_API_HEADERS``, ``DWAVE_METADATA_API_ENDPOINT``.

    Environment variables are described in :mod:`dwave.cloud.config`.

    Args:
        config_file (str/[str]/None/False/True, default=None):
            Path to configuration file(s).

            If `None`, the value is taken from `DWAVE_CONFIG_FILE` environment
            variable if defined. If the environment variable is undefined or empty,
            auto-detection searches for existing configuration files in the standard
            directories of :func:`get_configfile_paths`.

            If `False`, loading from file(s) is skipped; if `True`, forces auto-detection
            (regardless of the `DWAVE_CONFIG_FILE` environment variable).

        profile (str, default=None):
            Profile name (name of the profile section in the configuration file).

            If undefined, inferred from `DWAVE_PROFILE` environment variable if
            defined. If the environment variable is undefined or empty, a profile is
            selected in the following order:

            1. From the default section if it includes a profile key.
            2. The first section (after the default section).
            3. If no other section is defined besides `[defaults]`, the defaults
               section is promoted and selected.

        **kwargs (dict, optional):
            :class:`~dwave.cloud.client.Client` constructor arguments.

    Returns:
        dict:
            Mapping of configuration keys to values for the profile (section),
            as read from the configuration file and optionally overridden by
            environment values and specified keyword arguments.

    Raises:
        :exc:`ValueError`:
            Invalid (non-existing) profile name.

        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

    Note:
        Prior to 0.8.0, some keyword arguments did not overwrite config
        variables when their value was ``None``. Now we consistently do
        :meth:`dict.update` on the config read from file/env for all ``kwargs``.

    Examples:
        This example loads the configuration from an auto-detected configuration file
        in the home directory of a Windows system user.

        >>> from dwave.cloud import config
        >>> config.load_config()         # doctest: +SKIP
        {'client': 'qpu',
         'endpoint': 'https://url.of.some.dwavesystem.com/sapi',
         'proxy': None,
         'solver': 'EXAMPLE_2000Q_SYSTEM_A',
         'token': 'DEF-987654321987654321987654321',
         'headers': None}
        ... # See which configuration file was loaded
        >>> config.get_configfile_paths()             # doctest: +SKIP
        ['C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf']

        Additional examples are given in :mod:`dwave.cloud.config`.

    """

    logger.trace("load_config(config_file=%r, profile=%r, kwargs=%r)",
                 config_file, profile, kwargs)

    if profile is None:
        profile = os.getenv("DWAVE_PROFILE")

    if config_file == False:
        # skip loading from file altogether
        section = {}
    elif config_file == True:
        # force auto-detection, disregarding DWAVE_CONFIG_FILE
        section = load_profile_from_files(None, profile)
    else:
        # auto-detect if not specified with arg or env
        if config_file is None:
            # note: both empty and undefined DWAVE_CONFIG_FILE treated as None
            config_file = os.getenv("DWAVE_CONFIG_FILE")

        # handle ''/None/str/[str] for `config_file` (after env)
        filenames = None
        if config_file:
            if isinstance(config_file, str):
                filenames = [config_file]
            else:
                filenames = config_file

        section = load_profile_from_files(filenames, profile)

    logger.trace("config (from files) = %r", section)

    # override with env
    update_config_from_environment(section)
    logger.trace("config (from files+env) = %r", section)

    # override with supplied kwarg options
    update_config(section, kwargs)
    logger.trace("config (from files+env+kwargs) = %r", section)

    return section


def update_config_from_environment(section):
    """Update config profile/``section`` with values from environment variables.

    Supported environment variables are listed as keys in
    :attr:`.ENV_OPTION_MAP`, with corresponding config option names as values.
    """

    envopts = {opt: os.getenv(env) for env, opt in ENV_OPTION_MAP.items()}

    update_config(section, envopts)


def update_config(config: dict, options: dict) -> None:
    """Update `config` inplace with `options`, ignoring None and blank string
    values, clearing mutually exclusive options on different levels.
    """
    # skip null and empty option values
    # TODO: not needed when we switch to structured/typed config (like YAML)
    updates = {k: v for k,v in options.items() if v is not None and v != ''}

    # handle mutually exclusive options first (so it's order invariant)
    for group, optionset in MUTUALLY_EXCLUSIVE_OPTIONS.items():
        if updates.keys() & optionset:
            for excluded in optionset:
                config.pop(excluded, None)

    config.update(updates)

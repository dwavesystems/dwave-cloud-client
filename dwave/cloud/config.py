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

1. Values specified as keyword arguments
2. Values specified as environment variables
3. Values specified in the configuration file

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

    ``DWAVE_API_CLIENT``: API client class. Supported values are ``qpu`` or ``sw``.

    ``DWAVE_API_ENDPOINT``: API endpoint URL.

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

    >>> import dwave.cloud as dc
    >>> import os
    >>> os.environ['DWAVE_API_SOLVER'] = 'EXAMPLE_2000Q_SYSTEM'   # doctest: +SKIP
    >>> dc.config.load_config("./dwave_c.conf")   # doctest: +SKIP
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
import configparser
from collections import OrderedDict

import homebase

from dwave.cloud.utils import uniform_get
from dwave.cloud.exceptions import ConfigFileReadError, ConfigFileParseError

__all__ = ['get_configfile_paths', 'get_configfile_path', 'get_default_configfile_path',
           'load_config_from_files', 'load_profile_from_files',
           'load_config', 'legacy_load_config']

CONF_APP = "dwave"
CONF_AUTHOR = "dwavesystem"
CONF_FILENAME = "dwave.conf"


def parse_float(s):
    """Parse value as returned by ConfigParse as float.

    NB: we need this instead of ``ConfigParser.getfloat`` when we're parsing
    values downstream."""

    if s is None or s == '':
        return None
    return float(s)


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

        >>> import dwave.cloud as dc
        >>> # Display paths
        >>> dc.config.get_configfile_paths(only_existing=False)   # doctest: +SKIP
        ['C:\\ProgramData\\dwavesystem\\dwave\\dwave.conf',
         'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf',
         '.\\dwave.conf']
        >>> # Find existing files
        >>> dc.config.get_configfile_paths()   # doctest: +SKIP
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
        This example displays the highest-priority configuration file on a Windows system
        running Python 2.7.

        >>> import dwave.cloud as dc
        >>> # Display paths
        >>> dc.config.get_configfile_paths(only_existing=False)   # doctest: +SKIP
        ['C:\\ProgramData\\dwavesystem\\dwave\\dwave.conf',
         'C:\\Users\\jane\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf',
         '.\\dwave.conf']
        >>> # Find highest-priority local configuration file
        >>> dc.config.get_configfile_path()   # doctest: +SKIP
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
        This example displays the default configuration file on an Ubuntu Unix system
        running IPython 2.7.

        >>> import dwave.cloud as dc
        >>> # Display paths
        >>> dc.config.get_configfile_paths(only_existing=False)   # doctest: +SKIP
        ['/etc/xdg/xdg-ubuntu/dwave/dwave.conf',
         '/usr/share/upstart/xdg/dwave/dwave.conf',
         '/etc/xdg/dwave/dwave.conf',
         '/home/mary/.config/dwave/dwave.conf',
         './dwave.conf']
        >>> # Find default configuration path
        >>> dc.config.get_default_configfile_path()   # doctest: +SKIP
        '/home/mary/.config/dwave/dwave.conf'

    """
    base = homebase.user_config_dir(
        app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
        use_virtualenv=False, create=False)
    path = os.path.join(base, CONF_FILENAME)
    return path


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
        # QPU resource (QPU sampler), and software samplers.
        # Possible values: `qpu`, `sw`, `base` (equal to unspecified)
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


def load_config(config_file=None, profile=None, client=None,
                endpoint=None, token=None, solver=None,
                proxy=None, headers=None):
    """Load D-Wave Cloud Client configuration based on a configuration file.

    Configuration values can be specified in multiple ways, ranked in the following
    order (with 1 the highest ranked):

    1. Values specified as keyword arguments in :func:`load_config()`. These values replace
       values read from a configuration file, and therefore must be **strings**, including float
       values for timeouts, boolean flags (tested for "truthiness"), and solver feature
       constraints (a dictionary encoded as JSON).
    2. Values specified as environment variables.
    3. Values specified in the configuration file.

    Configuration-file format is described in :mod:`dwave.cloud.config`.

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
    ``DWAVE_API_CLIENT``, ``DWAVE_API_ENDPOINT``, ``DWAVE_API_TOKEN``,
    ``DWAVE_API_SOLVER``, ``DWAVE_API_PROXY``, ``DWAVE_API_HEADERS``.

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

        client (str, default=None):
            Client type used for accessing the API. Supported values are `qpu`
            for :class:`dwave.cloud.qpu.Client` and `sw` for
            :class:`dwave.cloud.sw.Client`.

        endpoint (str, default=None):
            API endpoint URL.

        token (str, default=None):
            API authorization token.

        solver (dict/str, default=None):
            :term:`solver` features, as a JSON-encoded dictionary of feature constraints,
            the client should use. See :meth:`~dwave.cloud.client.Client.get_solvers` for
            semantics of supported feature constraints.

            If undefined, the client uses a solver definition from environment variables,
            a configuration file, or falls back to the first available online solver.

            For backward compatibility, solver name in string format is accepted and
            converted to ``{"name": <solver name>}``.

        proxy (str, default=None):
            URL for proxy to use in connections to D-Wave API. Can include
            username/password, port, scheme, etc. If undefined, client
            uses the system-level proxy, if defined, or connects directly to the API.

        headers (dict/str, default=None):
            Header lines to include in API calls, each line formatted as
             ``Key: value``, or a parsed dictionary.

    Returns:
        dict:
            Mapping of configuration keys to values for the profile (section),
            as read from the configuration file and optionally overridden by
            environment values and specified keyword arguments. Always contains
            the `client`, `endpoint`, `token`, `solver`, and `proxy` keys.

    Raises:
        :exc:`ValueError`:
            Invalid (non-existing) profile name.

        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

    Examples
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

    # override a selected subset of values via env or kwargs,
    # pass-through the rest unmodified
    section['client'] = client or os.getenv("DWAVE_API_CLIENT", section.get('client'))
    section['endpoint'] = endpoint or os.getenv("DWAVE_API_ENDPOINT", section.get('endpoint'))
    section['token'] = token or os.getenv("DWAVE_API_TOKEN", section.get('token'))
    section['solver'] = solver or os.getenv("DWAVE_API_SOLVER", section.get('solver'))
    section['proxy'] = proxy or os.getenv("DWAVE_API_PROXY", section.get('proxy'))
    section['headers'] = headers or os.getenv("DWAVE_API_HEADERS", section.get('headers'))

    return section


def legacy_load_config(profile=None, endpoint=None, token=None, solver=None,
                       proxy=None, **kwargs):
    """Load configured URLs and token for the SAPI server.

    .. warning:: Included only for backward compatibility. Please use
        :func:`load_config` or the client factory
        :meth:`~dwave.cloud.client.Client.from_config` instead.

    This method tries to load a legacy configuration file from ``~/.dwrc``, select a
    specified `profile` (or, if not specified, the first profile), and override
    individual keys with values read from environment variables or
    specified explicitly as key values in the function.

    Configuration values can be specified in multiple ways, ranked in the following
    order (with 1 the highest ranked):

    1. Values specified as keyword arguments in :func:`legacy_load_config()`
    2. Values specified as environment variables
    3. Values specified in the legacy ``~/.dwrc`` configuration file

    Environment variables searched for are:

     - ``DW_INTERNAL__HTTPLINK``
     - ``DW_INTERNAL__TOKEN``
     - ``DW_INTERNAL__HTTPPROXY``
     - ``DW_INTERNAL__SOLVER``

    Legacy configuration file format is a modified CSV where the first comma is
    replaced with a bar character (``|``). Each line encodes a single profile. Its
    columns are::

        profile_name|endpoint_url,authentication_token,proxy_url,default_solver_name

    All its fields after ``authentication_token`` are optional.

    When there are multiple connections in a file, the first one is
    the default. Any commas in the URLs must be percent-encoded.

    Args:
        profile (str):
            Profile name in the legacy configuration file.

        endpoint (str, default=None):
            API endpoint URL.

        token (str, default=None):
            API authorization token.

        solver (str, default=None):
            Default solver to use in :meth:`~dwave.cloud.client.Client.get_solver`.
            If undefined, all calls to :meth:`~dwave.cloud.client.Client.get_solver`
            must explicitly specify the solver name/id.

        proxy (str, default=None):
            URL for proxy to use in connections to D-Wave API. Can include
            username/password, port, scheme, etc. If undefined, client uses a
            system-level proxy, if defined, or connects directly to the API.

    Returns:
        Dictionary with keys: endpoint, token, solver, and proxy.

    Examples:
        This example creates a client using the :meth:`~dwave.cloud.client.Client.from_config`
        method, which falls back on the legacy file by default when it fails to
        find a D-Wave Cloud Client configuration file (setting its `legacy_config_fallback`
        parameter to False precludes this fall-back operation). For this example,
        no D-Wave Cloud Client configuration file is present on the local system;
        instead the following ``.dwrc`` legacy configuration file is present in the
        user's home directory::

            profile-a|https://one.com,token-one
            profile-b|https://two.com,token-two

        The following example code creates a client without explicitly specifying
        key values, therefore auto-detection searches for existing (non-legacy) configuration
        files in the standard directories of :func:`get_configfile_paths` and, failing to
        find one, falls back on the existing legacy configuration file above.

        >>> import dwave.cloud as dc
        >>> client = dwave.cloud.Client.from_config()    # doctest: +SKIP
        >>> client.endpoint   # doctest: +SKIP
        'https://one.com'
        >>> client.token    # doctest: +SKIP
        'token-one'

        The following examples specify a profile and/or token.

        >>> # Explicitly specify a profile
        >>> client = dwave.cloud.Client.from_config(profile='profile-b')  # doctest: +SKIP
        ... # Will try to connect with the url `https://two.com` and the token `token-two`.
        >>> client = dwave.cloud.Client.from_config(profile='profile-b', token='new-token')    # doctest: +SKIP
        ... # Will try to connect with the url `https://two.com` and the token `new-token`.

    """

    def _parse_config(fp, filename):
        fields = ('endpoint', 'token', 'proxy', 'solver')
        config = OrderedDict()
        for line in fp:
            # strip whitespace, skip blank and comment lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # parse each record, store in dict with label as key
            try:
                label, data = line.split('|', 1)
                values = [v.strip() or None for v in data.split(',')]
                config[label] = dict(zip(fields, values))
            except:
                raise ConfigFileParseError(
                    "Failed to parse {!r}, line {!r}".format(filename, line))
        return config

    def _read_config(filename):
        try:
            with open(filename, 'r') as f:
                return _parse_config(f, filename)
        except (IOError, OSError):
            raise ConfigFileReadError("Failed to read {!r}".format(filename))

    config = {}
    filename = os.path.expanduser('~/.dwrc')
    if os.path.exists(filename):
        config = _read_config(filename)

    # load profile if specified, or first one in file
    if profile:
        try:
            section = config[profile]
        except KeyError:
            raise ValueError("Config profile {!r} not found".format(profile))
    else:
        try:
            _, section = next(iter(config.items()))
        except StopIteration:
            section = {}

    # override config variables (if any) with environment and then with arguments
    section['endpoint'] = endpoint or os.getenv("DW_INTERNAL__HTTPLINK", section.get('endpoint'))
    section['token'] = token or os.getenv("DW_INTERNAL__TOKEN", section.get('token'))
    section['proxy'] = proxy or os.getenv("DW_INTERNAL__HTTPPROXY", section.get('proxy'))
    section['solver'] = solver or os.getenv("DW_INTERNAL__SOLVER", section.get('solver'))
    section.update(kwargs)

    return section

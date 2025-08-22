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

from __future__ import annotations

import os
import ast
import logging
import configparser
from typing import Optional, Union
from urllib.parse import quote, unquote

import homebase

from dwave.cloud.config.exceptions import ConfigFileReadError, ConfigFileParseError
from dwave.cloud.package_info import __version__, __packagename__

__all__ = ['get_configfile_paths', 'get_configfile_path',
           'get_default_configfile_path', 'get_default_config',
           'load_config_from_files', 'load_profile_from_files',
           'load_config', 'update_config',
           # XXX: for backward compat only / temporary
           'parse_float', 'parse_int', 'parse_boolean', 'get_cache_dir']

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
    'DWAVE_LEAP_API_ENDPOINT': 'leap_api_endpoint',
    'DWAVE_LEAP_CLIENT_ID': 'leap_client_id',
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


def _solver_id_as_identity(id: str) -> dict:
    """Construct a solver identity dictionary from the unique string representation
    generated with :func:`_dump_solver_id`.

    Note: for internal use only. For public interface, see
    :meth:`dwave.cloud.api.models.SolverIdentity.from_id`.
    """
    name, *version = id.strip().split(';')
    identity = dict(name=unquote(name))
    if not identity['name']:
        raise ValueError('Invalid id string: missing "name"')

    version = dict(map(unquote, v.split('=', maxsplit=1)) for v in version)
    identity.update(dict(version=version) if version else {})

    return identity


def _solver_identity_as_id(identity: dict) -> str:
    """Serialize solver identity dictionary to a unique string representation
    that includes the ``name`` and all of the ``version`` fields.

    Note: for internal use only. For public interface, see
    :meth:`dwave.cloud.api.models.SolverIdentity.to_id`.
    """
    s = quote(identity.get('name', ''))
    if not s:
        raise ValueError('Invalid "identity" dict: missing "name"')

    if v := identity.get('version'):
        v = ";".join(f"{quote(str(k))}={quote(str(v))}" for k,v in v.items())
        s = f"{s};{v}"

    return s


def get_configfile_paths(
        *, system: bool = True, user: bool = True, local: bool = True,
        only_existing: bool = True, app_author: str = CONF_AUTHOR,
        app_name: str = CONF_APP, filename: str = CONF_FILENAME) -> list[str]:
    """Return a list of local configuration file paths.

    Search paths for configuration files on the local system
    are based on homebase_ and depend on operating system; for example, for Linux systems
    these might include ``dwave.conf`` in the current working directory (CWD),
    user-local ``.config/dwave/``, and system-wide ``/etc/dwave/``.

    .. _homebase: https://github.com/dwavesystems/homebase

    Args:
        system:
            Search for system-wide configuration files.
        user:
            Search for user-local configuration files.
        local:
            Search for local configuration files (in CWD).
        only_existing:
            Return only paths for files that exist on the local system.
        app_author:
            Application author, used by ``homebase`` to determine config file paths.
        app_name:
            Application name, used by ``homebase`` to determine config file paths.
        filename:
            Configuration filename.

    Returns:
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
            app_author=app_author, app_name=app_name,
            use_virtualenv=False, create=False))

    # user-local will override it, `~/.config/dwave/dwave.conf`
    if user:
        candidates.append(homebase.user_config_dir(
            app_author=app_author, app_name=app_name, roaming=False,
            use_virtualenv=False, create=False))

    # highest priority (overrides all): `./dwave.conf`
    if local:
        candidates.append(".")

    paths = [os.path.join(base, filename) for base in candidates]
    if only_existing:
        paths = list(filter(os.path.exists, paths))

    return paths


def get_configfile_path(**kwargs) -> str:
    """Return the highest-priority local configuration file.

    Selects the top-ranked configuration file path from a list of candidates returned
    by :func:`get_configfile_paths()`, or ``None`` if no candidate path exists.

    Args:
        **kwargs:
            Arguments passed-thru to :func:`get_configfile_paths`.

    Returns:
        Configuration file path.
    """
    paths = get_configfile_paths(**kwargs)
    return paths[-1] if paths else None


def get_default_configfile_path(**kwargs) -> str:
    """Return the default configuration-file path.

    Typically returns a user-local configuration file; e.g:
    ``~/.config/dwave/dwave.conf``.

    Args:
        **kwargs:
            Arguments passed-thru to :func:`get_configfile_paths`.

    Returns:
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
    kwargs.update(system=False, user=True, local=False, only_existing=False)
    return get_configfile_paths(**kwargs)[0]


def get_cache_dir(create: bool = False) -> str:
    """Return a directory path convenient for storing user-local,
    package-local and version-specific cache data.
    """
    path = homebase.user_cache_dir(
        app_name=__packagename__, app_author=CONF_AUTHOR,
        version=__version__, use_virtualenv=False, create=False)

    # avoid possible race condition on create (https://github.com/dwavesystems/homebase/issues/37)
    if create:
        os.makedirs(path, exist_ok=True)

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
        :exc:`~dwave.cloud.config.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.config.exceptions.ConfigFileParseError`:
            Config file parse failed.

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
        :exc:`~dwave.cloud.config.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.config.exceptions.ConfigFileParseError`:
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


def load_config(config_file: Optional[Union[str, bool]] = None,
                profile: Optional[str] = None,
                **kwargs) -> dict:
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
    :exc:`~dwave.cloud.config.exceptions.ConfigFileReadError`. Loading fails
    with :exc:`~dwave.cloud.config.exceptions.ConfigFileParseError` if the file is
    readable but invalid as a configuration file.

    Similarly, if a profile explicitly specified, via an argument or
    environment variable, is not present in the loaded configuration, loading fails
    with :exc:`ValueError`. Explicit profile selection also fails if the configuration
    file is not explicitly specified, detected on the system, or defined via
    an environment variable.

    Environment variables: ``DWAVE_CONFIG_FILE``, ``DWAVE_PROFILE``,
    ``DWAVE_API_CLIENT``, ``DWAVE_API_REGION``, ``DWAVE_API_ENDPOINT``,
    ``DWAVE_API_TOKEN``, ``DWAVE_API_SOLVER``, ``DWAVE_API_PROXY``,
    ``DWAVE_API_HEADERS``, ``DWAVE_LEAP_API_ENDPOINT``, ``DWAVE_LEAP_CLIENT_ID``,
    ``DWAVE_METADATA_API_ENDPOINT``.

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

        :exc:`~dwave.cloud.config.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.config.exceptions.ConfigFileParseError`:
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
        {'solver': '{"qpu": true, "num_qubits__gt": 5000}',
         'token': 'ABC-123456789123456789123456789'}
        >>> # See which configuration file was loaded
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

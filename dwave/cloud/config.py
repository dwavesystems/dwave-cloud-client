import os
import configparser
from collections import OrderedDict

import homebase

from dwave.cloud.utils import uniform_get
from dwave.cloud.exceptions import ConfigFileReadError, ConfigFileParseError


CONF_APP = "dwave"
CONF_AUTHOR = "dwavesystem"
CONF_FILENAME = "dwave.conf"


def get_configfile_paths(system=True, user=True, local=True, only_existing=True):
    """Returns a list of (existing) config files found on disk.

    Candidates examined depend on the OS, but for Linux possible list is:
    ``dwave.conf`` in CWD, user-local ``.config/dwave/``, system-wide
    ``/etc/dwave/``. For details, see :func:`load_config_from_files`.

    Args:
        system (boolean, default=True):
            Search for system-wide config files.

        user (boolean, default=True):
            Search for user-local config files.

        local (boolean, default=True):
            Search for local config files (in CWD).

        only_existing (boolean, default=True):
            Return only paths for files that exist on disk.

    Returns:
        list[str]:
            A list of config file paths.
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
    """Returns the highest-priority existing config file from a list
    of possible candidates returned by `get_configfile_paths()`, and
    ``None`` if no candidate config file exists."""
    paths = get_configfile_paths()
    return paths[-1] if paths else None


def get_default_configfile_path():
    """Returns the preferred config file path: a user-local config, e.g:
    ``~/.config/dwave/dwave.conf``."""
    base = homebase.user_config_dir(
        app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
        use_virtualenv=False, create=False)
    path = os.path.join(base, CONF_FILENAME)
    return path


def load_config_from_files(filenames=None):
    """Load D-Wave cloud client configuration from a list of ``filenames``.

    The format of the config file is the standard Windows INI-like format,
    parsable with the Python's :mod:`configparser`.

    Each filename in the list (each config file loaded) progressively upgrades
    the final configuration (on a key by key basis, per each section).

    The section containing default values inherited by other sections is called
    ``defaults``. For example::

        [defaults]
        endpoint = https://cloud.dwavesys.com/sapi
        client = qpu

        [dw2000]
        solver = DW_2000Q_1
        token = ...

        [software]
        client = sw
        solver = c4-sw_sample
        token = ...

        [alpha]
        endpoint = https://url.to.alpha/api
        proxy = http://user:pass@myproxy.com:8080/
        token = ...

    Args:
        filenames (list[str], default=None):
            D-Wave cloud client configuration file locations.

            If set to ``None``, a config file named ``dwave.conf`` is searched for
            in all system-wide config dirs, then in the user-local config dir,
            and finally in the current directory. For example, on Unix, we try
            to load the config from these paths (in order) and possibly others
            (depending on your Unix flavour)::

                /usr/share/dwave/dwave.conf
                /usr/local/share/dwave/dwave.conf
                ~/.config/dwave/dwave.conf
                ./dwave.conf

            On Windows 7+, config file should be located in:
            ``C:\\Users\\<username>\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf``,
            and on Mac OS X in: ``~/Library/Application Support/dwave/dwave.conf``.
            For details on user/system config paths see homebase_.

            .. _homebase: https://github.com/dwavesystems/homebase

    Returns:
        :obj:`~configparser.ConfigParser`:
            A :class:`dict`-like mapping of config sections (profiles) to
            mapping of per-profile keys holding values.

    Raises:
        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.
    """
    if filenames is None:
        filenames = get_configfile_paths()

    config = configparser.ConfigParser(default_section="defaults")
    for filename in filenames:
        try:
            with open(filename, 'r') as f:
                config.read_file(f, filename)
        except (IOError, OSError):
            raise ConfigFileReadError("Failed to read {!r}".format(filename))
        except configparser.Error:
            raise ConfigFileParseError("Failed to parse {!r}".format(filename))
    return config


def load_profile_from_files(filenames=None, profile=None):
    """Load config from a list of `filenames`, returning only section
    defined with `profile`.

    Note:
        Config files and profile name are **not** read from process environment.

    Args:
        filenames (list[str], default=None):
            D-Wave cloud client configuration file locations. Set to ``None`` to
            auto-detect config files, as described in
            :func:`load_config_from_files`.

        profile (str, default=None):
            Name of the profile to return from configuration read from config
            file(s). Set to ``None`` fallback to ``profile`` key under
            ``[defaults]`` section, or the first non-defaults section, or the
            actual ``[defaults]`` section.

    Returns:
        dict:
            Mapping of config keys to config values. If no valid config/profile
            found, returns an empty dict.

    Raises:
        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.

        :exc:`ValueError`:
            Profile name not found.
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
    config = configparser.ConfigParser(default_section="defaults")
    config.read_string(u"""
        [defaults]
        # This section provides default values for some variables.
        # Any of these can be overridden in a profile definition below.

        # D-Wave solver API endpoint URL defaults to a production endpoint
        endpoint = https://cloud.dwavesys.com/sapi

        # Default client is for the QPU resource (QPU sampler)
        # Possible values: qpu, sw
        client = qpu

        # Profile name to use if otherwise unspecified.
        # If undefined, the first section below will be used as the default profile.
        #profile = prod

        # Solver name used for sampling. If defining the solver in config,
        # make sure that solver is provided on the endpoint used.
        #solver = DW_2000Q_1

        # Proxy URL (including authentication credentials) that shall be used
        # for all requests to D-Wave API endpoint URL.
        #proxy = http://user:pass@proxy.org:8080/
    """)
    return config


def load_config(config_file=None, profile=None, client=None,
                endpoint=None, token=None, solver=None, proxy=None):
    """Load D-Wave cloud client configuration from ``config_file`` (either
    explicitly specified, or auto-detected) for ``profile``, but override the
    values from config file with values defined in process environment, and
    override those with values specified with keyword arguments.

    Config file uses the standard Windows INI format, with one profile per
    section. A section providing default values for all other sections is called
    ``defaults``.

    If the location of ``config_file`` is not specified, an auto-detection is
    performed (looking first for ``dwave.conf`` in process' current working
    directory, then in user-local config directories, and finally in system-wide
    config dirs). For details on format and location detection, see
    :func:`load_config_from_files`.

    If location of ``config_file`` is explicitly specified (via arguments or
    environment variable), but the file does not exits, or is not readable,
    config loading will fail with
    :exc:`~dwave.cloud.exceptions.ConfigFileReadError`. Config loading will fail
    with :exc:`~dwave.cloud.exceptions.ConfigFileParseError` if file is
    readable, but it's not a valid config file.

    Similarly, if ``profile`` is explicitly specified (via arguments or
    environment variable), config loading will fail with :exc:`ValueError` if
    that profile is not present in the config loaded. If config file is not
    specified explicitly, nor detected on file system, not defined via
    environment, resulting in an empty config, explicit profile selection will
    also fail.

    If profile is not explicitly specified, selection of profile is described
    under ``profile`` argument below.

    Environment variables:

        ``DWAVE_CONFIG_FILE``:
            Config file path used if ``config_file`` not specified.

        ``DWAVE_PROFILE``:
            Name of config profile (section) to use if ``profile`` not specified.

        ``DWAVE_API_CLIENT``:
            API client class used (can be: ``qpu`` or ``sw``). Overrides values
            from config file, but is overridden with ``client``.

        ``DWAVE_API_ENDPOINT``:
            API endpoint URL to use instead of the URL given in config file,
            if ``endpoint`` not given.

        ``DWAVE_API_TOKEN``:
            API authorization token. Overrides values from config file, but is
            overridden with ``token``.

        ``DWAVE_API_SOLVER``:
            Default solver. Overrides values from config file, but is overridden
            with ``solver``.

        ``DWAVE_API_PROXY``:
            URL for proxy to use in connections to D-Wave API. Overrides values
            from config file, but is overridden with ``proxy``.

    Args:

        config_file (str/None/False/True, default=None):
            Path to config file.

            If undefined (set to ``None``), the name of the config file is
            taken from ``DWAVE_CONFIG_FILE`` environment variable.
            If that env var is undefined or empty, the location of configuration
            files is auto-detected, as described in :func:`load_config_from_files`.

            Config loading from files, including auto-detected ones, can be
            skipped if `config_file` is set to ``False``.

            Auto-detection is forced (disregarding ``DWAVE_CONFIG_FILE`` env
            var) by setting `config_file` to ``True``.

        profile (str, default=None):
            Profile name (config file section name). If undefined (by default),
            it is inferred from ``DWAVE_PROFILE`` environment variable, and if
            that variable is not present, ``profile`` key is looked-up in the
            ``[defaults]`` config section. If ``profile`` is not defined under
            ``[defaults]``, the first section is used. If no other sections are
            defined besides ``[defaults]``, the ``[defaults]`` section is
            promoted to profile.

        client (str, default=None):
            Client (selected by name) to use for accessing the API. Use ``qpu``
            to specify the :class:`dwave.cloud.qpu.Client` and ``sw`` for
            :class:`dwave.cloud.sw.Client`.

        endpoint (str, default=None):
            API endpoint URL.

        token (str, default=None):
            API authorization token.

        solver (str, default=None):
            Default solver to use in :meth:`~dwave.cloud.client.Client.get_solver`.
            If undefined, you'll have to explicitly specify the solver name/id
            in all calls to :meth:`~dwave.cloud.client.Client.get_solver`.

        proxy (str, default=None):
            URL for proxy to use in connections to D-Wave API. Can include
            username/password, port, scheme, etc. If undefined, client will
            connect directly to the API (unless you use a system-level proxy).

    Returns:
        dict:
            Mapping of config keys to config values, for a specific profile
            (section), as read from the config file, overridden with
            environment values, overridden with immediate keyword arguments.

            A set of keys guaranteed to be present: ``client``,
            ``endpoint``, ``token``, ``solver``, ``proxy``.

            Example::

                {
                    'client': 'qpu'
                    'endpoint': 'https://cloud.dwavesys.com/sapi',
                    'token': '123',
                    'solver': None,
                    'proxy': None
                }

    Raises:
        :exc:`ValueError`:
            Invalid (non-existing) profile name.

        :exc:`~dwave.cloud.exceptions.ConfigFileReadError`:
            Config file specified or detected could not be opened or read.

        :exc:`~dwave.cloud.exceptions.ConfigFileParseError`:
            Config file parse failed.
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
        section = load_profile_from_files(
            [config_file] if config_file else None, profile)

    # override a selected subset of values via env or kwargs,
    # pass-through the rest unmodified
    section['client'] = client or os.getenv("DWAVE_API_CLIENT", section.get('client'))
    section['endpoint'] = endpoint or os.getenv("DWAVE_API_ENDPOINT", section.get('endpoint'))
    section['token'] = token or os.getenv("DWAVE_API_TOKEN", section.get('token'))
    section['solver'] = solver or os.getenv("DWAVE_API_SOLVER", section.get('solver'))
    section['proxy'] = proxy or os.getenv("DWAVE_API_PROXY", section.get('proxy'))

    return section


def legacy_load_config(profile=None, endpoint=None, token=None, solver=None,
                       proxy=None, **kwargs):
    """Load the configured URLs and token for the SAPI server.

    .. warning:: Included only for backward compatibility, please use
        :func:`load_config` instead, or the client factory
        :meth:`~dwave.cloud.client.Client.from_config`.

    This method tries to load a configuration file from ``~/.dwrc``, select a
    specified `profile` (or first if not specified), and then override
    individual keys with the values read from environment variables, and finally
    with values given explicitly through function arguments.

    The environment variables searched are:

     - ``DW_INTERNAL__HTTPLINK``
     - ``DW_INTERNAL__TOKEN``
     - ``DW_INTERNAL__HTTPPROXY``
     - ``DW_INTERNAL__SOLVER``

    The configuration file format is a modified CSV where the first comma is
    replaced with a bar character ``|``. Each line encodes a single profile.

    The columns are::

        profile_name|endpoint_url,authentication_token,proxy_url,default_solver_name

    Everything after the ``authentication_token`` is optional.

    When there are multiple connections in a file, the first one is taken to be
    the default. Any commas in the urls must be percent encoded.

    Example:

        For the configuration file ``./.dwrc``::

            profile-a|https://one.com,token-one
            profile-b|https://two.com,token-two

        Assuming the new config file ``dwave.conf`` is not found (in any of the
        standard locations, see :meth:`~dwave.cloud.config.load_config_from_files`
        and :meth:`~dwave.cloud.config.load_config`), then:

        >>> client = dwave.cloud.Client.from_config()
        # Will try to connect with the url `https://one.com` and the token `token-one`.

        >>> client = dwave.cloud.Client.from_config(profile='profile-b')
        # Will try to connect with the url `https://two.com` and the token `token-two`.

        >>> client = dwave.cloud.Client.from_config(profile='profile-b', token='new-token')
        # Will try to connect with the url `https://two.com` and the token `new-token`.

    Args:
        profile (str):
            The profile name in the legacy config file.

        endpoint (str, default=None):
            API endpoint URL. Overrides environment/config file.

        token (str, default=None):
            API authorization token. Overrides environment/config file.

        solver (str, default=None):
            Default solver to use in :meth:`~dwave.cloud.client.Client.get_solver`.
            If undefined, you'll have to explicitly specify the solver name/id
            in all calls to :meth:`~dwave.cloud.client.Client.get_solver`.

        proxy (str, default=None):
            URL for proxy to use in connections to D-Wave API. Can include
            username/password, port, scheme, etc. If undefined, client will
            connect directly to the API (unless you use a system-level proxy).

    Returns:
        A dictionary with keys: endpoint, token, solver, proxy.
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

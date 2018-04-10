import os
import configparser
import homebase

from dwave.cloud.exceptions import ConfigFileReadError, ConfigFileParseError

CONF_APP = "dwave"
CONF_AUTHOR = "dwavesystem"
CONF_FILENAME = "dwave.conf"


def detect_existing_configfile_paths():
    """Returns the list of existing config files found on disk.

    Candidates examined depend on the OS, but for Linux possible list is:
    ``dwave.conf`` in CWD, user-local ``.config/dwave/``, system-wide
    ``/etc/dwave/``. For details, see :func:`load_config_from_file`.
    """

    # system-wide has the lowest priority, `/etc/dwave/dwave.conf`
    candidates = homebase.site_config_dir_list(
        app_author=CONF_AUTHOR, app_name=CONF_APP,
        use_virtualenv=False, create=False)

    # user-local will override it, `~/.config/dwave/dwave.conf`
    candidates.append(homebase.user_config_dir(
        app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
        use_virtualenv=False, create=False))

    # highest priority (overrides all): `./dwave.conf`
    candidates.append(".")

    paths = [os.path.join(base, CONF_FILENAME) for base in candidates]
    existing_paths = [path for path in paths if os.path.exists(path)]

    return existing_paths


def get_configfile_path():
    """Returns the highest-priority existing config file from a list
    of possible candidates returned by `detect_existing_configfile_paths()`, and
    ``None`` if no candidate config file exists."""
    paths = detect_existing_configfile_paths()
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
        filenames = detect_existing_configfile_paths()

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
    :func:`load_config_from_file`.

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

        config_file (str, default=None):
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

    def _get_section(filenames, profile):
        """Load config from a list of `filenames`, returning only section
        defined with `profile`."""

        # progressively build config from a file, or a list of auto-detected files
        # raises ConfigFileReadError/ConfigFileParseError on error
        config = load_config_from_files(filenames)

        # determine profile name fallback:
        #  (1) profile key under [defaults],
        #  (2) first non-[defaults] section
        first_section = next(iter(config.sections() + [None]))
        config_defaults = config.defaults()
        default_profile = config_defaults.get('profile', first_section)

        # select profile from the config
        if profile is None:
            profile = os.getenv("DWAVE_PROFILE", default_profile)
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

    if config_file == False:
        # skip loading from file altogether
        section = {}
    elif config_file == True:
        # force auto-detection, disregarding DWAVE_CONFIG_FILE
        section = _get_section(None, profile)
    else:
        # auto-detect if not specified with arg or env
        if config_file is None:
            config_file = os.getenv("DWAVE_CONFIG_FILE")
        section = _get_section([config_file] if config_file else None, profile)

    # override a selected subset of values via env or kwargs,
    # pass-through the rest unmodified
    section['client'] = client or os.getenv("DWAVE_API_CLIENT", section.get('client'))
    section['endpoint'] = endpoint or os.getenv("DWAVE_API_ENDPOINT", section.get('endpoint'))
    section['token'] = token or os.getenv("DWAVE_API_TOKEN", section.get('token'))
    section['solver'] = solver or os.getenv("DWAVE_API_SOLVER", section.get('solver'))
    section['proxy'] = proxy or os.getenv("DWAVE_API_PROXY", section.get('proxy'))

    return section


def legacy_load_config(key=None, endpoint=None, token=None, solver=None, proxy=None):
    """Load the configured URLs and token for the SAPI server.

    .. warning:: Included only for backward compatibility, please use
        :func:`load_config` instead, or the client factory
        :meth:`~dwave.cloud.client.Client.from_config`.

    First, this method tries to read from environment variables.
    If these are not set, it tries to load a configuration file from ``~/.dwrc``.

    The environment variables searched are:

     - ``DW_INTERNAL__HTTPLINK``
     - ``DW_INTERNAL__TOKEN``
     - ``DW_INTERNAL__HTTPPROXY`` (optional)
     - ``DW_INTERNAL__SOLVER`` (optional)

    The configuration file format is a modified CSV where the first comma is
    replaced with a bar character ``|``. Each line encodes a single connection.

    The columns are::

        profile_name|endpoint_url,authentication_token,proxy_url,default_solver_name

    Everything after the ``authentication_token`` is optional.

    When there are multiple connections in a file, the first one is taken to be
    the default. Any commas in the urls are percent encoded.

    Example:

        For the configuration file ``./.dwrc``::

            profile-a|https://one.com,token-one
            profile-b|https://two.com,token-two

        Assuming the new config file ``dwave.conf`` is not found (in any of the
        standard locations, see :meth:`~dwave.cloud.config.load_config_from_file`
        and :meth:`~dwave.cloud.config.load_config`), then:

        >>> client = dwave.cloud.Client.from_config()
        # Will try to connect with the url `https://one.com` and the token `token-one`.

        >>> client = dwave.cloud.Client.from_config(profile='profile-b')
        # Will try to connect with the url `https://two.com` and the token `token-two`.

        >>> client = dwave.cloud.Client.from_config(profile='profile-b', token='new-token')
        # Will try to connect with the url `https://two.com` and the token `new-token`.

    Args:
        key (str):
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
        A tuple of SAPI info, as (url, token, proxy, default_solver_name)
    """
    # Try to load environment variables
    url = endpoint or os.environ.get('DW_INTERNAL__HTTPLINK')
    token = token or os.environ.get('DW_INTERNAL__TOKEN')
    proxy = proxy or os.environ.get('DW_INTERNAL__HTTPPROXY')
    solver = solver or os.environ.get('DW_INTERNAL__SOLVER')

    if url is not None and token is not None:
        return url, token, proxy, solver

    # Load the configuration file
    user_path = os.path.expanduser('~')
    file_path = os.path.join(user_path, '.dwrc')

    # Parse the config file
    try:
        with open(file_path, 'r') as handle:
            lines = handle.readlines()
    except (IOError, OSError):
        # Make sure python 2 and 3 raise the same error
        raise IOError("Could not load configuration from {}".format(file_path))

    # Clean whitespace and empty lines
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

    # Go through the connections and select entry matching the key
    for line in lines:
        try:
            label, data = line.split('|', 1)
            data = {index: value for index, value in enumerate(data.split(','))}

            if label == key or data[0] == key or key is None:
                return (endpoint or data[0] or None,
                        token or data[1] or None,
                        proxy or data.get(2),
                        solver or data.get(3))
        except:
            pass  # Just ignore any malformed lines
            # TODO issue a warning

    raise ValueError("No configuration for the client could be discovered.")

import os
import configparser
import homebase


CONF_APP = "dwave"
CONF_AUTHOR = "dwavesystem"
CONF_FILENAME = "dwave.conf"


def detect_configfile_path():
    """Returns the first existing file that it finds in a list of possible
    candidates, and `None` if the list was exhausted, but no candidate config
    file exists.

    For details, see :func:`load_config_from_file`.
    """
    # look for `./dwave.conf`
    candidates = ["."]

    # then for something like `~/.config/dwave/dwave.conf`
    candidates.append(homebase.user_config_dir(
        app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
        use_virtualenv=False, create=False))

    # and finally for e.g. `/etc/dwave/dwave.conf`
    candidates.extend(homebase.site_config_dir_list(
        app_author=CONF_AUTHOR, app_name=CONF_APP,
        use_virtualenv=False, create=False))

    for base in candidates:
        path = os.path.join(base, CONF_FILENAME)
        if os.path.exists(path):
            return path

    return None


def get_default_configfile_path():
    """Returns the preferred config file path: a user-local config, e.g:
    ``~/.config/dwave/dwave.conf``."""
    base = homebase.user_config_dir(
        app_author=CONF_AUTHOR, app_name=CONF_APP, roaming=False,
        use_virtualenv=False, create=False)
    path = os.path.join(base, CONF_FILENAME)
    return path


def load_config_from_file(filename=None):
    """Load D-Wave cloud client configuration from ``filename``.

    The format of the config file is the standard Windows INI-like format,
    parsable with the Python's :mod:`configparser`.

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
        filename (str, default=None):
            D-Wave cloud client configuration file location.

            If unspecified, a config file named ``dwave.conf`` is searched for in
            the current directory, then in the user-local config dir, and then
            in all system-wide config dirs. For example, on Unix, we try to load
            the config from these paths (in order) and possibly others
            (depending on your Unix flavour)::

                ./dwave.conf
                ~/.config/dwave/dwave.conf
                /usr/local/share/dwave/dwave.conf
                /usr/share/dwave/dwave.conf

            On Windows, config file should be located in:
            ``C:\\Users\\<username>\\AppData\\Local\\dwave\\client\\dwave.conf``,
            and on MacOS in: ``~/Library/Application Support/dwave/dwave.conf``.
            For details on user/system config paths see homebase_.

            .. _homebase: https://github.com/dwavesystems/homebase

    Returns:
        :obj:`~configparser.ConfigParser`:
            A :class:`dict`-like mapping of config sections (profiles) to
            mapping of per-profile keys holding values.

    Raises:
        :exc:`ValueError`:
            Config file not found, or format invalid (parsing failed).
    """
    if filename is None:
        filename = detect_configfile_path()
        if not filename:
            raise ValueError("Config filename not given, and could not be detected")

    config = configparser.ConfigParser(default_section="defaults")
    try:
        files_read = config.read(filename)
    except configparser.Error:
        files_read = []

    if not files_read:
        raise ValueError("Failed to parse the config file "\
                         "given or detected: {}".format(filename))

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


def load_profile(name, filename=None):
    """Load profile with ``name`` from config file ``filename``."""
    return load_config_from_file(filename)[name]


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
            Path to config file. Auto-detected by default.

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
        Nothing, except in unexpected circumstances.
            Cases of invalid config file, file/disk read error, invalid
            environment values, etc. are handled and result in `None` values
            for one or all of the keys in the result.

    """
    # load config file
    # lookup priority: explicitly specified, environment specified, auto-detected
    if config_file is None:
        config_file = os.getenv("DWAVE_CONFIG_FILE")
    try:
        config = load_config_from_file(config_file)
        # last-resort profile name:
        #  (1) profile key under [defaults],
        #  (2) first non-[defaults] section
        first_section = next(iter(config.sections() + [None]))
        config_defaults = config.defaults()
        default_profile = config_defaults.get('profile', first_section)
    except ValueError:
        config = {}
        config_defaults = {}
        default_profile = None

    if profile is None:
        profile = os.getenv("DWAVE_PROFILE", default_profile)
    if profile:
        try:
            section = dict(config[profile])
        except KeyError:
            raise ValueError("Profile {!r} not defined in config file".format(profile))
    else:
        # as the very last resort (unspecified profile name and
        # no profiles defined in config), try to use [defaults]
        if config_defaults:
            section = config_defaults
        else:
            section = {}

    return {
        'client': client or os.getenv("DWAVE_API_CLIENT", section.get('client')),
        'endpoint': endpoint or os.getenv("DWAVE_API_ENDPOINT", section.get('endpoint')),
        'token': token or os.getenv("DWAVE_API_TOKEN", section.get('token')),
        'solver': solver or os.getenv("DWAVE_API_SOLVER", section.get('solver')),
        'proxy': proxy or os.getenv("DWAVE_API_PROXY", section.get('proxy')),
    }


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

    raise ValueError("No configuration for the connection could be discovered.")

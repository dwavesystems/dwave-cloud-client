import os
import configparser
import homebase


def detect_configfile_path():
    """Returns the first existing file that finds in a list of possible
    candidates, and `None` if the list was exhausted, but no candidate config
    file exists.

    For details, see :func:`load_config_from_file`.
    """
    app = "dwave"
    author = "dwavesystem"
    filename = "dwave.conf"

    # look for `./dwave.conf`
    candidates = ["."]

    # then for something like `~/.config/dwave/dwave.conf`
    candidates.append(homebase.user_config_dir(
        app_author=author, app_name=app, roaming=False,
        use_virtualenv=False, create=False))

    # and finally for e.g. `/etc/dwave/dwave.conf`
    candidates.extend(homebase.site_config_dir_list(
        app_author=author, app_name=app,
        use_virtualenv=False, create=False))

    for base in candidates:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path

    return None


def load_config_from_file(filename=None):
    """Load D-Wave cloud client configuration from `filename`.

    The format of the config file is the standard Windows INI-like format,
    parsable with the Python's `configparser`.

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

            If unspecified, config file named ``dwave.conf`` is searched for in
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
        :obj:`configparser.ConfigParser`:
            A `dict`-like mapping of config sections (profiles) to mapping of
            per-profile keys holding values.

    Raises:
        :exc:`ValueError`:
            Config file not found, or format invalid (parsing failed).
    """
    if filename is None:
        filename = detect_configfile_path()
        if not filename:
            raise ValueError("Config filename not given, and could not be detected")

    config = configparser.ConfigParser(default_section="defaults")
    if not config.read(filename):
        raise ValueError("Failed to parse the config file "\
                         "given or detected: {}".format(filename))

    return config


def load_profile(name, filename=None):
    """Load profile with `name` from `filename` config file."""
    return load_config_from_file(filename)[name]


def load_config(config_file=None, profile=None,
                endpoint=None, token=None, solver=None, proxy=None):
    """Load config. Explicitly supplied values override environment values, and
    environment values override config file values.
    """
    # load config file
    # lookup priority: explicitly specified, environment specified, auto-detected
    if config_file is None:
        config_file = os.getenv("DWAVE_CONFIG_FILE")
    try:
        config = load_config_from_file(config_file)
    except ValueError:
        config = {}

    if profile is None:
        profile = os.getenv("DWAVE_PROFILE")
    if profile:
        try:
            section = dict(config[profile])
        except KeyError:
            raise ValueError("Profile {!r} not defined in config file".format(profile))
    else:
        section = {}

    return {
        'endpoint': endpoint or os.getenv("DWAVE_API_ENDPOINT", section.get('endpoint')),
        'token': token or os.getenv("DWAVE_API_TOKEN", section.get('token')),
        'solver': solver or os.getenv("DWAVE_API_SOLVER", section.get('solver')),
        'proxy': proxy or os.getenv("DWAVE_API_PROXY", section.get('proxy')),
    }


def load_configuration(key=None):
    """Load the configured URLs and token for the SAPI server.

    First, this method tries to read from environment variables.
    If these are not set, it tries to load a configuration file from ``~/.dwrc``.

    The environment variables searched are:

     - ``DW_INTERNAL__HTTPLINK``
     - ``DW_INTERNAL__TOKEN``
     - ``DW_INTERNAL__HTTPPROXY`` (optional)
     - ``DW_INTERNAL__SOLVER`` (optional)


    The configuration file is text file where each line encodes a connection as:

        {connection_name}|{sapi_url},{authentication_token},{proxy_url},{default_solver_name}

    An example configuration file::

        prod|https://qubist.dwavesys.com/sapi,prodtokenstring,,DW2X
        alpha|https://alpha.server.url/sapi,alphatokenstring,https://alpha.proxy.url,dummy-solver

    When there are multiple connections in a file, the first one is taken to be
    the default. Any commas in the urls are percent encoded.

    Args:
        key: The name or URL of the SAPI connection.

    Returns:
        A tuple of SAPI info, as (url, token, proxy, default_solver_name)
    """
    # Try to load environment variables
    url = os.environ.get('DW_INTERNAL__HTTPLINK')
    token = os.environ.get('DW_INTERNAL__TOKEN')
    proxy = os.environ.get('DW_INTERNAL__HTTPPROXY')
    solver = os.environ.get('DW_INTERNAL__SOLVER')

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
                return data[0] or None, data[1] or None, data.get(2), data.get(3)
        except:
            pass  # Just ignore any malformed lines
            # TODO issue a warning

    raise ValueError("No configuration for the connection could be discovered.")

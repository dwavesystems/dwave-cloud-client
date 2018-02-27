import os

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

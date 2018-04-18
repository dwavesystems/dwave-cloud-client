.. _configuration:

Configuration
*************

Configuration loading
---------------------

.. automodule:: dwave.cloud.config
   :members: load_config, legacy_load_config,
        load_config_from_files, load_profile_from_files,
        get_configfile_paths


Configuration examples
----------------------

Direct :class:`~dwave.cloud.client.Client` initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most basic initialization of a new :class:`~dwave.cloud.client.Client`
instance (like :class:`dwave.cloud.qpu.Client` or
:class:`dwave.cloud.sw.Client`) is via class constructor arguments. You should
specify values for at least ``endpoint`` and ``token``::

    from dwave.cloud.qpu import Client

    client = Client(endpoint='https://cloud.dwavesys.com/sapi', token='secret')

Initializing :class:`~dwave.cloud.client.Client` from explicitly given config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration parameters for ``Client`` classes can be read from a config file
(and process environment variables) via ``Client`` factory method
:func:`~dwave.cloud.client.Client.from_config`. Configuration loading is
delegated to :func:`~dwave.cloud.config.load_config`.

Assuming ``/path/to/config`` file contains::

    [prod]
    endpoint = https://cloud.dwavesys.com/sapi
    token = secret
    client = qpu
    solver = DW_2000Q_1

The code::

    from dwave.cloud import Client

    client = Client.from_config(config_file='/path/to/config')

will create a client object which will connect to D-Wave production QPU,
using :class:`dwave.cloud.qpu.Client` and ``DW_2000Q_1`` as a default solver.

Note: in case the config file specified does not exist, or the file is
unreadable (e.g. no read permission), or format is invalid,
:exc:`~dwave.cloud.exceptions.ConfigFileReadError` or
:exc:`~dwave.cloud.exceptions.ConfigFileParseError` will be raised.

Config file auto-detection
^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``config_file`` parameter to :func:`~dwave.cloud.client.Client.from_config`
factory method is not specified (or is explicitly set to ``None``), config file
location is auto-detected. Lookup order of paths examined is described in
:func:`~dwave.cloud.config.load_config_from_files`.

Assuming (on Linux) the file ``~/.config/dwave/dwave.conf`` contains::

    [prod]
    endpoint = https://cloud.dwavesys.com/sapi
    token = secret

The code can be simplified to::

    from dwave.cloud import Client

    client = Client.from_config()

Note: config file read/parse exceptions are not raised in the auto-detect case
if no suitable file is found. If a file is found, but it's unreadable or
unparseable, exception are still raised.

Defaults and profiles
^^^^^^^^^^^^^^^^^^^^^

One config file can contain multiple profiles, each defining a separate
(endpoint, token, solver, etc.) combination. Since config file conforms to a
standard Windows INI-style format, profiles are defined by sections like:
``[profile-a]`` and ``[profile-b]``.

Default values for undefined profile keys are taken from the ``[defaults]``
section.

For example, assuming ``~/.config/dwave/dwave.conf`` contains::

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

We can instantiate a client for D-Wave 2000Q QPU endpoint with::

    from dwave.cloud import Client

    client = Client.from_config(profile='dw2000')

and a client for remote software solver with::

    client = Client.from_config(profile='software')

``alpha`` profile will connect to a pre-release API endpoint via defined HTTP
proxy server.

Progressive config file override
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

System-wide configuration files are overridden (on a key-by-key,
section-by-section basis) with user-local config files, and then by current
directory config files.

For example, if ``/usr/local/share/dwave/dwave.conf`` has::

    [defaults]
    endpoint = <production>
    client = qpu

    [prod]
    token = <token>

and ``~/.config/dwave/dwave.conf`` has::

    [alpha]
    endpoint = <alpha>
    token = <token>

and ``./dwave.conf`` has::

    [alpha]
    proxy = <proxy>

then both profiles can be loaded::

    from dwave.cloud import Client

    production_client = Client.from_config(profile='prod')

    alpha_client = Client.from_config(profile='alpha')

Note: ``alpha`` profile will use alpha endpoint, but also a proxy and a QPU
client.

Environment variables and explicit argument override
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the basic ``Client`` configuration variables can be specified **via
environment variables** (see :func:`~dwave.cloud.config.load_config`). Values
from the environment will **override** values read from config file(s).

In addition, location of the very config file and profile can also be specified
within process environment.

For example, with ``/tests/test.conf`` containing::

    [defaults]
    endpoint = <production>

    [prod]
    token = <token-1>

    [test]
    token = <token-2>

and environment variables set as::

    DWAVE_CONFIG_FILE=/tests/test.conf
    DWAVE_PROFILE=test
    DWAVE_API_SOLVER=DW_2000Q_1

then the standard::

    client = Client.from_config()

will construct a client to connect with ``test`` profile from ``test.conf``
file, and in addition, to use the ``DW_2000Q_1`` solver.

However, **explicit keyword argument** values in
:func:`~dwave.cloud.client.Client.from_config` factory will override both file
and environment values::

    client = Client.from_config(token='token-3', proxy='...')

Custom variables in config files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All (unrecognized) config file keys are passed-through to the respective
``Client`` class constructor as string keyword arguments.

This makes config-parsing future-proof and enables passing of (undocumented)
variables like ``permissive_ssl``. With::

    [defaults]
    ...

    [testing]
    endpoint = <testing-endpoint>
    permissive_ssl = 1

The ``Client`` is constructed with ``Client(endpoint=..., permissive_ssl='1')``.

Managing config files with ``dwave configure``
----------------------------------------------

TODO: document ``dwave`` CLI.
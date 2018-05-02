import os

import click
from timeit import default_timer as timer

from dwave.cloud import Client
from dwave.cloud.utils import readline_input, click_info_switch
from dwave.cloud.package_info import __title__, __version__
from dwave.cloud.exceptions import (
    SolverAuthenticationError, InvalidAPIResponseError, UnsupportedSolverError,
    ConfigFileReadError, ConfigFileParseError)
from dwave.cloud.config import (
    load_profile_from_files, load_config_from_files, get_default_config,
    get_configfile_path, get_default_configfile_path,
    get_configfile_paths)


@click.group()
@click.version_option(prog_name=__title__, version=__version__)
def cli():
    """D-Wave Cloud Client interactive configuration tool."""


@cli.group()
def config():
    """Create, update or inspect cloud client configuration file(s)."""


@config.command()
@click.option('--system', is_flag=True,
              help='List paths of system-wide configuration files')
@click.option('--user', is_flag=True,
              help='List paths of user-local configuration files')
@click.option('--local', is_flag=True,
              help='List paths of local configuration files')
@click.option('--include-missing', '-m', is_flag=True,
              help='List all examined paths, not only used paths')
def ls(system, user, local, include_missing):
    """List configuration files detected (and/or examined paths)."""

    # default action is to list *all* auto-detected files
    if not (system or user or local):
        system = user = local = True

    for path in get_configfile_paths(system=system, user=user, local=local,
                                     only_existing=not include_missing):
        click.echo(path)


@config.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None,
              help='Connection profile (section) name')
def inspect(config_file, profile):
    """Inspect existing configuration/profile."""

    try:
        section = load_profile_from_files(
            [config_file] if config_file else None, profile)

        click.echo("Configuration file: {}".format(config_file if config_file else "auto-detected"))
        click.echo("Profile: {}".format(profile if profile else "auto-detected"))
        click.echo("---")
        for key, val in section.items():
            click.echo("{} = {}".format(key, val))

    except (ValueError, ConfigFileReadError, ConfigFileParseError) as e:
        click.echo(e)


@config.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=False, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None,
              help='Connection profile (section) name')
def create(config_file, profile):
    """Create and/or update cloud client configuration file."""

    # determine the config file path
    if config_file:
        click.echo("Using configuration file: {}".format(config_file))
    else:
        # path not given, try to detect; or use default, but allow user to override
        config_file = get_configfile_path()
        if config_file:
            click.echo("Found existing configuration file: {}".format(config_file))
        else:
            config_file = get_default_configfile_path()
            click.echo("Configuration file not found; the default location is: {}".format(config_file))
        config_file = readline_input("Confirm configuration file path (editable): ", config_file)

    # create config_file path
    config_base = os.path.dirname(config_file)
    if config_base and not os.path.exists(config_base):
        if click.confirm("Configuration file path does not exist. Create it?", abort=True):
            try:
                os.makedirs(config_base)
            except Exception as e:
                click.echo("Error creating configuration path: {}".format(e))
                return 1

    # try loading existing config, or use defaults
    try:
        config = load_config_from_files([config_file])
    except:
        config = get_default_config()

    # determine profile
    if profile:
        click.echo("Using profile: {}".format(profile))
    else:
        existing = config.sections()
        if existing:
            profiles = 'create new or choose from: {}'.format(', '.join(existing))
            default_profile = ''
        else:
            profiles = 'create new'
            default_profile = 'prod'
        while not profile:
            profile = readline_input("Profile (%s): " % profiles, default_profile)
            if not profile:
                click.echo("Profile name cannot be empty.")

    if not config.has_section(profile):
        config.add_section(profile)

    # fill out the profile variables
    variables = 'endpoint token client solver proxy'.split()
    prompts = ['API endpoint URL (editable): ',
               'Authentication token (editable): ',
               'Client class (qpu or sw): ',
               'Solver (can be left blank): ',
               'Proxy URL (can be left blank): ']
    for var, prompt in zip(variables, prompts):
        default_val = config.get(profile, var, fallback=None)
        val = readline_input(prompt, default_val)
        if val != default_val:
            config.set(profile, var, val)

    try:
        with open(config_file, 'w') as fp:
            config.write(fp)
    except Exception as e:
        click.echo("Error writing to configuration file: {}".format(e))
        return 2

    click.echo("Configuration saved.")
    return 0


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None,
              help='Connection profile (section) name')
def ping(config_file, profile):
    """Ping the QPU by submitting a single-qubit problem."""

    try:
        client = Client.from_config(config_file=config_file, profile=profile)
    except Exception as e:
        click.echo("Invalid configuration: {}".format(e))
        return 1
    if config_file:
        click.echo("Using configuration file: {}".format(config_file))
    if profile:
        click.echo("Using profile: {}".format(profile))
    click.echo("Using endpoint: {}".format(client.endpoint))

    t0 = timer()
    try:
        solvers = client.get_solvers()
    except SolverAuthenticationError:
        click.echo("Authentication error. Check credentials in your configuration file.")
        return 1
    except (InvalidAPIResponseError, UnsupportedSolverError):
        click.echo("Invalid or unexpected API response.")
        return 2

    try:
        solver = client.get_solver()
    except (ValueError, KeyError):
        # if not otherwise defined (ValueError), or unavailable (KeyError),
        # just use the first solver
        if solvers:
            _, solver = next(iter(solvers.items()))
        else:
            click.echo("No solvers available.")
            return 1

    t1 = timer()
    click.echo("Using solver: {}".format(solver.id))

    timing = solver.sample_ising({0: 1}, {}).timing
    t2 = timer()

    click.echo("\nWall clock time:")
    click.echo(" * Solver definition fetch: {:.3f} ms".format((t1-t0)*1000.0))
    click.echo(" * Problem submit and results fetch: {:.3f} ms".format((t2-t1)*1000.0))
    click.echo(" * Total: {:.3f} ms".format((t2-t0)*1000.0))
    click.echo("\nQPU timing:")
    for component, duration in timing.items():
        click.echo(" * {} = {} us".format(component, duration))
    return 0


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None, help='Connection profile name')
@click.option('--id', default=None, help='Solver ID/name')
def solvers(config_file, profile, id):
    """Get solver details.

    Unless solver name/id specified, fetch and display details for
    all solvers available on configured endpoint.
    """

    def trunc(s, maxlen=60):
        s = str(s)
        return s[:(maxlen-3)]+'...' if len(s) > maxlen else s

    with Client.from_config(config_file=config_file, profile=profile) as client:
        solvers = client.get_solvers().values()

        if id:
            solvers = filter(lambda s: s.id == id, solvers)
            if not solvers:
                click.echo("Solver {} not found.".format(id))
                return 1

        # ~YAML output
        for solver in solvers:
            click.echo("Solver: {}".format(solver.id))
            click.echo("  Parameters:")
            for param, desc in sorted(solver.parameters.items()):
                click.echo("    {}: {}".format(param, trunc(desc) if desc else '?'))
            solver.properties.pop('parameters', None)
            click.echo("  Properties:")
            for k,v in sorted(solver.properties.items()):
                click.echo("    {}: {}".format(k, trunc(v)))
            click.echo()

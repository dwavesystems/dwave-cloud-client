from __future__ import print_function

from functools import wraps

import click
from timeit import default_timer as timer

from dwave.cloud import Client
from dwave.cloud.utils import readline_input
from dwave.cloud.package_info import __title__, __version__
from dwave.cloud.exceptions import (
    SolverAuthenticationError, InvalidAPIResponseError, UnsupportedSolverError)
from dwave.cloud.config import (
    load_config_from_files, get_default_config,
    get_configfile_path, get_default_configfile_path,
    get_configfile_paths)


def click_info_switch(f):
    """Decorator to create Click eager info switch option, as described in docs:
    http://click.pocoo.org/6/options/#callbacks-and-eager-options.

    Takes a no-argument function and abstracts the boilerplate required by
    Click (value checking, exit on done).

    Example:

        @click.option('--my-option', is_flag=True, callback=my_option,
                    expose_value=False, is_eager=True)
        def test():
            pass

        @click_info_switch
        def my_option()
            click.echo('some info related to my switch')
    """

    @wraps(f)
    def wrapped(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        f()
        ctx.exit()
    return wrapped


@click_info_switch
def list_config_files():
    for path in get_configfile_paths():
        click.echo(path)

@click_info_switch
def list_system_config():
    for path in get_configfile_paths(user=False, local=False, only_existing=False):
        click.echo(path)

@click_info_switch
def list_user_config():
    for path in get_configfile_paths(system=False, local=False, only_existing=False):
        click.echo(path)

@click_info_switch
def list_local_config():
    for path in get_configfile_paths(system=False, user=False, only_existing=False):
        click.echo(path)


@click.group()
@click.version_option(prog_name=__title__, version=__version__)
def cli():
    """D-Wave cloud tool."""


@cli.command()
@click.option('--config-file', default=None, help='Config file path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--profile', default=None,
              help='Connection profile name (config section name)')
@click.option('--list-config-files', is_flag=True, callback=list_config_files,
              expose_value=False, is_eager=True,
              help='List paths of all config files detected on this system')
@click.option('--list-system-paths', is_flag=True, callback=list_system_config,
              expose_value=False, is_eager=True,
              help='List paths of system-wide config files examined')
@click.option('--list-user-paths', is_flag=True, callback=list_user_config,
              expose_value=False, is_eager=True,
              help='List paths of user-local config files examined')
@click.option('--list-local-paths', is_flag=True, callback=list_local_config,
              expose_value=False, is_eager=True,
              help='List paths of local config files examined')
def configure(config_file, profile):
    """Create and/or update cloud client configuration file."""

    # determine the config file path
    if config_file:
        print("Using config file:", config_file)
    else:
        # path not given, try to detect; or use default, but allow user to override
        config_file = get_configfile_path()
        if config_file:
            print("Found existing config file:", config_file)
        else:
            config_file = get_default_configfile_path()
            print("Config file not found, the default location is:", config_file)
        config_file = readline_input("Confirm config file path (editable): ", config_file)

    # try loading existing config, or use defaults
    try:
        config = load_config_from_files([config_file])
    except:
        config = get_default_config()

    # determine profile
    if profile:
        print("Using profile:", profile)
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
                print("Profile name can't be empty.")

    if not config.has_section(profile):
        config.add_section(profile)

    # fill out the profile variables
    variables = 'endpoint token client solver proxy'.split()
    prompts = ['API endpoint URL (editable): ',
               'Auth token (editable): ',
               'Client class (qpu or sw): ',
               'Solver (can be left blank): ',
               'Proxy URL (can be left blank): ']
    for var, prompt in zip(variables, prompts):
        default_val = config.get(profile, var, fallback=None)
        val = readline_input(prompt, default_val)
        if val != default_val:
            config.set(profile, var, val)

    with open(config_file, 'w') as fp:
        config.write(fp)

    print("Config saved.")
    return 0


@cli.command()
@click.option('--config-file', default=None, help='Config file path',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--profile', default=None, help='Connection profile name')
def ping(config_file, profile):
    """Ping the QPU by submitting a single-qubit problem."""

    try:
        client = Client.from_config(config_file=config_file, profile=profile)
    except Exception as e:
        print("Invalid config: {}".format(e))
        return 1
    if config_file:
        print("Using config file:", config_file)
    if profile:
        print("Using profile:", profile)
    print("Using endpoint:", client.endpoint)

    t0 = timer()
    try:
        solvers = client.get_solvers()
    except SolverAuthenticationError:
        print("Authentication error. Check credentials in your config file.")
        return 1
    except (InvalidAPIResponseError, UnsupportedSolverError):
        print("Invalid or unexpected API response.")
        return 2

    try:
        solver = client.get_solver()
    except (ValueError, KeyError):
        # if not otherwise defined (ValueError), or unavailable (KeyError),
        # just use the first solver
        if solvers:
            _, solver = next(iter(solvers.items()))
        else:
            print("No solvers available.")
            return 1

    t1 = timer()
    print("Using solver: {}".format(solver.id))

    timing = solver.sample_ising({0: 1}, {}).timing
    t2 = timer()

    print("\nWall clock time:")
    print(" * Solver definition fetch web request: {:.3f} ms".format((t1-t0)*1000.0))
    print(" * Problem submit and results fetch: {:.3f} ms".format((t2-t1)*1000.0))
    print(" * Total: {:.3f} ms".format((t2-t0)*1000.0))
    print("\nQPU timing:")
    for component, duration in timing.items():
        print(" * {} = {} us".format(component, duration))
    return 0

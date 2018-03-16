from __future__ import print_function

import click
from timeit import default_timer as timer

from dwave.cloud.qpu import Client
from dwave.cloud.utils import readline_input
from dwave.cloud.exceptions import (
    SolverAuthenticationError, InvalidAPIResponseError, UnsupportedSolverError)
from dwave.cloud.config import (
    load_config_from_file, get_default_config,
    detect_configfile_path, get_default_configfile_path)


@click.group()
def cli():
    """D-Wave cloud tool."""


@cli.command()
@click.option('--config-file', default=None, help='Config file path')
@click.option('--profile', default=None,
              help='Connection profile name (config section name)')
def configure(config_file, profile):
    """Create and/or update cloud client configuration file."""

    # determine the config file path
    if config_file:
        print("Using config file:", config_file)
    else:
        # path not given, try to detect; or use default, but allow user to override
        config_file = detect_configfile_path()
        if config_file:
            print("Found existing config file:", config_file)
        else:
            config_file = get_default_configfile_path()
            print("Config file not found, the default location is:", config_file)
        config_file = readline_input("Confirm config file path: ", config_file)

    # try loading existing config, or use defaults
    try:
        config = load_config_from_file(config_file)
    except ValueError:
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
    prompts = ['API endpoint URL: ', 'Auth token: ', 'Client class (qpu or sw): ',
               'Solver (can be left blank): ', 'Proxy URL (can be left blank): ']
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
@click.option('--config-file', default=None, help='Config file path')
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
        client.get_solver()
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

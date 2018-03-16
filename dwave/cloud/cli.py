from __future__ import print_function

import click

from dwave.cloud.qpu import Client
from dwave.cloud.utils import readline_input
from dwave.cloud.config import (
    load_config_from_file, get_default_config,
    detect_configfile_path, get_default_configfile_path)


@click.group()
def cli():
    """D-Wave cloud tool."""


@cli.command()
@click.option('--config-file', default=None, help='Config file path')
@click.option('--profile', default=None,
              help='Connection profile name (config section name).')
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
def ping():
    """Ping the QPU by submitting a single-qubit problem."""

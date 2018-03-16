from __future__ import print_function

import click


@click.group()
def cli():
    """D-Wave cloud tool."""


@cli.command()
@click.option('--config-file', default=None, help='Config file path')
@click.option('--profile', prompt='Connection/client profile name',
              help='The name of connection profile (also section name in config).')
def configure(config_file, profile):
    """Create and update cloud client configuration files."""


@cli.command()
def ping():
    """Ping the QPU by submitting a single-qubit problem."""

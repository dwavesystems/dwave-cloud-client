import os
import ast
import json
import logging
import datetime

import click
from timeit import default_timer as timer
from datetime import datetime, timedelta

from dwave.cloud import Client
from dwave.cloud.utils import (
    default_text_input, click_info_switch, generate_valid_random_problem,
    datetime_to_timestamp, utcnow, strtrunc)
from dwave.cloud.package_info import __title__, __version__
from dwave.cloud.exceptions import (
    SolverAuthenticationError, InvalidAPIResponseError, UnsupportedSolverError,
    ConfigFileReadError, ConfigFileParseError,
    RequestTimeout, PollingTimeout)
from dwave.cloud.config import (
    load_profile_from_files, load_config_from_files, get_default_config,
    get_configfile_path, get_default_configfile_path,
    get_configfile_paths)


def enable_logging(ctx, param, value):
    if value and not ctx.resilient_parsing:
        logging.getLogger('dwave.cloud').setLevel(logging.DEBUG)


@click.group()
@click.version_option(prog_name=__title__, version=__version__)
@click.option('--debug', is_flag=True, callback=enable_logging,
              help='Enable debug logging.')
def cli(debug=False):
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
        config_file = default_text_input("Configuration file path", config_file)
        config_file = os.path.expanduser(config_file)

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
        profile = default_text_input("Profile (%s)" % profiles, default_profile, optional=False)

    if not config.has_section(profile):
        config.add_section(profile)

    # fill out the profile variables
    variables = 'endpoint token client solver proxy'.split()
    prompts = ['API endpoint URL',
               'Authentication token',
               'Default client class (qpu or sw)',
               'Default solver']
    for var, prompt in zip(variables, prompts):
        default_val = config.get(profile, var, fallback=None)
        val = default_text_input(prompt, default_val)
        if val:
            val = os.path.expandvars(val)
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
@click.option('--request-timeout', default=None, type=float,
              help='Connection and read timeouts (in seconds) for all API requests')
@click.option('--polling-timeout', default=None, type=float,
              help='Problem polling timeout in seconds (time-to-solution timeout)')
@click.option('--json', 'json_output', default=False, is_flag=True,
              help='JSON output')
def ping(config_file, profile, json_output, request_timeout, polling_timeout):
    """Ping the QPU by submitting a single-qubit problem."""

    def output_error(msg, *values):
        if json_output:
            click.echo(json.dumps({"error": msg.format(*values)}))
        else:
            click.echo(msg.format(*values))

    now = utcnow()
    info = dict(datetime=now.isoformat(), timestamp=datetime_to_timestamp(now))

    def stage_info(msg, **kwargs):
        info.update(kwargs)
        if not json_output:
            click.echo(msg.format(**kwargs))

    def flush_info():
        if json_output:
            click.echo(json.dumps(info))

    config = dict(config_file=config_file, profile=profile)
    if request_timeout is not None:
        config.update(request_timeout=request_timeout)
    if polling_timeout is not None:
        config.update(polling_timeout=polling_timeout)
    try:
        client = Client.from_config(**config)
    except Exception as e:
        output_error("Invalid configuration: {!r}", e)
        return 1
    if config_file:
        stage_info("Using configuration file: {config_file}", config_file=config_file)
    if profile:
        stage_info("Using profile: {profile}", profile=profile)
    stage_info("Using endpoint: {endpoint}", endpoint=client.endpoint)

    t0 = timer()
    try:
        solvers = client.get_solvers()
    except SolverAuthenticationError:
        output_error("Authentication error. Check credentials in your configuration file.")
        return 2
    except (InvalidAPIResponseError, UnsupportedSolverError):
        output_error("Invalid or unexpected API response.")
        return 3
    except RequestTimeout:
        output_error("API connection timed out.")
        return 4
    except Exception as e:
        output_error("Unexpected error: {!r}", e)
        return 5

    try:
        solver = client.get_solver()
    except (ValueError, KeyError):
        # if not otherwise defined (ValueError), or unavailable (KeyError),
        # just use the first solver
        if solvers:
            _, solver = next(iter(solvers.items()))
        else:
            output_error("No solvers available.")
            return 6
    except RequestTimeout:
        output_error("API connection timed out.")
        return 7

    t1 = timer()
    stage_info("Using solver: {solver_id}", solver_id=solver.id)

    try:
        future = solver.sample_ising({0: 1}, {})
        timing = future.timing
    except RequestTimeout:
        output_error("API connection timed out.")
        return 8
    except PollingTimeout:
        output_error("Polling timeout exceeded.")
        return 9
    except Exception as e:
        output_error("Sampling error: {!r}", e)
        return 10
    t2 = timer()

    stage_info("\nWall clock time:")
    stage_info(" * Solver definition fetch: {wallclock_solver_definition:.3f} ms", wallclock_solver_definition=(t1-t0)*1000.0)
    stage_info(" * Problem submit and results fetch: {wallclock_sampling:.3f} ms", wallclock_sampling=(t2-t1)*1000.0)
    stage_info(" * Total: {wallclock_total:.3f} ms", wallclock_total=(t2-t0)*1000.0)
    stage_info("\nQPU timing:")
    for component, duration in timing.items():
        stage_info(" * %(name)s = {%(name)s} us" % {"name": component}, **{component: duration})

    flush_info()


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None, help='Connection profile name')
@click.option('--id', 'solver_id', default=None, help='Solver ID/name')
@click.option('--list', 'list_solvers', default=False, is_flag=True,
              help='List available solvers, one per line')
def solvers(config_file, profile, solver_id, list_solvers):
    """Get solver details.

    Unless solver name/id specified, fetch and display details for
    all solvers available on configured endpoint.
    """

    with Client.from_config(config_file=config_file, profile=profile) as client:
        solvers = client.get_solvers().values()

        if solver_id:
            solvers = filter(lambda s: s.id == solver_id, solvers)
            if not solvers:
                click.echo("Solver {} not found.".format(solver_id))
                return 1

        if list_solvers:
            for solver in solvers:
                click.echo(solver.id)
            return

        # ~YAML output
        for solver in solvers:
            click.echo("Solver: {}".format(solver.id))
            click.echo("  Parameters:")
            for param, desc in sorted(solver.parameters.items()):
                click.echo("    {}: {}".format(param, strtrunc(desc) if desc else '?'))
            solver.properties.pop('parameters', None)
            click.echo("  Properties:")
            for k,v in sorted(solver.properties.items()):
                click.echo("    {}: {}".format(k, strtrunc(v)))
            click.echo()


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None,
              help='Connection profile (section) name')
@click.option('--solver', '-s', 'solver_name', default=None,
              help='Solver name to use')
@click.option('--biases', '-h', default=None,
              help='List/dict of biases for Ising model problem formulation')
@click.option('--couplings', '-j', default=None,
              help='List/dict of couplings for Ising model problem formulation')
@click.option('--random-problem', '-r', default=False, is_flag=True,
              help='Submit a valid random problem using all qubits')
@click.option('--num-reads', '-n', default=1, type=int,
              help='Number of reads/samples')
@click.option('--verbose', '-v', default=False, is_flag=True,
              help='Increase output verbosity')
def sample(config_file, profile, solver_name, biases, couplings, random_problem,
           num_reads, verbose):
    """Submit Ising-formulated problem and return samples."""

    # TODO: de-dup wrt ping

    def echo(s, maxlen=100):
        click.echo(s if verbose else strtrunc(s, maxlen))

    try:
        client = Client.from_config(config_file=config_file, profile=profile)
    except Exception as e:
        click.echo("Invalid configuration: {}".format(e))
        return 1
    if config_file:
        echo("Using configuration file: {}".format(config_file))
    if profile:
        echo("Using profile: {}".format(profile))
    echo("Using endpoint: {}".format(client.endpoint))

    try:
        solvers = client.get_solvers()
    except SolverAuthenticationError:
        click.echo("Authentication error. Check credentials in your configuration file.")
        return 1
    except (InvalidAPIResponseError, UnsupportedSolverError):
        click.echo("Invalid or unexpected API response.")
        return 2

    if solver_name and solver_name in solvers:
        solver = solvers[solver_name]
    else:
        try:
            solver = client.get_solver()
        except (ValueError, KeyError):
            if solvers:
                _, solver = next(iter(solvers.items()))
            else:
                click.echo("No solvers available.")
                return 1

    echo("Using solver: {}".format(solver.id))

    if random_problem:
        linear, quadratic = generate_valid_random_problem(solver)
    else:
        try:
            linear = ast.literal_eval(biases) if biases else []
        except Exception as e:
            click.echo("Invalid biases: {}".format(e))
        try:
            quadratic = ast.literal_eval(couplings) if couplings else {}
        except Exception as e:
            click.echo("Invalid couplings: {}".format(e))

    echo("Using qubit biases: {!r}".format(linear))
    echo("Using qubit couplings: {!r}".format(quadratic))
    echo("Number of samples: {}".format(num_reads))

    try:
        result = solver.sample_ising(linear, quadratic, num_reads=num_reads).result()
    except Exception as e:
        click.echo(e)
        return 2

    if verbose:
        click.echo("Result: {!r}".format(result))

    echo("Samples: {!r}".format(result['samples']))
    echo("Occurrences: {!r}".format(result['occurrences']))
    echo("Energies: {!r}".format(result['energies']))

# Copyright 2017 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import ast
import json
import datetime

import click
import requests.exceptions
from timeit import default_timer as timer
from datetime import datetime, timedelta

import dwave.cloud
from dwave.cloud import Client
from dwave.cloud.utils import (
    default_text_input, click_info_switch, generate_random_ising_problem,
    datetime_to_timestamp, utcnow, strtrunc, CLIError, set_loglevel)
from dwave.cloud.coders import encode_problem_as_bq
from dwave.cloud.package_info import __title__, __version__
from dwave.cloud.exceptions import (
    SolverAuthenticationError, InvalidAPIResponseError, UnsupportedSolverError,
    ConfigFileReadError, ConfigFileParseError, SolverNotFoundError, SolverOfflineError,
    RequestTimeout, PollingTimeout)
from dwave.cloud.config import (
    load_profile_from_files, load_config_from_files, get_default_config,
    get_configfile_path, get_default_configfile_path,
    get_configfile_paths)


def enable_logging(ctx, param, value):
    if value and not ctx.resilient_parsing:
        set_loglevel(dwave.cloud.logger, param.name)

def enable_loglevel(ctx, param, value):
    if value and not ctx.resilient_parsing:
        set_loglevel(dwave.cloud.logger, value)


@click.group()
@click.version_option(prog_name=__title__, version=__version__)
@click.option('--debug', is_flag=True, callback=enable_logging,
              help='Enable debug logging.')
@click.option('--trace', is_flag=True, callback=enable_logging,
              help='Enable trace-level debug logging.')
@click.option('--log', 'loglevel', metavar='LEVEL', callback=enable_loglevel,
              help='Set custom numeric or symbolic log level.')
def cli(debug=False, trace=False, loglevel=None):
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


def _ping(config_file, profile, solver_def, request_timeout, polling_timeout, output):
    """Helper method for the ping command that uses `output()` for info output
    and raises `CLIError()` on handled errors.

    This function is invariant to output format and/or error signaling mechanism.
    """

    config = dict(config_file=config_file, profile=profile, solver=solver_def)
    if request_timeout is not None:
        config.update(request_timeout=request_timeout)
    if polling_timeout is not None:
        config.update(polling_timeout=polling_timeout)
    try:
        client = Client.from_config(**config)
    except Exception as e:
        raise CLIError("Invalid configuration: {}".format(e), code=1)
    if config_file:
        output("Using configuration file: {config_file}", config_file=config_file)
    if profile:
        output("Using profile: {profile}", profile=profile)
    output("Using endpoint: {endpoint}", endpoint=client.endpoint)

    t0 = timer()
    try:
        solver = client.get_solver()
    except SolverAuthenticationError:
        raise CLIError("Authentication error. Check credentials in your configuration file.", 2)
    except SolverNotFoundError:
        raise CLIError("Solver not available.", 6)
    except (InvalidAPIResponseError, UnsupportedSolverError):
        raise CLIError("Invalid or unexpected API response.", 3)
    except RequestTimeout:
        raise CLIError("API connection timed out.", 4)
    except requests.exceptions.SSLError as e:
        # we need to handle `ssl.SSLError` wrapped in several exceptions,
        # with differences between py2/3; greping the message is the easiest way
        if 'CERTIFICATE_VERIFY_FAILED' in str(e):
            raise CLIError(
                "Certificate verification failed. Please check that your API endpoint "
                "is correct. If you are connecting to a private or third-party D-Wave "
                "system that uses self-signed certificate(s), please see "
                "https://support.dwavesys.com/hc/en-us/community/posts/360018930954.", 5)
        raise CLIError("Unexpected SSL error while fetching solver: {!r}".format(e), 5)
    except Exception as e:
        raise CLIError("Unexpected error while fetching solver: {!r}".format(e), 5)

    if hasattr(solver, 'nodes'):
        # structured solver: use the first existing node
        problem = ({min(solver.nodes): 0}, {})
    else:
        # unstructured solver doesn't constrain problem graph
        problem = ({0: 1}, {})

    t1 = timer()
    output("Using solver: {solver_id}", solver_id=solver.id)

    try:
        future = solver.sample_ising(*problem)
        timing = future.timing
    except RequestTimeout:
        raise CLIError("API connection timed out.", 8)
    except PollingTimeout:
        raise CLIError("Polling timeout exceeded.", 9)
    except Exception as e:
        raise CLIError("Sampling error: {!r}".format(e), 10)
    output("Submitted problem ID: {problem_id}", problem_id=future.id)

    t2 = timer()
    output("\nWall clock time:")
    output(" * Solver definition fetch: {wallclock_solver_definition:.3f} ms", wallclock_solver_definition=(t1-t0)*1000.0)
    output(" * Problem submit and results fetch: {wallclock_sampling:.3f} ms", wallclock_sampling=(t2-t1)*1000.0)
    output(" * Total: {wallclock_total:.3f} ms", wallclock_total=(t2-t0)*1000.0)
    if timing:
        output("\nQPU timing:")
        for component, duration in timing.items():
            output(" * %(name)s = {%(name)s} us" % {"name": component}, **{component: duration})
    else:
        output("\nQPU timing data not available.")


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None,
              help='Connection profile (section) name')
@click.option('--solver', '-s', 'solver_def', default=None, help='Feature-based solver definition')
@click.option('--request-timeout', default=None, type=float,
              help='Connection and read timeouts (in seconds) for all API requests')
@click.option('--polling-timeout', default=None, type=float,
              help='Problem polling timeout in seconds (time-to-solution timeout)')
@click.option('--json', 'json_output', default=False, is_flag=True,
              help='JSON output')
def ping(config_file, profile, solver_def, json_output, request_timeout, polling_timeout):
    """Ping the QPU by submitting a single-qubit problem."""

    now = utcnow()
    info = dict(datetime=now.isoformat(), timestamp=datetime_to_timestamp(now), code=0)

    def output(fmt, **kwargs):
        info.update(kwargs)
        if not json_output:
            click.echo(fmt.format(**kwargs))

    def flush():
        if json_output:
            click.echo(json.dumps(info))

    try:
        _ping(config_file, profile, solver_def, request_timeout, polling_timeout, output)
    except CLIError as error:
        output("Error: {error} (code: {code})", error=str(error), code=error.code)
        sys.exit(error.code)
    except Exception as error:
        output("Unhandled error: {error}", error=str(error))
        sys.exit(127)
    finally:
        flush()


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None, help='Connection profile name')
@click.option('--solver', '-s', 'solver_def', default=None, help='Feature-based solver filter')
@click.option('--list', '-l', 'list_solvers', default=False, is_flag=True,
              help='List available solvers, one per line')
def solvers(config_file, profile, solver_def, list_solvers):
    """Get solver details.

    Unless solver name/id specified, fetch and display details for
    all online solvers available on the configured endpoint.
    """

    with Client.from_config(
            config_file=config_file, profile=profile, solver=solver_def) as client:

        try:
            solvers = client.get_solvers(**client.default_solver)
        except SolverNotFoundError:
            click.echo("Solver(s) {} not found.".format(solver_def))
            return 1

        if list_solvers:
            for solver in solvers:
                click.echo(solver.id)
            return

        # ~YAML output
        for solver in solvers:
            click.echo("Solver: {}".format(solver.id))
            click.echo("  Parameters:")
            for name, val in sorted(solver.parameters.items()):
                click.echo("    {}: {}".format(name, strtrunc(val) if val else '?'))
            solver.properties.pop('parameters', None)
            click.echo("  Properties:")
            for name, val in sorted(solver.properties.items()):
                click.echo("    {}: {}".format(name, strtrunc(val)))
            click.echo("  Derived properties:")
            for name in sorted(solver.derived_properties):
                click.echo("    {}: {}".format(name, strtrunc(getattr(solver, name))))
            click.echo()


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None,
              help='Connection profile (section) name')
@click.option('--solver', '-s', 'solver_def', default=None,
              help='Feature-based solver filter')
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
def sample(config_file, profile, solver_def, biases, couplings, random_problem,
           num_reads, verbose):
    """Submit Ising-formulated problem and return samples."""

    # TODO: de-dup wrt ping

    def echo(s, maxlen=100):
        click.echo(s if verbose else strtrunc(s, maxlen))

    try:
        client = Client.from_config(
            config_file=config_file, profile=profile, solver=solver_def)
    except Exception as e:
        click.echo("Invalid configuration: {}".format(e))
        return 1
    if config_file:
        echo("Using configuration file: {}".format(config_file))
    if profile:
        echo("Using profile: {}".format(profile))
    echo("Using endpoint: {}".format(client.endpoint))

    try:
        solver = client.get_solver()
    except SolverAuthenticationError:
        click.echo("Authentication error. Check credentials in your configuration file.")
        return 1
    except (InvalidAPIResponseError, UnsupportedSolverError):
        click.echo("Invalid or unexpected API response.")
        return 2
    except SolverNotFoundError:
        click.echo("Solver with the specified features does not exist.")
        return 3

    echo("Using solver: {}".format(solver.id))

    if random_problem:
        linear, quadratic = generate_random_ising_problem(solver)
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
        result = solver.sample_ising(linear, quadratic, num_reads=num_reads)
        result.result()
    except Exception as e:
        click.echo(e)
        return 4

    if verbose:
        click.echo("Result: {!r}".format(result))

    echo("Samples: {!r}".format(result.samples))
    echo("Occurrences: {!r}".format(result.occurrences))
    echo("Energies: {!r}".format(result.energies))


@cli.command()
@click.option('--config-file', '-c', default=None,
              type=click.Path(exists=True, dir_okay=False), help='Configuration file path')
@click.option('--profile', '-p', default=None,
              help='Connection profile (section) name')
@click.option('--problem-id', '-i', default=None,
              help='Problem ID (optional)')
@click.option('--format', '-f', default='bq-zlib',
              type=click.Choice(['coo', 'bq-zlib'], case_sensitive=False),
              help='Problem data encoding')
@click.argument('input_file', metavar='FILE', type=click.File('rb'))
def upload(config_file, profile, problem_id, format, input_file):
    """Multipart problem upload with cold restart support."""

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

    click.echo(("Preparing to upload a problem from {!r} "
                "in {!r} format.").format(input_file.name, format))

    if format == 'coo':
        click.echo("Transcoding 'coo' to 'bq-zlib'.")

        try:
            import dimod
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't decode 'coo' format without dimod. "
                               "Re-install the library with 'bqm' support.")

        # note: `BQM.from_coo` doesn't support files opened in binary (yet);
        # fallback to reopen for now
        with open(input_file.name, 'rt') as fp:
            bqm = dimod.BinaryQuadraticModel.from_coo(fp)
            problem = encode_problem_as_bq(bqm, compress=True)['data']

    elif format == 'bq-zlib':
        problem = input_file

    click.echo("Uploading...")

    try:
        future = client.upload_problem_encoded(
            problem=problem, problem_id=problem_id)
        remote_problem_id = future.result()
    except Exception as e:
        click.echo(e)
        return 2

    click.echo("Upload done. Problem ID: {!r}".format(remote_problem_id))

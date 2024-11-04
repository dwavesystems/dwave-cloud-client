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
import orjson
import subprocess
from collections import abc
from configparser import ConfigParser
from datetime import datetime
from functools import wraps, partial
from timeit import default_timer as timer
from typing import Optional

import click
import requests.exceptions

from dwave.cloud import api
from dwave.cloud.client import Client
from dwave.cloud.solver import StructuredSolver, BaseUnstructuredSolver
from dwave.cloud.utils.cli import default_text_input, strtrunc, CLIError
from dwave.cloud.utils.dist import (
    get_contrib_packages, get_distribution, PackageNotFoundError, VersionNotFoundError)
from dwave.cloud.utils.http import user_agent
from dwave.cloud.utils.logging import configure_logging
from dwave.cloud.utils.qubo import generate_random_ising_problem
from dwave.cloud.utils.time import datetime_to_timestamp, utcnow, epochnow
from dwave.cloud.package_info import __title__, __version__
from dwave.cloud.exceptions import (
    SolverAuthenticationError, InvalidAPIResponseError, UnsupportedSolverError,
    ConfigFileReadError, ConfigFileParseError, SolverNotFoundError, SolverOfflineError,
    RequestTimeout, PollingTimeout)
from dwave.cloud.config import (
    load_profile_from_files, load_config_from_files, load_config, get_default_config,
    get_configfile_path, get_default_configfile_path, get_configfile_paths)
from dwave.cloud.config.models import validate_config_v1
from dwave.cloud.regions import get_regions
from dwave.cloud.auth.flows import LeapAuthFlow, OAuthError


# show defaults for all click options when printing --help
click.option = partial(click.option, show_default=True)

def enable_logging(ctx, param, value):
    if value and not ctx.resilient_parsing:
        configure_logging(level=param.name)

def enable_loglevel(ctx, param, value):
    if value and not ctx.resilient_parsing:
        configure_logging(level=value)

def show_platform(ctx, param, value):
    if value and not ctx.resilient_parsing:
        click.echo(user_agent())
        sys.exit()


def config_file_options(exists=True):
    """Decorate `fn` with `--config-file` and `--profile` options.

    Optionally skip existency check by click with `exists=False`.
    """

    def decorator(fn):
        fn = click.option(
            '--config-file', '-f', default=None, is_eager=True,
            type=click.Path(exists=exists, dir_okay=False),
            help='Configuration file path')(fn)
        fn = click.option(
            '--profile', '-p', default=None,
            help='Connection profile (section) name')(fn)
        return fn

    return decorator


def solver_options(fn):
    """Decorate `fn` with `--client` and `--solver` options."""

    fn = click.option(
        '--client', 'client_type', default=None,
        type=click.Choice(['base', 'qpu', 'sw', 'hybrid'], case_sensitive=False),
        help='Client type used [default: from config]')(fn)
    fn = click.option(
        '--solver', '-s', 'solver_def', default=None,
        help='Feature-based solver filter [default: from config]')(fn)

    return fn


def endpoint_options(fn):
    """Decorate `fn` with `--endpoint` and `--region` options."""

    fn = click.option(
        '--endpoint', default=None, metavar='URL',
        help='Solver API endpoint [default: from config]')(fn)
    # TODO: provide choice for region
    fn = click.option(
        '--region', default=None, metavar='CODE',
        help='Solver API region [default: from config]')(fn)

    return fn


def json_output(fn):
    """Decorate `fn` with `--json` option."""

    fn = click.option('--json', 'json_output', default=False, is_flag=True,
                      help='JSON output')(fn)

    return fn


def raw_output(fn):
    """Decorate `fn` with `--raw` option."""

    fn = click.option('--raw', 'raw_output', default=False, is_flag=True,
                      help='Raw output')(fn)

    return fn


@click.group()
@click.version_option(prog_name=__title__, version=__version__)
@click.option('--debug', is_flag=True, callback=enable_logging,
              help='Enable debug logging.')
@click.option('--trace', is_flag=True, callback=enable_logging,
              help='Enable trace-level debug logging.')
@click.option('--log', 'loglevel', metavar='LEVEL', callback=enable_loglevel,
              help='Set custom numeric or symbolic log level.')
@click.option('--platform', is_flag=True, is_eager=True, callback=show_platform,
              help='Show the platform tags and exit.')
def cli(debug=False, trace=False, loglevel=None, platform=False):
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
@config_file_options()
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
@config_file_options(exists=False)
@click.option('--full', 'ask_full', is_flag=True,
              help='Configure non-essential options (such as endpoint and solver).')
@click.option('--auto-token', '--auto', 'auto_token', is_flag=True, default=False,
              help='Pull token from Leap API, if "dwave auth login" has been run.')
@click.option('--project', 'project', default=None,
              help='Leap project for which token is pulled. Defaults to active project.')
def create(*, config_file, profile, ask_full, auto_token, project):
    """Create or update cloud client configuration file."""

    try:
        _config_create(config_file=config_file, profile=profile,
                       ask_full=ask_full, auto_token=auto_token, project=project)
    except CLIError as error:
        click.echo(f"Error: {error!s} (code: {error.code})")
        sys.exit(error.code)
    except Exception as error:
        click.echo(f"Unhandled error: {error!s}")
        sys.exit(127)


def _input_config_variables(config: ConfigParser,
                            profile: str,
                            prompts: dict[str, dict[str, str]]) -> ConfigParser:
    """Update config variables in place with user-provided values."""

    for var, prompt in prompts.items():
        default_val = config.get(profile, var, fallback=None)
        prompt.setdefault('default', default_val)
        val = default_text_input(**prompt)
        if val:
            val = os.path.expandvars(val)
        if val and val != default_val:
            config.set(profile, var, val)
    return config

def _load_config(config_file: str) -> ConfigParser:
    """Load from config_file, or create new with defaults."""

    try:
        return load_config_from_files([config_file])
    except:
        return get_default_config()

def _write_config(config: ConfigParser, config_file: str):
    """Write config to config_file."""

    config_base = os.path.dirname(config_file)
    if config_base and not os.path.exists(config_base):
        try:
            os.makedirs(config_base)
        except Exception as e:
            raise CLIError(f"Error creating configuration path: {e!r}", code=1)

    try:
        with open(config_file, 'w') as fp:
            config.write(fp)
    except Exception as e:
        raise CLIError(f"Error writing to configuration file: {e!r}", code=2)

def _config_create(config_file, profile, ask_full=False, auto_token=False, project=None):
    """Full/simplified dwave create flows."""

    if ask_full:
        # try using existing file/env/kwarg=cli config for metadata api access,
        # but fall back to Regions defaults
        try:
            raw_config = load_config(config_file=config_file, profile=profile)
        except:
            raw_config = {}

        regions = get_regions(config=raw_config)
        region_choices = [r.code for r in regions]
        prompts = dict(
            region=dict(prompt="Solver API region", choices=region_choices),
            endpoint=dict(prompt="Solver API endpoint URL (overwrites 'region')"),
            token=dict(prompt="Solver API token"),
            client=dict(prompt="Client class", choices='base qpu sw hybrid'.split()),
            solver=dict(prompt="Solver"))

    else:
        prompts = dict(
            token=dict(prompt="Solver API token"))

        click.echo("Using the simplified configuration flow.\n"
                   "Try 'dwave config create --full' for more options.\n")

    # resolve config file path
    ask_to_confirm_config_path = not config_file
    if not config_file:
        config_file = get_configfile_path()
        if not config_file:
            config_file = get_default_configfile_path()

    config_file_exists = os.path.exists(config_file)
    verb = "Updating existing" if config_file_exists else "Creating new"
    click.echo(f"{verb} configuration file: {config_file}")

    if ask_full and ask_to_confirm_config_path:
        config_file = default_text_input("Confirm configuration file path", config_file)
        config_file = os.path.expanduser(config_file)

    config = _load_config(config_file)
    default_section = config.default_section

    # resolve profile
    if not profile:
        existing = config.sections()
        if default_section in config:   # not included in sections
            existing.insert(0, default_section)
        if config_file_exists:
            click.echo(f"Available profiles: {', '.join(existing)}")
        default_profile = next(iter(existing))

        if ask_full:
            _note = " (select existing or create new)" if config_file_exists else ""
            profile = default_text_input(f"Profile{_note}", default_profile)
        else:
            profile = default_profile

    profile_exists = profile in config
    verb = "Updating existing" if profile_exists else "Creating new"
    click.echo(f"{verb} profile: {profile}")

    if profile != default_section and not config.has_section(profile):
        config.add_section(profile)

    # pull token from Leap API
    sapi_token = None
    if auto_token:
        try:
            project, sapi_token = _get_sapi_token_for_leap_project(
                config_file=config_file if config_file_exists else False,
                profile=profile if profile_exists else None,
                project_hint=project, output=click.echo)
            click.echo(f"Fetched SAPI token for project {project.name!r} ({project.code}) from Leap API.")
        except Exception as error:
            click.echo(f"Failed to fetch SAPI token from Leap API ({error!s}).")

    if sapi_token:
        del prompts['token']
        config.set(profile, 'token', sapi_token)

    _input_config_variables(config, profile, prompts)

    _write_config(config, config_file)

    click.echo("Configuration saved.")


def _get_client_solver(config, output=None):
    """Helper function to return an instantiated client, and solver, validating
    parameters in the process, while wrapping errors in `CLIError` and using
    `output` writer as a centralized printer.
    """
    if output is None:
        output = click.echo

    # get client
    try:
        client = Client.from_config(**config)
    except Exception as e:
        raise CLIError("Invalid configuration: {}".format(e), code=1)

    config_file = config.get('config_file')
    if config_file:
        output("Using configuration file: {config_file}", config_file=config_file)

    profile = config.get('profile')
    if profile:
        output("Using profile: {profile}", profile=profile)

    output("Using endpoint: {endpoint}", endpoint=client.config.endpoint)
    output("Using region: {region}", region=client.config.region)

    # get solver
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

    output("Using solver: {solver_id}", solver_id=solver.id)

    return (client, solver)


def _sample(solver, problem, params, output):
    """Blocking sample call with error handling and using custom printer."""

    try:
        response = solver.sample_ising(*problem, **params)
        problem_id = response.wait_id()
        output("Submitted problem ID: {problem_id}", problem_id=problem_id)
        response.result()
    except RequestTimeout:
        raise CLIError("API connection timed out.", 8)
    except PollingTimeout:
        raise CLIError("Polling timeout exceeded.", 9)
    except Exception as e:
        raise CLIError("Sampling error: {!r}".format(e), 10)

    return response


def standardized_output(fn):
    """Decorator that captures `CLIError`s from `fn` and formats output.

    The decorated function (cli command) receives `output()` for info output
    and should raise `CLIError()` (for handled errors) to output error messages.

    The function itself can be invariant to output format and/or error signaling
    mechanism.
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        # text/json output taken from callee args
        json_output = kwargs.get('json_output', False)
        raw_output = kwargs.get('raw_output', False)

        now = utcnow()
        info = dict(datetime=now.isoformat(), timestamp=datetime_to_timestamp(now), code=0)

        def output(fmt=None, maxlen=None, **params):
            raw = params.pop('raw', None)
            if raw_output:
                if raw is not None:
                    click.echo(raw)
            elif json_output:
                info.update(params)
            else:
                if fmt is not None:
                    msg = fmt.format(**params)
                else:
                    # logfmt-like output
                    msg = ' '.join(f'{k}={v}' for k,v in params.items())
                if maxlen is not None:
                    msg = strtrunc(msg, maxlen)
                click.echo(msg)

        def flush():
            if json_output and not raw_output:
                click.echo(orjson.dumps(info))

        try:
            fn(*args, output=output, **kwargs)
        except CLIError as error:
            output("Error: {error} (code: {code})", error=str(error), code=error.code)
            sys.exit(error.code)
        except OAuthError as error:
            output("Auth error: {error}", error=str(error))
            output('Re-authorize with "dwave auth login" and retry.')
            sys.exit(127)
        except Exception as error:
            output("Unhandled error: {error}", error=str(error))
            sys.exit(127)
        finally:
            flush()

    return wrapped


@cli.command()
@config_file_options()
@endpoint_options
@solver_options
@click.option('--sampling-params', '-m', default=None, help='Sampling parameters (JSON encoded)')
@click.option('--request-timeout', default=None, type=float,
              help='Connection and read timeouts (in seconds) for all API requests')
@click.option('--polling-timeout', default=None, type=float,
              help='Problem polling timeout in seconds (time-to-solution timeout)')
@click.option('--label', default='dwave ping', type=str, help='Problem label')
@json_output
@standardized_output
def ping(*, config_file, profile, endpoint, region, client_type, solver_def,
         sampling_params, request_timeout, polling_timeout, label, json_output,
         output):
    """Ping the QPU by submitting a single-qubit problem."""

    # parse params (TODO: move to click validator)
    params = {}
    if sampling_params is not None:
        try:
            params = orjson.loads(sampling_params)
            assert isinstance(params, dict)
        except:
            raise CLIError("sampling parameters required as JSON-encoded "
                           "map of param names to values", code=99)

    if label:
        params.update(label=label)

    # gently prefer the least busy QPU
    def order_by(solver):
        category = solver.properties.get('category')
        avg_load = solver.properties.get('avg_load')
        return (0 if category == 'qpu' else 1, avg_load)

    config = dict(
        config_file=config_file, profile=profile,
        endpoint=endpoint, region=region,
        client=client_type, solver=solver_def,
        request_timeout=request_timeout, polling_timeout=polling_timeout,
        defaults=dict(solver=dict(order_by=order_by)))

    t0 = timer()
    client, solver = _get_client_solver(config, output)

    # generate problem
    if hasattr(solver, 'nodes'):
        # structured solver: use the first existing node
        problem = ({min(solver.nodes): 0}, {})
    else:
        # unstructured solver doesn't constrain problem graph
        problem = ({0: 1}, {})

    t1 = timer()
    response = _sample(solver, problem, params, output)

    t2 = timer()
    output("\nWall clock time:")
    output(" * Solver definition fetch: {wallclock_solver_definition:.3f} ms", wallclock_solver_definition=(t1-t0)*1000.0)
    output(" * Problem submit and results fetch: {wallclock_sampling:.3f} ms", wallclock_sampling=(t2-t1)*1000.0)
    output(" * Total: {wallclock_total:.3f} ms", wallclock_total=(t2-t0)*1000.0)
    if response.timing:
        output("\nQPU timing:")
        for component, duration in sorted(response.timing.items()):
            output(" * %(name)s = {%(name)s} us" % {"name": component}, **{component: duration})
    else:
        output("\nQPU timing data not available.")


@cli.command()
@config_file_options()
@endpoint_options
@solver_options
@click.option('--list', '-l', 'list_solvers', default=False, is_flag=True,
              help='Print filtered list of solver names, one per line')
@click.option('--all', '-a', 'list_all', default=False, is_flag=True,
              help='Ignore solver filter (list/print all solvers)')
def solvers(config_file, profile, endpoint, region, client_type, solver_def,
            list_solvers, list_all):
    """Get solver details.

    Solver filter is inherited from environment or the specified configuration
    file and profile.
    """

    if list_all:
        client_type = 'base'
        solver_def = '{}'

    # TODO: handle exceptions like in `_get_client_solver` and `standardized_output`
    with Client.from_config(
            config_file=config_file, profile=profile,
            endpoint=endpoint, region=region,
            client=client_type, solver=solver_def) as client:

        try:
            solvers = client.get_solvers(**client.config.solver)
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
@config_file_options()
@endpoint_options
@solver_options
@click.option('--biases', '-h', default=None,
              help='List/dict of biases for Ising model problem formulation')
@click.option('--couplings', '-j', default=None,
              help='List/dict of couplings for Ising model problem formulation')
@click.option('--random-problem', '-r', default=False, is_flag=True,
              help='Submit a valid random problem using all qubits on structured '
                   'solvers, or a random problem on a complete graph of size "-k" '
                   'on unstructured solvers')
@click.option('--clique-size', '--size', '-k', 'problem_size', default=3,
              help='Clique size for random problems generated for unstructured '
                   'solvers (complete graph of size K with random node/edge biases)')
@click.option('--num-reads', '-n', default=None, type=int,
              help='Number of reads/samples')
@click.option('--label', default='dwave sample', type=str, help='Problem label')
@click.option('--sampling-params', '-m', default=None,
              help='Sampling parameters, JSON encoded')
@click.option('--verbose', '-v', default=False, is_flag=True,
              help='Increase output verbosity')
@json_output
@standardized_output
def sample(*, config_file, profile, endpoint, region, client_type, solver_def,
           biases, couplings, random_problem, problem_size, num_reads, label,
           sampling_params, verbose, json_output, output):
    """Submit Ising-formulated problem and return samples."""

    # we'll limit max line len in non-verbose mode
    maxlen = None if verbose else 120

    # parse params (TODO: move to click validator)
    params = {}
    if sampling_params is not None:
        try:
            params = orjson.loads(sampling_params)
            assert isinstance(params, dict)
        except:
            raise CLIError("sampling parameters required as JSON-encoded "
                           "map of param names to values", code=99)

    if num_reads is not None:
        params.update(num_reads=num_reads)

    if label:
        params.update(label=label)

    # TODO: add other params, like timeout?
    config = dict(
        config_file=config_file, profile=profile,
        endpoint=endpoint, region=region,
        client=client_type, solver=solver_def)

    t0 = timer()
    client, solver = _get_client_solver(config, output)

    if random_problem:
        if isinstance(solver, StructuredSolver):
            linear, quadratic = generate_random_ising_problem(solver)

        elif isinstance(solver, BaseUnstructuredSolver):
            try:
                from dimod.generators import uniform
            except ImportError: # pragma: no cover
                raise RuntimeError("Can't sample from unstructured solver without dimod. "
                                   "Re-install the library with 'bqm' support.")
            linear, quadratic, _ = uniform(problem_size, 'SPIN').to_ising()

        else:
            raise CLIError(f"Unhandled solver type: {solver!r}", code=99)

    else:
        try:
            linear = ast.literal_eval(biases) if biases else {}
            if isinstance(linear, abc.Sequence):
                linear = dict(enumerate(linear))
        except Exception as e:
            raise CLIError(f"Invalid biases: {e}", code=99)
        try:
            quadratic = ast.literal_eval(couplings) if couplings else {}
        except Exception as e:
            raise CLIError(f"Invalid couplings: {e}", code=99)

    output("Using biases: {linear}", linear=list(linear.items()), maxlen=maxlen)
    output("Using couplings: {quadratic}", quadratic=list(quadratic.items()), maxlen=maxlen)
    output("Sampling parameters: {sampling_params}", sampling_params=params)

    t1 = timer()
    response = _sample(
        solver, problem=(linear, quadratic), params=params, output=output)

    t2 = timer()

    if verbose:
        output("Result: {response!r}", response=response.result())

    output("Samples: {samples!r}", samples=response.samples, maxlen=maxlen)
    output("Occurrences: {num_occurrences!r}", num_occurrences=response.num_occurrences, maxlen=maxlen)
    output("Energies: {energies!r}", energies=response.energies, maxlen=maxlen)

    output("\nWall clock time:")
    output(" * Solver definition fetch: {wallclock_solver_definition:.3f} ms", wallclock_solver_definition=(t1-t0)*1000.0)
    output(" * Problem submit and results fetch: {wallclock_sampling:.3f} ms", wallclock_sampling=(t2-t1)*1000.0)
    output(" * Total: {wallclock_total:.3f} ms", wallclock_total=(t2-t0)*1000.0)
    if response.timing:
        output("\nQPU timing:")
        for component, duration in sorted(response.timing.items()):
            output(" * %(name)s = {%(name)s} us" % {"name": component}, **{component: duration})
    else:
        output("\nQPU timing data not available.")


@cli.command()
@config_file_options()
@endpoint_options
@solver_options
@click.option('--problem-id', '-i', default=None,
              help='Problem ID (optional)')
@click.option('--format', '--fmt', default='dimodbqm',
              type=click.Choice(['coo', 'dimodbqm'], case_sensitive=False),
              help='Problem data encoding')
@click.argument('input_file', metavar='FILE', type=click.File('rb'))
def upload(config_file, profile, endpoint, region, client_type, solver_def,
           problem_id, format, input_file):
    """Multipart problem upload with cold restart support."""

    try:
        client = Client.from_config(
            config_file=config_file, profile=profile,
            endpoint=endpoint, region=region,
            client=client_type)
    except Exception as e:
        click.echo("Invalid configuration: {}".format(e))
        return 1
    if config_file:
        click.echo("Using configuration file: {}".format(config_file))
    if profile:
        click.echo("Using profile: {}".format(profile))
    click.echo("Using endpoint: {}".format(client.config.endpoint))

    click.echo(("Preparing to upload a problem from {!r} "
                "in {!r} format.").format(input_file.name, format))

    if format == 'coo':
        click.echo("Transcoding 'coo' to 'dimodbqm'.")

        try:
            import dimod
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't decode 'coo' format without dimod. "
                               "Re-install the library with 'bqm' support.")

        # note: `BQM.from_coo` doesn't support files opened in binary (yet);
        # fallback to reopen for now
        with open(input_file.name, 'rt') as fp:
            bqm = dimod.BinaryQuadraticModel.from_coo(fp)
            problem_file = bqm.to_file()

    elif format == 'dimodbqm':
        problem_file = input_file

    click.echo("Uploading...")

    try:
        future = client.upload_problem_encoded(
            problem=problem_file, problem_id=problem_id)
        remote_problem_id = future.result()
    except Exception as e:
        click.echo(e)
        return 2
    finally:
        problem_file.close()

    click.echo("Upload done. Problem ID: {!r}".format(remote_problem_id))


@cli.command()
@click.option('--list', '-l', 'list_all', default=False, is_flag=True,
              help='List available contrib (non-OSS) packages')
@click.option('--all', '-a', 'install_all', default=False, is_flag=True,
              help='Install all contrib (non-OSS) packages')
@click.option('--update', '-u', 'update_all', default=False, is_flag=True,
              help='Reinstall all installed contrib packages')
@click.option('--accept-license', '--yes', '-y', default=False, is_flag=True,
              help='Accept license(s) without prompting')
@click.option('--verbose', '-v', count=True,
              help='Increase output verbosity (additive, up to 4 times)')
@click.argument('packages', nargs=-1)
def install(list_all, install_all, update_all, accept_license, verbose, packages):
    """Install optional non-open-source Ocean packages."""

    contrib = get_contrib_packages()

    if list_all:
        if verbose:
            # ~YAML output
            for pkg, specs in contrib.items():
                click.echo("Package: {}".format(pkg))
                click.echo("  Title: {}".format(specs['title']))
                click.echo("  Description: {}".format(specs['description']))
                click.echo("  License: {}".format(specs['license']['name']))
                click.echo("  License-URL: {}".format(specs['license']['url']))
                click.echo("  Requires: {}".format(', '.join(specs['requirements'])))
                click.echo()
        else:
            # concise list of available packages
            if contrib:
                click.echo("Available packages: {}.".format(', '.join(contrib.keys())))
            else:
                click.echo("No available packages.")
        return

    if install_all:
        packages = list(contrib)

    if update_all:
        packages = list(filter(_contrib_package_maybe_installed, contrib))

    if not packages:
        click.echo('Nothing to do. Try "dwave install --help".')
        return

    # check all packages requested are registered/available
    for pkg in packages:
        if pkg not in contrib:
            click.echo("Package {!r} not found.".format(pkg))
            return 1

    for pkg in packages:
        _install_contrib_package(pkg, verbose=verbose, prompt=not accept_license)


def _contrib_package_maybe_installed(name: str) -> bool:
    """Check if contrib package `name` is installed (even partially)."""

    contrib = get_contrib_packages()
    pkg = contrib[name]

    maybe_installed = False
    for req in pkg['requirements']:
        try:
            get_distribution(req)
            maybe_installed = True
        except VersionNotFoundError:
            # dependency installed, but wrong version
            maybe_installed = True
        except PackageNotFoundError:
            # dependency not installed
            pass

    return maybe_installed


def _install_contrib_package(name, verbose=0, prompt=True):
    """pip install non-oss package `name` from dwave's pypi repo."""

    contrib = get_contrib_packages()
    dwave_contrib_repo = "https://pypi.dwavesys.com/simple"

    assert name in contrib
    pkg = contrib[name]
    title = pkg['title']

    # check if `name` package is already installed
    # right now the only way to check that is to check if all dependencies from
    # requirements are installed
    reinstall = False
    try:
        if all(get_distribution(req) for req in pkg['requirements']):
            click.echo("{} installed and up to date.\n".format(title))
            return
    except VersionNotFoundError:
        click.echo("{} dependency version mismatch.\n".format(title))
        reinstall = True
    except PackageNotFoundError:
        pass    # dependency not installed, proceed with install

    action = 'Reinstall' if reinstall else 'Install'

    # basic pkg info
    click.echo(title)
    click.echo(pkg['description'])

    # license prompt
    license = pkg['license']
    msgtpl = ("This package is available under the {name} license.\n"
              "The terms of the license are available online: {url}")
    click.echo(msgtpl.format(name=license['name'], url=license['url']))

    if prompt:
        val = default_text_input('{} (y/n)?'.format(action),
                                 default='y', optional=False)
        if val.lower() != 'y':
            click.echo('Skipping: {}.\n'.format(title))
            return

    click.echo('{}ing: {}'.format(action, title))
    for req in pkg['requirements']:

        cmd = [sys.executable, "-m", "pip", "install", req,
               "--extra-index-url", dwave_contrib_repo]
        if verbose > 1:
            cmd.append("-{}".format('v' * (verbose - 1)))

        res = subprocess.run(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        if res.returncode or verbose:
            click.echo(res.stdout)

        if res.returncode:
            click.echo('Failed to install {}.\n'.format(title))
            return

    click.echo('Successfully installed {}.\n'.format(title))


@cli.command()
@click.option('--install-all', '--all', '-a', default=False, is_flag=True,
              help='Install all non-open-source packages '\
                   'available and accept licenses without prompting')
@click.option('--auth', default=False, is_flag=True,
              help="Authorize Ocean to access Leap API on user's behalf. "
              "Implies --auto-token during 'dwave config create' and it's "
              "mutually exclusive with --full.")
@click.option('--project', 'project', default=None,
              help='Leap project for which SAPI token is retrieved. Defaults to active project.')
@click.option('--oob', default=False, is_flag=True,
              help="Same as '--auth', but using OAuth out-of-band flow. "
              "Use when 'localhost' is not available in your browser.")
@click.option('--full', 'ask_full', default=False, is_flag=True,
              help='Manually configure non-essential options such as endpoint and solver.')
@click.option('--verbose', '-v', count=True,
              help='Increase output verbosity (additive, up to 4 times)')
@standardized_output
def setup(install_all, auth, project, oob, ask_full, verbose, output):
    """Setup optional Ocean packages and configuration file(s).

    Equivalent to running `dwave install [--all]`, followed by
    an optional `dwave auth login --skip-valid [--oob]` and then by
    `dwave config create [--full] [--auto-token]`.
    """

    contrib = get_contrib_packages()
    packages = list(contrib)

    if not packages:
        install = False
    elif install_all:
        click.echo("Installing all optional non-open-source packages.\n")
        install = True
    else:
        # The default flow: SDK installed, so some contrib packages registered
        # and `dwave setup` ran without `--all` flag.
        click.echo("Optionally install non-open-source packages and "
                   "configure your environment.\n")

        # Skip the prompt if all contrib packages already installed
        if all(_contrib_package_maybe_installed(name) for name in contrib):
            click.echo("All optional packages already installed.")
            install = False
        else:
            prompt = "Do you want to select non-open-source packages to install (y/n)?"
            val = default_text_input(prompt, default='y')
            install = val.lower() == 'y'
        click.echo()

    if install:
        for pkg in packages:
            _install_contrib_package(pkg, verbose=verbose, prompt=not install_all)

    auto_token = False
    if auth or oob:
        click.echo("Authorizing Leap access.\n")
        _login(config_file=None, profile=None, oob=oob, skip_valid=True, output=output)
        click.echo()
        auto_token = True
        ask_full = False

    click.echo("Creating the D-Wave configuration file.\n")
    return _config_create(config_file=None, profile=None, ask_full=ask_full,
                          auto_token=auto_token, project=project)


@cli.group()
def auth():
    """Authorize Leap access and fetch Leap/Solver API tokens."""

@auth.command()
@config_file_options()
@click.option('--oob', is_flag=True,
              help='Run OAuth 2.0 Authorization Code flow out-of-band, '
                   'without the use of locally hosted redirect URL.')
@click.option('--skip-valid', is_flag=True,
              help='Skip authorization if access token is valid, or it '
                   'can be refreshed.')
@standardized_output
def login(*, config_file, profile, oob, skip_valid, output):
    """Authorize Ocean to access Leap API on user's behalf."""

    return _login(config_file=config_file, profile=profile, oob=oob,
                  skip_valid=skip_valid, output=output)

def _login(*, config_file, profile, oob, skip_valid, output):
    config = validate_config_v1(load_config(config_file=config_file, profile=profile))
    flow = LeapAuthFlow.from_config_model(config)

    if skip_valid:
        if flow.ensure_active_token():
            output('Valid token found, skipping authorization.')
            return

    if oob:
        flow.run_oob_flow(open_browser=True)
    else:
        flow.run_redirect_flow(open_browser=True)

    output('Authorization completed successfully. '
           'You can now use "dwave auth get" to fetch your token.')


@auth.command()
@config_file_options()
@click.argument('token_type', default='access-token',
                type=click.Choice(['access-token', 'refresh-token', 'id-token']))
@json_output
@raw_output
@standardized_output
def get(*, config_file, profile, token_type, json_output, raw_output, output):
    """Fetch Leap API token."""

    config = validate_config_v1(load_config(config_file=config_file, profile=profile))
    flow = LeapAuthFlow.from_config_model(config)

    token_key = token_type.replace('-', '_')
    if not flow.token or not token_key in flow.token:
        raise CLIError('Token not found. Please run "dwave auth login".', code=100)

    token_pretty = token_type.replace('-', ' ').capitalize()
    token_val = flow.token[token_key]
    output(f"{token_pretty}: {{%s}}" % token_key, **{token_key: token_val})
    output(raw=token_val)

    if token_type == 'access-token':
        expires_at = flow.token.get('expires_at')
        if not expires_at:
            return
        expires_at = int(expires_at)
        expired = expires_at < epochnow()

        output("Expires at: {expires_at_iso}Z (timestamp={expires_at}) ({is_valid})",
            expires_at_iso=datetime.utcfromtimestamp(expires_at).isoformat(),
            expires_at=expires_at,
            is_valid="expired" if expired else "valid")

        if expired:
            output('\nTo refresh the token, please run "dwave auth refresh".')


@auth.command()
@config_file_options()
@standardized_output
def refresh(*, config_file, profile, output):
    """Refresh Leap API access token."""

    config = validate_config_v1(load_config(config_file=config_file, profile=profile))
    flow = LeapAuthFlow.from_config_model(config)

    # check we have a token
    if not flow.token or 'refresh_token' not in flow.token:
        raise CLIError('Refresh token not found. Please run "dwave auth login".', code=100)

    # refresh
    flow.refresh_token()

    output('Access and refresh tokens successfully refreshed. '
           'You can now use "dwave auth get" to view them.')


@auth.command()
@config_file_options()
@click.argument('token_type', default='access-token',
                type=click.Choice(['access-token', 'refresh-token']))
@json_output
@standardized_output
def revoke(*, config_file, profile, token_type, json_output, output):
    """Revoke access and/or refresh token(s) for Leap API."""

    config = validate_config_v1(load_config(config_file=config_file, profile=profile))
    flow = LeapAuthFlow.from_config_model(config)

    token_key = token_type.replace('-', '_')
    token_pretty = token_type.replace('-', ' ').capitalize()
    if not flow.token or not token_key in flow.token:
        raise CLIError('Token not found. Please run "dwave auth login".', code=100)

    # revoke
    revoked = flow.revoke_token(token=flow.token[token_key], token_type_hint=token_key)

    if not revoked:
        raise CLIError(f'{token_pretty} revocation failed.', code=102)

    output(f'{token_pretty} successfully revoked.')


@cli.group()
def leap():
    """Interact with Leap API."""


@leap.group()
def project():
    """Leap projects."""


@project.command(name='ls')
@config_file_options()
@json_output
@standardized_output
def list_leap_projects(*, config_file, profile, json_output, output):
    """List available Leap projects."""

    config = validate_config_v1(load_config(config_file=config_file, profile=profile))
    flow = LeapAuthFlow.from_config_model(config)

    if not flow.token or 'access_token' not in flow.token:
        raise CLIError('Leap API access token not found. Please run "dwave auth login".', code=100)

    if flow.token_expires_soon(within=60):
        output("Access token expired (or expires soon), refreshing it.")
        flow.refresh_token()

    account = api.LeapAccount.from_config(config=flow.config,
                                          token=flow.token.get('access_token'))
    projects = account.list_projects()
    output("Leap projects:", projects=[p.model_dump() for p in projects])
    for project in projects:
        output(f" - {project.name} (code={project.code!r}, id={project.id})")


@project.command(name='token')
@config_file_options()
@click.option('--project', 'project_hint', type=str, default=None,
              help='Leap project ID, name or code. If unspecified, currently active project is used.')
@json_output
@raw_output
@standardized_output
def leap_project_token(*, config_file, profile, project_hint, json_output, raw_output, output):
    """Get Solver API token for a selected Leap project."""

    project, token = _get_sapi_token_for_leap_project(
        config_file=config_file, profile=profile,
        project_hint=project_hint, output=output)

    output(f"Solver API token for project {project.name} ({project.code}) is {token}.",
           token=token, project=project.model_dump(), raw=token)


def _get_sapi_token_for_leap_project(
        *, config_file: str, profile: str,
        project_hint: Optional[str], output: abc.Callable) -> tuple[api.models.LeapProject, str]:

    config = validate_config_v1(load_config(config_file=config_file, profile=profile))
    flow = LeapAuthFlow.from_config_model(config)

    if not flow.token or 'access_token' not in flow.token:
        raise CLIError('Leap API access token not found. '
                       'Please run "dwave auth login".', code=100)

    if flow.token_expires_soon(within=120):
        output("Access token expired (or expires soon), refreshing it.")
        flow.refresh_token()

    account = api.LeapAccount.from_config(config=flow.config,
                                          token=flow.token.get('access_token'))

    if project_hint is None:
        # use active project
        project = account.get_active_project()

    else:
        # find project using project_hint
        projects = account.list_projects()
        needle = str(project_hint).strip().lower()
        project = [p for p in projects
                   if needle in (str(p.id), p.name.lower(), p.code.lower())]

        if not project:
            raise CLIError(f'Project with {project_hint!r} ID, name or code not found. '
                           'Please run "dwave leap project ls" to list available projects.', code=101)

        project = project[0]

    # get token
    token = account.get_project_token(project=project)

    return (project, token)

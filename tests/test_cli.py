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

import contextlib
import importlib.metadata
import json
import os
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from parameterized import parameterized

from dwave.cloud.cli import cli
from dwave.cloud.config import load_config
from dwave.cloud.config.models import validate_config_v1
from dwave.cloud.testing import isolated_environ
from dwave.cloud.auth.creds import Credentials, CREDS_FILENAME
from dwave.cloud.api.client import _create_default_cache_store
from dwave.cloud.api.models import LeapProject
from dwave.cloud.api.resources import Regions

from tests import config, test_config_path, test_config_profile
from tests.test_mock_solver_loading import solver_object


def touch(path):
    """Implements UNIX `touch`."""
    with open(path, 'a'):
        os.utime(path, None)


class TestConfigCreate(unittest.TestCase):

    @parameterized.expand([
        ("simple", [], dict(token="token")),
        ("full", ["--full"], dict(region="na-west-1", endpoint="endpoint",
                                  token="token", client=None, solver=None)),
    ])
    def test_create(self, name, extra_opts, inputs):
        config_file = 'path/to/dwave.conf'
        profile = 'profile'

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'config', 'create', '--config-file', config_file, '--profile', profile
            ] + extra_opts, input=''.join(f"{'' if v is None else v}\n" for v in inputs.values()))
            self.assertEqual(result.exit_code, 0)

            # load and verify config
            with isolated_environ(remove_dwave=True):
                config = load_config(config_file, profile=profile)
                for var, val in inputs.items():
                    self.assertEqual(config.get(var), val)

    @parameterized.expand([
        ("simple", [], dict(token="token")),
        ("full", ["--full"], dict(config_file=None, profile=None, region="na-west-1",
                                  endpoint="endpoint", token="token", client=None, solver=None)),
    ])
    def test_default_flows(self, name, extra_opts, inputs):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with mock.patch("dwave.cloud.config.loaders.get_configfile_paths", lambda: ['dwave.conf']):
                with mock.patch("dwave.cloud.utils.cli.input", side_effect=inputs):
                    result = runner.invoke(cli, [
                        'config', 'create'
                    ] + extra_opts, input=''.join(f"{'' if v is None else v}\n" for v in inputs.values()))
                    self.assertEqual(result.exit_code, 0)

                # load and verify config
                with isolated_environ(remove_dwave=True):
                    config = load_config()
                    for var, val in inputs.items():
                        if val:   # skip empty default confirmations
                            self.assertEqual(config.get(var), val)

    @isolated_environ(empty=True)
    @mock.patch('dwave.cloud.cli._get_sapi_token_for_leap_project',
                return_value=(LeapProject(id=1, name='Project', code='PRJ'), 'auto-token'))
    def test_auto_create(self, mock_fetch_sapi_token):
        runner = CliRunner()
        with runner.isolated_filesystem():
            local_config_file = './dwave.conf'
            result = runner.invoke(cli, [
                'config', 'create', '--config-file', local_config_file, '--auto'
            ])
            self.assertEqual(result.exit_code, 0)

            # load and verify config
            config = load_config(config_file=local_config_file)
            self.assertEqual(config.get('token'), 'auto-token')

    @parameterized.expand([
        ("simple", [], dict(token="token")),
        ("full", ["--full"], dict(region="na-west-1", endpoint="endpoint",
                                  token="token", client="base", solver="solver")),
    ])
    def test_update(self, name, extra_opts, inputs):
        config_file = 'dwave.conf'
        profile = 'profile'

        runner = CliRunner()
        with runner.isolated_filesystem():
            # create config
            config_body = '\n'.join(f"{k} = old-{v}" for k,v in inputs.items())
            with open(config_file, 'w') as fp:
                fp.write(f"[{profile}]\n{config_body}")

            # verify config before update
            with isolated_environ(remove_dwave=True):
                config = load_config(config_file=config_file)
                for var, val in inputs.items():
                    self.assertEqual(config.get(var), f"old-{val}")

            # update config
            result = runner.invoke(cli, [
                'config', 'create', '-f', config_file, '-p', profile,
            ] + extra_opts, input='\n'.join('' if v is None else v for v in inputs.values()))
            self.assertEqual(result.exit_code, 0)

            # verify config updated
            with isolated_environ(remove_dwave=True):
                config = load_config(config_file=config_file)
                for var, val in inputs.items():
                    self.assertEqual(config.get(var), val)


class TestCli(unittest.TestCase):

    def test_config_ls(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            touch('dwave.conf')
            with mock.patch('dwave.cloud.config.loaders.homebase.site_config_dir_list',
                            lambda **kw: ['system1', 'system2']):
                with mock.patch('dwave.cloud.config.loaders.homebase.user_config_dir',
                                lambda **kw: 'user'):
                    with mock.patch('os.path.exists', lambda *x: True):
                        # test listing of all auto-detected config files
                        result = runner.invoke(cli, [
                            'config', 'ls'
                        ])
                        self.assertEqual(result.output.strip(), '\n'.join([
                            os.path.join('system1', 'dwave.conf'),
                            os.path.join('system2', 'dwave.conf'),
                            os.path.join('user', 'dwave.conf'),
                            os.path.join('.', 'dwave.conf')
                        ]))

                        # test --system
                        result = runner.invoke(cli, [
                            'config', 'ls', '--system'
                        ])
                        self.assertEqual(result.output.strip(), '\n'.join([
                            os.path.join('system1', 'dwave.conf'),
                            os.path.join('system2', 'dwave.conf')
                        ]))

                        # test --user
                        result = runner.invoke(cli, [
                            'config', 'ls', '--user'
                        ])
                        self.assertEqual(result.output.strip(),
                                         os.path.join('user', 'dwave.conf'))

                        # test --local
                        result = runner.invoke(cli, [
                            'config', 'ls', '--local'
                        ])
                        self.assertEqual(result.output.strip(),
                                         os.path.join('.', 'dwave.conf'))

                    # test --include-missing (none of the examined paths exist)
                    with mock.patch('os.path.exists', lambda *x: False):
                        # test listing of all examined paths
                        result = runner.invoke(cli, [
                            'config', 'ls', '--include-missing'
                        ])
                        self.assertEqual(result.output.strip(), '\n'.join([
                            os.path.join('system1', 'dwave.conf'),
                            os.path.join('system2', 'dwave.conf'),
                            os.path.join('user', 'dwave.conf'),
                            os.path.join('.', 'dwave.conf')
                        ]))

                        # test none exists
                        result = runner.invoke(cli, [
                            'config', 'ls', '--system', '--user', '--local'
                        ])
                        self.assertEqual(result.output.strip(), '')

                        # test --system
                        result = runner.invoke(cli, [
                            'config', 'ls', '--system', '--include-missing'
                        ])
                        self.assertEqual(result.output.strip(), '\n'.join([
                            os.path.join('system1', 'dwave.conf'),
                            os.path.join('system2', 'dwave.conf')
                        ]))

                        # test --user
                        result = runner.invoke(cli, [
                            'config', 'ls', '--user', '--include-missing'
                        ])
                        self.assertEqual(result.output.strip(),
                                         os.path.join('user', 'dwave.conf'))

                        # test --local
                        result = runner.invoke(cli, [
                            'config', 'ls', '--local', '--include-missing'
                        ])
                        self.assertEqual(result.output.strip(),
                                         os.path.join('.', 'dwave.conf'))

    def test_configure_inspect(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            config_file = 'dwave.conf'
            with open(config_file, 'w') as f:
                f.write('''
                    [defaults]
                    endpoint = 1
                    [a]
                    endpoint = 2
                    [b]
                    token = 3''')

            # test auto-detected case
            with mock.patch('dwave.cloud.config.loaders.get_configfile_paths',
                            lambda **kw: [config_file]):
                result = runner.invoke(cli, [
                    'config', 'inspect'
                ])
                self.assertIn('endpoint = 2', result.output)

            # test explicit config
            result = runner.invoke(cli, [
                'config', 'inspect', '--config-file', config_file
            ])
            self.assertIn('endpoint = 2', result.output)

            # test explicit profile
            result = runner.invoke(cli, [
                'config', 'inspect', '--config-file', config_file, '--profile', 'b'
            ])
            self.assertIn('endpoint = 1', result.output)
            self.assertIn('token = 3', result.output)

    @parameterized.expand([
        ("--config-file", ),
        ("-f", ),
    ])
    def test_ping(self, config_file_option):
        config_file = 'dwave.conf'
        profile = 'profile'
        client_type = 'qpu'
        params = dict(num_reads=10)
        label = 'label'

        with mock.patch('dwave.cloud.cli.Client') as m:
            # mock returned solver
            client = m.from_config.return_value
            client.get_solver.return_value.nodes = [5, 7, 3]

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                result = runner.invoke(cli, ['ping',
                                             config_file_option, config_file,
                                             '--profile', profile,
                                             '--client', client_type,
                                             '--sampling-params', json.dumps(params),
                                             '--request-timeout', '.5',
                                             '--polling-timeout', '30',
                                             '--label', label])

            # proper arguments passed to Client.from_config?
            expected_config = dict(
                config_file=config_file, profile=profile,
                endpoint=None, region=None,
                client=client_type, solver=None,
                request_timeout=0.5, polling_timeout=30)
            call = m.from_config.call_args.kwargs
            self.assertTrue(callable(call.pop('defaults', {}).get('solver', {}).get('order_by')))
            self.assertEqual(call, expected_config)

            # get solver called?
            client.get_solver.assert_called_with()

            # sampling method called on solver with correct params?
            solver = client.get_solver.return_value
            solver.sample_ising.assert_called_with(
                {3: 0}, {}, label=label, **params)

            # verify output contains timing data
            self.assertIn('Wall clock time', result.output)

        self.assertEqual(result.exit_code, 0)

    @parameterized.expand([
        ("--config-file", ),
        ("-f", ),
    ])
    def test_sample(self, config_file_option):
        config_file = 'dwave.conf'
        profile = 'profile'
        client = 'qpu'
        solver = '{"qpu": true}'
        biases = '[0]'
        couplings = '{(0, 4): 1}'
        num_reads = '10'
        label = 'label'

        with mock.patch('dwave.cloud.cli.Client') as m:

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                result = runner.invoke(cli, ['sample',
                                             config_file_option, config_file,
                                             '--profile', profile,
                                             '--client', client,
                                             '--solver', solver,
                                             '--label', label,
                                             '-h', biases,
                                             '-j', couplings,
                                             '-n', num_reads])

            # proper arguments passed to Client.from_config?
            m.from_config.assert_called_with(
                config_file=config_file, profile=profile,
                endpoint=None, region=None,
                client=client, solver=solver)

            # get solver called?
            c = m.from_config.return_value
            c.get_solver.assert_called_with()

            # sampling method called on solver?
            s = c.get_solver.return_value
            s.sample_ising.assert_called_with(
                {0: 0}, {(0, 4): 1}, num_reads=10, label=label)

            # verify output contains timing data
            self.assertIn('Wall clock time', result.output)

        self.assertEqual(result.exit_code, 0)

    @parameterized.expand([
        ("--config-file", ),
        ("-f", ),
    ])
    def test_solvers(self, config_file_option):
        config_file = 'dwave.conf'
        profile = 'profile'
        client_type = 'base'
        solver = '{"qpu": true}'
        solvers = [solver_object('A'), solver_object('B')]

        with mock.patch('dwave.cloud.cli.Client') as m:
            # mock returned solvers
            client = m.from_config.return_value.__enter__.return_value
            client.get_solvers.return_value = solvers

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                result = runner.invoke(cli, ['solvers',
                                             config_file_option, config_file,
                                             '--profile', profile,
                                             '--client', client_type,
                                             '--solver', solver])

            # proper arguments passed to Client.from_config?
            m.from_config.assert_called_with(
                config_file=config_file, profile=profile,
                endpoint=None, region=None,
                client=client_type, solver=solver)

            # get solvers called?
            client.get_solvers.assert_called_with()

        # verify exit code and stdout printout
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.count('Solver:'), 2)
        self.assertIn('Solver: A', result.output)
        self.assertIn('Solver: B', result.output)

    @parameterized.expand([
        ("--config-file", ),
        ("-f", ),
    ])
    def test_upload(self, config_file_option):
        config_file = 'dwave.conf'
        profile = 'profile'
        client_type = 'base'
        format = 'dimodbqm'
        problem_id = 'prob:lem:id'
        filename = 'filename'

        with mock.patch('dwave.cloud.cli.Client') as m:

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                touch(filename)
                result = runner.invoke(cli, ['upload',
                                             config_file_option, config_file,
                                             '--profile', profile,
                                             '--client', client_type,
                                             '--format', format,
                                             '--problem-id', problem_id,
                                             filename])

                # proper arguments passed to Client.from_config?
                m.from_config.assert_called_with(
                    config_file=config_file, profile=profile,
                    endpoint=None, region=None,
                    client=client_type)

                # upload method called on client?
                c = m.from_config.return_value
                self.assertTrue(c.upload_problem_encoded.called)

                # verify problem_id
                args, kwargs = c.upload_problem_encoded.call_args
                self.assertEqual(kwargs.get('problem_id'), problem_id)

        self.assertEqual(result.exit_code, 0)

    def test_platform(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['--platform'])

        # verify exit code and stdout printout
        self.assertEqual(result.exit_code, 0)

        from dwave.cloud.package_info import __packagename__, __version__
        self.assertNotIn(__packagename__, result.output)
        required = ['python', 'machine', 'system', 'platform']
        for key in required:
            self.assertIn(key, result.output)


class TestAuthCli(unittest.TestCase):

    def setUp(self):
        self.env = isolated_environ(empty=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    @mock.patch('dwave.cloud.auth.flows.LeapAuthFlow.from_config_model')
    def test_login(self, flow_factory):
        flow = flow_factory.return_value

        with self.subTest('dwave auth login'):
            flow.reset_mock()
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'login'])

            flow.run_redirect_flow.assert_called_once()
            self.assertEqual(result.exit_code, 0)

        with self.subTest('dwave auth login --skip-valid'):
            flow.reset_mock()
            flow.ensure_active_token.return_value = True
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'login', '--skip-valid'])

            flow.ensure_active_token.assert_called_once()
            flow.run_redirect_flow.assert_not_called()
            self.assertEqual(result.exit_code, 0)

        with self.subTest('dwave auth login --oob'):
            flow.reset_mock()
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'login', '--oob'])

            flow.run_oob_flow.assert_called_once()
            self.assertEqual(result.exit_code, 0)

        with self.subTest('dwave auth login --config-file <file> --profile <profile>'):
            config_file = 'dwave.conf'
            profile = 'profile'
            leap_api_endpoint = 'https://mock.dwavesys.com/leap/api'
            config_lines = [f'[{profile}]', f'leap_api_endpoint = {leap_api_endpoint}']

            flow_factory.reset_mock()
            runner = CliRunner()
            with runner.isolated_filesystem():
                # create config
                with open(config_file, 'w') as fp:
                    fp.write('\n'.join(config_lines))

                result = runner.invoke(cli, ['auth', 'login',
                                             '--config-file', config_file,
                                             '--profile', profile])

            config = validate_config_v1(dict(leap_api_endpoint=leap_api_endpoint))
            flow_factory.assert_called_with(config)
            self.assertEqual(result.exit_code, 0)

    @mock.patch('dwave.cloud.auth.flows.LeapAuthFlow.from_config_model')
    def test_get(self, flow_factory):
        flow = flow_factory.return_value
        token = dict(access_token='123', refresh_token='456')
        type(flow).token = mock.PropertyMock(return_value=token)

        with self.subTest('dwave auth get access-token'):
            flow.reset_mock()
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'get', 'access-token'])

            self.assertEqual(result.exit_code, 0)
            self.assertIn(token['access_token'], result.output)

        with self.subTest('dwave auth get access-token --raw'):
            flow.reset_mock()
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'get', 'access-token', '--raw'])

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(token['access_token'], result.output.strip())

        with self.subTest('dwave auth get refresh-token'):
            flow.reset_mock()
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'get', 'refresh-token'])

            self.assertEqual(result.exit_code, 0)
            self.assertIn(token['refresh_token'], result.output)

    @mock.patch('dwave.cloud.auth.flows.LeapAuthFlow.from_config_model')
    def test_refresh(self, flow_factory):
        flow = flow_factory.return_value
        type(flow).token = mock.PropertyMock(return_value=dict(refresh_token='123'))

        with self.subTest('dwave auth refresh'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'refresh'])

            self.assertEqual(result.exit_code, 0)
            flow.refresh_token.assert_called_once()

        flow.reset_mock()
        type(flow).token = mock.PropertyMock(return_value=dict())

        with self.subTest('dwave auth refresh (no token)'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'refresh'])

            self.assertEqual(result.exit_code, 100)

    @mock.patch('dwave.cloud.auth.flows.LeapAuthFlow.from_config_model')
    def test_revoke(self, flow_factory):
        flow = flow_factory.return_value
        token = dict(access_token='123', refresh_token='456')
        type(flow).token = mock.PropertyMock(return_value=token)

        with self.subTest('dwave auth revoke (default: access-token)'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'revoke'])

            self.assertEqual(result.exit_code, 0)
            flow.revoke_token.assert_called_with(
                token=token['access_token'], token_type_hint='access_token')

        with self.subTest('dwave auth revoke access-token'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'revoke', 'access-token'])

            self.assertEqual(result.exit_code, 0)
            flow.revoke_token.assert_called_with(
                token=token['access_token'], token_type_hint='access_token')

        with self.subTest('dwave auth revoke refresh-token'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'revoke', 'refresh-token'])

            self.assertEqual(result.exit_code, 0)
            flow.revoke_token.assert_called_with(
                token=token['refresh_token'], token_type_hint='refresh_token')

    @mock.patch('dwave.cloud.auth.flows.LeapAuthFlow.from_config_model')
    def test_revoke_failure_modes(self, flow_factory):
        flow = flow_factory.return_value
        type(flow).token = mock.PropertyMock(return_value={})

        with self.subTest('dwave auth revoke: token missing'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'revoke'])

            self.assertEqual(result.exit_code, 100)

        flow.reset_mock()
        flow.revoke_token.return_value = False
        type(flow).token = mock.PropertyMock(return_value=dict(access_token='123'))

        with self.subTest('dwave auth revoke: server-side failure'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['auth', 'revoke', 'access-token'])

            self.assertEqual(result.exit_code, 102)

    @mock.patch('dwave.cloud.api.resources.LeapAccount.from_config')
    @mock.patch('dwave.cloud.auth.flows.LeapAuthFlow.from_config_model')
    def test_leap_project_ls(self, flow_factory, account_factory):
        flow = flow_factory.return_value
        account = account_factory.return_value

        type(flow).token = mock.PropertyMock(return_value=dict(access_token='123'))
        flow.token_expires_soon.return_value = False

        with self.subTest('dwave leap project ls'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, ['leap', 'project', 'ls'])

            self.assertEqual(result.exit_code, 0)
            account.list_projects.assert_called_once()

    @mock.patch('dwave.cloud.api.resources.LeapAccount.from_config')
    @mock.patch('dwave.cloud.auth.flows.LeapAuthFlow.from_config_model')
    def test_leap_project_token(self, flow_factory, account_factory):
        flow = flow_factory.return_value
        account = account_factory.return_value

        type(flow).token = mock.PropertyMock(return_value=dict(access_token='123'))
        flow.token_expires_soon.return_value = False

        projects = [
            LeapProject(id=1, name='Project A', code='A'),
            LeapProject(id=2, name='Project B', code='B'),
        ]
        project = 'b'
        token = 'token-b'

        account.list_projects.return_value = projects
        account.get_project_token.return_value = token

        with self.subTest('dwave leap project token'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, [
                    'leap', 'project', 'token', '--project', project])

            self.assertEqual(result.exit_code, 0)
            account.list_projects.assert_called_once()
            account.get_project_token.assert_called_with(project=projects[1])

        with self.subTest('dwave leap project token --raw'):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, [
                    'leap', 'project', 'token', '--project', project, '--raw'])

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.output.strip(), token)


class TestCacheCli(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        # suppress tmp dir cleanup failures on windows when files are still in use
        # note: the temp dir is cleaned up automatically when `self.tmpdir` object is gc'ed
        with contextlib.suppress(OSError):
            self._tmpdir.cleanup()

    def test_api_cache(self):
        runner = CliRunner()

        with mock.patch.multiple(
            'dwave.cloud.cli',
            get_cache_dir=lambda **kw: str(self.tmpdir),
            _get_creds_paths=lambda **kw: []
        ):
            with self.subTest('empty cache'):
                result = runner.invoke(cli, ['cache', 'purge'])
                self.assertIn("Cache purged.", result.output)

                result = runner.invoke(cli, ['cache', 'ls'])
                self.assertIn("Cache empty.", result.output)

            # populate api cache
            store = partial(_create_default_cache_store, directory=str(self.tmpdir / 'api'))
            with Regions(cache=dict(store=store)) as regions:
                regions.list_regions()

            with self.subTest('one api request cached (data+meta)'):
                result = runner.invoke(cli, ['cache', 'info'])
                self.assertIn(f"Path: {self.tmpdir}", result.output)
                self.assertIn("Items: 2", result.output)

                result = runner.invoke(cli, ['cache', 'ls'])
                self.assertIn(str(self.tmpdir), result.output)

    def test_creds_cache(self):
        creds_path = self.tmpdir / CREDS_FILENAME
        runner = CliRunner()

        with mock.patch.multiple(
            'dwave.cloud.cli',
            get_cache_dir=lambda **kw: str(self.tmpdir),
            _get_creds_paths=lambda **kw: [str(creds_path)]
        ):
            with self.subTest('empty cache'):
                result = runner.invoke(cli, ['cache', 'purge'])
                self.assertIn("Cache purged.", result.output)
                self.assertFalse(creds_path.exists())

            # populate creds cache
            Credentials(creds_file=creds_path)

            with self.subTest('creds db created'):
                result = runner.invoke(cli, ['cache', 'info'])
                self.assertIn(f"Path: {creds_path}", result.output)
                self.assertIn("Items: 0", result.output)

                result = runner.invoke(cli, ['cache', 'ls'])
                self.assertIn(str(creds_path), result.output)


@unittest.skipUnless(config, "No live server configuration available.")
class TestCliLive(unittest.TestCase):

    def test_ping(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile])
        self.assertEqual(result.exit_code, 0)

    def test_ping_json(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile,
                                     '--json'])

        res = json.loads(result.output)
        self.assertIn('timestamp', res)
        self.assertIn('datetime', res)
        self.assertIn('solver_name', res)
        self.assertIn('code', res)
        self.assertEqual(result.exit_code, 0)

    def test_ping_json_timeout_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile,
                                     '--polling-timeout', '0.00001',
                                     '--json'])

        res = json.loads(result.output)
        self.assertIn('timestamp', res)
        self.assertIn('datetime', res)
        self.assertIn('code', res)
        self.assertIn('error', res)
        self.assertEqual(result.exit_code, 9)

    @parameterized.expand([
        ("bqm", 3),
        ("cqm", 5),
        ("dqm", 5),
        ("nl", 1),
    ])
    def test_ping_unstructured_solver(self, problem_type, time_limit):
        solver = json.dumps({"supported_problem_types__contains": problem_type})
        params = json.dumps({"time_limit": time_limit})
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile,
                                     '--solver', solver,
                                     '--sampling-params', params])
        self.assertEqual(result.exit_code, 0)


class TestLogging(unittest.TestCase):

    # TODO: simplify the logic by requiring click>8.2 once we drop py39
    @classmethod
    def setUpClass(cls):
        cls.clirunner_kwargs = {}
        # in click<8.2 we need to use `mix_stderr=False` to split streams
        # (`mix_stderr` is removed in click 8.2, and split streams are now the default)
        click_version = tuple(map(int, importlib.metadata.version('click').split('.')))
        if click_version < (8, 2, 0):
            cls.clirunner_kwargs.update(mix_stderr=False)

    @isolated_environ(remove_dwave=True)
    def test_json_logs(self):
        env = {
            'DWAVE_CONFIG_FILE': test_config_path,
            'DWAVE_LOG_LEVEL': 'debug',
            'DWAVE_LOG_FORMAT': 'json',
        }

        runner = CliRunner(env=env, **self.clirunner_kwargs)
        result = runner.invoke(cli, ['ping'])

        # ping will fail because API token is undefined
        self.assertEqual(result.exit_code, 1)

        # but stderr should still contain one json log record per line
        for line in result.stderr.splitlines():
            rec = json.loads(line)
            self.assertIsInstance(rec, dict)
            self.assertIn('message', rec)
            self.assertIn('created', rec)


if __name__ == '__main__':
    unittest.main()

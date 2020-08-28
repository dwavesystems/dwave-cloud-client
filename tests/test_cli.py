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
import json
import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner

from dwave.cloud.cli import cli
from dwave.cloud.config import load_config
from dwave.cloud.testing import isolated_environ

from tests import config, test_config_path, test_config_profile
from tests.test_mock_solver_loading import solver_object


def touch(path):
    """Implements UNIX `touch`."""
    with open(path, 'a'):
        os.utime(path, None)


class TestCli(unittest.TestCase):

    def test_config_create(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        values = 'endpoint token client solver'.split()

        runner = CliRunner()
        with runner.isolated_filesystem():
            # create config file through simulated user input in `dwave configure`
            touch(config_file)
            with mock.patch("dwave.cloud.utils.input", side_effect=values):
                result = runner.invoke(cli, [
                    'config', 'create', '--config-file', config_file, '--profile', profile
                ], input='\n'.join(values))
                self.assertEqual(result.exit_code, 0)

            # load and verify config
            with isolated_environ(remove_dwave=True):
                config = load_config(config_file, profile=profile)
                for val in values:
                    self.assertEqual(config.get(val), val)

    def test_config_ls(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            touch('dwave.conf')
            with mock.patch('dwave.cloud.config.homebase.site_config_dir_list',
                            lambda **kw: ['system1', 'system2']):
                with mock.patch('dwave.cloud.config.homebase.user_config_dir',
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
            with mock.patch('dwave.cloud.config.get_configfile_paths',
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

    def test_ping(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        params = dict(num_reads=10)

        with mock.patch('dwave.cloud.cli.Client') as m:
            # mock returned solver
            client = m.from_config.return_value
            client.get_solver.return_value.nodes = [5, 7, 3]

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                result = runner.invoke(cli, ['ping',
                                             '--config-file', config_file,
                                             '--profile', profile,
                                             '--sampling-params', json.dumps(params),
                                             '--request-timeout', '.5',
                                             '--polling-timeout', '30'])

            # proper arguments passed to Client.from_config?
            m.from_config.assert_called_with(
                config_file=config_file, profile=profile, solver=None,
                request_timeout=0.5, polling_timeout=30)

            # get solver called?
            client.get_solver.assert_called_with()

            # sampling method called on solver with correct params?
            solver = client.get_solver.return_value
            solver.sample_ising.assert_called_with({3: 0}, {}, **params)

        self.assertEqual(result.exit_code, 0)

    @unittest.skipUnless(config, "No live server configuration available.")
    def test_ping_live(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile])
        self.assertEqual(result.exit_code, 0)

    @unittest.skipUnless(config, "No live server configuration available.")
    def test_ping_json_live(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile,
                                     '--json'])

        res = json.loads(result.output)
        self.assertIn('timestamp', res)
        self.assertIn('datetime', res)
        self.assertIn('solver_id', res)
        self.assertIn('code', res)
        self.assertEqual(result.exit_code, 0)

    @unittest.skipUnless(config, "No live server configuration available.")
    def test_ping_json_timeout_error_live(self):
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

    def test_sample(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        solver = '{"qpu": true}'
        biases = '[0]'
        couplings = '{(0, 4): 1}'
        num_reads = '10'

        with mock.patch('dwave.cloud.cli.Client') as m:

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                result = runner.invoke(cli, ['sample',
                                             '--config-file', config_file,
                                             '--profile', profile,
                                             '--solver', solver,
                                             '-h', biases,
                                             '-j', couplings,
                                             '-n', num_reads])

            # proper arguments passed to Client.from_config?
            m.from_config.assert_called_with(config_file=config_file, profile=profile, solver=solver)

            # get solver called?
            c = m.from_config.return_value
            c.get_solver.assert_called_with()

            # sampling method called on solver?
            s = c.get_solver.return_value
            s.sample_ising.assert_called_with([0], {(0, 4): 1}, num_reads=10)

        self.assertEqual(result.exit_code, 0)

    def test_solvers(self):
        config_file = 'dwave.conf'
        profile = 'profile'
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
                                             '--config-file', config_file,
                                             '--profile', profile,
                                             '--solver', solver])

            # proper arguments passed to Client.from_config?
            m.from_config.assert_called_with(
                config_file=config_file, profile=profile, solver=solver)

            # get solvers called?
            client.get_solvers.assert_called_with()

        # verify exit code and stdout printout
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.count('Solver:'), 2)
        self.assertIn('Solver: A', result.output)
        self.assertIn('Solver: B', result.output)

    def test_upload(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        format = 'bq-zlib'
        problem_id = 'prob:lem:id'
        filename = 'filename'

        with mock.patch('dwave.cloud.cli.Client') as m:

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                touch(filename)
                result = runner.invoke(cli, ['upload',
                                             '--config-file', config_file,
                                             '--profile', profile,
                                             '--format', format,
                                             '--problem-id', problem_id,
                                             filename])

                # proper arguments passed to Client.from_config?
                m.from_config.assert_called_with(config_file=config_file, profile=profile)

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


if __name__ == '__main__':
    unittest.main()

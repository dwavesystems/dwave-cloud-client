import unittest
import tempfile
import os

from click.testing import CliRunner

from dwave.cloud.cli import cli
from dwave.cloud.config import load_config
from dwave.cloud.testing import mock, isolated_environ

from tests import config, test_config_path, test_config_profile


def touch(path):
    """Implements UNIX `touch`."""
    with open(path, 'a'):
        os.utime(path, None)


class TestCli(unittest.TestCase):

    def test_config_create(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        values = 'endpoint token client solver proxy'.split()

        runner = CliRunner()
        with runner.isolated_filesystem():
            # create config file through simulated user input in `dwave configure`
            touch(config_file)
            with mock.patch("six.moves.input", side_effect=values, create=True):
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

        with mock.patch('dwave.cloud.cli.Client') as m:

            runner = CliRunner()
            with runner.isolated_filesystem():
                touch(config_file)
                result = runner.invoke(cli, ['ping',
                                            '--config-file', config_file,
                                            '--profile', profile])

            # proper arguments passed to Client.from_config?
            m.from_config.assert_called_with(config_file=config_file, profile=profile)

            # get solver called?
            c = m.from_config(config_file=config_file, profile=profile)
            c.get_solvers.assert_called_with()
            c.get_solver.assert_called_with()

            # sampling method called on solver?
            s = c.get_solver()
            s.sample_ising.assert_called_with({0: 1}, {})

        self.assertEqual(result.exit_code, 0)

    @unittest.skipUnless(config, "No live server configuration available.")
    def test_ping_live(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile])
        self.assertEqual(result.exit_code, 0)

    def test_sample(self):
        config_file = 'dwave.conf'
        profile = 'profile'
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
                                            '-h', biases,
                                            '-j', couplings,
                                            '-n', num_reads])

            # proper arguments passed to Client.from_config?
            m.from_config.assert_called_with(config_file=config_file, profile=profile)

            # get solver called?
            c = m.from_config(config_file=config_file, profile=profile)
            c.get_solvers.assert_called_with()
            c.get_solver.assert_called_with()

            # sampling method called on solver?
            s = c.get_solver()
            s.sample_ising.assert_called_with([0], {(0, 4): 1}, num_reads=10)

        self.assertEqual(result.exit_code, 0)


if __name__ == '__main__':
    unittest.main()

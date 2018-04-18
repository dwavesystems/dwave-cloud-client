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

    def test_configure(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        values = 'endpoint token client solver proxy'.split()

        runner = CliRunner()
        with runner.isolated_filesystem():
            # create config file through simulated user input in `dwave configure`
            touch(config_file)
            with mock.patch("six.moves.input", side_effect=values, create=True):
                result = runner.invoke(cli, [
                    'configure', '--config-file', config_file, '--profile', profile
                ], input='\n'.join(values))
                self.assertEqual(result.exit_code, 0)

            # load and verify config
            with isolated_environ(remove_dwave=True):
                config = load_config(config_file, profile=profile)
                for val in values:
                    self.assertEqual(config.get(val), val)

    def test_configure_list_config(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            touch('dwave.conf')
            with mock.patch('dwave.cloud.config.homebase.site_config_dir_list',
                            lambda **kw: ['/system1', '/system2']):
                with mock.patch('dwave.cloud.config.homebase.user_config_dir',
                                lambda **kw: '/user'):
                    with mock.patch('os.path.exists', lambda *x: True):
                        # test --list-config-files
                        result = runner.invoke(cli, [
                            'configure', '--list-config-files'
                        ])
                        self.assertEqual(result.output.strip(), '\n'.join([
                            '/system1/dwave.conf', '/system2/dwave.conf',
                            '/user/dwave.conf', './dwave.conf']))

                        # test --list-system-paths
                        result = runner.invoke(cli, [
                            'configure', '--list-system-paths'
                        ])
                        self.assertEqual(result.output.strip(), '\n'.join([
                            '/system1/dwave.conf', '/system2/dwave.conf']))

                        # test --list-user-paths
                        result = runner.invoke(cli, [
                            'configure', '--list-user-paths'
                        ])
                        self.assertEqual(result.output.strip(), '/user/dwave.conf')

                        # test --list-local-paths
                        result = runner.invoke(cli, [
                            'configure', '--list-local-paths'
                        ])
                        self.assertEqual(result.output.strip(), './dwave.conf')

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
                    'configure', '--inspect'
                ])
                self.assertIn('endpoint = 2', result.output)

            # test explicit config
            result = runner.invoke(cli, [
                'configure', '--inspect', '--config-file', config_file
            ])
            self.assertIn('endpoint = 2', result.output)

            # test explicit profile
            result = runner.invoke(cli, [
                'configure', '--inspect', '--config-file', config_file,
                '--profile', 'b'
            ])
            self.assertIn('endpoint = 1', result.output)
            self.assertIn('token = 3', result.output)

            # test eagerness of config-file ane profile
            result = runner.invoke(cli, [
                'configure', '--config-file', config_file,
                '--profile', 'b', '--inspect'
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


if __name__ == '__main__':
    unittest.main()

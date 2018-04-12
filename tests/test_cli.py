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

    @unittest.skipUnless(config, "No live server configuration available.")
    def test_ping(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['ping',
                                     '--config-file', test_config_path,
                                     '--profile', test_config_profile])
        self.assertEqual(result.exit_code, 0)


if __name__ == '__main__':
    unittest.main()

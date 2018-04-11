import unittest
import tempfile
import os

from click.testing import CliRunner

from dwave.cloud.cli import cli
from dwave.cloud.config import load_config
from dwave.cloud.testing import mock, isolated_environ


def touch(path):
    """Implements UNIX `touch`."""
    with open(path, 'a'):
        os.utime(path, None)


class TestCli(unittest.TestCase):

    def test_configure(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        values = 'endpoint token client solver proxy'.split()

        with mock.patch("six.moves.input", side_effect=values, create=True):
            runner = CliRunner()
            with runner.isolated_filesystem(), isolated_environ(remove_dwave=True):
                touch(config_file)
                result = runner.invoke(cli, [
                    'configure', '--config-file', config_file, '--profile', profile
                ], input='\n'.join(values))
                self.assertEqual(result.exit_code, 0)

                config = load_config(config_file, profile=profile)
                for val in values:
                    self.assertEqual(config.get(val), val)


if __name__ == '__main__':
    unittest.main()

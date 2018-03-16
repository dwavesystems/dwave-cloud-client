import unittest
import tempfile
import os

from click.testing import CliRunner

try:
    import unittest.mock as mock
except ImportError:
    import mock

from dwave.cloud.cli import cli
from dwave.cloud.config import load_config


class TestCli(unittest.TestCase):

    def test_configure(self):
        config_file = 'dwave.conf'
        profile = 'profile'
        values = 'endpoint token client solver proxy'.split()

        with mock.patch("six.moves.input", side_effect=values, create=True):
            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(cli, [
                    'configure', '--config-file', config_file, '--profile', profile
                ], input='\n'.join(values))
                self.assertEqual(result.exit_code, 0)

                config = load_config(config_file, profile=profile)
                for val in values:
                    self.assertEqual(config.get(val), val)


if __name__ == '__main__':
    unittest.main()

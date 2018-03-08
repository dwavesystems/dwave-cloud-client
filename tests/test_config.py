from __future__ import absolute_import, print_function

import os
import sys
import unittest
import configparser

from functools import partial

try:
    # python 3
    import unittest.mock as mock

    def iterable_mock_open(data):
        m = mock.mock_open(read_data=data)
        m.return_value.__iter__ = lambda self: self
        m.return_value.__next__ = lambda self: next(iter(self.readline, ''))
        return m

    configparser_open_namespace = "configparser.open"

except ImportError:
    # python 2
    import mock

    def iterable_mock_open(data):
        m = mock.mock_open(read_data=data)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        return m

    configparser_open_namespace = "backports.configparser.open"


from dwave.cloud.config import (
    detect_configfile_path, load_config_from_file, load_profile, load_config)


class TestConfig(unittest.TestCase):

    config_body = u"""
        [defaults]
        endpoint = https://cloud.dwavesys.com/sapi
        client = qpu
        profile = software

        [dw2000]
        solver = DW_2000Q_1
        token = ...

        [software]
        client = sw
        solver = c4-sw_sample
        token = ...

        [alpha]
        endpoint = https://url.to.alpha/api
        proxy = http://user:pass@myproxy.com:8080/
        token = alpha-token
    """

    def parse_config_string(self, text):
        config = configparser.ConfigParser(default_section="defaults")
        config.read_string(text)
        return config

    def test_config_load_from_file(self):
        with mock.patch(configparser_open_namespace, iterable_mock_open(self.config_body), create=True):
            config = load_config_from_file(filename="filename")
            self.assertEqual(config.sections(), ['dw2000', 'software', 'alpha'])
            self.assertEqual(config['dw2000']['client'], 'qpu')
            self.assertEqual(config['software']['client'], 'sw')

    def test_config_load_from_file__invalid_format__duplicate_sections(self):
        """Config loading should fail with `ValueError` on file load error."""
        myconfig = u"""
            [section]
            key = val
            [section]
            key = val
        """
        with mock.patch(configparser_open_namespace, iterable_mock_open(myconfig), create=True):
            self.assertRaises(ValueError, load_config_from_file, filename="filename")
            self.assertRaises(ValueError, load_config, config_file="filename", profile="section")

    def test_no_config_detected(self):
        with mock.patch("dwave.cloud.config.detect_configfile_path", lambda: None):
            self.assertRaises(ValueError, load_config_from_file)

    def test_invalid_filename_given(self):
        self.assertRaises(ValueError, load_config_from_file, filename='/path/to/non/existing/config')

    def test_config_load_profile(self):
        with mock.patch(configparser_open_namespace, iterable_mock_open(self.config_body), create=True):
            profile = load_profile(name="alpha", filename="filename")
            self.assertEqual(profile['token'], 'alpha-token')
            self.assertRaises(KeyError, load_profile, name="non-existing-section", filename="filename")

    def test_config_file_detection_cwd(self):
        configpath = "./dwave.conf"
        with mock.patch("os.path.exists", lambda path: path == configpath):
            self.assertEqual(detect_configfile_path(), configpath)

    def test_config_file_detection_user(self):
        if sys.platform == 'win32':
            # TODO
            pass
        elif sys.platform == 'darwin':
            configpath = os.path.expanduser("~/Library/Application Support/dwave/dwave.conf")
        else:
            configpath = os.path.expanduser("~/.config/dwave/dwave.conf")

        with mock.patch("os.path.exists", lambda path: path == configpath):
            self.assertEqual(detect_configfile_path(), configpath)

    def test_config_file_detection_system(self):
        if sys.platform == 'win32':
            # TODO
            pass
        elif sys.platform == 'darwin':
            configpath = os.path.expanduser("/Library/Application Support/dwave/dwave.conf")
        else:
            configpath = "/etc/xdg/dwave/dwave.conf"

        with mock.patch("os.path.exists", lambda path: path == configpath):
            self.assertEqual(detect_configfile_path(), configpath)

    def test_config_file_detection_nonexisting(self):
        with mock.patch("os.path.exists", lambda path: False):
            self.assertEqual(detect_configfile_path(), None)


    def _assert_config_valid(self, config):
        # profile 'alpha' is loaded
        self.assertEqual(config['endpoint'], "https://url.to.alpha/api")
        # default values are inherited
        self.assertEqual(config['client'], "qpu")

    def _load_config_from_file(self, asked, provided, data=None):
        self.assertEqual(asked, provided)
        if data is None:
            data = self.config_body
        return self.parse_config_string(data)


    def test_config_load_configfile_arg(self):
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided='myfile')):
            self._assert_config_valid(load_config(config_file='myfile', profile='alpha'))

    def test_config_load_configfile_env(self):
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided='myfile')):
            os.environ = {'DWAVE_CONFIG_FILE': 'myfile'}
            self._assert_config_valid(load_config(config_file=None, profile='alpha'))

    def test_config_load_configfile_detect(self):
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided=None)):
            self._assert_config_valid(load_config(config_file=None, profile='alpha'))

    def test_config_load_configfile_detect_profile_env(self):
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided=None)):
            os.environ = {'DWAVE_PROFILE': 'alpha'}
            self._assert_config_valid(load_config())

    def test_config_load_configfile_env_profile_env(self):
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided='myfile')):
            os.environ = {'DWAVE_CONFIG_FILE': 'myfile', 'DWAVE_PROFILE': 'alpha'}
            self._assert_config_valid(load_config())

    def test_config_load_configfile_env_profile_env_key_arg(self):
        """Explicitly provided values should override env/file."""
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided='myfile')):
            os.environ = {'DWAVE_CONFIG_FILE': 'myfile', 'DWAVE_PROFILE': 'alpha'}
            self.assertEqual(load_config(endpoint='manual')['endpoint'], 'manual')
            self.assertEqual(load_config(token='manual')['token'], 'manual')
            self.assertEqual(load_config(client='manual')['client'], 'manual')
            self.assertEqual(load_config(solver='manual')['solver'], 'manual')
            self.assertEqual(load_config(proxy='manual')['proxy'], 'manual')

    def test_config_load__profile_arg_nonexisting(self):
        """load_config should fail if the profile specified in kwargs or env in
        non-existing.
        """
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided=None)):
            self.assertRaises(ValueError, load_config, profile="nonexisting")
            with mock.patch.dict(os.environ, {'DWAVE_PROFILE': 'nonexisting'}):
                self.assertRaises(ValueError, load_config)

    def test_config_load_configfile_arg_profile_default(self):
        """Check the right profile is loaded when `profile` specified only in
        [defaults] section.
        """
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file, provided='myfile')):
            profile = load_config(config_file='myfile')
            self.assertEqual(profile['solver'], 'c4-sw_sample')

    def test_config_load_configfile_arg_profile_default_nonexisting(self):
        """load_config should fail if the profile specified in the defaults
        section is non-existing.
        """
        myconfig = u"""
            [defaults]
            profile = nonexisting

            [some]
            solver = DW_2000Q_1
        """
        with mock.patch("dwave.cloud.config.load_config_from_file",
                        partial(self._load_config_from_file,
                                provided='myfile', data=myconfig)):
            self.assertRaises(ValueError, load_config, config_file='myfile')

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
import unittest
import configparser
from unittest import mock
from functools import partial
from collections import abc
from typing import Any

from parameterized import parameterized

from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.testing import iterable_mock_open
from dwave.cloud.config import (
    get_configfile_paths, load_config_from_files, load_config,
    parse_float, parse_int, parse_boolean, get_cache_dir, update_config)
from dwave.cloud.config.loaders import (
    _solver_id_as_identity, _solver_identity_as_id)
from dwave.cloud.config.constants import DEFAULT_METADATA_API_ENDPOINT
from dwave.cloud.config.exceptions import ConfigFileParseError, ConfigFileReadError
from dwave.cloud.config.models import ClientConfig, PollingStrategy
from dwave.cloud.config.models import validate_config_v1, load_config_v1, dump_config_v1
from dwave.cloud.testing import isolated_environ


class TestConfigParsing(unittest.TestCase):

    config_body = """
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
        headers =  key-1:value-1
          key-2: value-2
    """

    def parse_config_string(self, text):
        config = configparser.ConfigParser(default_section="defaults")
        config.read_string(text)
        return config

    def test_config_load_from_file(self):
        with mock.patch('dwave.cloud.config.loaders.open', iterable_mock_open(self.config_body), create=True):
            config = load_config_from_files(filenames=["filename"])
            self.assertEqual(config.sections(), ['dw2000', 'software', 'alpha'])
            self.assertEqual(config['dw2000']['client'], 'qpu')
            self.assertEqual(config['software']['client'], 'sw')

    def setUp(self):
        # clear `config_load`-relevant environment variables before testing, so
        # we only need to patch the ones that we are currently testing
        # also, make sure the env is isolated to prevent interference with other tests
        self._env = isolated_environ(remove_dwave=True).start()

    def tearDown(self):
        self._env.stop()

    def test_config_load_from_file__invalid_format__duplicate_sections(self):
        """Config loading should fail with ``ConfigFileParseError`` for invalid
        config files."""
        myconfig = """
            [section]
            key = val
            [section]
            key = val
        """
        with mock.patch('dwave.cloud.config.loaders.open', iterable_mock_open(myconfig), create=True):
            self.assertRaises(ConfigFileParseError, load_config_from_files, filenames=["filename"])
            self.assertRaises(ConfigFileParseError, load_config, config_file="filename", profile="section")

    def test_no_config_detected(self):
        """When no config file detected, `load_config_from_files` should return
        empty config."""
        with mock.patch("dwave.cloud.config.loaders.get_configfile_paths", lambda: []):
            self.assertFalse(load_config_from_files().sections())

    def test_invalid_filename_given(self):
        self.assertRaises(ConfigFileReadError, load_config_from_files, filenames=['/path/to/non/existing/config'])

    def test_config_file_detection_cwd(self):
        configpath = os.path.join('.', 'dwave.conf')
        with mock.patch("os.path.exists", lambda path: path == configpath):
            self.assertEqual(get_configfile_paths(), [configpath])

    def test_config_file_detection_user(self):
        if sys.platform == 'win32':
            configpath = os.path.expanduser("~\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf")
        elif sys.platform == 'darwin':
            configpath = os.path.expanduser("~/Library/Application Support/dwave/dwave.conf")
        else:
            configpath = os.path.expanduser("~/.config/dwave/dwave.conf")

        with mock.patch("os.path.exists", lambda path: path == configpath):
            self.assertEqual(get_configfile_paths(), [configpath])

    def test_config_file_detection_system(self):
        if sys.platform == 'win32':
            configpath = os.path.expandvars("%SYSTEMDRIVE%\\ProgramData\\dwavesystem\\dwave\\dwave.conf")
        elif sys.platform == 'darwin':
            configpath = "/Library/Application Support/dwave/dwave.conf"
        else:
            configpath = "/etc/xdg/dwave/dwave.conf"

        with mock.patch("os.path.exists", lambda path: path == configpath):
            self.assertEqual(get_configfile_paths(), [configpath])

    def test_config_file_detection_nonexisting(self):
        with mock.patch("os.path.exists", lambda path: False):
            self.assertEqual(get_configfile_paths(), [])

    def test_config_file_path_expansion(self):
        """Home dir and env vars are expanded when resolving config path."""

        env = {"var": "val"}
        config_file = "~/path/${var}/to/$var/my.conf"
        expected_path = os.path.expanduser("~/path/val/to/val/my.conf")
        profile = "profile"

        conf_content = """
            [{}]
            valid = yes
        """.format(profile)

        def mock_open(filename, *pa, **kw):
            self.assertEqual(filename, expected_path)
            return iterable_mock_open(conf_content)()

        # config file via kwarg
        with mock.patch.dict(os.environ, env):
            with mock.patch('dwave.cloud.config.loaders.open', mock_open) as m:
                conf = load_config(config_file=config_file, profile=profile)
                self.assertEqual(conf['valid'], 'yes')

        # config file via env var
        with mock.patch.dict(os.environ, env, DWAVE_CONFIG_FILE=config_file):
            with mock.patch('dwave.cloud.config.loaders.open', mock_open) as m:
                conf = load_config(profile=profile)
                self.assertEqual(conf['valid'], 'yes')

    def _assert_config_valid(self, config):
        # profile 'alpha' is loaded
        self.assertEqual(config['endpoint'], "https://url.to.alpha/api")
        # default values are inherited
        self.assertEqual(config['client'], "qpu")
        # multi-line values are read
        self.assertEqual(config['headers'], "key-1:value-1\nkey-2: value-2")

    def _load_config_from_files(self, asked, provided=None, data=None):
        self.assertEqual(asked, provided)
        if data is None:
            data = self.config_body
        return self.parse_config_string(data)


    def test_config_load_configfile_arg(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=['myfile'])):
            self._assert_config_valid(load_config(config_file='myfile', profile='alpha'))

    def test_config_load_configfile_env(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=['myfile'])):
            with mock.patch.dict(os.environ, {'DWAVE_CONFIG_FILE': 'myfile'}):
                self._assert_config_valid(load_config(config_file=None, profile='alpha'))

    def test_config_load_configfile_detect(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=None)):
            self._assert_config_valid(load_config(config_file=None, profile='alpha'))

    def test_config_load_skip_configfiles(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        self._load_config_from_files):

            # don't load from file, use arg or env
            self.assertNotIn('endpoint', load_config(config_file=False))
            with mock.patch.dict(os.environ, {'DWAVE_API_ENDPOINT': 'test'}):
                self.assertEqual(load_config(config_file=False)['endpoint'], 'test')

            # specifying a profile doesn't affect outcome
            self.assertNotIn('endpoint', load_config(config_file=False, profile='alpha'))
            with mock.patch.dict(os.environ, {'DWAVE_API_ENDPOINT': 'test'}):
                self.assertEqual(load_config(config_file=False, profile='alpha')['endpoint'], 'test')
            with mock.patch.dict(os.environ, {'DWAVE_PROFILE': 'profile'}):
                self.assertEqual(load_config(config_file=False, endpoint='test')['endpoint'], 'test')

    def test_config_load_force_autodetection(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=None)):

            # load from file
            self._assert_config_valid(load_config(config_file=True, profile='alpha'))

            # load from file, even when config_file overridden in env (to path)
            with mock.patch.dict(os.environ, {'DWAVE_CONFIG_FILE': 'nonexisting'}):
                self._assert_config_valid(load_config(config_file=True, profile='alpha'))
                with mock.patch.dict(os.environ, {'DWAVE_PROFILE': 'alpha'}):
                    self._assert_config_valid(load_config(config_file=True))

            # load from file, even when config_file overridden in env (to None)
            with mock.patch.dict(os.environ, {'DWAVE_CONFIG_FILE': ''}):
                self._assert_config_valid(load_config(config_file=True, profile='alpha'))
                with mock.patch.dict(os.environ, {'DWAVE_PROFILE': 'alpha'}):
                    self._assert_config_valid(load_config(config_file=True))

    def test_config_load_configfile_detect_profile_env(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=None)):
            with mock.patch.dict(os.environ, {'DWAVE_PROFILE': 'alpha'}):
                self._assert_config_valid(load_config())

    def test_config_load_configfile_env_profile_env(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=['myfile'])):
            with mock.patch.dict(os.environ, {'DWAVE_CONFIG_FILE': 'myfile',
                                              'DWAVE_PROFILE': 'alpha'}):
                self._assert_config_valid(load_config())

    def test_config_load_configfile_env_profile_env_key_arg(self):
        """Explicitly provided values should override env/file."""
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=['myfile'])):
            with mock.patch.dict(os.environ, {'DWAVE_CONFIG_FILE': 'myfile',
                                              'DWAVE_PROFILE': 'alpha'}):
                self.assertEqual(load_config(endpoint='manual')['endpoint'], 'manual')
                self.assertEqual(load_config(token='manual')['token'], 'manual')
                self.assertEqual(load_config(client='manual')['client'], 'manual')
                self.assertEqual(load_config(solver='manual')['solver'], 'manual')
                self.assertEqual(load_config(proxy='manual')['proxy'], 'manual')
                self.assertEqual(load_config(headers='headers')['headers'], 'headers')

    def test_config_load__profile_arg_nonexisting(self):
        """load_config should fail if the profile specified in kwargs or env in
        non-existing.
        """
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=None)):
            self.assertRaises(ValueError, load_config, profile="nonexisting")
            with mock.patch.dict(os.environ, {'DWAVE_PROFILE': 'nonexisting'}):
                self.assertRaises(ValueError, load_config)

    def test_config_load_configfile_arg_profile_default(self):
        """Check the right profile is loaded when `profile` specified only in
        [defaults] section.
        """
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, provided=['myfile'])):
            profile = load_config(config_file='myfile')
            self.assertEqual(profile['solver'], 'c4-sw_sample')

    def test_config_load__profile_first_section(self):
        """load_config should load the first section for profile, if profile
        is nowhere else specified.
        """
        myconfig = """
            [first]
            solver = DW_2000Q_1
        """
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files,
                                provided=None, data=myconfig)):
            profile = load_config()
            self.assertIn('solver', profile)
            self.assertEqual(profile['solver'], 'DW_2000Q_1')

    def test_config_load__profile_from_defaults(self):
        """load_config should promote [defaults] section to profile, if profile
        is nowhere else specified *and* not even a single non-[defaults] section
        exists.
        """
        myconfig = """
            [defaults]
            solver = DW_2000Q_1
        """
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files,
                                provided=None, data=myconfig)):
            profile = load_config()
            self.assertIn('solver', profile)
            self.assertEqual(profile['solver'], 'DW_2000Q_1')

    def test_config_load_configfile_arg_profile_default_nonexisting(self):
        """load_config should fail if the profile specified in the defaults
        section is non-existing.
        """
        myconfig = """
            [defaults]
            profile = nonexisting

            [some]
            solver = DW_2000Q_1
        """
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files,
                                provided=['myfile'], data=myconfig)):
            self.assertRaises(ValueError, load_config, config_file='myfile')

    def test_config_load_multiple_autodetected_configfiles(self):
        """Test more specific config overrides less specific one,
        on a key by key basis, in a list of auto-detected config files."""

        config_system = """
            [alpha]
            endpoint = alpha
            solver = DW_2000Q_1
        """
        config_user = """
            [alpha]
            solver = DW_2000Q_2
            [beta]
            endpoint = beta
        """

        with mock.patch("dwave.cloud.config.loaders.get_configfile_paths",
                        lambda: ['config_system', 'config_user']):

            # test per-key override
            with mock.patch('dwave.cloud.config.loaders.open', create=True) as m:
                m.side_effect = [iterable_mock_open(config_system)(),
                                 iterable_mock_open(config_user)()]
                section = load_config(profile='alpha')
                self.assertEqual(section['endpoint'], 'alpha')
                self.assertEqual(section['solver'], 'DW_2000Q_2')

            # test per-section override (section addition)
            with mock.patch('dwave.cloud.config.loaders.open', create=True) as m:
                m.side_effect = [iterable_mock_open(config_system)(),
                                 iterable_mock_open(config_user)()]
                section = load_config(profile='beta')
                self.assertEqual(section['endpoint'], 'beta')

    def test_config_load_multiple_explicit_configfiles(self):
        """Test more specific config overrides less specific one,
        on a key by key basis, in a list of explicitly given files."""

        file1 = """
            [alpha]
            endpoint = alpha
            solver = DW_2000Q_1
        """
        file2 = """
            [alpha]
            solver = DW_2000Q_2
        """

        with mock.patch('dwave.cloud.config.loaders.open', create=True) as m:
            m.side_effect=[iterable_mock_open(file1)(),
                           iterable_mock_open(file2)()]
            section = load_config(config_file=['file1', 'file2'], profile='alpha')
            m.assert_has_calls([mock.call('file1', 'r'), mock.call('file2', 'r')])
            self.assertEqual(section['endpoint'], 'alpha')
            self.assertEqual(section['solver'], 'DW_2000Q_2')

    def test_config_load_env_override(self):
        with mock.patch("dwave.cloud.config.loaders.load_config_from_files",
                        partial(self._load_config_from_files, data="", provided=['myfile'])):

            with mock.patch.dict(os.environ, {'DWAVE_API_CLIENT': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['client'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_API_ENDPOINT': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['endpoint'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_API_TOKEN': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['token'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_API_SOLVER': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['solver'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_API_PROXY': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['proxy'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_API_HEADERS': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['headers'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_API_REGION': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['region'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_METADATA_API_ENDPOINT': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['metadata_api_endpoint'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_LEAP_API_ENDPOINT': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['leap_api_endpoint'], 'test')

            with mock.patch.dict(os.environ, {'DWAVE_LEAP_CLIENT_ID': 'test'}):
                self.assertEqual(load_config(config_file='myfile')['leap_client_id'], 'test')


class TestConfigUtils(unittest.TestCase):

    def test_parse_float(self):
        self.assertEqual(parse_float(None), None)
        self.assertEqual(parse_float(''), None)
        self.assertEqual(parse_float('', default=1), 1)

        self.assertEqual(parse_float('1.5'), 1.5)
        self.assertEqual(parse_float(1.5), 1.5)
        self.assertEqual(parse_float(1), 1.0)

    def test_parse_int(self):
        self.assertEqual(parse_int(None), None)
        self.assertEqual(parse_int(''), None)
        self.assertEqual(parse_int('', default=1), 1)

        with self.assertRaises(ValueError):
            parse_int('1.5')
        with self.assertRaises(ValueError):
            parse_int(1.5)

        self.assertEqual(parse_int(123), 123)
        self.assertEqual(parse_int(0), 0)
        self.assertEqual(parse_int(-123), -123)

    def test_parse_boolean(self):
        self.assertEqual(parse_boolean(None), None)
        self.assertEqual(parse_boolean(''), None)
        self.assertEqual(parse_boolean('', default=1), 1)

        self.assertEqual(parse_boolean(True), True)
        self.assertEqual(parse_boolean('on'), True)
        self.assertEqual(parse_boolean('On'), True)
        self.assertEqual(parse_boolean('true'), True)
        self.assertEqual(parse_boolean('True'), True)
        self.assertEqual(parse_boolean('1'), True)
        self.assertEqual(parse_boolean('123'), True)
        self.assertEqual(parse_boolean('{"a": 1}'), True)

        self.assertEqual(parse_boolean(False), False)
        self.assertEqual(parse_boolean('off'), False)
        self.assertEqual(parse_boolean('Off'), False)
        self.assertEqual(parse_boolean('false'), False)
        self.assertEqual(parse_boolean('False'), False)
        self.assertEqual(parse_boolean('0'), False)
        self.assertEqual(parse_boolean('{}'), False)

        with self.assertRaises(ValueError):
            parse_boolean('x')

    @parameterized.expand([
        (dict(name='name'), 'name'),
        (dict(name='qpu', version=dict(graph_id='123')), 'qpu;graph_id=123'),
        (dict(name='qpu', version=dict(a='a', b='b')), 'qpu;a=a;b=b'),
        (dict(name='qpu;a=b', version=dict(a=';', b='=')), 'qpu%3Ba%3Db;a=%3B;b=%3D'),
        (dict(name='|_%.:', version={";": '"'}), '%7C_%25.%3A;%3B=%22'),
    ])
    def test_solver_identity_serialization(self, identity_dict, id_string):
        # note: some overlap with SolverIdentity.to_id/.from_id tests

        with self.subTest('serialization'):
            self.assertEqual(_solver_identity_as_id(identity_dict), id_string)

        with self.subTest('deserialization'):
            self.assertEqual(_solver_id_as_identity(id_string), identity_dict)

        with self.subTest('str-dict-str'):
            self.assertEqual(_solver_identity_as_id(_solver_id_as_identity(id_string)), id_string)

        with self.subTest('dict-str-dict'):
            self.assertEqual(_solver_id_as_identity(_solver_identity_as_id(identity_dict)), identity_dict)

    def test_solver_identity_edge_cases(self):
        with self.assertRaises(ValueError):
            _solver_identity_as_id({})

        with self.assertRaises(ValueError):
            _solver_id_as_identity(';')

        with self.assertRaises(ValueError):
            _solver_id_as_identity('x;')

        self.assertEqual(_solver_id_as_identity('a;b==1'), {'name': 'a', 'version': {'b': '=1'}})
        self.assertEqual(_solver_id_as_identity(' x '), {'name': 'x'})
        self.assertEqual(_solver_id_as_identity(' x%20 '), {'name': 'x '})

        self.assertEqual(_solver_identity_as_id({'name': 'name', 'version': {}}), 'name')
        self.assertEqual(_solver_identity_as_id({'name': 'name', 'version': None}), 'name')

    def test_cache_dir(self):
        path = get_cache_dir(create=True)
        self.assertTrue(os.path.isdir(path))
        self.assertIn(__packagename__, path)
        self.assertIn(__version__, path)

    def test_config_update_simple_override(self):
        config = {'token': 1}
        update = {'token': 2, 'solver': 2}
        update_config(config, update)
        self.assertEqual(config, {'token': 2, 'solver': 2})

        config = {'endpoint': 1}
        update = {'endpoint': 2}
        update_config(config, update)
        self.assertEqual(config, {'endpoint': 2})

        config = {'region': 1}
        update = {'region': 2}
        update_config(config, update)
        self.assertEqual(config, {'region': 2})

    def test_config_update_mutually_exclusive_on_different_levels(self):
        # region/endpoint on higher level override the lower level
        config = {'endpoint': 1}
        update = {'region': 2}
        update_config(config, update)
        self.assertEqual(config, {'region': 2})

        config = {'region': 1}
        update = {'endpoint': 2}
        update_config(config, update)
        self.assertEqual(config, {'endpoint': 2})

        config = {'endpoint': 1, 'region': 1}
        update = {'region': 2}
        update_config(config, update)
        self.assertEqual(config, {'region': 2})

        config = {'endpoint': 1, 'region': 1}
        update = {'endpoint': 2}
        update_config(config, update)
        self.assertEqual(config, {'endpoint': 2})

    def test_config_update_mutually_exclusive_on_same_level(self):
        # config update should not conflate same-level mutually exclusive options
        # updates should be minimal, only when options are in direct conflict
        config = {'endpoint': 1}
        update = {'endpoint': 2, 'region': 2}
        update_config(config, update)
        self.assertEqual(config, update)

        config = {'region': 1}
        update = {'endpoint': 2, 'region': 2}
        update_config(config, update)
        self.assertEqual(config, update)

        config = {'endpoint': 1, 'region': 1}
        update = {'endpoint': 2, 'region': 2}
        update_config(config, update)
        self.assertEqual(config, update)

        config = {}
        update = {'endpoint': 2, 'region': 2}
        update_config(config, update)
        self.assertEqual(config, update)


class TestConfigModel(unittest.TestCase):

    OMITTED = object()

    def _verify(self,
                raw_config: dict[str, Any],
                get_field: abc.Callable[[ClientConfig], Any],
                model_value: Any):
        """For a given ``raw_config``, test both model validate and dump."""

        config = validate_config_v1(raw_config)

        with self.subTest("validate"):
            self.assertEqual(get_field(config), model_value)

        with self.subTest("dump"):
            config = validate_config_v1(dump_config_v1(config))
            self.assertEqual(get_field(config), model_value)

    @parameterized.expand([
        ("null", {"client_cert": None, "client_cert_key": None}, None),
        ("pem", {"client_cert": "/path/to/pem", "client_cert_key": None}, "/path/to/pem"),
        ("pair", {"client_cert": "cert", "client_cert_key": "key"}, ("cert", "key")),
    ])
    def test_cert(self, name, raw_config, model_value):
        self._verify(raw_config=raw_config,
                     get_field=lambda config: config.cert,
                     model_value=model_value)

    @parameterized.expand([
        ("null", None, {}),
        ("blank", '', {}),
        ("empty", '  ', {}),
        ("dict", {"Accept": "*/*"}, {"Accept": "*/*"}),
        ("valid string", "Field-1: value-1\nField-2: value-2",
                         {"Field-1": "value-1", "Field-2": "value-2"}),
        ("invalid string", "Field", {}),
        ("empty field", "Field:", {"Field": ""}),
        ("empty value", "\n", {}),
    ])
    def test_headers(self, name, raw_value, model_value):
        self._verify(raw_config={"headers": raw_value},
                     get_field=lambda config: config.headers,
                     model_value=model_value)

    @parameterized.expand([
        ("null", None, {}),
        ("blank", '', {}),
        ("empty", '  ', {}),
        ("string", "name", dict(name__eq="name")),
        ("identity string", "name;graph_id=123", dict(identity__eq=dict(name="name", version=dict(graph_id="123")))),
        ("simple dict", dict(name="name"), dict(name="name")),
        ("features dict", dict(feature__op="val", x="y"), dict(feature__op="val", x="y")),
        ("valid json", '{"key": "val"}', dict(key="val")),
        ("valid json non-dict", '"id"', dict(id__eq='"id"')),
        ("invalid json, partial identity", "{name}", dict(name__eq="{name}")),
        ("invalid json, invalid identity", ";;", dict(id__eq=";;")),
    ])
    def test_solver(self, name, raw_value, model_value):
        self._verify(raw_config={"solver": raw_value},
                     get_field=lambda config: config.solver,
                     model_value=model_value)

    @parameterized.expand([
        ("null meta", "metadata_api_endpoint", None, None),
        ("null leap", "leap_api_endpoint", None, None),
        ("null sapi", "endpoint", None, None),
        ("omitted meta", "metadata_api_endpoint", OMITTED, DEFAULT_METADATA_API_ENDPOINT),
        ("omitted leap", "leap_api_endpoint", OMITTED, None),
        ("omitted sapi", "endpoint", OMITTED, None),
        ("url meta", "metadata_api_endpoint", "https://metadata.api/v1", "https://metadata.api/v1"),
        ("url leap", "leap_api_endpoint", "https://leap.api/v1", "https://leap.api/v1"),
        ("url sapi", "endpoint", "https://solver.api/v1", "https://solver.api/v1"),
    ])
    def test_endpoints(self, name, key, raw_value, model_value):
        raw_config = {} if raw_value is self.OMITTED else {key: raw_value}
        self._verify(raw_config=raw_config,
                     get_field=lambda config: config[key],
                     model_value=model_value)

    @parameterized.expand([
        ("cert missing", {"client_cert": None, "client_cert_key": "key"}, ValueError),
        ("invalid headers", {"headers": [1,2,3]}, ValueError),
        ("invalid solver", {"solver": {1,2,3}}, ValueError),
    ])
    def test_validation_errors(self, name, raw_config, error):
        with self.assertRaises(error):
            validate_config_v1(raw_config)

    @parameterized.expand([
        ("total", 1, 1),
        ("total", 1.0, 1),
        ("total", False, False),
        ("connect", 1.0, 1),
        ("read", 1, 1),
        ("redirect", 1, 1),
        ("redirect", False, False),
        ("status", 1, 1),
        ("other", 1, 1),
        ("backoff_factor", 1, 1.0),
        ("backoff_max", 1, 1.0),
    ])
    def test_request_retry(self, key, raw_value, model_value):
        self._verify(raw_config={f"http_retry_{key}": raw_value},
                     get_field=lambda config: getattr(config.request_retry, key),
                     model_value=model_value)

    @parameterized.expand([
        ("backoff_min", 1, 1.0),
        ("backoff_min", 1.5, 1.5),
        ("backoff_max", 1, 1.0),
        ("backoff_max", 1.5, 1.5),
        ("backoff_base", 1, 1.0),
        ("backoff_base", 1.5, 1.5),
    ])
    def test_polling_schedule(self, key, raw_value, model_value):
        self._verify(raw_config={f"poll_{key}": raw_value},
                     get_field=lambda config: getattr(config.polling_schedule, key),
                     model_value=model_value)

    @parameterized.expand([
        ("default",
            {}, "strategy", PollingStrategy.BACKOFF),
        ("explicit backoff",
            {"poll_strategy": "backoff"}, "strategy", PollingStrategy.BACKOFF),
        ("explicit long-polling",
            {"poll_strategy": "long-polling"}, "strategy", PollingStrategy.LONG_POLLING),
        ("implicit strategy, conformant value accepted",
            {"poll_backoff_min": 1}, "backoff_min", 1.0),
        ("implicit strategy, non-conformant value ignored",
            {"poll_wait_time": 1}, "wait_time", OMITTED),
        ("explicit strategy, conformant value accepted",
            {"poll_wait_time": 1, "poll_strategy": "long-polling"}, "wait_time", 1.0),
        ("explicit strategy, non-conformant value ignored",
            {"poll_backoff_min": 1, "poll_strategy": "long-polling"}, "backoff_min", OMITTED),
    ])
    def test_polling_schedule_union(self, desc, raw_config, key, model_value):
        self._verify(raw_config=raw_config,
                     get_field=lambda config: getattr(config.polling_schedule, key, self.OMITTED),
                     model_value=model_value)

    @parameterized.expand([
        ("1", True), ("on", True), ("On", True), ("true", True), ("True", True),
        ("0", False), ("off", False), ("Off", False), ("false", False), ("False", False),
    ])
    def test_scalars(self, value, model_value):
        for flag in ("permissive_ssl", "connection_close"):
            with self.subTest(flag):
                self._verify(raw_config={flag: value},
                             get_field=lambda config: getattr(config, flag),
                             model_value=model_value)

    @parameterized.expand([
        ("null", None, None),
        ("omitted", OMITTED, (60.0, 120.0)),    # default
        ("int", 10, 10.0),
        ("float", 10.0, 10.0),
        ("tuple[int]", (10, 20), (10.0, 20.0)),
        ("tuple[float]", (10.0, 20.0), (10.0, 20.0)),
        ("literal scalar", '10', 10.0),
        ("literal tuple", '(20, 30)', (20.0, 30.0)),
        ("literal list", '[20, 30]', (20.0, 30.0)),
    ])
    def test_request_timeout(self, name, raw_value, model_value):
        raw_config = {} if raw_value is self.OMITTED else {"request_timeout": raw_value}
        self._verify(raw_config=raw_config,
                     get_field=lambda config: config.request_timeout,
                     model_value=model_value)

    def test_model_getter_mixin(self):
        val = "https://some.url/path/"
        config = validate_config_v1({"endpoint": val})
        self.assertEqual(config.endpoint, val)
        self.assertEqual(config['endpoint'], val)

    def test_full_load_defaults(self):
        # just a smoke test of `load_config_v1` for now
        from dwave.cloud.config.models import _V1_CONFIG_DEFAULTS
        self.assertEqual(load_config_v1({}),  validate_config_v1(_V1_CONFIG_DEFAULTS))

# Copyright 2023 D-Wave Systems Inc.
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

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dwave.cloud.auth.creds import (
    Credentials, _get_creds_paths, _get_default_creds_path, CREDS_FILENAME)
from dwave.cloud.auth._creds import _Cache


class TestCredentials(unittest.TestCase):

    def test_utils(self):
        with mock.patch('dwave.cloud.auth.creds.get_configfile_paths') as m:
            _get_creds_paths()
            m.assert_called_once_with(
                system=True, user=True, local=True,
                only_existing=True, filename=CREDS_FILENAME)

        with mock.patch('dwave.cloud.auth.creds.get_default_configfile_path') as m:
            _get_default_creds_path()
            m.assert_called_once_with(filename=CREDS_FILENAME)

        paths = _get_creds_paths(only_existing=False)
        self.assertGreater(len(paths), 0)

        default = _get_default_creds_path()
        self.assertGreater(len(default), 0)

    def test_cache_subclass_smoke(self):
        dbname = 'custom.db'
        cache = _Cache(dbname=dbname)
        self.assertEqual(cache.dbname, dbname)

    def setUp(self):
        self.default_file = Path('/path/to/creds.db').resolve()
        self.mocker = mock.patch('dwave.cloud.auth.creds._get_default_creds_path',
                                 return_value=str(self.default_file))
        self.mocker.start()

    def tearDown(self):
        self.mocker.stop()

    def test_autodetect(self):
        # fallback to default
        with mock.patch('dwave.cloud.auth.creds._get_creds_paths',
                        return_value=[]):
            c = Credentials(create=False)
            self.assertEqual(c.creds_file, self.default_file)

        # creds file exists
        existing_file = Path('/path/to/existing.db').resolve()
        with mock.patch('dwave.cloud.auth.creds._get_creds_paths',
                        return_value=[str(existing_file)]):
            c = Credentials(create=False)
            self.assertEqual(c.creds_file, existing_file)

    def test_use_default(self):
        c = Credentials(creds_file=None, create=False)
        self.assertEqual(c.creds_file, self.default_file)

    def test_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir).joinpath('some/deep/path', CREDS_FILENAME).resolve()
            self.assertFalse(path.exists())

            c = Credentials(creds_file=path, create=True)
            self.assertTrue(path.exists())

            self.assertEqual(c.creds_file, path)
            self.assertEqual(c.directory, str(path.parent))

            # enable temp dir deletion on windows
            c.close()

    def test_no_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir).joinpath('some/deep/path', CREDS_FILENAME).resolve()
            self.assertFalse(path.exists())

            c = Credentials(creds_file=path, create=False)
            self.assertFalse(path.exists())

            self.assertEqual(c.creds_file, path)
            self.assertIn('diskcache-', c.directory)

            # enable temp dir deletion on windows
            c.close()

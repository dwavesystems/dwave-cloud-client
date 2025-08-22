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

"""
Tests the ability to connect to the SAPI server with the `dwave.cloud.qpu.Client`.

test_mock_solver_loading.py duplicates some of these tests against a mock server.
"""

import json
import sys
import threading
import time
import unittest
import weakref
from collections import defaultdict
from contextlib import contextmanager
from unittest import mock

import requests
from parameterized import parameterized
from plucky import merge

from dwave.cloud.api import models, Regions
from dwave.cloud.client import Client
from dwave.cloud.config import constants
from dwave.cloud.config.models import dump_config_v1, PollingStrategy
from dwave.cloud.exceptions import (
    SolverAuthenticationError, SolverError, SolverNotFoundError)
from dwave.cloud.regions import get_regions
from dwave.cloud.solver import StructuredSolver, UnstructuredSolver
from dwave.cloud.testing import iterable_mock_open, isolated_environ
from dwave.cloud.utils.decorators import cached
import dwave.cloud

from tests import config


DEFAULT_REGIONS = [
    models.Region(
        code=constants.DEFAULT_REGION,
        name=constants.DEFAULT_REGION,
        endpoint=constants.DEFAULT_SOLVER_API_ENDPOINT
    ),
]

def get_default_regions(*args, **kwargs):
    return DEFAULT_REGIONS


@unittest.skipUnless(config, "No live server configuration available.")
class ConnectivityTests(unittest.TestCase):
    """Test connecting and related failure modes."""

    def test_bad_url(self):
        """Connect with a bad URL."""
        with self.assertRaises(IOError):
            invalid_config = config.copy()
            invalid_config.update(endpoint='invalid-endpoint')
            with Client(**invalid_config) as client:
                client.get_solvers()

    def test_bad_token(self):
        """Connect with a bad token."""
        with self.assertRaises(SolverAuthenticationError):
            invalid_config = config.copy()
            invalid_config.update(token='invalid-token')
            with Client(**invalid_config) as client:
                client.get_solvers()

    def test_good_connection(self):
        """Connect with valid connection settings (url/token/proxy/etc)."""
        with Client(**config) as client:
            self.assertTrue(len(client.get_solvers()) > 0)


@unittest.skipUnless(config, "No live server configuration available.")
class TestTimeouts(unittest.TestCase):
    """Test timeout works for all Client connections."""

    def test_request_timeout(self):
        with self.assertRaises(dwave.cloud.exceptions.RequestTimeout):
            with Client(request_timeout=0.00001, **config) as client:
                client.get_solvers(refresh=True)

    def test_polling_timeout(self):
        with self.assertRaises(dwave.cloud.exceptions.PollingTimeout):
            with Client(polling_timeout=0.00001, **config) as client:
                solver = client.get_solver()
                solver.sample_qubo({next(iter(solver.edges)): 0}).result()


@unittest.skipUnless(config, "No live server configuration available.")
class TestRetrieveAnswer(unittest.TestCase):
    """Test loading a problem from its id"""

    def test_retrieve_answer(self):
        """Answer retrieved based on problem_id in a new client."""

        with Client(**config) as client:
            solver = client.get_solver()

            h = {v: -1 for v in solver.nodes}

            f = solver.sample_ising(h, {})

            # the id is not set right away
            while f.id is None:
                time.sleep(.01)

            id_ = f.id
        
        with Client(**config) as client:
            # get a "new" client
            f2 = client.retrieve_answer(id_)

            self.assertIn('solutions', f2.result())

    def test_retrieve_invalid_answer_id(self):
        """Loading result fails when problem_id invalid."""

        with Client(**config) as client:
            f = client.retrieve_answer('invalid_id')
            with self.assertRaises(SolverError):
                f.result()

@unittest.skipUnless(config, "No live server configuration available.")
class SolverLoading(unittest.TestCase):
    """Test loading solvers in a few different configurations."""

    def test_list_all_solvers(self):
        """List all the solvers."""
        with Client(**config) as client:
            self.assertTrue(len(client.get_solvers()) > 0)

    def test_load_all_solvers(self):
        """List and retrieve all the solvers."""
        with Client(**config) as client:
            for solver in client.get_solvers():
                self.assertEqual(client.get_solver(name=solver.name).name, solver.name)

    def test_load_bad_solvers(self):
        """Try to load a nonexistent solver."""
        with Client(**config) as client:
            with self.assertRaises(SolverNotFoundError):
                client.get_solver("not-a-solver")

    def test_load_any_solver(self):
        """Load a single solver without calling get_solvers."""
        with Client(**config) as client:
            self.assertTrue(client.get_solver(software=True).software)

    def test_get_solver_no_defaults(self):
        """Specialized client returns the correct solver by default."""
        from dwave.cloud.client import qpu, sw, hybrid

        conf = config.copy()
        del conf['solver']

        with Client(**conf) as base_client:

            with qpu.Client(**conf) as client:
                solvers = {s.name for s in base_client.get_solvers(qpu=True)}
                if solvers:
                    self.assertIn(client.get_solver().name, solvers)
                else:
                    self.assertRaises(SolverError, client.get_solver)

            with sw.Client(**conf) as client:
                solvers = {s.name for s in base_client.get_solvers(software=True)}
                if solvers:
                    self.assertIn(client.get_solver().name, solvers)
                else:
                    self.assertRaises(SolverError, client.get_solver)

            with hybrid.Client(**conf) as client:
                solvers = {s.name for s in base_client.get_solvers(hybrid=True)}
                if solvers:
                    self.assertIn(client.get_solver().name, solvers)
                else:
                    self.assertRaises(SolverError, client.get_solver)


@mock.patch("dwave.cloud.regions.get_regions", get_default_regions)
class ClientConstruction(unittest.TestCase):
    """Test Client constructor and Client.from_config() factory."""

    def setUp(self):
        # prevent run-time env vars interfering with tests
        self._env = isolated_environ(empty=True).start()

    def tearDown(self):
        self._env.stop()

    def test_default(self):
        conf = {k: k for k in 'endpoint token'.split()}
        with mock.patch("dwave.cloud.client.base.load_config", lambda **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.config.endpoint, 'endpoint')
                self.assertEqual(client.config.token, 'token')
                self.assertIsInstance(client, dwave.cloud.client.Client)
                self.assertNotIsInstance(client, dwave.cloud.client.sw.Client)
                self.assertNotIsInstance(client, dwave.cloud.client.qpu.Client)
                self.assertNotIsInstance(client, dwave.cloud.client.hybrid.Client)

    def test_region_default(self):
        conf = {k: k for k in 'token'.split()}
        with mock.patch("dwave.cloud.client.base.load_config", lambda **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.config.region, constants.DEFAULT_REGION)
                self.assertEqual(client.config.endpoint, constants.DEFAULT_SOLVER_API_ENDPOINT)
                self.assertEqual(client.config.leap_api_endpoint, constants.DEFAULT_LEAP_API_ENDPOINT)
                self.assertEqual(client.config.metadata_api_endpoint, constants.DEFAULT_METADATA_API_ENDPOINT)
                self.assertEqual(client.config.token, 'token')
                self.assertIsInstance(client, dwave.cloud.client.Client)
                self.assertNotIsInstance(client, dwave.cloud.client.sw.Client)
                self.assertNotIsInstance(client, dwave.cloud.client.qpu.Client)
                self.assertNotIsInstance(client, dwave.cloud.client.hybrid.Client)

    def test_region_selection_over_defaults(self):
        conf = {k: k for k in 'token'.split()}
        region_code = 'region-code'
        region_name = 'region-name'
        region_endpoint = 'region-endpoint'
        region_conf = [models.Region(code=region_code, name=region_name, endpoint=region_endpoint)]

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with mock.patch("dwave.cloud.regions.get_regions", lambda config: region_conf):
                with dwave.cloud.Client.from_config(region=region_code) as client:
                    self.assertEqual(client.config.region, region_code)
                    self.assertEqual(client.config.endpoint, region_endpoint)

    def test_region_kwarg_overrides_endpoint_from_config(self):
        conf = {k: k for k in 'endpoint token'.split()}
        region_code = 'region-code'
        region_name = 'region-name'
        region_endpoint = 'region-endpoint'
        region_conf = [models.Region(code=region_code, name=region_name, endpoint=region_endpoint)]

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with mock.patch("dwave.cloud.regions.get_regions", lambda config: region_conf):
                with dwave.cloud.Client.from_config(region=region_code) as client:
                    self.assertEqual(client.config.region, region_code)
                    self.assertEqual(client.config.endpoint, region_endpoint)

    def test_region_endpoint_pair_kwarg_overrides_region_endpoint_pair_from_config(self):
        conf = {k: k for k in 'region endpoint token'.split()}
        region_code = 'region-code'
        region_name = 'region-name'
        region_endpoint = 'region-endpoint'
        region_conf = [models.Region(code=region_code, name=region_name, endpoint=region_endpoint)]

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with mock.patch("dwave.cloud.regions.get_regions", lambda config: region_conf):
                with dwave.cloud.Client.from_config(region=region_code, endpoint=region_endpoint) as client:
                    self.assertEqual(client.config.region, region_code)
                    self.assertEqual(client.config.endpoint, region_endpoint)

    def test_region_from_env_overrides_endpoint_from_config(self):
        conf = {k: k for k in 'endpoint token'.split()}
        region_code = 'region-code'
        region_name = 'region-name'
        region_endpoint = 'region-endpoint'
        region_conf = [models.Region(code=region_code, name=region_name, endpoint=region_endpoint)]

        with isolated_environ(add={'DWAVE_API_REGION': region_code}):
            with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
                with mock.patch("dwave.cloud.regions.get_regions", lambda config: region_conf):
                    with dwave.cloud.Client.from_config() as client:
                        self.assertEqual(client.config.region, region_code)
                        self.assertEqual(client.config.endpoint, region_endpoint)

    @parameterized.expand([
        ("DWAVE_METADATA_API_ENDPOINT", "metadata-api-endpoint", "metadata_api_endpoint"),
        ("DWAVE_LEAP_API_ENDPOINT", "leap-api-endpoint", "leap_api_endpoint"),
    ])
    def test_metadata_api_endpoint_from_env_accepted(self, env_key, env_val, config_key):
        with isolated_environ(add={env_key: env_val}):
            conf = {k: k for k in 'endpoint token'.split()}
            with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
                with dwave.cloud.Client.from_config() as client:
                    self.assertEqual(client.config[config_key], env_val)

    def test_client_type(self):
        conf = {k: k for k in 'endpoint token'.split()}

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.client.Client)

            with dwave.cloud.client.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.client.Client)

            with dwave.cloud.client.qpu.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.client.qpu.Client)

            with dwave.cloud.client.sw.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.client.sw.Client)

            with dwave.cloud.client.hybrid.Client.from_config() as client:
                self.assertIsInstance(client, dwave.cloud.client.hybrid.Client)

            with dwave.cloud.client.qpu.Client.from_config(client='base') as client:
                self.assertIsInstance(client, dwave.cloud.client.Client)

            with dwave.cloud.Client.from_config(client='qpu') as client:
                self.assertIsInstance(client, dwave.cloud.client.qpu.Client)

            with dwave.cloud.Client.from_config(client='sw') as client:
                self.assertIsInstance(client, dwave.cloud.client.sw.Client)

            with dwave.cloud.Client.from_config(client='hybrid') as client:
                self.assertIsInstance(client, dwave.cloud.client.hybrid.Client)

    def test_custom_kwargs(self):
        conf = {k: k for k in 'endpoint token'.split()}
        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with mock.patch("dwave.cloud.client.Client.__init__", return_value=None) as init:
                dwave.cloud.Client.from_config(custom='custom')
                init.assert_called_once_with(
                    endpoint='endpoint', token='token', custom='custom')

    def test_custom_kwargs_overrides_config(self):
        conf = {k: k for k in 'endpoint token custom'.split()}
        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with mock.patch("dwave.cloud.client.Client.__init__", return_value=None) as init:
                dwave.cloud.Client.from_config(custom='new-custom')
                init.assert_called_once_with(
                    endpoint='endpoint', token='token', custom='new-custom')

    @parameterized.expand([
        ("json", '{"qpu": true}', {"qpu": True}),
        ("name", "solver", {"name__eq": "solver"}),
        ("id", "solver;graph_id=123", {"identity__eq": {"name": "solver", "version": {"graph_id": "123"}}}),
    ])
    def test_solver_from_config(self, name, solver_inp, solver_def):
        conf = dict(endpoint='endpoint', token='token', solver=solver_inp)

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.config.solver, solver_def)

    def test_solver_features_kwargs_override_config(self):
        new_solver_def = {"software": True}
        conf = {k: k for k in 'endpoint token solver'.split()}

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config(solver=new_solver_def) as client:
                self.assertEqual(client.config.solver, new_solver_def)

    def test_none_kwargs_do_not_override_config(self):
        """kwargs with value ``None`` should be ignored (issue #430)"""
        conf = {k: k for k in 'endpoint token'.split()}
        solver_json = '{"qpu": true}'
        conf.update(solver=solver_json)
        solver = json.loads(solver_json)

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config(endpoint=None, solver=None) as client:
                self.assertEqual(client.config.endpoint, conf['endpoint'])
                self.assertEqual(client.config.solver, solver)

    def test_solver_name_overrides_config_features(self):
        conf = {k: k for k in 'endpoint token solver'.split()}
        conf.update(solver=json.dumps({"software": True}))

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config(solver='solver') as client:
                self.assertEqual(client.config.solver, {"name__eq": "solver"})

    def test_boolean_options_parsed_from_config(self):
        conf = {'connection_close': 'off', 'permissive_ssl': 'true'}

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config(token='token') as client:
                self.assertFalse(client.config.connection_close)
                self.assertTrue(client.config.permissive_ssl)

    def test_class_defaults(self):
        token = 'value'
        DEFAULTS = Client.DEFAULTS.copy()
        DEFAULTS.update(token=token)

        with mock.patch("dwave.cloud.client.base.load_config", lambda **kw: {}):
            with mock.patch.multiple("dwave.cloud.Client", DEFAULTS=DEFAULTS):
                with dwave.cloud.Client.from_config() as client:
                    self.assertEqual(client.config.token, token)

                # None defaults are ignored
                with dwave.cloud.Client(defaults=None) as client:
                    self.assertEqual(client.config.token, token)

                # explicit None kwargs do not modify defaults
                with dwave.cloud.Client(
                        endpoint=None, token=None, solver=None,
                        connection_close=None, poll_backoff_min=None,
                        metadata_api_endpoint=None, leap_api_endpoint=None) as client:

                    self.assertEqual(client.config.endpoint, client.DEFAULT_API_ENDPOINT)
                    self.assertEqual(client.config.leap_api_endpoint, constants.DEFAULT_LEAP_API_ENDPOINT)
                    self.assertEqual(client.config.metadata_api_endpoint, constants.DEFAULT_METADATA_API_ENDPOINT)
                    self.assertEqual(client.config.token, token)
                    self.assertEqual(client.config.solver, {})

                    self.assertEqual(client.config.connection_close, DEFAULTS['connection_close'])
                    self.assertEqual(client.config.polling_schedule.wait_time, DEFAULTS['poll_wait_time'])

    def test_defaults_as_kwarg(self):
        token = 'value'
        defaults = dict(token=token)

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: {}):
            with dwave.cloud.Client.from_config(defaults=defaults) as client:
                self.assertEqual(client.config.token, token)

    def test_defaults_partial_update(self):
        """Some options come from DEFAULTS, some from defaults, some from config, and some from kwargs"""

        token = 'value'
        solver = {'feature': 'value'}
        request_timeout = 10

        DEFAULTS = Client.DEFAULTS.copy()
        DEFAULTS.update(token='wrong')

        defaults = dict(solver='wrong')

        conf = dict(solver=solver, request_timeout=request_timeout)

        kwargs = dict(token=token, defaults=defaults, request_timeout=None)

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with mock.patch.multiple("dwave.cloud.Client", DEFAULTS=DEFAULTS):
                with dwave.cloud.Client.from_config(**kwargs) as client:

                    # token: set on DEFAULTS, overwritten in kwargs
                    self.assertEqual(client.config.token, token)

                    # solver: set in defaults, overwritten in file conf
                    self.assertEqual(client.config.solver, solver)

                    # endpoint: derived from region
                    self.assertEqual(client.config.endpoint, constants.DEFAULT_SOLVER_API_ENDPOINT)

                    # None kwarg: used from class defaults
                    self.assertEqual(client.config.request_timeout, request_timeout)

    def test_headers_from_config(self):
        headers_dict = {"key-1": "value-1", "key-2": "value-2"}
        headers_str = """  key-1:value-1
            key-2: value-2
        """
        conf = dict(token='token', headers=headers_str)

        with mock.patch("dwave.cloud.client.base.load_config", lambda **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertDictEqual(client.config.headers, headers_dict)

    def test_headers_from_kwargs(self):
        headers_dict = {"key-1": "value-1", "key-2": "value-2"}
        headers_str = "key-2:value-2\nkey-1:value-1"
        conf = dict(token='token')

        # headers as dict
        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config(headers=headers_dict) as client:
                self.assertDictEqual(client.config.headers, headers_dict)

        # headers as str
        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config(headers=headers_str) as client:
                self.assertDictEqual(client.config.headers, headers_dict)

    def test_client_cert_from_config(self):
        crt = '/path/to/crt'
        key = '/path/to/key'

        # single file with cert+key
        client_cert = crt
        conf = dict(token='token', client_cert=crt)

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.config.cert, client_cert)

                session = client.create_session()
                self.assertEqual(session.cert, client_cert)

        # separate cert and key files
        client_cert = (crt, key)
        conf = dict(token='token', client_cert=crt, client_cert_key=key)

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.config.cert, client_cert)

                session = client.create_session()
                self.assertEqual(session.cert, client_cert)

    def test_client_cert_from_kwargs(self):
        crt = '/path/to/crt'
        key = '/path/to/key'

        def load_config(**kwargs):
            file_conf = dict(token='token')
            return merge(kwargs, file_conf, op=lambda a, b: a or b)

        # single file with cert+key
        client_cert = crt
        conf = dict(client_cert=crt)

        with mock.patch("dwave.cloud.client.base.load_config", load_config):
            with dwave.cloud.Client.from_config(**conf) as client:
                self.assertEqual(client.config.cert, client_cert)

        # separate cert and key files
        client_cert = (crt, key)
        conf = dict(client_cert=crt, client_cert_key=key)

        with mock.patch("dwave.cloud.client.base.load_config", load_config):
            with dwave.cloud.Client.from_config(**conf) as client:
                self.assertEqual(client.config.cert, client_cert)

        # client_cert as tuple (direct `requests` format)
        client_cert = (crt, key)
        conf = dict(client_cert=client_cert)

        with mock.patch("dwave.cloud.client.base.load_config", load_config):
            with dwave.cloud.Client.from_config(**conf) as client:
                self.assertEqual(client.config.cert, client_cert)

    def test_polling_params_from_config(self):
        with self.subTest("defaults"):
            conf = {"token": "token"}
            with mock.patch("dwave.cloud.config.loaders.load_profile_from_files",
                            lambda *pa, **kw: conf):
                with dwave.cloud.Client.from_config() as client:
                    ps = client.config.polling_schedule
                    self.assertEqual(ps.strategy, Client.DEFAULTS['poll_strategy'])
                    self.assertEqual(ps.wait_time, Client.DEFAULTS['poll_wait_time'])
                    self.assertEqual(ps.pause, Client.DEFAULTS['poll_pause'])

        with self.subTest("backoff"):
            conf = {"poll_strategy": "backoff",
                    "poll_backoff_min": "0.1",
                    "poll_backoff_max": "1",
                    "token": "token"}
            with mock.patch("dwave.cloud.config.loaders.load_profile_from_files",
                            lambda *pa, **kw: conf):
                with dwave.cloud.Client.from_config() as client:
                    ps = client.config.polling_schedule
                    self.assertEqual(ps.strategy, PollingStrategy.BACKOFF)
                    self.assertEqual(ps.backoff_min, 0.1)
                    self.assertEqual(ps.backoff_max, 1.0)

        with self.subTest("long polling"):
            conf = {"poll_strategy": "long-polling",
                    "poll_wait_time": "10",
                    "poll_pause": "1",
                    "token": "token"}
            with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
                with dwave.cloud.Client.from_config() as client:
                    ps = client.config.polling_schedule
                    self.assertEqual(ps.strategy, PollingStrategy.LONG_POLLING)
                    self.assertEqual(ps.wait_time, 10)
                    self.assertEqual(ps.pause, 1.0)

    @mock.patch("dwave.cloud.config.loaders.load_profile_from_files",
                lambda *pa, **kw: dict(token='token'))
    def test_polling_params_from_kwargs(self):
        with self.subTest("backoff"):
            with dwave.cloud.Client.from_config(poll_strategy="backoff",
                                                poll_backoff_min=0.5) as client:
                self.assertEqual(client.config.polling_schedule.backoff_min, 0.5)

        with self.subTest("long polling"):
            with dwave.cloud.Client.from_config(poll_strategy="long-polling",
                                                poll_wait_time=10) as client:
                self.assertEqual(client.config.polling_schedule.wait_time, 10)

    def _verify_retry_config(self, retry, opts):
        self.assertEqual(retry.total, opts['http_retry_total'])
        self.assertEqual(retry.connect, opts['http_retry_connect'])
        self.assertEqual(retry.read, opts['http_retry_read'])
        self.assertEqual(retry.redirect, opts['http_retry_redirect'])
        self.assertEqual(retry.status, opts['http_retry_status'])
        self.assertEqual(retry.backoff_factor, opts['http_retry_backoff_factor'])
        backoff_max = getattr(retry, 'backoff_max', getattr(retry, 'BACKOFF_MAX'))
        self.assertEqual(backoff_max, opts['http_retry_backoff_max'])

    def test_http_retry_params_from_config(self):
        retry_opts = {
            "http_retry_total": 3,
            "http_retry_connect": 2,
            "http_retry_read": 2,
            "http_retry_redirect": 0,
            "http_retry_status": 2,
            "http_retry_backoff_factor": 0.5,
            "http_retry_backoff_max": 30,
        }
        retry_conf = {k: str(v) for k, v in retry_opts.items()}
        conf = dict(token='token', **retry_conf)

        # http retry params from config file propagated to client object
        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                session = client.create_session()
                retry = session.get_adapter('https://').max_retries
                self._verify_retry_config(retry, retry_opts)

        # test defaults
        conf = dict(token='token')
        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                v1_config = dump_config_v1(client.config)
                for param in retry_conf:
                    self.assertEqual(v1_config.get(param), Client.DEFAULTS[param])

    def test_http_retry_params_from_kwargs(self):
        retry_kwargs = {
            "http_retry_total": 3,
            "http_retry_connect": 2,
            "http_retry_read": None,
            "http_retry_redirect": 0,
            "http_retry_status": None,
            "http_retry_backoff_factor": 0.5,
            "http_retry_backoff_max": 30,
        }
        conf = dict(token='token')

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with dwave.cloud.Client.from_config(**retry_kwargs) as client:
                session = client.create_session()
                retry = session.get_adapter('https://').max_retries
                self._verify_retry_config(retry, retry_kwargs)


class VerifyLazyClientImport(unittest.TestCase):

    def test_client_available(self):
        import dwave.cloud
        import dwave.cloud.client

        self.assertEqual(dwave.cloud.client.Client, dwave.cloud.Client)


class VerifyClientCleanup(unittest.TestCase):

    def test_client_close(self):
        thread_cnt_base = threading.active_count()

        client = Client(endpoint='endpoint', token='token')
        ref = weakref.ref(client)

        self.assertGreater(threading.active_count(), thread_cnt_base)
        self.assertGreater(sys.getrefcount(client), 1)

        client.close()
        del client

        # verify all threads closed and no client referrers
        self.assertEqual(threading.active_count(), thread_cnt_base)
        self.assertIsNone(ref())


@mock.patch("dwave.cloud.regions.get_regions", get_default_regions)
class ClientConfigIntegration(unittest.TestCase):

    def test_custom_options(self):
        """Test custom options (request_timeout, polling_timeout, permissive_ssl)
        are propagated to Client."""

        request_timeout = (15, 30)
        polling_timeout = 180

        config_body = """
            [custom]
            token = 123
            permissive_ssl = on
            request_timeout = {}
            polling_timeout = {}
        """.format(request_timeout, polling_timeout)

        with mock.patch("dwave.cloud.config.loaders.open", iterable_mock_open(config_body)):
            with Client.from_config('config_file', profile='custom') as client:
                # check permissive_ssl and timeouts custom params passed-thru
                self.assertTrue(client.config.permissive_ssl)
                self.assertEqual(client.config.request_timeout, request_timeout)
                self.assertEqual(client.config.polling_timeout, polling_timeout)

                # verify client uses those properly
                def mock_send(*args, **kwargs):
                    self.assertEqual(kwargs.get('timeout'), request_timeout)
                    response = requests.Response()
                    response.status_code = 200
                    response._content = b'[]'
                    response.headers = {'Content-Type': 'application/vnd.dwave.sapi.solver-definition-list+json; version=3.0'}
                    return response

                with mock.patch("requests.adapters.HTTPAdapter.send", mock_send):
                    client.get_solvers()


class MultiRegionSupport(unittest.TestCase):

    @isolated_environ(empty=True)
    def test_region_selection_mocked_end_to_end(self):
        """Given `metadata_api_endpoint` and `region` in a config file,
        when Client is initialized from config,
        then it is set-up with sapi endpoint retrieved from the metadata api."""

        metadata_api_endpoint = 'metadata-api'
        regions_response = [
            {"code": "a", "endpoint": "endpoint-a", "name": "A"},
            {"code": "b", "endpoint": "endpoint-b", "name": "B"}
        ]
        selected_region = regions_response[0]['code']
        selected_endpoint = regions_response[0]['endpoint']

        permissive_ssl = True
        proxy = 'http://127.0.0.1/'

        config_body = f"""
            [defaults]
            metadata_api_endpoint = {metadata_api_endpoint}
            token = token
            region = {selected_region}
            proxy = {proxy}
            permissive_ssl = {permissive_ssl}
        """

        expected_get_regions_return = [
            models.Region(**region) for region in regions_response
        ]

        def _mocked_session():
            session = mock.Mock()
            session.headers = {}
            session.hooks = defaultdict(list)

            def get(url, **kwargs):
                response = mock.Mock(['text', 'json'])
                response.status_code = 200
                response.content = json.dumps(regions_response).encode('utf8')
                return response
            session.get = get

            return session

        def _mocked_regions_factory(config, **kwargs):
            # verify full client config is passed to `Regions.from_config` factory
            self.assertEqual(config.metadata_api_endpoint, metadata_api_endpoint)
            self.assertEqual(config.permissive_ssl, permissive_ssl)
            self.assertEqual(config.proxy, proxy)

            regions = Regions()
            regions._session = _mocked_session()
            return regions

        with mock.patch("dwave.cloud.config.loaders.open", iterable_mock_open(config_body)):
            with cached.disabled():
                with mock.patch("dwave.cloud.client.base.api.Regions.from_config", _mocked_regions_factory):
                    # note: we specify config_file, otherwise reading from files
                    # might be skipped altogether if zero config files found on disk
                    # (i.e. config open mock above fails on CI)
                    with Client.from_config(config_file='config_file') as client:
                        # verify get_regions() returns data from metadata_api_endpoint
                        regions = get_regions(config=client.config)
                        self.assertEqual(regions, expected_get_regions_return)

                        # check region endpoint initialized from custom metadata_api_endpoint
                        self.assertEqual(client.config.metadata_api_endpoint, metadata_api_endpoint)
                        self.assertEqual(client.config.region, selected_region)
                        self.assertEqual(client.config.endpoint, selected_endpoint)

    @isolated_environ(empty=True)
    def test_region_endpoint_fallback_when_no_metadata_api(self):
        """Given region in config, and endpoint omitted,
        when Client is initialized from config and MetadataAPI is down,
        then the client is initialized to use the default (old) endpoint and region.
        """
        # test regions.resolve_endpoint

        conf = dict(token='token', metadata_api_endpoint='invalid')
        region = Client.DEFAULT_API_REGION

        with mock.patch("dwave.cloud.client.base.load_config", lambda **kw: conf):
            with dwave.cloud.Client.from_config(region=region) as client:
                self.assertEqual(client.config.region, Client.DEFAULT_API_REGION)
                self.assertEqual(client.config.endpoint, Client.DEFAULT_API_ENDPOINT)

    @mock.patch("dwave.cloud.regions.get_regions", get_default_regions)
    @isolated_environ(empty=True)
    def test_region_endpoint_fallback_when_region_unknown(self):
        """Given invalid region in config, and endpoint omitted,
        when Client is initialized from config,
        then ValueError is raised.
        """
        # test regions.resolve_endpoint

        conf = dict(token='token')
        region = 'invalid-region-code'

        with mock.patch("dwave.cloud.config.loaders.load_profile_from_files", lambda *pa, **kw: conf):
            with self.assertRaises(ValueError):
                Client.from_config(region=region)

    @mock.patch("dwave.cloud.regions.get_regions", get_default_regions)
    @isolated_environ(empty=True)
    def test_region_endpoint_null_case(self):
        """Given region as None, and endpoint as None,
        when Client is initialized from config,
        then the client is initialized to use the default (old) endpoint and region.
        """
        # test regions.resolve_endpoint

        conf = dict(token='token', region='')

        with mock.patch("dwave.cloud.client.base.load_config", lambda **kw: conf):
            with dwave.cloud.Client.from_config() as client:
                self.assertEqual(client.config.region, Client.DEFAULT_API_REGION)
                self.assertEqual(client.config.endpoint, Client.DEFAULT_API_ENDPOINT)

    @unittest.skipUnless(config, "No live server configuration available.")
    @isolated_environ(empty=True)
    def test_region_selection_default(self):
        with Client(**config) as client:
            self.assertEqual(client.config.region, constants.DEFAULT_REGION)
            self.assertIn(constants.DEFAULT_SOLVER_API_ENDPOINT, client.config.endpoint)
            self.assertGreater(len(client.get_regions()), 0)

    @unittest.skipUnless(config, "No live server configuration available.")
    @isolated_environ(empty=True)
    def test_region_selection_default(self):
        region = 'eu-central-1'
        with Client(region=region, **config) as client:
            self.assertEqual(client.config.region, region)
            self.assertIn(region, client.config.endpoint)


class FeatureBasedSolverSelection(unittest.TestCase):
    """Test Client.get_solvers(**filters)."""

    def setUp(self):
        # mock solvers
        self.qpu1 = StructuredSolver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1, 2],
                "couplers": [[0, 1], [0, 2], [1, 2]],
                "num_qubits": 3,
                "num_reads_range": [0, 100],
                "parameters": {
                    "num_reads": "Number of samples to return.",
                    "postprocess": "either 'sampling' or 'optimization'"
                },
                "topology": {
                    "type": "chimera",
                    "shape": [16, 16, 4]
                },
                "category": "qpu",
                "tags": ["lower_noise"]
            },
            "identity": {
                "name": "qpu1",
                "version": {
                    "graph_id": "wg1",
                },
            },
            "description": "QPU Chimera solver",
            "status": "online",
            "avg_load": 0.1
        })
        self.qpu2 = StructuredSolver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1, 2, 3, 4],
                "couplers": [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]],
                "num_qubits": 5,
                "num_reads_range": [0, 200],
                "parameters": {
                    "num_reads": "Number of samples to return.",
                    "flux_biases": "Supported ...",
                    "anneal_schedule": "Supported ..."
                },
                "topology": {
                    "type": "pegasus",
                    "shape": [6, 6, 12]
                },
                "category": "qpu",
                "vfyc": True
            },
            "identity": {
                "name": "qpu2",
                "version": {
                    "graph_id": "wg1",
                },
            },
            "description": "QPU Pegasus solver",
            "avg_load": 0.2
        })
        self.software = StructuredSolver(client=None, data={
            "properties": {
                "supported_problem_types": ["qubo", "ising"],
                "qubits": [0, 1],
                "couplers": [[0, 1]],
                "num_qubits": 7,
                "num_reads_range": [0, 1000],
                "parameters": {"num_reads": "Number of samples to return."},
                "vfyc": False,
                "topology": {
                    "type": "chimera",
                    "shape": [4, 4, 4]
                },
                "category": "software",
                # the following are only present in this solver
                "some_set": [1, 2],
                "some_range": [1, 2],
                "some_string": "x",
                "tags": ["tag"]
            },
            "identity": {
                "name": "sw_solver1",
                "version": {
                    "graph_id": "sw1",
                },
            },
            "description": "Software solver",
            "avg_load": 0.7
        })
        self.hybrid = UnstructuredSolver(client=None, data={
            "properties": {
                "supported_problem_types": ["bqm"],
                "maximum_number_of_variables": 10000,
                "maximum_time_limit_hrs": 24.0,
                "minimum_time_limit": [[1, 3.0], [1024, 3.0], [4096, 10.0], [10000, 40.0]],
                "quota_conversion_rate": 20,
                "parameters": {
                    "time_limit": ""
                },
                "category": "hybrid",
            },
            "identity": {
                "name": "hybrid_v1",
            },
            "description": "Hybrid solver"
        })

        self.qpu_solvers = [self.qpu1, self.qpu2]
        self.software_solvers = [self.software]
        self.hybrid_solvers = [self.hybrid]

        self.structured_solvers = self.qpu_solvers + self.software_solvers
        self.unstructured_solvers = self.hybrid_solvers

        self.solvers = self.structured_solvers + self.unstructured_solvers

        # mock client
        self.client = Client(endpoint='endpoint', token='token')
        self.client._fetch_solvers = lambda **kw: self.solvers

    def shutDown(self):
        self.client.close()

    def assertSolvers(self, container, members):
        self.assertEqual(set(container), set(members))

    def test_default(self):
        self.assertSolvers(self.client.get_solvers(), self.solvers)

    def test_online(self):
        self.assertSolvers(self.client.get_solvers(online=True), self.solvers)
        self.assertSolvers(self.client.get_solvers(online=False), [])

    def test_derived_category_properties(self):
        self.assertSolvers(self.client.get_solvers(qpu=True), self.qpu_solvers)
        self.assertSolvers(self.client.get_solvers(qpu=False), self.software_solvers + self.hybrid_solvers)
        self.assertSolvers(self.client.get_solvers(software=True), self.software_solvers)
        self.assertSolvers(self.client.get_solvers(software=False), self.qpu_solvers + self.hybrid_solvers)
        self.assertSolvers(self.client.get_solvers(hybrid=True), self.hybrid_solvers)
        self.assertSolvers(self.client.get_solvers(hybrid=False), self.qpu_solvers + self.software_solvers)

    # Test fallback for legacy solvers without the `category` property
    # TODO: remove when all production solvers are updated
    def test_derived_category_properties_without_category(self):
        "Category-based filtering works without explicit `category` property."

        @contextmanager
        def multi_solver_properties_patch(solvers, update):
            """Update properties for all `solvers` at once."""
            patchers = [mock.patch.dict(s.properties, update) for s in solvers]
            try:
                yield (p.start() for p in patchers)
            finally:
                return (p.stop() for p in patchers)

        with mock.patch.object(self.software.identity, 'name', 'c4-sw_solver3'):
            # patch categories and re-run the category-based filtering test
            with multi_solver_properties_patch(self.solvers, {'category': ''}):
                self.test_derived_category_properties()

            # verify patching
            with multi_solver_properties_patch(self.solvers, {'category': 'x'}):
                with self.assertRaises(AssertionError):
                    self.test_derived_category_properties()

    def test_category(self):
        self.assertSolvers(self.client.get_solvers(category='qpu'), self.qpu_solvers)
        self.assertSolvers(self.client.get_solvers(category='software'), self.software_solvers)
        self.assertSolvers(self.client.get_solvers(category='hybrid'), self.hybrid_solvers)

    def test_name(self):
        self.assertSolvers(self.client.get_solvers(name='qpu1'), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(name='qpu2'), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(name='solver'), [])
        self.assertSolvers(self.client.get_solvers(name='olver1'), [])
        self.assertSolvers(self.client.get_solvers(name__regex='.*1'), [self.qpu1, self.software, self.hybrid])
        self.assertSolvers(self.client.get_solvers(name__regex='.*_v1'), [self.hybrid])
        self.assertSolvers(self.client.get_solvers(name__regex='.*[12].*'), self.solvers)
        self.assertSolvers(self.client.get_solvers(name__regex='qpu[12]'), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(name__regex='^qpu(1|2)$'), [self.qpu1, self.qpu2])

    def test_graph_id(self):
        self.assertSolvers(self.client.get_solvers(graph_id='wg1'), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(graph_id='sw1'), [self.software])
        self.assertSolvers(self.client.get_solvers(version__graph_id='wg1'), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(version=dict(graph_id='wg1')), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(graph_id__eq=None), [self.hybrid])

    def test_identity(self):
        # model-to-model and model-to-dict equality
        identity = self.qpu1.identity
        self.assertSolvers(self.client.get_solvers(identity=identity), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(identity=identity.dict()), [self.qpu1])
        # partial identity isn't matched
        self.assertSolvers(self.client.get_solvers(identity=dict(name=identity.name)), [])
        # mix
        self.assertSolvers(self.client.get_solvers(name='qpu1', graph_id='wg1'), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(name='qpu2', graph_id='wg1'), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(name='qpu1', graph_id='x'), [])

    def test_num_qubits(self):
        self.assertSolvers(self.client.get_solvers(num_qubits=5), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_active_qubits=2), [self.software])
        self.assertSolvers(self.client.get_solvers(num_active_qubits__in=[2, 3]), [self.qpu1, self.software])

    def test_lower_noise_derived_property(self):
        self.assertSolvers(self.client.get_solvers(lower_noise=True), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(lower_noise=False), [self.qpu2, self.software])

    def test_parameter_availability_check(self):
        self.assertSolvers(self.client.get_solvers(postprocess__available=True), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(postprocess=True), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(parameters__contains='flux_biases'), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(parameters__contains='num_reads'), self.structured_solvers)

    def test_property_availability_check(self):
        self.assertSolvers(self.client.get_solvers(vfyc__available=True), [self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(vfyc__eq=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(vfyc=True), [self.qpu2])

        # inverse of vfyc=True
        self.assertSolvers(self.client.get_solvers(vfyc__in=[False, None]), [self.qpu1, self.software, self.hybrid])

        # vfyc unavailable or unadvertized
        self.assertSolvers(self.client.get_solvers(vfyc__available=False), [self.qpu1, self.hybrid])
        self.assertSolvers(self.client.get_solvers(vfyc__eq=False), [self.software])
        self.assertSolvers(self.client.get_solvers(vfyc=False), [self.software])

        # non-existing params/props have value of None
        self.assertSolvers(self.client.get_solvers(vfyc__eq=None), [self.qpu1, self.hybrid])
        self.assertSolvers(self.client.get_solvers(vfyc=None), [self.qpu1, self.hybrid])

    def test_availability_combo(self):
        self.assertSolvers(self.client.get_solvers(vfyc=False, flux_biases=True), [])
        self.assertSolvers(self.client.get_solvers(vfyc=True, flux_biases=True), [self.qpu2])

    def test_relational_ops(self):
        self.assertSolvers(self.client.get_solvers(num_qubits=3), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits__eq=3), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits=4), [])
        self.assertSolvers(self.client.get_solvers(num_qubits=5), [self.qpu2])

        self.assertSolvers(self.client.get_solvers(num_qubits__gte=2), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__gte=5), [self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(num_qubits__gt=5), [self.software])

        self.assertSolvers(self.client.get_solvers(num_qubits__lte=8), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__lte=7), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__lt=7), [self.qpu1, self.qpu2])

        # skip solver if LHS value not defined (None)
        self.assertSolvers(self.client.get_solvers(avg_load__gt=0), [self.qpu1, self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__gte=0), [self.qpu1, self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__lt=1), [self.qpu1, self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__lte=1), [self.qpu1, self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(avg_load=0.7), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__eq=0.7), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load=None), [self.hybrid])
        self.assertSolvers(self.client.get_solvers(avg_load__eq=None), [self.hybrid])

    def test_range_ops(self):
        # value within range
        self.assertSolvers(self.client.get_solvers(num_qubits__within=[3, 7]), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_qubits__within=[3, 5]), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=[2, 4]), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=(2, 4)), [self.qpu1])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=(4, 2)), [self.qpu1])

        # range within (covered by) range
        self.assertSolvers(self.client.get_solvers(num_reads_range__within=(0, 500)), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_reads_range__within=(1, 500)), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_range__within=[0, 2]), [self.software])

        # range covering a value (value included in range)
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=0), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=150), [self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=550), [self.software])
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=1000), [self.software])
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=1001), [])

        # range covering a range
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=(10, 90)), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads_range__covers=(110, 200)), [self.qpu2, self.software])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_range__covers=1.5), [self.software])

    def test_membership_ops(self):
        # property contains
        self.assertSolvers(self.client.get_solvers(supported_problem_types__contains="qubo"), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__contains="undef"), [])
        self.assertSolvers(self.client.get_solvers(supported_problem_types__contains="bqm"), self.unstructured_solvers)
        self.assertSolvers(self.client.get_solvers(couplers__contains=[0, 1]), [self.qpu1, self.qpu2, self.software])
        self.assertSolvers(self.client.get_solvers(couplers__contains=[0, 2]), [self.qpu1, self.qpu2])

        # property in
        self.assertSolvers(self.client.get_solvers(num_qubits__in=[3, 5]), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits__in=[7]), [self.software])
        self.assertSolvers(self.client.get_solvers(num_qubits__in=[]), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_set__contains=1), [self.software])
        self.assertSolvers(self.client.get_solvers(avg_load__in=[None]), [self.hybrid])
        self.assertSolvers(self.client.get_solvers(avg_load__in=[None, 0.1, 0.2, 0.7]), self.solvers)

    def test_set_ops(self):
        # property issubset
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=("qubo", "ising", "bqm", "other")), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=["qubo", "ising", "bqm"]), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=["bqm", "ising", "qubo"]), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset={"ising", "qubo", "bqm"}), self.solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset=("unicorn", "ising", "other")), [])

        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset={"ising", "qubo"}), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(supported_problem_types__issubset={"bqm", "other"}), self.unstructured_solvers)

        # property issuperset
        self.assertSolvers(self.client.get_solvers(qubits__issuperset={0, 1}), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(qubits__issuperset={1, 2}), [self.qpu1, self.qpu2])

        # unhashable types
        self.assertSolvers(self.client.get_solvers(couplers__issuperset=[[0, 1]]), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1)}), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1), (0, 2)}), [self.qpu1, self.qpu2])
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1), (0, 2), (2, 3)}), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(couplers__issuperset={(0, 1), (0, 2), (2, 3), (0, 5)}), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_set__issubset={0, 1, 2}), [self.software])
        self.assertSolvers(self.client.get_solvers(some_set__issuperset={1}), [self.software])

    def test_regex(self):
        self.assertSolvers(self.client.get_solvers(num_reads__regex='.*number.*'), [])
        self.assertSolvers(self.client.get_solvers(num_reads__regex='.*Number.*'), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads__regex='Number.*'), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(num_reads__regex='Number'), [])

        # invalid LHS
        self.assertSolvers(self.client.get_solvers(some_string__regex='x'), [self.software])

    def test_range_boolean_combo(self):
        self.assertSolvers(self.client.get_solvers(num_qubits=3, vfyc=True), [])
        self.assertSolvers(self.client.get_solvers(num_qubits__gte=3, vfyc=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits__lte=4, vfyc=True), [])
        self.assertSolvers(self.client.get_solvers(num_qubits__within=(3, 6), flux_biases=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(num_qubits=5, flux_biases=True), [self.qpu2])

    def test_nested_properties_leaf_lookup(self):
        self.assertSolvers(self.client.get_solvers(topology__type="chimera"), [self.qpu1, self.software])
        self.assertSolvers(self.client.get_solvers(topology__type="pegasus"), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(topology__type__eq="pegasus"), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(topology__shape=[6,6,12]), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(topology__type="chimera", topology__shape__contains=16), [self.qpu1])

    def test_nested_properties_intermediate_key_lookup(self):
        self.assertSolvers(self.client.get_solvers(topology__contains="shape"), self.structured_solvers)
        self.assertSolvers(self.client.get_solvers(topology={"type": "pegasus", "shape": [6, 6, 12]}), [self.qpu2])

    def test_anneal_schedule(self):
        self.assertSolvers(self.client.get_solvers(anneal_schedule__available=True), [self.qpu2])
        self.assertSolvers(self.client.get_solvers(anneal_schedule=True), [self.qpu2])

    def test_order_by_edgecases(self):
        # default: sort by avg_load
        self.assertEqual(self.client.get_solvers(), [self.qpu1, self.qpu2, self.software, self.hybrid])

        # explicit no sort
        self.assertEqual(self.client.get_solvers(order_by=None), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by=''), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by=False), self.solvers)

        # reverse without sorting
        self.assertEqual(self.client.get_solvers(order_by='-'), list(reversed(self.solvers)))

        # invalid type of `order_by`
        with self.assertRaises(TypeError):
            self.client.get_solvers(order_by=1)
        with self.assertRaises(TypeError):
            self.client.get_solvers(order_by=list)

    def test_order_by_respects_default_solver(self):
        """order_by used in isolation should not affect default_solver filters (issue #401)"""

        with Client(endpoint='endpoint', token='token', solver=dict(name='qpu2')) as client:
            # mock the network call to fetch all solvers
            client._fetch_solvers = lambda **kw: self.solvers

            # the default solver was set on client init
            self.assertEqual(client.get_solver(), self.qpu2)

            # the default solver should not change when we add order_by
            self.assertEqual(client.get_solver(order_by='name'), self.qpu2)

        with Client(endpoint='endpoint', token='token', solver=dict(category='qpu')) as client:
            # mock the network call to fetch all solvers
            client._fetch_solvers = lambda **kw: self.solvers

            # test default order_by is avg_load
            self.assertEqual(client.get_solver(), self.qpu1)

            # but we can change it, without affecting solver filters
            self.assertEqual(client.get_solver(order_by='-avg_load'), self.qpu2)

    def test_order_by_in_default_solver(self):
        """order_by can be specified as part of default_solver filters (issue #407)"""

        with Client(endpoint='endpoint', token='token', solver=dict(order_by='id')) as client:
            # mock the network call to fetch all solvers
            client._fetch_solvers = lambda **kw: self.solvers

            # the default solver was set on client init
            self.assertEqual(client.get_solver(), self.hybrid)

            # the default solver can be overridden
            self.assertEqual(client.get_solver(order_by='-name'), self.software)

        with Client(endpoint='endpoint', token='token', solver=dict(qpu=True, order_by='-num_active_qubits')) as client:
            # mock the network call to fetch all solvers
            client._fetch_solvers = lambda **kw: self.solvers

            # the default solver was set on client init
            self.assertEqual(client.get_solver(), self.qpu2)

            # adding order_by doesn't change other default solver features
            self.assertEqual(client.get_solver(order_by='num_active_qubits'), self.qpu1)

    def test_order_by_string(self):
        # sort by Solver inferred properties
        self.assertEqual(self.client.get_solvers(order_by='id'), [self.hybrid, self.qpu1, self.qpu2, self.software])
        self.assertEqual(self.client.get_solvers(order_by='qpu'), [self.software, self.hybrid, self.qpu1, self.qpu2])
        self.assertEqual(self.client.get_solvers(order_by='num_qubits'), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by='num_active_qubits'), [self.software, self.qpu1, self.qpu2, self.hybrid])

        # sort by solver property
        self.assertEqual(self.client.get_solvers(order_by='properties.num_qubits'), self.solvers)

        # sort (and reverse sort) by upper bound of a range property
        self.assertEqual(self.client.get_solvers(order_by='properties.num_reads_range[1]'), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by='-properties.num_reads_range[1]'), [self.hybrid, self.software, self.qpu2, self.qpu1])

        # check solvers with None values for key end up last
        self.assertEqual(self.client.get_solvers(order_by='properties.vfyc'), [self.software, self.qpu2, self.qpu1, self.hybrid])
        self.assertEqual(self.client.get_solvers(order_by='-properties.vfyc'), [self.hybrid, self.qpu1, self.qpu2, self.software])

        # check invalid keys don't fail, and effectively don't sort the list
        self.assertEqual(self.client.get_solvers(order_by='non_existing_key'), self.solvers)
        self.assertEqual(self.client.get_solvers(order_by='-non_existing_key'), list(reversed(self.solvers)))

    def test_order_by_callable(self):
        # sort by Solver inferred properties
        by_name = [self.hybrid, self.qpu1, self.qpu2, self.software]
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.id), by_name)
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.name), by_name)
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.avg_load), [self.qpu1, self.qpu2, self.software, self.hybrid])

        # sort by solver property
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.properties.get('num_qubits')), self.solvers)

        # sort None`s last (here: False, True, None)
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: solver.properties.get('vfyc')), [self.software, self.qpu2, self.qpu1, self.hybrid])

        # test no sort
        self.assertEqual(self.client.get_solvers(order_by=lambda solver: None), self.solvers)


if __name__ == '__main__':
    unittest.main()

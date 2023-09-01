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

import copy
import json
import logging
from collections import abc
from typing import Optional, Union, Literal, Tuple, Dict, Any

from pydantic import BaseModel

from dwave.cloud.config.loaders import update_config
from dwave.cloud.api.constants import (
    DEFAULT_REGION, DEFAULT_METADATA_API_ENDPOINT)

__all__ = ['RequestRetryConfig', 'PollingSchedule', 'ClientConfig',
           'validate_config_v1', 'load_config_v1']

logger = logging.getLogger(__name__)


class GetterMixin:
    """Expose class attributes via item getter protocol."""

    def __getitem__(self, item):
        return getattr(self, item)


class RequestRetryConfig(BaseModel, GetterMixin):
    """Request retry config conformant with :class:`urllib3.Retry` interface."""

    #: Total number of retries to allow. Takes precedence over other counts.
    total: Optional[Union[int, Literal[False]]] = 10

    #: How many connection-related errors to retry on.
    connect: Optional[int] = None

    #: How many times to retry on read errors.
    read: Optional[int] = None

    #: How many redirects to perform. Limit this to avoid infinite redirect loops.
    redirect: Optional[Union[int, Literal[False]]] = 10

    #: How many times to retry on bad status codes.
    status: Optional[int] = None

    #: How many times to retry on other errors.
    other: Optional[int] = None

    #: A backoff factor to apply between attempts after the second try.
    backoff_factor: Optional[float] = 0.01

    #: No backoff will ever be longer than `backoff_max` seconds.
    backoff_max: Optional[float] = 60


class PollingSchedule(BaseModel, GetterMixin):
    """Problem status polling exponential back-off schedule params."""

    #: Duration of the first interval (between first and second poll), in seconds.
    backoff_min: Optional[float] = 0.05

    #: Maximum back-off period, in seconds.
    backoff_max: Optional[float] = 60

    #: Exponential function base. For poll `i`, back-off period (in seconds) is
    #: defined as `backoff_min * (backoff_base ** i)`.
    backoff_base: Optional[float] = 1.3


class ClientConfig(BaseModel, GetterMixin):
    # api region
    metadata_api_endpoint: str = DEFAULT_METADATA_API_ENDPOINT
    region: Optional[str] = DEFAULT_REGION

    # resolved api endpoint
    endpoint: Optional[str] = None
    token: Optional[str] = None

    # [sapi client specific] feature-based solver selection query
    client: Optional[str] = 'base'
    solver: Optional[Dict[str, Any]] = None

    # [sapi client specific] poll back-off schedule defaults [sec]
    polling_schedule: PollingSchedule = PollingSchedule()
    polling_timeout: Optional[float] = None

    # general http(s) connection params
    cert: Optional[Union[str, Tuple[str, str]]] = None
    headers: Optional[Dict[str, str]] = None
    proxy: Optional[str] = None

    # specific connection options
    permissive_ssl: bool = False
    connection_close: bool = False

    # api request retry params
    request_retry: RequestRetryConfig = RequestRetryConfig()
    request_timeout: Optional[Union[float, Tuple[float, float]]] = (60, 120)


def validate_config_v1(raw_config: dict) -> ClientConfig:
    """Validate raw config data (as obtained from the current INI-style config
    via :func:`~dwave.cloud.config.loaders.load_config`) and construct a
    :class:`ClientConfig` object.

    Note: v1 config data is a flat dictionary of config options we transform
    and structure to fit the :class:`ClientConfig` model.
    """

    config = raw_config.copy()

    # parse string headers
    headers = config.get('headers')
    if not headers:
        headers_dict = {}
    elif isinstance(headers, abc.Mapping):
        headers_dict = headers
    elif isinstance(headers, str):
        try:
            # valid  headers = "Field-1: value-1\nField-2: value-2"
            headers_dict = {key.strip(): val.strip()
                            for key, val in [line.split(':') for line in
                                             headers.strip().split('\n')]}
        except Exception as e:
            logger.debug("Invalid headers: %r", headers)
            headers_dict = {}
    else:
        raise ValueError("HTTP headers expected in a dict, or a string")
    logger.debug("parsed headers: %r", headers_dict)
    config['headers'] = headers_dict

    # parse optional client certificate
    # (if string, path to ssl client cert file (.pem). If tuple, (‘cert’, ‘key’) pair.)
    client_cert = config.get('client_cert')
    client_cert_key = config.get('client_cert_key')
    if client_cert_key is not None:
        if client_cert is not None:
            client_cert = (client_cert, client_cert_key)
        else:
            raise ValueError(
                "Client certificate key given, but the cert is missing")
    logger.debug("parsed client cert: %r", client_cert)
    config['cert'] = client_cert

    # parse solver
    solver = config.get('solver')
    if isinstance(solver, str):
        solver = solver.strip()
    if not solver:
        solver_def = {}
    elif isinstance(solver, abc.Mapping):
        solver_def = solver
    elif isinstance(solver, str):
        # support features dict encoded as JSON in our config INI file
        try:
            solver_def = json.loads(solver)
        except Exception:
            # unparseable (or non-dict) json, assume string name for solver
            # we'll deprecate this eventually, but for now just convert it to
            # features dict (equality constraint on full solver name)
            logger.debug("Invalid solver JSON, assuming string name: %r", solver)
            solver_def = dict(name__eq=solver)
        else:
            # if valid json, has to be a dict
            if not isinstance(solver_def, abc.Mapping):
                logger.debug("Non-dict solver JSON, assuming string name: %r", solver)
                solver_def = dict(name__eq=solver)
    else:
        raise ValueError("Expecting a features dictionary or a string name for 'solver'")
    logger.debug("Parsed solver definition: %r", solver_def)
    config['solver'] = solver_def

    # polling
    prefix = 'poll_'
    config['polling_schedule'] = {k[len(prefix):]: v for k, v in config.items()
                                  if k.startswith(prefix)}

    # retry
    prefix = 'http_retry_'
    config['request_retry'] = {k[len(prefix):]: v for k, v in config.items()
                               if k.startswith(prefix)}

    return ClientConfig.model_validate(config)


# XXX: replace with model defaults?
_V1_CONFIG_DEFAULTS = {
    'client': 'base',
    'metadata_api_endpoint': DEFAULT_METADATA_API_ENDPOINT,
    'region': DEFAULT_REGION,
    'endpoint': None,
    'token': None,
    'solver': None,
    'proxy': None,
    'permissive_ssl': False,
    'request_timeout': 60,
    'polling_timeout': None,
    'connection_close': False,
    'headers': None,
    'client_cert': None,
    'client_cert_key': None,
    # poll back-off schedule defaults
    'poll_backoff_min': 0.05,
    'poll_backoff_max': 60,
    'poll_backoff_base': 1.3,
    # idempotent http requests retry params
    'http_retry_total': 10,
    'http_retry_connect': None,
    'http_retry_read': None,
    'http_retry_redirect': None,
    'http_retry_status': None,
    'http_retry_backoff_factor': 0.01,
    'http_retry_backoff_max': 60,
}

def load_config_v1(raw_config: dict, defaults: Optional[dict] = None) -> ClientConfig:
    if defaults is None:
        defaults = {}

    config = copy.deepcopy(_V1_CONFIG_DEFAULTS)
    update_config(config, defaults)
    logger.debug("defaults (global + local) = %r", config)

    update_config(config, raw_config)
    logger.debug("raw config with defaults = %r", config)

    return validate_config_v1(config)

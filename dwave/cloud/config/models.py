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

from __future__ import annotations

import ast
import copy
import enum
import logging
import orjson
from collections import abc
from typing import Optional, Union, Literal, Any, Annotated

import urllib3
from pydantic import BaseModel, BeforeValidator, NonNegativeInt

from dwave.cloud.config import constants
from dwave.cloud.config.loaders import update_config, _solver_id_as_identity

__all__ = ['RequestRetryConfig', 'ClientConfig',
           'BackoffPollingSchedule', 'LongPollingSchedule',
           'validate_config_v1', 'dump_config_v1', 'load_config_v1']

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
    backoff_max: Optional[float] = 60.0

    def to_urllib3_retry(self) -> urllib3.Retry:
        """Return :class:`urllib3.Retry` configuration matching the model.
        """
        # dev note: after we drop support for urllib3<2, this whole function can
        # be replaced with `Retry(**self.model_dump())`

        params = self.model_dump()

        # `urllib3<2` doesn't support `backoff_max` in `Retry()`
        del params['backoff_max']

        retry = urllib3.Retry(**params)

        if self.backoff_max is not None:
            # handle `urllib3>=1.21.1,<1.27` AND `urllib3>=1.21.1,<3`
            retry.BACKOFF_MAX = retry.backoff_max = self.backoff_max

        return retry


class PollingStrategy(str, enum.Enum):
    BACKOFF = "backoff"
    LONG_POLLING = "long-polling"


class BackoffPollingSchedule(BaseModel):
    """Problem status polling exponential back-off schedule params."""

    #: Polling with exponential backoff
    strategy: Literal[PollingStrategy.BACKOFF] = PollingStrategy.BACKOFF

    #: Duration of the first interval (between first and second poll), in seconds.
    backoff_min: Optional[float] = 0.05

    #: Maximum back-off period, in seconds.
    backoff_max: Optional[float] = 60.0

    #: Exponential function base. For poll `i`, back-off period (in seconds) is
    #: defined as `backoff_min * (backoff_base ** i)`.
    backoff_base: Optional[float] = 1.3


class LongPollingSchedule(BaseModel):
    """Problem status long polling params."""

    #: Long polling strategy
    strategy: Literal[PollingStrategy.LONG_POLLING] = PollingStrategy.LONG_POLLING

    #: Maximum duration a long polling connection is kept open, in whole number of seconds.
    #: Note: SAPI requires a non-negative integer wait_time
    wait_time: Optional[NonNegativeInt] = 30

    #: Pause between two successive long polling connections, in seconds.
    pause: Optional[float] = 0.0


def _literal_eval(obj):
    if isinstance(obj, str):
        return ast.literal_eval(obj)
    return obj


class ClientConfig(BaseModel, GetterMixin):
    # api region
    metadata_api_endpoint: Optional[str] = constants.DEFAULT_METADATA_API_ENDPOINT
    region: Optional[str] = None

    # resolved api endpoint
    leap_api_endpoint: Optional[str] = None
    endpoint: Optional[str] = None
    token: Optional[str] = None

    # oauth 2.0 support
    leap_client_id: Optional[str] = None

    # [sapi client specific] feature-based solver selection query
    client: Optional[str] = None
    solver: Optional[abc.Mapping[str, Any]] = None

    # [sapi client specific] polling schedule defaults [sec]
    # note: discriminated unions are faster than unions, but we want to be able
    # to set a defautl value when `strategy` is not specified
    polling_schedule: Optional[Union[BackoffPollingSchedule,
                                     LongPollingSchedule]] = LongPollingSchedule()
    polling_timeout: Optional[float] = None

    # [sapi client specific] preemptive compression on qpu problem upload
    compress_qpu_problem_data: Optional[bool] = True

    # general http(s) connection params
    cert: Optional[Union[str, tuple[str, str]]] = None
    headers: Optional[abc.Mapping[str, str]] = None
    proxy: Optional[str] = None

    # specific connection options
    permissive_ssl: Optional[bool] = False
    connection_close: Optional[bool] = False

    # api request retry params
    request_retry: Optional[RequestRetryConfig] = RequestRetryConfig()
    request_timeout: Annotated[Optional[Union[float, tuple[float, float]]],
                               BeforeValidator(_literal_eval)] = (60.0, 120.0)


def validate_config_v1(raw_config: abc.Mapping) -> ClientConfig:
    """Validate raw config data (as obtained from the current INI-style config
    via :func:`~dwave.cloud.config.loaders.load_config`) and construct a
    :class:`ClientConfig` object.

    Note: v1 config data is a flat dictionary of config options we transform
    and structure to fit the :class:`ClientConfig` model.
    """

    # shallow copy
    config = dict(raw_config)

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
    logger.trace("parsed headers: %r", headers_dict)
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
    logger.trace("parsed client cert: %r", client_cert)
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
            solver_def = orjson.loads(solver)
        except Exception:
            # unparseable (or non-dict) json, assume solver identity as string
            logger.info("Invalid solver JSON, parsing as string identity: %r", solver)
            try:
                identity = _solver_id_as_identity(solver)
            except Exception:
                logger.debug("Unknown solver identity string format, falling back to string")
                solver_def = dict(id__eq=solver)
            else:
                # use identity equality constraint only when full identity is indeed
                # specified; this is to prevent mismatches when user specifies just a name
                # (which is a partial identity for a structured solver)
                if identity.get('version'):
                    solver_def = dict(identity__eq=identity)
                else:
                    solver_def = dict(name__eq=identity['name'])
        else:
            # if valid json, has to be a dict
            if not isinstance(solver_def, abc.Mapping):
                logger.debug("Non-dict solver JSON, assuming string identity: %r", solver)
                solver_def = dict(id__eq=solver)
    else:
        raise ValueError("Expecting a features dictionary or a string identity for 'solver'")
    logger.trace("parsed solver definition: %r", solver_def)
    config['solver'] = solver_def

    # polling
    prefix = 'poll_'
    config['polling_schedule'] = {k[len(prefix):]: v for k, v in config.items()
                                  if k.startswith(prefix)}

    # set a default polling strategy
    config['polling_schedule'].setdefault('strategy', PollingStrategy.BACKOFF.value)

    # retry
    prefix = 'http_retry_'
    config['request_retry'] = {k[len(prefix):]: v for k, v in config.items()
                               if k.startswith(prefix)}

    return ClientConfig.model_validate(config)


def dump_config_v1(config: ClientConfig) -> dict:
    """Dump :class:`ClientConfig` to a flat dict of v1 config options.

    Note: consider this to be an approximate inverse of :func:`validate_config_v1`::

        dump_config_v1(validate_config_v1(raw_config)) ~= raw_config

    The inverse might not be exact due to field type casting during validation.
    E.g. original boolean value of 'off' or '0' will be dumped as 'false'.
    """

    raw_config = config.model_dump()

    # invert transformations during validation
    if h := raw_config['headers']:
        raw_config['headers'] = '\n'.join(f"{k}: {v}" for k, v in h.items())

    cert_pair = (None, None)
    if c := raw_config.get('cert'):
        if isinstance(c, tuple):
            cert_pair = c
        else:   # assume string
            cert_pair = (c, None)
    raw_config.update(zip(("client_cert", "client_cert_key"), cert_pair))
    del raw_config['cert']

    raw_config['solver'] = orjson.dumps(raw_config['solver']).decode('utf8')

    # expand/translate polling
    raw_config.update({f"poll_{k}": v for k, v in raw_config['polling_schedule'].items()})
    del raw_config['polling_schedule']

    # expand/translate retry config
    raw_config.update({f"http_retry_{k}": v for k, v in raw_config['request_retry'].items()})
    del raw_config['request_retry']

    return raw_config


# XXX: replace with model defaults?
_V1_CONFIG_DEFAULTS = {
    'client': 'base',
    'metadata_api_endpoint': constants.DEFAULT_METADATA_API_ENDPOINT,
    'leap_api_endpoint': constants.DEFAULT_LEAP_API_ENDPOINT,
    'region': constants.DEFAULT_REGION,
    'endpoint': None,
    'token': None,
    'leap_client_id': None,
    'solver': None,
    'proxy': None,
    'permissive_ssl': False,
    'request_timeout': 60,
    'polling_timeout': None,
    'connection_close': False,
    'compress_qpu_problem_data': True,
    'headers': None,
    'client_cert': None,
    'client_cert_key': None,
    'poll_strategy': 'long-polling',
    # poll back-off schedule defaults
    'poll_backoff_min': 0.05,
    'poll_backoff_max': 60,
    'poll_backoff_base': 1.3,
    # long polling parameters
    'poll_wait_time': 30,
    'poll_pause': 0,
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

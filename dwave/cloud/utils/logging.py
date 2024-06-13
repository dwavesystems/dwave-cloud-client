# Copyright 2024 D-Wave Inc.
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

"""Logging utilities for private and Ocean-internal use."""

import datetime
import inspect
import io
import json
import logging
import os
import re
import sys
import typing

__all__ = []


class ISOFormatter(logging.Formatter):
    # target timezone, e.g. `datetime.timezone.utc`, or `None` for naive timestamp
    as_tz: typing.Optional[datetime.timezone] = None

    def __init__(self, *args, as_tz: typing.Optional[datetime.timezone] = None, **kwargs):
        self.as_tz = as_tz
        super().__init__(*args, **kwargs)

    def formatTime(self, record: logging.LogRecord, datefmt: typing.Optional[str] = None) -> str:
        return datetime.datetime.fromtimestamp(record.created, tz=self.as_tz).isoformat()


class FilteredSecretsFormatter(logging.Formatter):
    """Logging formatter that filters out secrets (like Solver API tokens).

    Note: we assume, for easier disambiguation, a secret/token is prefixed with
    a short alphanumeric string, and comprises 40 or more hex digits.
    """

    # prefixed 160-bit+ hex tokens (sapi token format: `A{2,4}-X{40,}`)
    _SAPI_TOKEN_PATTERN = re.compile(
        r'\b([0-9A-Za-z]{2,4}-[0-9A-Fa-f]{3})([0-9A-Fa-f]{34,})([0-9A-Fa-f]{3})\b')
    # 128-bit+ hex tokens (`X{32,}`)
    _HEX_TOKEN_PATTERN = re.compile(
        r'\b([0-9A-Fa-f]{3})([0-9A-Fa-f]{26,})([0-9A-Fa-f]{3})\b')
    # 128-bit uuid tokens (`X{8}-X{4}-X{4}-X{4}-X{12}`)
    _UUID_TOKEN_PATTERN = re.compile(
        r'\b([0-9A-Fa-f]{3})([0-9A-Fa-f]{5}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{9})([0-9A-Fa-f]{3})\b')

    def format(self, record: logging.LogRecord) -> str:
        output = super().format(record)
        output = re.sub(self._SAPI_TOKEN_PATTERN, r'\1...\3', output)
        output = re.sub(self._HEX_TOKEN_PATTERN, r'\1...\3', output)
        output = re.sub(self._UUID_TOKEN_PATTERN, r'\1...\3', output)
        return output


class JSONFormatter(ISOFormatter):
    def format(self, record: logging.LogRecord) -> str:
        super().format(record)
        # filter out message template and potentially unserializable args
        rec = record.__dict__.copy()
        del rec['args']
        del rec['msg']
        return json.dumps(rec)


def parse_loglevel(level_name: typing.Union[str, int],
                   default: int = logging.NOTSET) -> int:
    """Resolve numeric and symbolic log level names to numeric levels."""

    try:
        level_name = str(level_name or '').strip().lower()
    except:
        return default

    # note: in py 3.11+ we can use `logging.getLevelNamesMapping()` instead
    known_levels = {
        'notset': logging.NOTSET,
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'warn': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
        'fatal': logging.CRITICAL
    }

    # support TRACE level, but make sure this function works without it
    if hasattr(logging, 'TRACE'):
        known_levels.update(trace=logging.TRACE)

    try:
        level = int(level_name)
    except ValueError:
        level = known_levels.get(level_name, default)

    return level


def set_loglevel(logger: logging.Logger, level: typing.Union[str, int]) -> None:
    level = parse_loglevel(level)
    logger.setLevel(level)
    logger.info("Log level for %r namespace set to %r", logger.name, level)


def add_loglevel(name: str, value: int) -> None:
    """Create a new globally available logging loglevel and a logging method
    on the base ``logging.Logger`` class.
    """

    name = name.upper()

    setattr(logging, name, value)
    logging.addLevelName(value, name)

    def log_method(logger, message, *args, **kwargs):
        logger.log(value, message, *args, **kwargs)

    setattr(logging.Logger, name.lower(), log_method)


def configure_logging(logger: typing.Optional[logging.Logger] = None,
                      *,
                      level: typing.Union[str, int] = logging.WARNING,
                      filter_secrets: bool = True,
                      output_stream: typing.Optional[io.IOBase] = None,
                      in_utc: bool = False,
                      structured_output: bool = False,
                      handler_level: typing.Optional[int] = None,
                      additive: bool = False,
                      ) -> logging.Logger:
    """Configure cloud-client's `dwave.cloud` base logger.

    Logging output from the cloud-client is suppressed by default. This utility
    function can be used to quickly setup basic logging from the library.

    .. note::
       This function is currently intended for internal/private use only.

    .. versionadded:: 0.12.0
       Explicit optional logging configuration. Previously, logger was minimally
       configured by default.
    """

    level = parse_loglevel(level)

    if logger is None:
        logger = logging.getLogger('dwave.cloud')
    if output_stream is None:
        output_stream = sys.stderr
    if handler_level is None:
        handler_level = level

    format = dict(
        fmt='%(asctime)s %(name)s %(levelname)s %(threadName)s [%(funcName)s] %(message)s',
        as_tz=datetime.timezone.utc if in_utc else None,
    )

    if structured_output:
        formatter_base = JSONFormatter
    else:
        formatter_base = ISOFormatter

    if filter_secrets:
        class Formatter(FilteredSecretsFormatter, formatter_base):
            pass
    else:
        Formatter = formatter_base

    if not additive:
        # make sure handlers are not accumulated
        while len(logger.handlers):
            logger.removeHandler(logger.handlers[-1])

    formatter = Formatter(**format)
    handler = logging.StreamHandler(stream=output_stream)
    handler.setFormatter(formatter)
    handler.setLevel(handler_level)

    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


def configure_logging_from_env(logger: logging.Logger) -> bool:
    """Use ``DWAVE_LOG_LEVEL`` (name or value) and ``DWAVE_LOG_FORMAT`` ("json"
    for structured output) environment variables to configure logging.
    """

    log_level = os.getenv('DWAVE_LOG_LEVEL', os.getenv('dwave_log_level'))
    log_format = os.getenv('DWAVE_LOG_FORMAT', os.getenv('dwave_log_format', ''))

    if not log_level:
        return False

    configure_logging(logger, additive=False,
                      level=parse_loglevel(log_level),
                      structured_output=(log_format.strip().lower() == 'json'))

    return True


def pretty_argvalues():
    """Pretty-formatted function call arguments, from the caller's frame."""
    return inspect.formatargvalues(*inspect.getargvalues(inspect.currentframe().f_back))

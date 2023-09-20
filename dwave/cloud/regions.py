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

import re
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit

from pydantic import TypeAdapter

from dwave.cloud import api
from dwave.cloud.api.constants import (
    DEFAULT_REGION, DEFAULT_SOLVER_API_ENDPOINT, DEFAULT_LEAP_API_ENDPOINT)
from dwave.cloud.config.models import ClientConfig, validate_config_v1
from dwave.cloud.utils import cached

__all__ = ['get_regions']

logger = logging.getLogger(__name__)

_REGIONS_CACHE_MAXAGE = 86400   # 1 day


@cached.ondisk(maxage=_REGIONS_CACHE_MAXAGE, key='cache_key')
def _fetch_available_regions(cache_key: Any, config: ClientConfig) -> Dict[str, str]:
    logger.debug("Fetching available regions from the Metadata API at %r",
                 config.metadata_api_endpoint)

    with api.Regions.from_config(config) as regions:
        regions = regions.list_regions()
        data = TypeAdapter(Any).dump_python(regions)

    logger.debug("Received region metadata: %r", data)
    return data


def get_regions(config: Optional[Union[ClientConfig, str, dict]] = None,
                *,
                refresh: bool = False,
                maxage: Optional[float] = None,
                no_cache: bool = False) -> List[api.models.Region]:
    """Retrieve available API regions.

    Args:
        config:
            Client configuration used for requests to Metadata API. Given as
            a :class:`~dwave.cloud.config.models.ClientConfig` object it defines
            connection parameters in full. Raw config (full or partial) loaded
            from a file/env with :meth:`~dwave.cloud.config.loaders.load_config`
            can be passed in as a :class:`dict` and it will be validated as
            :class:`~dwave.cloud.config.models.ClientConfig`. A string ``config``
            value is interpreted as ``metadata_api_endpoint``.
        refresh:
            Force regions cache refresh.
        maxage:
            Maximum allowed age of cached regions metadata.
        no_cache:
            Don't use cache.

    Returns:
        List of :class:`~dwave.cloud.api.models.Region` objects, each describing
        one region.

    .. versionadded:: 0.11.0
        Added :func:`.get_regions`.

    """
    if isinstance(config, str):
        config = ClientConfig(metadata_api_endpoint=config)
    elif isinstance(config, dict):
        config = validate_config_v1(config)
    elif config is None:
        config = ClientConfig()
    elif not isinstance(config, ClientConfig):
        raise TypeError(f"'config' type {type(config).__name__!r} not supported")

    # make sure we don't cache by noise in config
    cache_key = (config.metadata_api_endpoint, config.cert, config.headers,
                 config.proxy, config.permissive_ssl,
                 config.request_retry, config.request_timeout)

    fetch = partial(_fetch_available_regions,
        cache_key=cache_key, config=config, refresh_=refresh, maxage_=maxage)

    if no_cache:
        fetch = cached.disabled()(fetch)

    try:
        obj = fetch()
    except api.exceptions.RequestError as exc:
        logger.debug("Metadata API unavailable", exc_info=True)
        raise ValueError(
            f"Metadata API unavailable at {config.metadata_api_endpoint!r}")

    regions = TypeAdapter(List[api.models.Region]).validate_python(obj)
    return regions


def _infer_leap_api_endpoint(solver_api_endpoint: str,
                             region_code: Optional[str] = None) -> str:
    # shim until metadata api includes leap endpoint in region data

    parts = urlsplit(solver_api_endpoint)

    # infer region code if necessary
    if not region_code:
        if m := re.match(r'([a-z]+-[a-z]+-\d+)\.', parts.netloc, re.I):
            region_code = m.group(1)
        else:
            region_code = DEFAULT_REGION

    # update path
    path = urlsplit(DEFAULT_LEAP_API_ENDPOINT).path

    # strip region from netloc
    netloc = parts.netloc
    if parts.netloc.startswith(f"{region_code}."):
        netloc = parts.netloc[len(region_code)+1:]

    return parts._replace(netloc=netloc, path=path).geturl()


def resolve_endpoints(config: ClientConfig, *, inplace: bool = False) -> ClientConfig:
    """Use region and endpoint from config to resolve all config endpoints.

    Explicit endpoint will override the region (i.e. region extension is
    backwards-compatible).

    Regional endpoint is fetched from Metadata API. If Metadata API is not
    available, default global endpoint is used.
    """

    # Note: this function is currently indirectly tested with `Client`
    # multi-region support tests.

    if not inplace:
        config = config.model_copy(deep=True)

    def set_defaults(config):
        config.region = DEFAULT_REGION
        config.endpoint = DEFAULT_SOLVER_API_ENDPOINT
        config.leap_api_endpoint = DEFAULT_LEAP_API_ENDPOINT

    # for backward-compat: endpoint overrides region
    if config.endpoint:
        if not config.leap_api_endpoint:
            config.leap_api_endpoint = _infer_leap_api_endpoint(
                solver_api_endpoint=config.endpoint, region_code=config.region)
        return config

    if not config.region:
        set_defaults(config)
        return config

    try:
        regions = get_regions(config=config)
        regions = {r.code: r for r in regions}
    except (api.exceptions.RequestError, ValueError) as exc:
        logger.warning("Failed to fetch available regions: %r. "
                       "Using default endpoints.", exc)
        set_defaults(config)
        return config

    if config.region not in regions:
        raise ValueError(f"Region {config.region!r} unknown. "
                         f"Try one of {list(regions.keys())!r}.")

    region = regions[config.region]
    config.endpoint = region.solver_api_endpoint
    config.leap_api_endpoint = region.leap_api_endpoint
    return config

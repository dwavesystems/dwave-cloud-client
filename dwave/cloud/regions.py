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

import logging
import re
from typing import Optional, Union
from urllib.parse import urlsplit

from dwave.cloud import api
from dwave.cloud.config.constants import (
    DEFAULT_REGION, DEFAULT_SOLVER_API_ENDPOINT, DEFAULT_LEAP_API_ENDPOINT,
    DEFAULT_METADATA_API_ENDPOINT)
from dwave.cloud.config.models import ClientConfig, validate_config_v1

__all__ = ['get_regions']

logger = logging.getLogger(__name__)

_DEFAULT_REGIONS_CACHE_CONFIG = dict(
    enabled=True,
    maxage=7 * 86400    # 7 days
)


def _fetch_available_regions(config: ClientConfig, **kwargs) -> list[api.models.Region]:
    logger.debug("Fetching available regions from the Metadata API at %r",
                 config.metadata_api_endpoint)

    with api.Regions.from_config(
            config, cache=_DEFAULT_REGIONS_CACHE_CONFIG) as regions:
        regions = regions.list_regions(**kwargs)

    logger.debug("Received region metadata: %r", regions)
    return regions


def get_regions(config: Optional[Union[ClientConfig, str, dict]] = None,
                *,
                refresh: bool = False,
                maxage: Optional[float] = None,
                no_cache: bool = False) -> list[api.models.Region]:
    """Retrieve available Solver API regions.

    Args:
        config:
            Client configuration used for requests to Metadata API. Given as
            a :class:`~dwave.cloud.config.models.ClientConfig` object, it defines
            connection parameters in full. Raw config (full or partial) loaded
            from a file/env with :meth:`~dwave.cloud.config.loaders.load_config`
            can be passed in as a :class:`dict` and is validated as
            :class:`~dwave.cloud.config.models.ClientConfig`. A string ``config``
            value is interpreted as ``metadata_api_endpoint``.
        refresh:
            Force regions cache refresh.
        maxage:
            Default maximum allowed age, in seconds, of cached regions metadata.
            Overridden by Metadata API cache control.
        no_cache:
            Do not use cache.

    Returns:
        List of :class:`~dwave.cloud.api.models.Region` objects, each describing
        one region.

    .. versionadded:: 0.11.0
        Added :func:`.get_regions`.

    .. versionchanged:: 0.12.2
        ``maxage`` parameter is now just a default value for the maximum allowed
        cache age, and it's typically overridden by regions API's cache control.

    """
    if isinstance(config, str):
        config = ClientConfig(metadata_api_endpoint=config)
    elif isinstance(config, dict):
        config = validate_config_v1(config)
    elif config is None:
        config = ClientConfig()
    elif not isinstance(config, ClientConfig):
        raise TypeError(f"'config' type {type(config).__name__!r} not supported")

    try:
        return _fetch_available_regions(
            config=config, refresh_=refresh, maxage_=maxage, no_cache_=no_cache)

    except api.exceptions.RequestError:
        logger.debug("Metadata API unavailable", exc_info=True)
        raise ValueError(
            f"Metadata API unavailable at {config.metadata_api_endpoint!r}")


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


def _infer_solver_api_endpoint(leap_api_endpoint: str) -> str:
    # solver_api_endpoint from leap_api_endpoint

    # sapi path
    path = urlsplit(DEFAULT_SOLVER_API_ENDPOINT).path

    return urlsplit(leap_api_endpoint)._replace(path=path).geturl()


def resolve_endpoints(config: ClientConfig, *, inplace: bool = False,
                      shortcircuit: bool = True) -> ClientConfig:
    """Use region and endpoint from configuration to resolve all endpoints.

    Explicit endpoint overrides the region (i.e. region extension is
    backwards-compatible).

    Regional endpoint is fetched from Metadata API. If Metadata API is not
    available, default global endpoint is used.
    """

    if not inplace:
        config = config.model_copy(deep=True)

    def set_defaults(config):
        config.region = DEFAULT_REGION
        config.endpoint = DEFAULT_SOLVER_API_ENDPOINT
        config.leap_api_endpoint = DEFAULT_LEAP_API_ENDPOINT

    # we always need metadata endpoint
    if config.metadata_api_endpoint is None:
        config.metadata_api_endpoint = DEFAULT_METADATA_API_ENDPOINT

    # for backward-compat: endpoint overrides region
    if config.endpoint:
        if not config.leap_api_endpoint:
            config.leap_api_endpoint = _infer_leap_api_endpoint(
                solver_api_endpoint=config.endpoint, region_code=config.region)
        return config

    # for consistency: any endpoint overrides region
    if config.leap_api_endpoint:
        if not config.endpoint:
            config.endpoint = _infer_solver_api_endpoint(
                leap_api_endpoint=config.leap_api_endpoint)
        return config

    if not config.region:
        config.region = DEFAULT_REGION

    # short-circuit a Metadata API hit if we just need default values
    if (shortcircuit and config.region == DEFAULT_REGION
        and config.metadata_api_endpoint == DEFAULT_METADATA_API_ENDPOINT):
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

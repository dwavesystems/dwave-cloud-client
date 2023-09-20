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
from functools import partial
from typing import Any, Dict, List, Optional, Union

from pydantic import TypeAdapter

from dwave.cloud import api
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

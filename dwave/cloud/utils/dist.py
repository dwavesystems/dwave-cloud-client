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

"""Distribution utilities for private and Ocean-internal use.

.. versionchanged:: 0.12.0
   These functions previously lived under ``dwave.cloud.utils``.
"""

from collections import OrderedDict
from importlib.metadata import EntryPoint, Distribution, PackageNotFoundError
from typing import Union

from importlib_metadata import entry_points
from packaging.requirements import Requirement

__all__ = ['get_distribution', 'PackageNotFoundError', 'VersionNotFoundError']


def get_contrib_config() -> list[EntryPoint]:
    """Return all registered contrib (non-open-source) Ocean packages."""

    # Note: we use `entry_points` from `importlib_metadata>=5` to simplify access
    # and use py312 semantics. See "compatibility note" in `importlib.metadata`
    # docs for entry points.
    contrib = [ep.load() for ep in entry_points(group='dwave_contrib')]
    return contrib


def get_contrib_packages() -> OrderedDict[str, dict]:
    """Combine all contrib packages in an ordered dict. Assumes package names
    are unique.
    """

    contrib = get_contrib_config()

    packages = OrderedDict()
    for dist in contrib:
        for pkg in dist:
            packages[pkg['name']] = pkg

    return packages


class VersionNotFoundError(Exception):
    """Package version requirement not satisfied."""


def get_distribution(requirement: Union[str, Requirement],
                     prereleases: bool = True) -> Distribution:
    """Returns :class:`~importlib.metadata.Distribution` for a matching
    `requirement` specification.

    Note: this function re-implements :func:`pkg_resources.get_distribution`
    functionality for py38+ (including py312, where setuptools/pkg_resources
    is not available by default).

    Args:
        requirement:
            Package dependency requirement according to PEP-508, given as string,
            or :class:`~packaging.requirements.Requirement`.
        prereleases:
            Boolean flag to control if installed prereleases are allowed.

    Raises:
        :class:`~importlib.metadata.PackageNotFoundError`:
            Package by name not found.
        :class:`~dwave.cloud.utils.dist.VersionNotFoundError`:
            Version of the package found (distribution) does not match the
            requirement.
    """

    if not isinstance(requirement, Requirement):
        requirement = Requirement(requirement)

    dist = Distribution.from_name(requirement.name)

    if not requirement.specifier.contains(dist.version, prereleases=prereleases):
        raise VersionNotFoundError(
            f"Package {dist.name!r} version {dist.version} "
            f"does not match {requirement.specifier!s}")

    return dist

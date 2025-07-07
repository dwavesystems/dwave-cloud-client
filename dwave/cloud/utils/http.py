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

"""HTTP/requests utilities for private and Ocean-internal use.

.. versionchanged:: 0.12.0
   These functions previously lived under ``dwave.cloud.utils``.
"""

import platform
import sys
from typing import Optional
from urllib.parse import urljoin

import requests

from dwave.cloud.package_info import __packagename__, __version__

__all__ = []


class PretimedHTTPAdapter(requests.adapters.HTTPAdapter):
    """Sets a default timeout for all adapter (think session) requests. It is
    overridden with per-request timeout. But it can not be reset back to
    infinite wait (``None``).

    Usage:

        s = requests.Session()
        s.mount("http://", PretimedHTTPAdapter(timeout=5))
        s.mount("https://", PretimedHTTPAdapter(timeout=5))

        s.get('http://httpbin.org/delay/6')                 # -> timeouts after 5sec
        s.get('http://httpbin.org/delay/6', timeout=10)     # -> completes after 6sec

    The alternative is to set ``timeout`` on each request manually/explicitly,
    subclass ``Session``, or monkeypatch ``Session.request()``.
    """

    def __init__(self, timeout=None, *args, **kwargs):
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, *args, **kwargs):
        # can't use setdefault because caller always sets timeout kwarg
        kwargs['timeout'] = self.timeout
        return super().send(*args, **kwargs)


class TimeoutingHTTPAdapter(PretimedHTTPAdapter):
    """Alias for :class:`~dwave.cloud.utils.PretimedHTTPAdapter`. Deprecated,
    but retained for backward compatibility.
    """


# Note: BaseUrlSession is taken from https://github.com/requests/toolbelt under
# an Apache 2 license. This simple extension didn't warrant a new dependency.
# If we later decide to use additional features from `requests-toolbelt`,
# remove it from here.

class BaseUrlSessionMixin:
    """A :class:`~requests.Session` mixin that allows setting of a base URL for
    all session requests.

    See also :class:`.BaseUrlSession`.

    Args:
        base_url:
            Base URL.

    """
    base_url = None

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        if base_url:
            self.base_url = base_url
        super().__init__(**kwargs)

    def request(self, method: str, url: str, *args, **kwargs) -> requests.Response:
        """Send the request after generating the complete URL."""
        url = self.create_url(url)
        return super().request(method, url, *args, **kwargs)

    def create_url(self, url: str) -> str:
        """Create the URL based off this partial path."""
        return urljoin(self.base_url, url)


class BaseUrlSession(BaseUrlSessionMixin, requests.Session):
    """A Session with a URL that all requests will use as a base.

    See also :class:`.BaseUrlSessionMixin`.

    Example::

        session = BaseUrlSession(base_url='http://example.com/api/')
        session.get('version')

    """


def user_agent(name: Optional[str] = None,
               version: Optional[str] = None,
               *,
               include_platform_tags: bool = True) -> str:
    """Return User-Agent ~ "name/version language/version interpreter/version os/version".

    Args:
        name:
            Package name, primary UA component name.
        version:
            Package version, primary UA component version.
        include_platform_tags:
            Look for, query and include externally-contributed platform tags
            (via ``dwave.common.platform.tags`` entrypoint).
            See :func:`dwave.cloud.utils.get_platform_tags`.

    Return:
        User-Agent string composed of "key/value" pairs (joined with a space
        character), for following components: package, language, interpreter,
        machine, system and platform.
    """

    def _interpreter():
        name = platform.python_implementation()
        version = platform.python_version()
        if name == 'PyPy':
            version = '.'.join(map(str, sys.pypy_version_info[:3]))
        full_version = [version]
        is_64bits = sys.maxsize > 2**32
        if is_64bits:
            full_version.append('64bit')
        return name, "-".join(full_version)

    tags = []

    if name and version:
        tags.append((name, version))

    tags.extend([
        ("python", platform.python_version()),
        _interpreter(),
        ("machine", platform.machine() or 'unknown'),
        ("system", platform.system() or 'unknown'),
        ("platform", platform.platform() or 'unknown'),
    ])

    # add platform-specific tags
    if include_platform_tags:
        tags.extend(platform_tags())

    return ' '.join("{}/{}".format(name, version) for name, version in tags)


# defined as a function rather than constant because env might change during runtime
def default_user_agent() -> str:
    """Default user agent string to be used consistently across client(s)."""
    return user_agent(
        name=__packagename__, version=__version__, include_platform_tags=False)


def platform_tags() -> list[str]:
    """Return a list of platform tags generated from registered entry points."""

    # import only when needed, as it's pretty slow! (~10ms)
    from importlib_metadata import entry_points

    fs = [ep.load() for ep in entry_points(group='dwave.common.platform.tags')]
    tags = list(filter(None, [f() for f in fs]))
    return tags

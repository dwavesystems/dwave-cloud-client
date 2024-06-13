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

import json
import logging
import inspect

from collections import OrderedDict
from importlib.metadata import Distribution, PackageNotFoundError
from typing import Any, Optional, Union, List, Dict, Mapping, Sequence

from importlib_metadata import entry_points
from packaging.requirements import Requirement

# Use numpy if available for fast decoding
try:
    import numpy
except ImportError:  # pragma: no cover
    pass

__all__ = ['hasinstance', 'exception_chain', 'is_caused_by',
           'NumpyEncoder', 'coerce_numpy_to_python',
           'get_distribution', 'PackageNotFoundError', 'VersionNotFoundError',
           ]

logger = logging.getLogger(__name__)


def coerce_numpy_to_python(obj):
    """Numpy object serializer with support for basic scalar types and ndarrays."""

    if isinstance(obj, numpy.integer):
        return int(obj)
    elif isinstance(obj, numpy.floating):
        return float(obj)
    elif isinstance(obj, numpy.bool_):
        return bool(obj)
    elif isinstance(obj, numpy.ndarray):
        return [coerce_numpy_to_python(v) for v in obj.tolist()]
    elif isinstance(obj, (list, tuple)):    # be explicit to avoid recursing over string et al
        return type(obj)(coerce_numpy_to_python(v) for v in obj)
    elif isinstance(obj, dict):
        return {coerce_numpy_to_python(k): coerce_numpy_to_python(v) for k, v in obj.items()}
    return obj


# copied from dwave-hybrid utils
# (https://github.com/dwavesystems/dwave-hybrid/blob/b9025b5bb3d88dce98ec70e28cfdb25400a10e4a/hybrid/utils.py#L43-L61)
# TODO: switch to `dwave.common` if and when we create it
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types.

    Supported types:
     - basic numeric types: booleans, integers, floats
     - arrays: ndarray, recarray
    """

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()

        return super().default(obj)


def hasinstance(iterable, class_or_tuple):
    """Extension of ``isinstance`` to iterables/sequences. Returns True iff the
    sequence contains at least one object which is instance of ``class_or_tuple``.
    """

    return any(isinstance(e, class_or_tuple) for e in iterable)


def exception_chain(exception):
    """Traverse the chain of embedded exceptions, yielding one at the time.

    Args:
        exception (:class:`Exception`): Chained exception.

    Yields:
        :class:`Exception`: The next exception in the input exception's chain.

    Examples:
        def f():
            try:
                1/0
            except ZeroDivisionError:
                raise ValueError

        try:
            f()
        except Exception as e:
            assert(hasinstance(exception_chain(e), ZeroDivisionError))

    See: PEP-3134.
    """

    while exception:
        yield exception

        # explicit exception chaining, i.e `raise .. from ..`
        if exception.__cause__:
            exception = exception.__cause__

        # implicit exception chaining
        elif exception.__context__:
            exception = exception.__context__

        else:
            return


def is_caused_by(exception, exception_types):
    """Check if any of ``exception_types`` is causing the ``exception``.
    Equivalently, check if any of ``exception_types`` is contained in the
    exception chain rooted at ``exception``.

    Args:
        exception (:class:`Exception`):
            Chained exception.

        exception_types (:class:`Exception` or tuple of :class:`Exception`):
            Exception type or a tuple of exception types to check for.

    Returns:
        bool:
            True when ``exception`` is caused by any of the exceptions in
            ``exception_types``.
    """

    return hasinstance(exception_chain(exception), exception_types)



def pretty_argvalues():
    """Pretty-formatted function call arguments, from the caller's frame."""
    return inspect.formatargvalues(*inspect.getargvalues(inspect.currentframe().f_back))


def get_contrib_config():
    """Return all registered contrib (non-open-source) Ocean packages."""

    # Note: we use `entry_points` from `importlib_metadata` to simplify access
    # and use py312 semantics. See "compatibility note" in `importlib.metadata`
    # docs for entry points.
    contrib = [ep.load() for ep in entry_points(group='dwave_contrib')]
    return contrib


def get_contrib_packages():
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
        :class:`~dwave.cloud.utils.VersionNotFoundError`:
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


def bqm_to_dqm(bqm):
    """Represent a :class:`dimod.BQM` as a :class:`dimod.DQM`."""
    try:
        from dimod import DiscreteQuadraticModel
    except ImportError: # pragma: no cover
        raise RuntimeError(
            "dimod package with support for DiscreteQuadraticModel required."
            "Re-install the library with 'dqm' support.")

    dqm = DiscreteQuadraticModel()

    ising = bqm.spin

    for v, bias in ising.linear.items():
        dqm.add_variable(2, label=v)
        dqm.set_linear(v, [-bias, bias])

    for (u, v), bias in ising.quadratic.items():
        biases = numpy.array([[bias, -bias], [-bias, bias]], dtype=numpy.float64)
        dqm.set_quadratic(u, v, biases)

    return dqm

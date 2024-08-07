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

"""Encoders/decoders of 3rd-party formats for private and Ocean-internal use.

.. versionchanged:: 0.12.0
   These functions previously lived under ``dwave.cloud.utils``.
"""

import json
import warnings

try:
    import numpy
except ImportError: # pragma: no cover
    pass    # numpy not required for all cloud-client functionality

__all__ = ['NumpyEncoder', 'coerce_numpy_to_python']


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
        warnings.warn(
            f"`dwave.cloud.utils.coders.NumpyEncoder` is deprecated since "
            f"dwave-cloud-client 0.12.2, and will be removed in 0.14.0. "
            f"Use `orjson.dumps()` with `OPT_SERIALIZE_NUMPY` option instead.",
            DeprecationWarning, stacklevel=2)

        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()

        return super().default(obj)

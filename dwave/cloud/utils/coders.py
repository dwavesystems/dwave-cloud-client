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

.. versionremoved:: 0.14.0
   ``NumpyEncoder`` replaced with ``orjson.dumps(..., option=OPT_SERIALIZE_NUMPY)``.
"""

try:
    import numpy
except ImportError: # pragma: no cover
    pass    # numpy not required for all cloud-client functionality

__all__ = ['coerce_numpy_to_python']


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

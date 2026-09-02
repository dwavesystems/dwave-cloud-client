# Copyright 2021 D-Wave Systems Inc.
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

import enum


# Default API version
DEFAULT_API_MEDIA_TYPE = 'application/vnd.dwave+json'
DEFAULT_API_RESPONSE_VERSION = '1.0.0'


class _OpenStrEnumMixin:
    """Extends a string :class:`enum.Enum` to handle unknown values.

    Instead of failing when we receive an unknown enum from SAPI, we'll create
    a its enum value on the fly.

    See: protobuf's `open enum <https://protobuf.dev/programming-guides/enum/>`_.
    """

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__} requires a str, got {type(value).__name__}")
        member = str.__new__(cls, value)
        member._name_ = str(value).upper()
        member._value_ = value
        # cache it so identity comparison works on repeat values
        cls._value2member_map_[value] = member
        return member


# TODO: replace (str, Enum) with StrEnum when we drop py310
class ProblemStatus(_OpenStrEnumMixin, str, enum.Enum):
    """Solver API problem status values.

    Initially a problem is in the PENDING state. When the D-Wave system starts
    to process a problem, its state changes to IN_PROGRESS. After completion,
    the problem status changes to either COMPLETED or FAILED (if an error
    occurred). COMPLETED, FAILED, and CANCELLED are all terminal states.

    After a problem enters a terminal state, its status does not change. Users
    can cancel a problem at any time before it reaches its terminal state.
    """

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProblemEncodingFormat(_OpenStrEnumMixin, str, enum.Enum):
    QP = "qp"
    BQ = "bq"   # deprecated for submission
    REF = "ref"


class AnswerEncodingFormat(_OpenStrEnumMixin, str, enum.Enum):
    QP = "qp"
    BQ = "bq"   # dimod (de-)serialization-based
    BINARY_REF = "binary-ref"


class BinaryRefAuthMethod(_OpenStrEnumMixin, str, enum.Enum):
    SAPI_TOKEN = "sapi-token"


class ProblemType(_OpenStrEnumMixin, str, enum.Enum):
    ISING = "ising"
    QUBO = "qubo"
    BQM = "bqm"
    CQM = "cqm"
    DQM = "dqm"
    NL = "nl"
    QCDL = "qcdl"


class DeprecationContext(_OpenStrEnumMixin, str, enum.Enum):
    API = "api"                 # API changes such as endpoints, data structures, and headers
    FEATURE = "feature"         # solver feature is deprecated
    PARAMETER = "parameter"     # solver parameter is deprecated
    SOLVER = "solver"           # solver or solver type is deprecated
    OTHER = "other"

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

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Optional, Union

from pydantic import BaseModel, RootModel, ConfigDict, Field
from pydantic.functional_validators import AfterValidator

from dwave.cloud.api import constants
from dwave.cloud.utils.coders import coerce_numpy_to_python


# coerce common numpy types to python types on validation (parsing)
AnyIncludingNumpy = Annotated[Any, AfterValidator(coerce_numpy_to_python)]


class _RootGetterMixin:
    """Proxy getter calls to root model."""

    def __getattr__(self, item):
        return getattr(self.root, item)

    def __getitem__(self, name):
        return getattr(self.root, name)

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default


class _RootSetterMixin:
    """Proxy setter calls to root model."""

    def __setattr__(self, name, value):
        return setattr(self.root, name, value)

    def __setitem__(self, name, value):
        return setattr(self.root, name, value)


class _DictMixin:
    """Add `.dict()` method."""

    def dict(self):
        return self.model_dump(exclude_unset=True, exclude_none=True)


class _DictEqualityMixin:
    """Enable equality testing against a dict. Requires `.dict` method."""

    def __eq__(self, other):
        if isinstance(other, dict):
            return self.dict() == other
        return super().__eq__(other)


class SolverVersion(_DictMixin, _DictEqualityMixin, BaseModel):
    # allow additional version specifiers in the future
    model_config = ConfigDict(extra='allow')

    graph_id: Optional[str] = None              # QPU solvers require graph_id


class SolverIdentity(_DictMixin, _DictEqualityMixin, BaseModel):
    name: str
    version: Optional[SolverVersion] = None     # only QPU solvers have `version` structure

    def __str__(self):
        return self.to_id()

    def to_id(self) -> str:
        s = self.name
        d = self.dict()
        if v := d.get('version'):
            v = ";".join(f"{k!s}={v!s}" for k,v in v.items())
            s = f"{s};{v}"
        return s

    @classmethod
    def from_id(cls, id: str) -> SolverIdentity:
        name, *version = id.split(';')
        version = dict(v.split('=', maxsplit=2) for v in version)
        return cls(name=name, version=version)


class SolverCompleteConfiguration(BaseModel):
    identity: SolverIdentity
    status: str
    description: str
    properties: dict
    avg_load: float


class SolverFilteredConfiguration(BaseModel):
    # no required fields, and no ignored fields
    model_config = ConfigDict(extra='allow')

    identity: Optional[SolverIdentity] = None


# NOTE: we implement getitem interface so that `SolverConfiguration` can be
# used as a drop-in replacement for data dict in `Solver(..., data=...)`
# TODO: break `Solver()` backwards compat and require pydantic model
class SolverConfiguration(_RootGetterMixin, _RootSetterMixin, RootModel):
    root: Union[SolverCompleteConfiguration, SolverFilteredConfiguration]


class ProblemInitialStatus(BaseModel):
    id: str
    type: constants.ProblemType
    solver: SolverIdentity
    label: Optional[str] = None
    status: constants.ProblemStatus
    submitted_on: datetime


class ProblemStatus(ProblemInitialStatus):
    solved_on: Optional[datetime] = None


class StructuredProblemAnswer(BaseModel):
    format: constants.AnswerEncodingFormat = constants.AnswerEncodingFormat.QP
    active_variables: str
    energies: str
    solutions: str
    timing: dict
    num_occurrences: str
    num_variables: int


class UnstructuredProblemAnswer(BaseModel):
    format: constants.AnswerEncodingFormat = constants.AnswerEncodingFormat.BQ
    data: dict


class UnstructuredProblemAnswerBinaryRef(BaseModel):
    format: constants.AnswerEncodingFormat = constants.AnswerEncodingFormat.BINARY_REF
    auth_method: constants.BinaryRefAuthMethod = constants.BinaryRefAuthMethod.SAPI_TOKEN
    url: str
    timing: dict
    shape: dict


class ProblemAnswer(_RootGetterMixin, RootModel):
    root: Union[StructuredProblemAnswer,
                UnstructuredProblemAnswer,
                UnstructuredProblemAnswerBinaryRef]


class ProblemStatusWithAnswer(ProblemStatus):
    answer: ProblemAnswer


class ProblemStatusMaybeWithAnswer(ProblemStatus):
    answer: Optional[ProblemAnswer] = None


class StructuredProblemData(BaseModel):
    format: constants.ProblemEncodingFormat = constants.ProblemEncodingFormat.QP
    lin: str
    quad: str
    offset: float = 0.0


class UnstructuredProblemData(BaseModel):
    format: constants.ProblemEncodingFormat = constants.ProblemEncodingFormat.REF
    data: str


class ProblemData(_RootGetterMixin, RootModel):
    root: Union[StructuredProblemData, UnstructuredProblemData]


class ProblemMetadata(BaseModel):
    solver: SolverIdentity
    type: constants.ProblemType
    label: Optional[str] = None
    status: constants.ProblemStatus
    submitted_by: str
    submitted_on: datetime
    solved_on: Optional[datetime] = None
    messages: Optional[list[dict]] = None


class ProblemInfo(BaseModel):
    id: str
    data: ProblemData
    params: dict[str, AnyIncludingNumpy]
    metadata: ProblemMetadata
    answer: Optional[ProblemAnswer] = None          # missing unless problem status is COMPLETED


class ProblemJob(BaseModel):
    data: ProblemData
    params: dict[str, AnyIncludingNumpy]
    solver: SolverIdentity
    type: constants.ProblemType
    label: Optional[str] = None

    @classmethod
    def from_info(cls, info: ProblemInfo):
        return cls(data=info.data,
                   params=info.params,
                   solver=info.metadata.solver,
                   type=info.metadata.type,
                   label=info.metadata.label)


class BatchItemError(BaseModel):
    error_code: int
    error_msg: str

class ProblemSubmitError(BatchItemError):
    pass

class ProblemCancelError(BatchItemError):
    pass


# region info on metadata api
class Region(BaseModel):
    code: str
    name: str
    endpoint: str

    @property
    def solver_api_endpoint(self) -> str:
        return self.endpoint

    @property
    def leap_api_endpoint(self) -> str:
        # guess until metadata api includes leap endpoint in region data
        from dwave.cloud.regions import _infer_leap_api_endpoint
        return _infer_leap_api_endpoint(self.endpoint, self.code)


# LeapAPI types, provisional

class LeapProject(BaseModel):
    id: int
    name: str
    code: str

class _LeapProjectWrapper(BaseModel):
    project: LeapProject

# LeapAPI / account / active project response
class _LeapActiveProjectResponse(BaseModel):
    data: _LeapProjectWrapper

# LeapAPI / account / projects response
class _LeapProjectsWrapper(BaseModel):
    projects: list[_LeapProjectWrapper]

class _LeapProjectsResponse(BaseModel):
    data: _LeapProjectsWrapper

# LeapAPI / account / token response
class _LeapTokenWrapper(BaseModel):
    token: Optional[str]

class _LeapProjectTokenResponse(BaseModel):
    data: _LeapTokenWrapper

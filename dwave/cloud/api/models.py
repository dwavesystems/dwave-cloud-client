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

import json
from typing import List, Union, Optional, Dict, Any, Annotated, Mapping, Sequence
from datetime import datetime

import numpy
from pydantic import BaseModel, RootModel, field_serializer, SerializerFunctionWrapHandler
from pydantic.functional_serializers import WrapSerializer

from dwave.cloud.api import constants
from dwave.cloud.utils import NumpyEncoder


class SolverConfiguration(BaseModel):
    id: str
    status: str
    description: str
    properties: dict
    avg_load: float


class ProblemInitialStatus(BaseModel):
    id: str
    type: constants.ProblemType
    solver: str
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


class ProblemAnswer(RootModel):
    root: Union[StructuredProblemAnswer, UnstructuredProblemAnswer]

    def __getattr__(self, item):
        return getattr(self.root, item)


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


class ProblemData(RootModel):
    root: Union[StructuredProblemData, UnstructuredProblemData]

    def __getattr__(self, item):
        return getattr(self.root, item)


class ProblemMetadata(BaseModel):
    solver: str
    type: constants.ProblemType
    label: Optional[str] = None
    status: constants.ProblemStatus
    submitted_by: str
    submitted_on: datetime
    solved_on: Optional[datetime] = None
    messages: Optional[List[dict]] = None


class ProblemInfo(BaseModel):
    id: str
    data: ProblemData
    params: dict
    metadata: ProblemMetadata
    answer: ProblemAnswer


def coerce_numpy_to_python(obj, handler):
    if isinstance(obj, numpy.integer):
        return int(obj)
    elif isinstance(obj, numpy.floating):
        return float(obj)
    elif isinstance(obj, numpy.bool_):
        return bool(obj)
    elif isinstance(obj, numpy.ndarray):
        return [coerce_numpy_to_python(v, handler) for v in obj.to_list()]
    elif isinstance(obj, Sequence):
        return [coerce_numpy_to_python(v, handler) for v in obj]
    elif isinstance(obj, Mapping):
        return {k: coerce_numpy_to_python(v, handler) for k,v in obj.items()}
    return handler(obj)

def np_serialize(v: Any, nxt: SerializerFunctionWrapHandler):
    return coerce_numpy_to_python(v, nxt)

AnyIncludingNumpy = Annotated[Any, WrapSerializer(np_serialize, when_used='json')]

class ProblemJob(BaseModel):
    data: ProblemData
    params: Dict[str, AnyIncludingNumpy]
    solver: str
    type: constants.ProblemType
    label: Optional[str] = None


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
    projects: List[_LeapProjectWrapper]

class _LeapProjectsResponse(BaseModel):
    data: _LeapProjectsWrapper

# LeapAPI / account / token response
class _LeapTokenWrapper(BaseModel):
    token: str

class _LeapProjectTokenResponse(BaseModel):
    data: _LeapTokenWrapper

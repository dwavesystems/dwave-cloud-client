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

from typing import List, Union, Optional
from datetime import datetime

import numpy
from pydantic import BaseModel

from dwave.cloud.api import constants


class BaseModelWithEncoders(BaseModel):
    class Config:
        json_encoders = {
            numpy.integer: int,
            numpy.floating: float,
            numpy.bool_: bool,
            numpy.ndarray: lambda obj: obj.tolist(),
        }


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
    label: Optional[str]
    status: constants.ProblemStatus
    submitted_on: datetime


class ProblemStatus(ProblemInitialStatus):
    solved_on: Optional[datetime]


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


class ProblemAnswer(BaseModel):
    __root__: Union[StructuredProblemAnswer, UnstructuredProblemAnswer]

    def __getattr__(self, item):
        return getattr(self.__root__, item)

    def dict(self, **kwargs):
        return super().dict(**kwargs)['__root__']


class ProblemStatusWithAnswer(ProblemStatus):
    answer: ProblemAnswer


class ProblemStatusMaybeWithAnswer(ProblemStatus):
    answer: Optional[ProblemAnswer]


class StructuredProblemData(BaseModel):
    format: constants.ProblemEncodingFormat = constants.ProblemEncodingFormat.QP
    lin: str
    quad: str
    offset: float = 0.0


class UnstructuredProblemData(BaseModel):
    format: constants.ProblemEncodingFormat = constants.ProblemEncodingFormat.REF
    data: str


class ProblemData(BaseModel):
    __root__: Union[StructuredProblemData, UnstructuredProblemData]

    def __getattr__(self, item):
        return getattr(self.__root__, item)

    def dict(self, **kwargs):
        return super().dict(**kwargs)['__root__']


class ProblemMetadata(BaseModel):
    solver: str
    type: constants.ProblemType
    label: Optional[str]
    status: constants.ProblemStatus
    submitted_by: str
    submitted_on: datetime
    solved_on: Optional[datetime]
    messages: Optional[List[dict]]


class ProblemInfo(BaseModel):
    id: str
    data: ProblemData
    params: dict
    metadata: ProblemMetadata
    answer: ProblemAnswer


class ProblemJob(BaseModelWithEncoders):
    data: ProblemData
    params: dict
    solver: str
    type: constants.ProblemType
    label: Optional[str]


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

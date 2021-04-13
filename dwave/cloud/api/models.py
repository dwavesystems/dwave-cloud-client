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

from pydantic import BaseModel

from dwave.cloud.api import constants


class SolverDescription(BaseModel):
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


class ProblemAnswer(BaseModel):
    format: constants.AnswerEncodingFormat
    active_variables: str
    energies: str
    solutions: str
    timing: dict
    num_occurrences: int
    num_variables: int


class ProblemStatusWithAnswer(ProblemStatus):
    answer: ProblemAnswer


class ProblemStatusMaybeWithAnswer(ProblemStatus):
    answer: Optional[ProblemAnswer]


class ProblemData(BaseModel):
    format: constants.ProblemEncodingFormat
    # qp format fields
    lin: str = None
    quad: str = None
    offset: float = 0
    # ref format fields
    data: str = None


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


class ProblemJob(BaseModel):
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

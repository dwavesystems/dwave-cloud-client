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

from dataclasses import dataclass
from typing import List, Union


@dataclass
class SolverDescription:
    id: str
    status: str
    description: str
    properties: dict
    avg_load: float

@dataclass
class ProblemInitialStatus:
    id: str
    solver: str
    type: str
    label: str
    status: str             # TODO: make enum?
    submitted_on: str       # TODO: convert to datetime?

@dataclass
class ProblemStatus(ProblemInitialStatus):
    solved_on: str = None

@dataclass
class ProblemAnswer:
    format: str
    active_variables: str
    energies: str
    solutions: str
    timing: dict
    num_occurrences: int
    num_variables: int

@dataclass
class ProblemStatusWithAnswer(ProblemInitialStatus):
    solved_on: str
    answer: ProblemAnswer

    def __post_init__(self):
        self.answer = ProblemAnswer(**self.answer)

@dataclass
class ProblemStatusMaybeWithAnswer(ProblemInitialStatus):
    solved_on: str = None
    answer: ProblemAnswer = None

    def __post_init__(self):
        if self.answer is not None:
            self.answer = ProblemAnswer(**self.answer)

@dataclass
class ProblemData:
    format: str
    # qp format fields
    lin: str = None
    quad: str = None
    offset: float = 0
    # ref format fields
    data: str = None

@dataclass
class ProblemMetadata:
    solver: str
    type: str
    label: str
    status: str             # TODO: make enum?
    submitted_by: str
    submitted_on: str       # TODO: convert to datetime?
    solved_on: str = None   # TODO: convert to datetime?
    messages: list = None

@dataclass
class ProblemInfo:
    id: str
    data: ProblemData
    params: dict
    metadata: ProblemMetadata
    answer: ProblemAnswer

    # unpack nested fields
    def __post_init__(self):
        self.data = ProblemData(**self.data)
        self.metadata = ProblemMetadata(**self.metadata)
        self.answer = ProblemAnswer(**self.answer)

@dataclass
class ProblemJob:
    data: ProblemData
    params: dict
    solver: str
    type: str
    label: str = None

@dataclass
class ProblemSubmitError:
    error_code: int
    error_msg: str

@dataclass
class ProblemCancelError(ProblemSubmitError):
    pass

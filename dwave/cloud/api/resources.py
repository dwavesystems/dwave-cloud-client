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

from dataclasses import asdict

from dwave.cloud.api.client import SAPIClient
from dwave.cloud.api.models import *

__all__ = ['Solvers', 'Problems']


class Resource(SAPIClient):
    """A class for interacting with a SAPI resource."""


class Solvers(Resource):

    # Content-Type: application/vnd.dwave.sapi.solver-definition-list+json; version=2.0.0
    def list_solvers(self) -> List[SolverDescription]:
        path = 'solvers/remote/'
        response = self.session.get(path)
        solvers = response.json()
        return [SolverDescription(**s) for s in solvers]

    # Content-Type: application/vnd.dwave.sapi.solver-definition+json; version=2.0.0
    def get_solver(self, solver_id: str) -> SolverDescription:
        path = 'solvers/remote/{}'.format(solver_id)
        response = self.session.get(path)
        solver = response.json()
        return SolverDescription(**solver)


class Problems(Resource):

    # Content-Type: application/vnd.dwave.sapi.problems+json; version=2.1.0
    def list_problems(self, **params) -> List[ProblemStatus]:
        # available params: id, label, max_results, status, solver
        path = 'problems'
        response = self.session.get(path, params=params)
        statuses = response.json()
        return [ProblemStatus(**s) for s in statuses]

    # Content-Type: application/vnd.dwave.sapi.problem+json; version=2.1.0
    def get_problem(self, problem_id: str) -> ProblemStatusMaybeWithAnswer:
        """Retrieve problem short status and answer if answer is available."""
        path = 'problems/{}'.format(problem_id)
        response = self.session.get(path)
        status = response.json()
        return ProblemStatusMaybeWithAnswer(**status)

    # Content-Type: application/vnd.dwave.sapi.problems+json; version=2.1.0
    def get_problem_status(self, problem_id: str) -> ProblemStatus:
        """Retrieve short status of a single problem."""
        path = 'problems/'
        params = dict(id=problem_id)
        response = self.session.get(path, params=params)
        status = response.json()[0]
        return ProblemStatus(**status)

    # Content-Type: application/vnd.dwave.sapi.problems+json; version=2.1.0
    def get_problem_statuses(self, problem_ids: List[str]) -> List[ProblemStatus]:
        """Retrieve short problem statuses for a list of problems."""
        if len(problem_ids) > 1000:
            raise ValueError('number of problem ids is limited to 1000')

        path = 'problems/'
        params = dict(id=','.join(problem_ids))
        response = self.session.get(path, params=params)
        statuses = response.json()
        return [ProblemStatus(**s) for s in statuses]

    # Content-Type: application/vnd.dwave.sapi.problem-data+json; version=2.1.0
    def get_problem_info(self, problem_id: str) -> ProblemInfo:
        """Retrieve complete problem info."""
        path = 'problems/{}/info'.format(problem_id)
        response = self.session.get(path)
        info = response.json()
        return ProblemInfo(**info)

    # Content-Type: application/vnd.dwave.sapi.problem-answer+json; version=2.1.0
    def get_problem_answer(self, problem_id: str) -> ProblemAnswer:
        """Retrieve problem answer."""
        path = 'problems/{}/answer'.format(problem_id)
        response = self.session.get(path)
        answer = response.json()
        return ProblemAnswer(**answer)

    # Content-Type: application/vnd.dwave.sapi.problem-message+json; version=2.1.0
    def get_problem_messages(self, problem_id: str) -> List[dict]:
        """Retrieve list of problem messages."""
        path = 'problems/{}/messages'.format(problem_id)
        response = self.session.get(path)
        return response.json()

    # Content-Type: application/vnd.dwave.sapi.problems+json; version=2.1.0
    def submit_problem(self,
                       data: ProblemData,
                       params: dict,
                       solver: str,
                       type: str,
                       label: str = None) -> ProblemStatusMaybeWithAnswer:
        """Finite-time blocking problem submit, returning final status and
        answer, if problem was solved within the (undisclosed) time limit.
        """
        path = 'problems/'
        body = dict(data=asdict(data), params=params, solver=solver,
                    type=type, label=label)
        response = self.session.post(path, json=body)
        return ProblemStatusMaybeWithAnswer(**response.json())

    # Content-Type: application/vnd.dwave.sapi.problems+json; version=2.1.0
    def submit_problems(self, problems: List[ProblemJob]) -> List[Union[ProblemInitialStatus, ProblemSubmitError]]:
        """Asynchronous multi-problem submit, returning initial statuses."""
        path = 'problems/'
        body = [asdict(p) for p in problems]
        response = self.session.post(path, json=body)
        statuses = response.json()
        return [ProblemInitialStatus(**s) if 'status' in s else ProblemSubmitError(**s) for s in statuses]

    # Content-Type: application/vnd.dwave.sapi.problem+json; version=2.1.0
    def cancel_problem(self, problem_id: str) -> ProblemStatus:
        """Initiate problem cancel by problem id."""
        path = 'problems/{}/'.format(problem_id)
        response = self.session.delete(path)
        status = response.json()
        return ProblemStatus(**status)

    # Content-Type: application/vnd.dwave.sapi.problems+json; version=2.1.0
    def cancel_problems(self, problem_ids: List[str]) -> List[Union[ProblemStatus, ProblemCancelError]]:
        """Initiate problem cancel for a list of problems."""
        path = 'problems/'
        response = self.session.delete(path, json=problem_ids)
        statuses = response.json()
        return [ProblemStatus(**s) if 'status' in s else ProblemCancelError(**s) for s in statuses]

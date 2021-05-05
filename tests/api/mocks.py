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
import uuid
from typing import Union, Tuple

from dwave.cloud.coders import encode_problem_as_qp
from dwave.cloud.utils import utcrel, generate_const_ising_problem
from dwave.cloud.solver import StructuredSolver, UnstructuredSolver
from dwave.cloud.testing.mocks import qpu_clique_solver_data, hybrid_bqm_solver_data


class SapiMockResponses:
    """SAPI mock response generator specialized for a prototypical solver and
    prototypical problem (e.g. structured solver and ising problem).
    """

    def __init__(self,
                 solver: Union[StructuredSolver, UnstructuredSolver],
                 problem: Union[Tuple[dict, dict], 'dimod.BQM'],
                 type: str,
                 answer: dict,
                 ):
        self.solver = solver
        self.problem = problem
        self.answer = answer

        self.solver_id = str(uuid.uuid4())
        self.problem_id = str(uuid.uuid4())
        self.problem_type = type

    def problem_data(self,
                     solver: Union[StructuredSolver, UnstructuredSolver] = None,
                     problem: Union[Tuple[dict, dict], 'dimod.BQM'] = None) -> dict:
        if solver is None:
            solver = self.solver
        if problem is None:
            problem = self.problem
        linear, quadratic = problem
        return encode_problem_as_qp(solver, linear, quadratic)

    def problem_messages(self) -> dict:
        return [{
            "timestamp": utcrel(-10).isoformat(),
            "message": "Internal SAPI error occurred.",
            "severity": "ERROR"
        }]

    def problem_metadata(self, **kwargs) -> dict:
        metadata = {
            "submitted_by":  str(uuid.uuid4()),
            "solver": self.solver_id,
            "type": self.problem_type,
            "submitted_on": utcrel(-20).isoformat(),
            "solved_on": utcrel(-10).isoformat(),
            "status": "COMPLETED",
            "messages": self.problem_messages(),
            "label": None,
        }
        metadata.update(**kwargs)
        return metadata

    def problem_answer(self) -> dict:
        return {
            "answer": self.answer.copy()
        }

    def problem_info(self, **kwargs) -> dict:
        response = {
            "id": self.problem_id,
            "data": self.problem_data(),
            "params": {},
            "metadata": self.problem_metadata(),
            "answer": self.problem_answer(),
        }
        response.update(**kwargs)
        return response

    def complete_reply(self, answer_patch=None, **kwargs) -> dict:
        """Reply with solutions for the test problem."""

        response = {
            "status": "COMPLETED",
            "solved_on": utcrel(-10).isoformat(),
            "solver": self.solver_id,
            "submitted_on": utcrel(-20).isoformat(),
            "answer": self.answer.copy(),
            "type": self.problem_type,
            "id": self.problem_id,
            "label": None,
        }

        # optional answer fields override
        if answer_patch:
            response['answer'].update(answer_patch)

        # optional top-level override
        response.update(**kwargs)

        return response

    def complete_no_answer_reply(self, **kwargs) -> dict:
        """A reply saying a problem is finished without providing the results."""

        response = {
            "status": "COMPLETED",
            "solved_on": utcrel(-10).isoformat(),
            "solver": self.solver_id,
            "submitted_on": utcrel(-20).isoformat(),
            "type": self.problem_type,
            "id": self.problem_id,
            "label": None,
        }
        response.update(**kwargs)
        return response

    def error_reply(self, error_message=None, **kwargs) -> dict:
        """A reply saying an error has occurred."""

        response = {
            "status": "FAILED",
            "solved_on": utcrel(-10).isoformat(),
            "solver": self.solver_id,
            "submitted_on": utcrel(-20).isoformat(),
            "type": self.problem_type,
            "id": self.problem_id,
            "label": None,
            "error_message": error_message or 'An error message',
        }
        response.update(**kwargs)
        return response

    def immediate_error_reply(self, code: int, msg: str) -> dict:
        """A reply saying an error has occurred (before scheduling for execution)."""

        return {
            "error_code": code,
            "error_msg": msg
        }

    def cancel_reply(self, **kwargs) -> dict:
        """A reply saying a problem was canceled."""

        return {
            "status": "CANCELLED",
            "solved_on": utcrel(-10).isoformat(),
            "solver": self.solver_id,
            "submitted_on": utcrel(-20).isoformat(),
            "type": self.problem_type,
            "id": self.problem_id,
            "label": None,
        }
        response.update(**kwargs)
        return response

    def continue_reply(self, **kwargs) -> dict:
        """A reply saying a problem is still in the queue."""

        response = {
            "status": "PENDING",
            "solved_on": None,
            "solver": self.solver_id,
            "submitted_on": utcrel(0).isoformat(),
            "type": self.problem_type,
            "id": self.problem_id,
            "label": None,
        }
        response.update(**kwargs)
        return response



class StructuredSapiMockResponses(SapiMockResponses):

    # TODO: accept problem, generate a consistent answer
    def _problem_answer(self, **kwargs) -> dict:
        answer = {
            "format": "qp",
            "num_variables": 5,
            "energies": "AAAAAAAALsA=",
            "num_occurrences": "ZAAAAA==",
            "active_variables": "AAAAAAEAAAACAAAAAwAAAAQAAAA=",
            "solutions": "AAAAAA==",
            "timing": {}
        }
        answer.update(**kwargs)
        return answer

    def __init__(self, **kwargs):
        kwargs.setdefault('solver', StructuredSolver(client=None, data=qpu_clique_solver_data(5)))
        kwargs.setdefault('problem', generate_const_ising_problem(kwargs['solver'], h=1, j=-1))
        kwargs.setdefault('type', 'ising')
        kwargs.setdefault('answer', self._problem_answer())
        super().__init__(**kwargs)


class UnstructuredSapiMockResponses(SapiMockResponses):

    def _problem_answer(self, sampleset, **kwargs) -> dict:
        answer = {
            "format": "bq",
            "data": sampleset.to_serializable()
        }
        answer.update(**kwargs)
        return answer

    def __init__(self, **kwargs):
        import dimod

        solver = UnstructuredSolver(client=None, data=hybrid_bqm_solver_data())
        bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})
        sampleset = dimod.ExactSolver().sample(bqm)

        kwargs.setdefault('solver', solver)
        kwargs.setdefault('problem', bqm)
        kwargs.setdefault('type', 'bqm')
        kwargs.setdefault('answer', self._problem_answer(sampleset))
        super().__init__(**kwargs)

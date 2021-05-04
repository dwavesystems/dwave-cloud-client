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

from dwave.cloud.utils import utcrel
from dwave.cloud.testing.mocks import qpu_clique_solver_data


def solver_data(**kwargs):
    """Mock solver compatible with problem mock replies below."""
    return qpu_clique_solver_data(5, **kwargs)


def _get_problem_id(id=None):
    return id if id is not None else str(uuid.uuid4())


def _get_solver_id(id=None):
    return id if id is not None else solver_data()['id']


def complete_reply(id=None, solver_id=None, answer=None, **kwargs):
    """Reply with solutions for the test problem."""

    response = {
        "status": "COMPLETED",
        "solved_on": utcrel(-10).isoformat(),
        "solver": _get_solver_id(solver_id),
        "submitted_on": utcrel(-20).isoformat(),
        "answer": {
            "format": "qp",
            "num_variables": 5,
            "energies": "AAAAAAAALsA=",
            "num_occurrences": "ZAAAAA==",
            "active_variables": "AAAAAAEAAAACAAAAAwAAAAQAAAA=",
            "solutions": "AAAAAA==",
            "timing": {}
        },
        "type": "ising",
        "id": _get_problem_id(id),
        "label": None,
    }

    # optional answer fields override
    if answer:
        response['answer'].update(answer)

    # optional top-level override
    response.update(**kwargs)

    return response


def complete_no_answer_reply(id=None, solver_id=None, **kwargs):
    """A reply saying a problem is finished without providing the results."""

    response = {
        "status": "COMPLETED",
        "solved_on": utcrel(-10).isoformat(),
        "solver": _get_solver_id(solver_id),
        "submitted_on": utcrel(-20).isoformat(),
        "type": "ising",
        "id": _get_problem_id(id),
        "label": None,
    }
    response.update(**kwargs)
    return response


def error_reply(id=None, solver_id=None, error_message=None, **kwargs):
    """A reply saying an error has occurred."""

    response = {
        "status": "FAILED",
        "solved_on": utcrel(-10).isoformat(),
        "solver": _get_solver_id(solver_id),
        "submitted_on": utcrel(-20).isoformat(),
        "type": "ising",
        "id": _get_problem_id(id),
        "label": None,
        "error_message": error_message or 'An error message',
    }
    response.update(**kwargs)
    return response


def immediate_error_reply(code, msg):
    """A reply saying an error has occurred (before scheduling for execution)."""

    return {
        "error_code": code,
        "error_msg": msg
    }


def cancel_reply(id=None, solver_id=None, **kwargs):
    """A reply saying a problem was canceled."""

    return {
        "status": "CANCELLED",
        "solved_on": utcrel(-10).isoformat(),
        "solver": _get_solver_id(solver_id),
        "submitted_on": utcrel(-20).isoformat(),
        "type": "ising",
        "id": _get_problem_id(id),
        "label": None,
    }
    response.update(**kwargs)
    return response


def continue_reply(id=None, solver_id=None, now=None, **kwargs):
    """A reply saying a problem is still in the queue."""

    if not now:
        now = utcrel(0)

    response = {
        "status": "PENDING",
        "solved_on": None,
        "solver": _get_solver_id(solver_id),
        "submitted_on": now.isoformat(),
        "type": "ising",
        "id": _get_problem_id(id),
        "label": None,
    }

    response.update(**kwargs)

    return response

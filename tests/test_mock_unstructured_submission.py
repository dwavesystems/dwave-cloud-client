# Copyright 2019 D-Wave Systems Inc.
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

"""Test problem submission to mock unstructured solvers."""

import json
import unittest

import dimod
import numpy

from dwave.cloud.client import Client
from dwave.cloud.solver import UnstructuredSolver
from dwave.cloud.testing import mock


def unstructured_solver_data():
    return {
        "properties": {
            "supported_problem_types": ["bqm"],
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "id": "test-unstructured-solver",
        "description": "A test unstructured solver"
    }

def complete_reply(sampleset):
    """Reply with the sampleset as a solution."""

    return json.dumps([{
        "status": "COMPLETED",
        "solver": "solver-name",
        "solved_on": "2019-07-31T12:34:56Z",
        "submitted_on": "2019-07-31T12:34:56Z",
        "answer": {
            "format": "bq",
            "data": sampleset.to_serializable()
        },
        "type": "bqm",
        "id": "problem-id"
    }])

def choose_reply(path, replies):
    """Choose the right response based on the path and make a mock response."""

    if path in replies:
        response = mock.Mock()
        response.status_code = 200
        response.json.side_effect = lambda: json.loads(replies[path])
        return response
    else:
        raise NotImplementedError(path)


class TestUnstructuredSolver(unittest.TestCase):

    def test_submit_immediate_reply(self):
        """Construction of and sampling from an unstructured solver works."""

        # build a test problem
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # construct a functional solver by mocking client and api response data
        with Client('endpoint', 'token') as client:
            client.session = mock.Mock()
            solver = UnstructuredSolver(client, unstructured_solver_data())

            # direct bqm sampling
            ss = dimod.ExactSolver().sample(bqm)
            client.session.post = lambda path, _: choose_reply(
                path, {'endpoint/problems/': complete_reply(ss)})

            fut = solver.sample_bqm(bqm)
            numpy.testing.assert_array_equal(fut.sampleset, ss)
            numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
            numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
            numpy.testing.assert_array_equal(fut.occurrences, ss.record.num_occurrences)

            # ising sampling
            lin, quad, _ = bqm.to_ising()
            ss = dimod.ExactSolver().sample_ising(lin, quad)
            client.session.post = lambda path, _: choose_reply(
                path, {'endpoint/problems/': complete_reply(ss)})

            fut = solver.sample_ising(lin, quad)
            numpy.testing.assert_array_equal(fut.sampleset, ss)
            numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
            numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
            numpy.testing.assert_array_equal(fut.occurrences, ss.record.num_occurrences)

            # qubo sampling
            qubo, _ = bqm.to_qubo()
            ss = dimod.ExactSolver().sample_qubo(qubo)
            client.session.post = lambda path, _: choose_reply(
                path, {'endpoint/problems/': complete_reply(ss)})

            fut = solver.sample_qubo(qubo)
            numpy.testing.assert_array_equal(fut.sampleset, ss)
            numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
            numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
            numpy.testing.assert_array_equal(fut.occurrences, ss.record.num_occurrences)

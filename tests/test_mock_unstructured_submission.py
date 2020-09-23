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
from unittest import mock

import dimod
import numpy

from dwave.cloud.client import Client
from dwave.cloud.solver import (
    BaseUnstructuredSolver, UnstructuredSolver, BQMSolver)
from dwave.cloud.concurrency import Present


def unstructured_solver_data(problem_type='bqm'):
    return {
        "properties": {
            "supported_problem_types": [problem_type],
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "id": "test-unstructured-solver",
        "description": "A test unstructured solver"
    }

def complete_reply(sampleset, problem_type='bqm'):
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
        "type": problem_type,
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

    def test_submit_bqm_immediate_reply(self):
        """Construction of and sampling from an unstructured BQM solver works."""

        # build a test problem
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, bqm):
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client('endpoint', 'token') as client:
                with mock.patch.object(BaseUnstructuredSolver, 'upload_problem', mock_upload):
                    solver = BQMSolver(client, unstructured_solver_data())

                    # make sure this still works
                    _ = UnstructuredSolver(client, unstructured_solver_data())

                    # direct bqm sampling
                    ss = dimod.ExactSolver().sample(bqm)
                    session.post = lambda path, _: choose_reply(
                        path, {'problems/': complete_reply(ss)})

                    fut = solver.sample_bqm(bqm)
                    numpy.testing.assert_array_equal(fut.sampleset, ss)
                    numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
                    numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
                    numpy.testing.assert_array_equal(fut.num_occurrences, ss.record.num_occurrences)

                    # submit of pre-uploaded bqm problem
                    fut = solver.sample_bqm(mock_problem_id)
                    numpy.testing.assert_array_equal(fut.sampleset, ss)
                    numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
                    numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
                    numpy.testing.assert_array_equal(fut.num_occurrences, ss.record.num_occurrences)

                    # ising sampling
                    lin, quad, _ = bqm.to_ising()
                    ss = dimod.ExactSolver().sample_ising(lin, quad)
                    session.post = lambda path, _: choose_reply(
                        path, {'problems/': complete_reply(ss)})

                    fut = solver.sample_ising(lin, quad)
                    numpy.testing.assert_array_equal(fut.sampleset, ss)
                    numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
                    numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
                    numpy.testing.assert_array_equal(fut.num_occurrences, ss.record.num_occurrences)

                    # qubo sampling
                    qubo, _ = bqm.to_qubo()
                    ss = dimod.ExactSolver().sample_qubo(qubo)
                    session.post = lambda path, _: choose_reply(
                        path, {'problems/': complete_reply(ss)})

                    fut = solver.sample_qubo(qubo)
                    numpy.testing.assert_array_equal(fut.sampleset, ss)
                    numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
                    numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
                    numpy.testing.assert_array_equal(fut.num_occurrences, ss.record.num_occurrences)

    def test_upload_failure(self):
        """Submit should gracefully fail if upload as part of submit fails."""

        # build a test problem
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_upload_exc = ValueError('error')
        def mock_upload(self, bqm):
            return Present(exception=mock_upload_exc)

        # construct a functional solver by mocking client and api response data
        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client('endpoint', 'token') as client:
                with mock.patch.object(BaseUnstructuredSolver, 'upload_problem', mock_upload):
                    solver = UnstructuredSolver(client, unstructured_solver_data())

                    # direct bqm sampling
                    ss = dimod.ExactSolver().sample(bqm)
                    session.post = lambda path, _: choose_reply(
                        path, {'problems/': complete_reply(ss)})

                    fut = solver.sample_bqm(bqm)

                    with self.assertRaises(type(mock_upload_exc)):
                        fut.result()

    def test_many_upload_failures(self):
        """Failure handling in high concurrency mode works correctly."""

        # build a test problem
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_upload_exc = ValueError('error')
        def mock_upload(self, bqm):
            return Present(exception=mock_upload_exc)

        # construct a functional solver by mocking client and api response data
        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client('endpoint', 'token') as client:
                with mock.patch.object(BaseUnstructuredSolver, 'upload_problem', mock_upload):
                    solver = BQMSolver(client, unstructured_solver_data())

                    futs = [solver.sample_bqm(bqm) for _ in range(100)]

                    for fut in futs:
                        with self.assertRaises(type(mock_upload_exc)):
                            fut.result()

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

import abc
import uuid
import unittest
from urllib.parse import urljoin
from itertools import chain

import requests_mock

try:
    import dimod
except ImportError:
    dimod = None

from dwave.cloud.client.base import Client
from dwave.cloud.coders import (
    encode_problem_as_qp, decode_qp,
    encode_problem_as_ref, decode_bq)
from dwave.cloud.api.resources import Problems
from dwave.cloud.api import exceptions, models, constants

from tests import config


class TestMockProblems(unittest.TestCase):

    token = str(uuid.uuid4())
    endpoint = 'http://test.com/path/'


@unittest.skipUnless(config, "SAPI access not configured")
class ProblemResourcesBaseTests(abc.ABC):
    """Basic tests for `dwave.cloud.api.resources.Problems`."""

    @classmethod
    @abc.abstractmethod
    def setUpClass(cls):
        """Create and submit one problem.

        Store as class attributes: `problem_data`, `problem_type`, sampling
        `params`, submitted `future`, `solver_id`.
        """

    def verify_problem_status(self, status: models.ProblemStatus, solved: bool = False):
        """Verify `status` consistent with the submitted problem."""
        self.assertEqual(status.id, self.future.id)
        self.assertEqual(status.type, self.problem_type)
        self.assertEqual(status.solver, self.future.solver.id)
        self.assertEqual(status.label, self.future.label)
        self.assertEqual(status.status, constants.ProblemStatus.COMPLETED)
        self.assertIsNotNone(status.submitted_on)
        if solved:
            self.assertIsNotNone(status.solved_on)

    @abc.abstractmethod
    def verify_problem_answer(self, answer: models.ProblemAnswer):
        """Verify `answer` consistent with the submitted problem."""

    def test_list_all(self):
        """List of all available problems retrieved."""

        ps = self.api.list_problems()
        self.assertIsInstance(ps, list)
        self.assertGreater(len(ps), 0)
        self.assertLessEqual(len(ps), 1000)     # sapi limit

        p = ps.pop()
        self.assertIsInstance(p, models.ProblemStatus)

    def test_list_filter(self):
        """Problem filtering works (on edges as well)."""

        # 404 on non-existing problem id
        with self.assertRaises(exceptions.ResourceNotFoundError):
            self.api.list_problems(id='nonexisting-problem-id')

        # result size limited
        ps = self.api.list_problems(max_results=1)
        self.assertEqual(len(ps), 1)

        # 400 on filtering by invalid status
        with self.assertRaises(exceptions.ResourceBadRequestError):
            self.api.list_problems(status='nonexisting-status-id')

        # empty result for non-existing solver
        ps = self.api.list_problems(solver='nonexisting-solver-id')
        self.assertEqual(len(ps), 0)

        # filtering by valid status
        ps = self.api.list_problems(status=constants.ProblemStatus.COMPLETED)
        self.assertGreater(len(ps), 0)

        # extract one completed problem for further tests
        problem = ps.pop()

        # filter by exact id
        ps = self.api.list_problems(id=problem.id)
        self.assertEqual(len(ps), 1)

        # verify all fields of the returned problem are identical to the original
        p = ps.pop()
        self.assertEqual(p, problem)

    def test_get_problem(self):
        """Problem status with answer retrieved by problem id."""

        problem_id = self.future.id
        status = self.api.get_problem(problem_id)

        self.assertIsInstance(status, models.ProblemStatusMaybeWithAnswer)
        self.verify_problem_status(status, solved=True)

        self.assertIsInstance(status.answer, models.ProblemAnswer)
        self.verify_problem_answer(status.answer)

    def test_get_problem_status(self):
        """Problem status retrieved by problem id."""

        problem_id = self.future.id
        status = self.api.get_problem_status(problem_id)

        self.assertIsInstance(status, models.ProblemStatus)
        self.verify_problem_status(status, solved=True)

    def test_get_problem_statuses(self):
        """Multiple problem statuses retrieved by problem ids."""

        problem_id = self.future.id
        ids = [problem_id] * 3
        statuses = self.api.get_problem_statuses(ids)

        self.assertEqual(len(statuses), len(ids))
        for status in statuses:
            self.assertIsInstance(status, models.ProblemStatus)
            self.verify_problem_status(status, solved=True)

    def test_get_problem_info(self):
        """Problem info (complete problem description) retrieved by problem id."""

        problem_id = self.future.id
        info = self.api.get_problem_info(problem_id)

        self.assertIsInstance(info, models.ProblemInfo)
        self.assertEqual(info.id, self.future.id)

        # data
        self.assertIsInstance(info.data, models.ProblemData)
        self.assertEqual(info.data, self.problem_data)

        # params
        self.assertEqual(info.params, self.params)

        # metadata
        self.assertIsInstance(info.metadata, models.ProblemMetadata)
        self.assertEqual(info.metadata.solver, self.future.solver.id)
        self.assertEqual(info.metadata.type, self.problem_type)
        self.assertEqual(info.metadata.label, self.future.label)
        self.assertEqual(info.metadata.status.value, self.future.remote_status)
        self.assertEqual(info.metadata.submitted_by, config['token'])
        self.assertIsNotNone(info.metadata.submitted_on)
        self.assertIsNotNone(info.metadata.solved_on)

        # answer
        self.assertIsInstance(info.answer, models.ProblemAnswer)
        self.verify_problem_answer(info.answer)

    def test_get_problem_answer(self):
        """Problem answer retrieved by problem id."""

        problem_id = self.future.id
        answer = self.api.get_problem_answer(problem_id)

        self.assertIsInstance(answer, models.ProblemAnswer)
        self.verify_problem_answer(answer)

    def test_get_problem_messages(self):
        """Problem messages retrieved by problem id."""

        problem_id = self.future.id
        messages = self.api.get_problem_messages(problem_id)

        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 0)

    def test_problem_submit(self):
        """Problem submitted."""

        status = self.api.submit_problem(
            data=self.problem_data,
            params=self.params,
            solver=self.solver_id,
            type=self.problem_type,
        )

        self.assertIsInstance(status, models.ProblemStatusMaybeWithAnswer)
        self.assertEqual(status.type, self.problem_type)
        self.assertEqual(status.solver, self.solver_id)
        self.assertIsNotNone(status.submitted_on)

        if status.status is constants.ProblemStatus.COMPLETED:
            self.assertIsInstance(status.answer, models.ProblemAnswer)
            self.verify_problem_answer(status.answer)

    def test_problem_submit_errors(self):
        """Problem submit fails due to invalid parameters."""

        with self.assertRaises(exceptions.ResourceBadRequestError):
            self.api.submit_problem(
                data=self.problem_data,
                params=dict(non_existing_param=1),
                solver=self.solver_id,
                type=self.problem_type,
            )

        with self.assertRaises(exceptions.ResourceNotFoundError):
            self.api.submit_problem(
                data=self.problem_data,
                params=self.params,
                solver='non-existing-solver',
                type=self.problem_type,
            )

    def test_problem_batch_submit(self):
        """Multiple problems are submitted and initial statuses returned."""

        job = models.ProblemJob(
            data=self.problem_data,
            params=self.params,
            solver=self.solver_id,
            type=self.problem_type,
        )

        statuses = self.api.submit_problems([job] * 3)

        self.assertIsInstance(statuses, list)
        self.assertEqual(len(statuses), 3)

        for status in statuses:
            self.assertIsInstance(status, models.ProblemInitialStatus)

    def test_problem_batch_submit_error(self):
        """Problem batch submit fails due to invalid parameters."""

        job = models.ProblemJob(
            data=self.problem_data,
            params=dict(non_existing_param=1),
            solver=self.solver_id,
            type=self.problem_type,
        )

        statuses = self.api.submit_problems([job])

        self.assertIsInstance(statuses, list)
        self.assertEqual(len(statuses), 1)

        for status in statuses:
            self.assertIsInstance(status, models.ProblemSubmitError)
            self.assertEqual(status.error_code, 400)


class TestCloudProblemsStructured(ProblemResourcesBaseTests, unittest.TestCase):
    """Verify `dwave.cloud.api.resources.Problems` handle structured problems."""

    @classmethod
    def setUpClass(cls):
        with Client(**config) as client:
            cls.api = Problems.from_client_config(client)

            # submit and solve an Ising problem as a fixture
            solver = client.get_solver(qpu=True)
            edge = next(iter(solver.edges))
            cls.linear = {}
            cls.quadratic = {edge: 1.0}
            qp = encode_problem_as_qp(solver, cls.linear, cls.quadratic)
            cls.problem_data = models.ProblemData.parse_obj(qp)
            cls.problem_type = constants.ProblemType.ISING
            cls.params = dict(num_reads=100)
            cls.future = solver.sample_ising(cls.linear, cls.quadratic, **cls.params)
            cls.solver_id = solver.id

            # double-check
            resolved = cls.future.result()
            assert cls.future.remote_status == constants.ProblemStatus.COMPLETED.value

    def verify_problem_answer(self, answer: models.ProblemAnswer):
        ans = decode_qp(msg=dict(answer=answer.dict(), type=self.problem_type.value))
        var = set(chain(*self.quadratic)) | self.linear.keys()
        self.assertEqual(set(ans['active_variables']), var)
        self.assertEqual(len(ans['energies']), len(ans['solutions']))
        self.assertEqual(sum(ans['num_occurrences']), self.params['num_reads'])


@unittest.skipUnless(dimod, "dimod not installed")
class TestCloudProblemsUnstructured(ProblemResourcesBaseTests, unittest.TestCase):
    """Verify `dwave.cloud.api.resources.Problems` handle unstructured problems."""

    @classmethod
    def setUpClass(cls):
        with Client(**config) as client:
            cls.api = Problems.from_client_config(client)

            # submit and solve an Ising problem as a fixture
            solver = client.get_solver(hybrid=True)
            cls.bqm = dimod.BQM.from_ising({}, {'ab': 1.0})
            problem_data_id = solver.upload_problem(cls.bqm).result()
            problem_data_ref = encode_problem_as_ref(problem_data_id)
            cls.problem_data = models.ProblemData.parse_obj(problem_data_ref)
            cls.problem_type = constants.ProblemType.BQM
            cls.params = dict(time_limit=3)
            cls.future = solver.sample_bqm(problem_data_id, **cls.params)
            cls.solver_id = solver.id

            # double-check
            resolved = cls.future.result()
            assert cls.future.remote_status == constants.ProblemStatus.COMPLETED.value

    def verify_problem_answer(self, answer: models.ProblemAnswer):
        ans = decode_bq(msg=dict(answer=answer.dict(), type=self.problem_type.value))
        ss = ans['sampleset']
        self.assertEqual(ss.variables, self.bqm.variables)
        dimod.testing.assert_sampleset_energies(ss, self.bqm)

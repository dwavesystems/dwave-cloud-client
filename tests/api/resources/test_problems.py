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
import io
import json
import uuid
import unittest
import datetime
from urllib.parse import urljoin, urlparse, parse_qs
from itertools import chain

import numpy
import requests_mock

from parameterized import parameterized

try:
    import dimod
except ImportError:
    dimod = None

from dwave.cloud.client.base import Client
from dwave.cloud.coders import (
    encode_problem_as_qp, decode_qp,
    encode_problem_as_ref, decode_bq, decode_binary_ref)
from dwave.cloud.api.resources import Problems
from dwave.cloud.api import exceptions, models, constants

from tests import config
from tests.api.mocks import (
    StructuredSapiMockResponses, UnstructuredSapiMockResponses,
    UnstructuredSapiMockResponsesWithBinaryRefAnswer)


class AssertionSatisfied(Exception):
    """Raise it to quickly unwrap the stack and signal assertion satisfied."""


class ProblemResourcesBaseTests(abc.ABC):
    """Basic tests for `dwave.cloud.api.resources.Problems`."""

    @classmethod
    def setUpClass(cls):
        """Create and submit one problem.

        Set the following attributes:
        - `linear`/`quadratic` (or `bqm`), `problem_data`
        - `problem_type`, `problem_id`, `problem_label`
        - `params`
        - `solver_id`

        Written to work with:
        - both mock data and live api
        - both structured and unstructured problems/answers/solvers
        """

    def verify_problem_status(self, status: models.ProblemStatus, solved: bool = False):
        """Verify `status` consistent with the submitted problem."""
        self.assertEqual(status.id, self.problem_id)
        self.assertEqual(status.type, self.problem_type)
        self.assertEqual(status.solver, self.solver_id)
        self.assertEqual(status.label, self.problem_label)
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

        # dev note: due to a SAPI bug/regression (SSW-11390), we can't simply
        # assert equality with: `self.assertEqual(p, problem)`; instead we have
        # to round all datetimes to a full second.
        def round_to_second(val):
            if isinstance(val, datetime.datetime):
                val = round(val.timestamp())
            return val
        def with_second_resolution_values(obj: dict):
            return {k: round_to_second(v) for k, v in obj.copy().items()}

        self.assertEqual(
            with_second_resolution_values(p.model_dump()),
            with_second_resolution_values(problem.model_dump()))

    def test_get_problem(self):
        """Problem status with answer retrieved by problem id."""

        problem_id = self.problem_id
        status = self.api.get_problem(problem_id)

        self.assertIsInstance(status, models.ProblemStatusMaybeWithAnswer)
        self.verify_problem_status(status, solved=True)

        self.assertIsInstance(status.answer, models.ProblemAnswer)
        self.verify_problem_answer(status.answer)

    def test_get_problem_status(self):
        """Problem status retrieved by problem id."""

        problem_id = self.problem_id
        status = self.api.get_problem_status(problem_id)

        self.assertIsInstance(status, models.ProblemStatus)
        self.verify_problem_status(status, solved=True)

    def test_get_problem_statuses(self):
        """Multiple problem statuses retrieved by problem ids."""

        problem_id = self.problem_id
        ids = [problem_id] * 3
        statuses = self.api.get_problem_statuses(ids)

        self.assertEqual(len(statuses), len(ids))
        for status in statuses:
            self.assertIsInstance(status, models.ProblemStatus)
            self.verify_problem_status(status, solved=True)

    def test_get_problem_info(self):
        """Problem info (complete problem description) retrieved by problem id."""

        problem_id = self.problem_id
        info = self.api.get_problem_info(problem_id)

        self.assertIsInstance(info, models.ProblemInfo)
        self.assertEqual(info.id, problem_id)

        # data
        self.assertIsInstance(info.data, models.ProblemData)
        self.assertEqual(info.data, self.problem_data)

        # params
        self.assertEqual(info.params, self.params)

        # metadata
        self.assertIsInstance(info.metadata, models.ProblemMetadata)
        self.assertEqual(info.metadata.solver, self.solver_id)
        self.assertEqual(info.metadata.type, self.problem_type)
        self.assertEqual(info.metadata.label, self.problem_label)
        self.assertEqual(info.metadata.status, constants.ProblemStatus.COMPLETED)
        self.assertEqual(info.metadata.submitted_by, self.token)
        self.assertIsNotNone(info.metadata.submitted_on)
        self.assertIsNotNone(info.metadata.solved_on)

        # answer
        self.assertIsInstance(info.answer, models.ProblemAnswer)
        self.verify_problem_answer(info.answer)

    def test_get_problem_answer(self):
        """Problem answer retrieved by problem id."""

        problem_id = self.problem_id
        answer = self.api.get_problem_answer(problem_id)

        self.assertIsInstance(answer, models.ProblemAnswer)
        self.verify_problem_answer(answer)

    def test_get_problem_answer_data(self):
        """Problem answer data retrieved from a binary-ref."""

        answer = self.api.get_problem_answer(problem_id=self.problem_id)

        if answer.format != constants.AnswerEncodingFormat.BINARY_REF:
            self.skipTest("no binary-ref answer data to download")

        data = self.api.get_answer_data(answer)

        self.assertIsInstance(data, io.IOBase)
        self.assertEqual(data.read(), self.p1.problem_answer_data().read())

    def test_get_problem_messages(self):
        """Problem messages retrieved by problem id."""

        problem_id = self.problem_id
        messages = self.api.get_problem_messages(problem_id)

        self.assertIsInstance(messages, list)
        if messages:
            self.assertIsInstance(messages[0], dict)

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

    def test_problem_cancel(self):
        """Cancel problem."""

        # first, submit a problem we can immediately cancel
        job = models.ProblemJob(
            data=self.problem_data,
            params=self.params,
            solver=self.solver_id,
            type=self.problem_type,
        )
        submitted = self.api.submit_problems(problems=[job]*2)
        problem_id = submitted[0].id

        # now cancel the submitted problem (hopefully hasn't finished already;
        # but catch `ResourceConflictError` in case it has, and `ProblemCancelError`
        # in case problem is in progress already)
        try:
            status = self.api.cancel_problem(problem_id=problem_id)
        except exceptions.ResourceConflictError:
            pass
        else:
            if isinstance(status, models.ProblemStatus):
                self.assertIs(status.status, constants.ProblemStatus.CANCELLED)
            elif isinstance(status, models.ProblemCancelError):
                self.assertIn('cancel problem in progress', status.error_msg)

    def test_finished_problem_cancel(self):
        """(Finished) problem cancel attempted."""

        with self.assertRaises(exceptions.ResourceConflictError):
            self.api.cancel_problem(self.problem_id)

    def test_problem_batch_cancel(self):
        """(Finished) problem cancel attempted."""

        problem_ids = [self.problem_id, self.problem_id]
        statuses = self.api.cancel_problems(problem_ids)

        self.assertIsInstance(statuses, list)
        self.assertEqual(len(statuses), len(problem_ids))

        for status in statuses:
            self.assertIsInstance(status, models.ProblemCancelError)
            self.assertEqual(status.error_code, 409)


class StructuredProblemTestsMixin:

    def verify_problem_answer(self, answer: models.ProblemAnswer):
        ans = decode_qp(msg=dict(answer=answer.model_dump(), type=self.problem_type.value))
        var = set(chain(*self.quadratic)) | self.linear.keys()
        self.assertEqual(set(ans['active_variables']), var)
        self.assertEqual(len(ans['energies']), len(ans['solutions']))
        self.assertEqual(sum(ans['num_occurrences']), self.params['num_reads'])


class UnstructuredProblemTestsMixin:

    def verify_problem_answer(self, answer: models.ProblemAnswer):
        if answer.format is constants.AnswerEncodingFormat.BQ:
            ans = decode_bq(
                msg=dict(answer=answer.model_dump(), type=self.problem_type.value))
            ss = ans['sampleset']
            self.assertEqual(ss.variables, self.bqm.variables)
            dimod.testing.assert_sampleset_energies(ss, self.bqm)
        elif answer.format is constants.AnswerEncodingFormat.BINARY_REF:
            mock_answer_data = b'123456'
            ans = decode_binary_ref(
                msg=dict(answer=answer.model_dump(by_alias=True), type=self.problem_type.value),
                ref_resolver=lambda **kw: mock_answer_data)
            answer_data = ans['answer']
            self.assertEqual(answer_data, mock_answer_data)
        else:
            raise ValueError(f"Unsupported answer encoding format: {answer.format!r}")


class ProblemResourcesMockerMixin:
    """Define and set up mock responses for all requests issued by
    `ProblemResourcesBaseTests` tests.
    """

    token = str(uuid.uuid4())
    endpoint = 'http://test.com/path/'

    def setUp(self):
        def url(path):
            return urljoin(self.endpoint, path)

        self.mocker = requests_mock.Mocker()

        headers = {'X-Auth-Token': self.token}
        self.mocker.get(requests_mock.ANY, status_code=401)
        self.mocker.get(requests_mock.ANY, status_code=404, request_headers=headers)

        # p1 is completed
        p1_id = self.p1.problem_id
        p1_status = self.p1.complete_no_answer_reply()
        p1_status_with_answer = self.p1.complete_reply()
        p1_params = self.params
        p1_metadata = self.p1.problem_metadata(submitted_by=self.token)
        p1_answer = self.p1.problem_answer()
        p1_answer_data = self.p1.problem_answer_data()
        p1_answer_data_uri = self.p1.problem_answer_data_uri()
        p1_messages = self.p1.problem_messages()
        p1_info = self.p1.problem_info(params=p1_params, metadata=p1_metadata)

        # p2 is submitted (and pending)
        p2_id = self.p2.problem_id
        p2_status = self.p2.continue_reply(id=p2_id)

        # p3 is cancelled
        p3_id = self.p3.problem_id
        p3_status = self.p3.cancel_reply(id=p3_id)

        all_problem_ids = {p1_id, p2_id, p3_id}
        all_solvers = [self.solver_id]
        all_statuses = {s.value for s in constants.ProblemStatus}

        # problem status et al
        self.mocker.get(
            url(f'problems/{p1_id}'),
            json=p1_status_with_answer,
            request_headers=headers)
        self.mocker.get(
            url(f'problems/?id={p1_id}'),
            complete_qs=True,
            json=[p1_status],
            request_headers=headers)
        self.mocker.get(
            url(f'problems/?id={p2_id}'),
            complete_qs=True,
            json=[p2_status],
            request_headers=headers)
        self.mocker.get(
            url(f'problems/?id={p3_id}'),
            complete_qs=True,
            json=[p3_status],
            request_headers=headers)
        self.mocker.get(
            url(f'problems/{p1_id}/info'),
            json=p1_info,
            request_headers=headers)
        self.mocker.get(
            url(f'problems/{p1_id}/answer'),
            json=p1_answer,
            request_headers=headers)
        self.mocker.get(
            p1_answer_data_uri,
            body=p1_answer_data,
            request_headers=headers)
        self.mocker.get(
            url(f'problems/{p1_id}/messages'),
            json=p1_messages,
            request_headers=headers)
        self.mocker.get(
            url(f'problems/?id={p1_id},{p1_id},{p1_id}'),
            complete_qs=True,
            json=[p1_status] * 3,
            request_headers=headers)

        # list problems
        self.mocker.get(
            url('problems/'),
            complete_qs=True,
            json=[p1_status, p2_status, p3_status],
            request_headers=headers)
        self.mocker.get(
            url(f'problems/?id={p1_id},{p2_id}'),
            complete_qs=True,
            json=[p1_status, p2_status],
            request_headers=headers)
        self.mocker.get(
            url('problems/?max_results=1'),
            json=[p1_status],
            request_headers=headers)
        self.mocker.get(
            url('problems/?status=COMPLETED'),
            json=[p1_status],
            request_headers=headers)

        def match_invalid_status(request):
            query = parse_qs(urlparse(request.url).query)
            status = query.get('status')
            return status is not None and status[0] not in all_statuses

        self.mocker.get(
            url('problems/'),
            additional_matcher=match_invalid_status,
            status_code=400,
            request_headers=headers)

        def match_invalid_solver(request):
            query = parse_qs(urlparse(request.url).query)
            solver = query.get('solver')
            return solver is not None and solver[0] not in all_solvers

        self.mocker.get(
            url('problems/'),
            additional_matcher=match_invalid_solver,
            json=[],
            request_headers=headers)

        def match_invalid_problem_id(request):
            query = parse_qs(urlparse(request.url).query)
            problem_id = query.get('id')
            return problem_id is not None and problem_id[0] not in all_problem_ids

        self.mocker.get(
            url('problems/'),
            complete_qs=True,
            additional_matcher=match_invalid_problem_id,
            status_code=404,
            request_headers=headers)

        # problem submit
        self.mocker.post(url('problems/'), json=p2_status, request_headers=headers)

        def match_invalid_problem_params(request):
            return request.json()['params'] != self.params

        self.mocker.post(
            url('problems/'),
            additional_matcher=match_invalid_problem_params,
            status_code=400,
            request_headers=headers)

        def match_invalid_problem_solver(request):
            return request.json()['solver'] != self.solver_id

        self.mocker.post(
            url('problems/'),
            additional_matcher=match_invalid_problem_solver,
            status_code=404,
            request_headers=headers)

        # problem batch submit
        def match_batch_submit(request):
            data = request.json()
            return isinstance(data, list) and len(data) == 3

        self.mocker.post(
            url('problems/'),
            additional_matcher=match_batch_submit,
            json=[p2_status] * 3,
            request_headers=headers)

        # problem submitted just to be cancelled (list of len 2)
        def match_batch_submit(request):
            data = request.json()
            return isinstance(data, list) and len(data) == 2

        self.mocker.post(
            url('problems/'),
            additional_matcher=match_batch_submit,
            json=[p2_status] * 2,
            request_headers=headers)

        def match_invalid_batch_submit(request):
            data = request.json()
            return isinstance(data, list) and len(data) == 1 and data[0]['params'] != self.params

        self.mocker.post(
            url('problems/'),
            additional_matcher=match_invalid_batch_submit,
            json=[self.p1.immediate_error_reply(code=400, msg='Unknown parameter')],
            request_headers=headers)

        # problem cancel (finished problem)
        self.mocker.delete(url(f'problems/{p1_id}'), status_code=409, request_headers=headers)
        self.mocker.delete(
            url('problems/'),
            json=[self.p1.immediate_error_reply(code=409, msg='Problem has been finished.')] * 2,
            request_headers=headers)

        # problem cancel (pending problem)
        self.mocker.delete(
            url(f'problems/{p2_id}'),
            json=self.p2.cancel_reply(),
            request_headers=headers)

        self.mocker.start()

        self.api = Problems(token=self.token, endpoint=self.endpoint, version_strict_mode=False)

    def tearDown(self):
        self.mocker.stop()

        self.api.close()


class TestMockProblemsStructured(StructuredProblemTestsMixin,
                                 ProblemResourcesMockerMixin,
                                 ProblemResourcesBaseTests,
                                 unittest.TestCase):
    """Verify `dwave.cloud.api.resources.Problems` handle structured problems
    using mocked SAPI responses.
    """

    @classmethod
    def setUpClass(cls):
        """Configure attributes required (used) by ProblemResourcesBaseTests."""

        cls.p1 = StructuredSapiMockResponses()
        cls.p2 = StructuredSapiMockResponses()
        cls.p3 = StructuredSapiMockResponses()

        cls.linear, cls.quadratic = cls.p1.problem
        cls.problem_data = models.ProblemData.model_validate(cls.p1.problem_data())
        cls.problem_type = constants.ProblemType(cls.p1.problem_type)
        cls.problem_id = cls.p1.problem_id
        cls.problem_label = cls.p1.problem_label

        cls.params = dict(num_reads=100)
        cls.solver_id = cls.p1.solver.id


@unittest.skipUnless(dimod, "dimod not installed")
class TestMockProblemsUnstructured(UnstructuredProblemTestsMixin,
                                   ProblemResourcesMockerMixin,
                                   ProblemResourcesBaseTests,
                                   unittest.TestCase):
    """Verify `dwave.cloud.api.resources.Problems` handle unstructured problems
    with answers in the `bq` format, using mocked SAPI responses.
    """

    @classmethod
    def setUpClass(cls):
        """Configure attributes required (used) by ProblemResourcesBaseTests."""

        cls.p1 = UnstructuredSapiMockResponses()
        cls.p2 = UnstructuredSapiMockResponses()
        cls.p3 = UnstructuredSapiMockResponses()

        cls.bqm = cls.p1.problem
        cls.problem_data = models.ProblemData.model_validate(cls.p1.problem_data())
        cls.problem_type = constants.ProblemType(cls.p1.problem_type)
        cls.problem_id = cls.p1.problem_id
        cls.problem_label = cls.p1.problem_label

        cls.params = dict(time_limit=3)
        cls.solver_id = cls.p1.solver.id


@unittest.skipUnless(dimod, "dimod not installed")
class TestMockProblemsUnstructuredWithBinaryRefAsnwer(TestMockProblemsUnstructured):
    """Verify `dwave.cloud.api.resources.Problems` handle unstructured problems
    with answers in the `binary-ref` format, using mocked SAPI responses.
    """

    @classmethod
    def setUpClass(cls):
        """Configure attributes required (used) by ProblemResourcesBaseTests."""

        cls.p1 = UnstructuredSapiMockResponsesWithBinaryRefAnswer()
        cls.p2 = UnstructuredSapiMockResponsesWithBinaryRefAnswer()
        cls.p3 = UnstructuredSapiMockResponsesWithBinaryRefAnswer()

        cls.bqm = cls.p1.problem
        cls.problem_data = models.ProblemData.model_validate(cls.p1.problem_data())
        cls.problem_type = constants.ProblemType(cls.p1.problem_type)
        cls.problem_id = cls.p1.problem_id
        cls.problem_label = cls.p1.problem_label

        cls.params = dict(time_limit=3)
        cls.solver_id = cls.p1.solver.id


@unittest.skipUnless(config, "SAPI access not configured")
class TestCloudProblemsStructured(StructuredProblemTestsMixin,
                                  ProblemResourcesBaseTests,
                                  unittest.TestCase):
    """Verify `dwave.cloud.api.resources.Problems` handle structured problems
    as returned by SAPI.
    """

    @classmethod
    def setUpClass(cls):
        """Configure attributes required (used) by ProblemResourcesBaseTests."""

        with Client(**config) as client:
            cls.token = client.config.token
            cls.api = Problems.from_config(client.config)

            # submit and solve an Ising problem as a fixture
            solver = client.get_solver(qpu=True)
            cls.solver_id = solver.id
            edge = next(iter(solver.edges))
            cls.linear = {}
            cls.quadratic = {edge: 1.0}
            qp = encode_problem_as_qp(solver, cls.linear, cls.quadratic)
            cls.problem_data = models.ProblemData.model_validate(qp)
            cls.problem_type = constants.ProblemType.ISING
            cls.params = dict(num_reads=100)
            future = solver.sample_ising(cls.linear, cls.quadratic, **cls.params)
            resolved = future.result()
            cls.problem_id = future.id
            cls.problem_label = future.label

            # double-check
            assert future.remote_status == constants.ProblemStatus.COMPLETED.value

    @classmethod
    def tearDownClass(cls):
        cls.api.close()


@unittest.skipUnless(dimod, "dimod not installed")
@unittest.skipUnless(config, "SAPI access not configured")
class TestCloudProblemsUnstructured(UnstructuredProblemTestsMixin,
                                    ProblemResourcesBaseTests,
                                    unittest.TestCase):
    """Verify `dwave.cloud.api.resources.Problems` handle unstructured problems
    as returned by SAPI.
    """

    @classmethod
    def setUpClass(cls):
        """Configure attributes required (used) by ProblemResourcesBaseTests."""

        with Client(**config) as client:
            cls.token = client.config.token
            cls.api = Problems.from_config(client.config)

            # submit and solve an Ising problem as a fixture
            solver = client.get_solver(hybrid=True)
            cls.solver_id = solver.id
            cls.bqm = dimod.BQM.from_ising({}, {'ab': 1.0})
            problem_data_id = solver.upload_problem(cls.bqm).result()
            problem_data_ref = encode_problem_as_ref(problem_data_id)
            cls.problem_data = models.ProblemData.model_validate(problem_data_ref)
            cls.problem_type = constants.ProblemType.BQM
            cls.params = dict(time_limit=3)
            future = solver.sample_bqm(problem_data_id, **cls.params)
            resolved = future.result()
            cls.problem_id = future.id
            cls.problem_label = future.label

            # double-check
            assert future.remote_status == constants.ProblemStatus.COMPLETED.value

    @classmethod
    def tearDownClass(cls):
        cls.api.close()


class NumpyParamsSerialization(unittest.TestCase):

    # basic scalar types we support for parameter values
    NUMPY_TYPES_AS_PYTHON = [
        (numpy.bool_(1), True),
        (numpy.byte(1), 1), (numpy.int8(1), 1),
        (numpy.ubyte(1), 1), (numpy.uint8(1), 1),
        (numpy.short(1), 1), (numpy.int16(1), 1),
        (numpy.ushort(1), 1), (numpy.uint16(1), 1),
        (numpy.intc(1), 1), (numpy.int32(1), 1),
        (numpy.uintc(1), 1), (numpy.uint32(1), 1),
        (numpy.int_(1), 1), (numpy.int32(1), 1),
        (numpy.uint(1), 1), (numpy.uint32(1), 1),
        (numpy.longlong(1), 1), (numpy.int64(1), 1),
        (numpy.ulonglong(1), 1), (numpy.uint64(1), 1),
        (numpy.half(1.0), 1.0), (numpy.float16(1.0), 1.0),
        (numpy.single(1.0), 1.0), (numpy.float32(1.0), 1.0),
        (numpy.double(1.0), 1.0), (numpy.float64(1.0), 1.0),
        (numpy.longdouble(1.0), 1.0)
    ] + ([
        (numpy.float128(1.0), 1.0)      # unavailable on windows
    ] if hasattr(numpy, 'float128') else [
    ])

    @classmethod
    def setUpClass(cls):
        _p = StructuredSapiMockResponses()
        cls.linear, cls.quadratic = _p.problem
        cls.problem_data = models.ProblemData.model_validate(_p.problem_data())
        cls.problem_type = constants.ProblemType(_p.problem_type)
        cls.problem_id = _p.problem_id
        cls.problem_label = _p.problem_label
        cls.params = {}
        cls.solver_id = _p.solver.id
        cls.api = Problems(token='token', endpoint='http://end.point/')

    @parameterized.expand(NUMPY_TYPES_AS_PYTHON)
    def test_problem_submit(self, np_val, py_val):
        user_params = dict(num_reads=np_val)
        expected_params = dict(num_reads=py_val)

        def verify_params(*args, **kwargs):
            body = json.loads(kwargs.get('data'))
            params = body.get('params')
            if params != expected_params:
                raise AssertionError("params don't match")
            raise AssertionSatisfied

        with unittest.mock.patch.object(self.api.session, 'post', verify_params):
            with self.assertRaises(AssertionSatisfied):
                self.api.submit_problem(
                    data=self.problem_data,
                    params=user_params,
                    solver=self.solver_id,
                    type=self.problem_type,
                )

    @parameterized.expand(NUMPY_TYPES_AS_PYTHON)
    def test_problem_batch_submit(self, np_val, py_val):
        user_params = dict(num_reads=np_val)
        expected_params = dict(num_reads=py_val)

        def verify_params(*args, **kwargs):
            body = json.loads(kwargs.get('data'))
            params = body[0].get('params')
            if params != expected_params:
                raise AssertionError("params don't match")
            raise AssertionSatisfied

        job = models.ProblemJob(
            data=self.problem_data,
            params=user_params,
            solver=self.solver_id,
            type=self.problem_type,
        )

        with unittest.mock.patch.object(self.api.session, 'post', verify_params):
            with self.assertRaises(AssertionSatisfied):
                self.api.submit_problems([job])

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

import io
import json
import unittest
from unittest import mock
from parameterized import parameterized

import numpy

from dwave.cloud.client import Client
from dwave.cloud.solver import (
    BaseUnstructuredSolver, UnstructuredSolver, BQMSolver, CQMSolver, DQMSolver)
from dwave.cloud.concurrency import Present

try:
    import dimod
except ImportError:
    raise unittest.SkipTest("dimod required for unstructured solver tests")


def unstructured_solver_data(problem_type='bqm'):
    return {
        "properties": {
            "supported_problem_types": [problem_type],
            "parameters": {"num_reads": "Number of samples to return."}
        },
        "id": "test-unstructured-solver",
        "description": "A test unstructured solver"
    }

def complete_reply_bq(sampleset, id_="problem-id", type_='bqm', label=None):
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
        "type": type_,
        "id": id_,
        "label": label
    }])

def complete_reply_binary_ref(answer_data_uri, id_="problem-id", type_='cqm', label=None):
    """Reply with `answer_data_uri` in binary-ref."""

    return json.dumps([{
        "status": "COMPLETED",
        "solver": "solver-name",
        "solved_on": "2019-07-31T12:34:56Z",
        "submitted_on": "2019-07-31T12:34:56Z",
        "answer": {
            "format": "binary-ref",
            "auth_method": "sapi-token",
            "url": answer_data_uri,
            "timing": {
                "qpu_access_time": 1
            }
        },
        "type": type_,
        "id": id_,
        "label": label
    }])

def answer_data_reply(data):
    return data

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

    def test_sample_bqm_immediate_reply(self):
        """Construction of and sampling from an unstructured BQM solver works."""

        # build a test problem
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, bqm_file):
            bqm_file.close()
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.multiple(Client, create_session=lambda self: session,
                                 upload_problem_encoded=mock_upload):
            with Client(endpoint='endpoint', token='token') as client:
                solver = BQMSolver(client, unstructured_solver_data())

                # make sure this still works
                _ = UnstructuredSolver(client, unstructured_solver_data())

                # direct bqm sampling
                ss = dimod.ExactSolver().sample(bqm)
                ss.info.update(problem_id=mock_problem_id)
                session.post = lambda path, _: choose_reply(
                    path, {'problems/': complete_reply_bq(ss, id_=mock_problem_id)})

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
                ss.info.update(problem_id=mock_problem_id)
                session.post = lambda path, _: choose_reply(
                    path, {'problems/': complete_reply_bq(ss, id_=mock_problem_id)})

                fut = solver.sample_ising(lin, quad)
                numpy.testing.assert_array_equal(fut.sampleset, ss)
                numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
                numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
                numpy.testing.assert_array_equal(fut.num_occurrences, ss.record.num_occurrences)

                # qubo sampling
                qubo, _ = bqm.to_qubo()
                ss = dimod.ExactSolver().sample_qubo(qubo)
                ss.info.update(problem_id=mock_problem_id)
                session.post = lambda path, _: choose_reply(
                    path, {'problems/': complete_reply_bq(ss, id_=mock_problem_id)})

                fut = solver.sample_qubo(qubo)
                numpy.testing.assert_array_equal(fut.sampleset, ss)
                numpy.testing.assert_array_equal(fut.samples, ss.record.sample)
                numpy.testing.assert_array_equal(fut.energies, ss.record.energy)
                numpy.testing.assert_array_equal(fut.num_occurrences, ss.record.num_occurrences)

    def test_sample_cqm_smoke_test(self):
        """Construction of and sampling from an unstructured CQM solver works."""

        # construct a small 3-variable CQM of mixed vartypes
        try:
            import dimod
            mixed = dimod.QM()
            mixed.add_variable('BINARY', 'a')
            mixed.add_variable('SPIN', 'b')
            mixed.add_variable('INTEGER', 'c')
            cqm = dimod.CQM()
            cqm.set_objective(mixed)
            cqm.add_constraint(mixed, rhs=1, sense='==')
        except:
            # dimod or dimod with CQM support not available, so just use a mock
            cqm = mock.Mock()
            cqm.to_file.return_value = io.BytesIO(b'123')

        problem_type = 'cqm'

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, bqm_file):
            bqm_file.close()
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.multiple(Client, create_session=lambda self: session,
                                 upload_problem_encoded=mock_upload):
            with Client(endpoint='endpoint', token='token') as client:
                solver = CQMSolver(client, unstructured_solver_data(problem_type=problem_type))

                # use bqm for mock response (for now)
                ss = dimod.ExactSolver().sample(dimod.BQM.empty('SPIN'))
                ss.info.update(problem_id=mock_problem_id)
                session.post = lambda path, _: choose_reply(
                    path, {'problems/': complete_reply_bq(ss, id_=mock_problem_id, type_=problem_type)})

                # verify decoding works
                fut = solver.sample_cqm(cqm)
                numpy.testing.assert_array_equal(fut.sampleset, ss)
                numpy.testing.assert_array_equal(fut.problem_type, problem_type)

    def test_sample_dqm_smoke_test(self):
        """Construction of and sampling from an unstructured DQM solver works."""

        try:
            import dimod
            dqm = dimod.DQM()
            dqm.add_variable(5)
            dqm.add_variable(7)
            dqm.set_linear_case(0, 3, 1.5)
            dqm.set_quadratic(0, 1, {(0, 1): 1.5, (3, 4): 1})
        except:
            # dimod or dimod with DQM support not available, so just use a mock
            dqm = mock.Mock()
            dqm.to_file.return_value = io.BytesIO(b'123')

        problem_type = 'dqm'

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, bqm_file):
            bqm_file.close()
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.multiple(Client, create_session=lambda self: session,
                                 upload_problem_encoded=mock_upload):
            with Client(endpoint='endpoint', token='token') as client:
                solver = DQMSolver(client, unstructured_solver_data(problem_type=problem_type))

                # use bqm for mock response (for now)
                ss = dimod.ExactSolver().sample(dimod.BQM.empty('SPIN'))
                ss.info.update(problem_id=mock_problem_id)
                session.post = lambda path, _: choose_reply(
                    path, {'problems/': complete_reply_bq(ss, id_=mock_problem_id, type_=problem_type)})

                # verify decoding works
                fut = solver.sample_dqm(dqm)
                numpy.testing.assert_array_equal(fut.sampleset, ss)
                numpy.testing.assert_array_equal(fut.problem_type, problem_type)

    def test_upload_failure(self):
        """Submit should gracefully fail if upload as part of submit fails."""

        # build a test problem
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_upload_exc = ValueError('error')
        def mock_upload(self, bqm_file):
            return Present(exception=mock_upload_exc)

        # construct a functional solver by mocking client and api response data
        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(endpoint='endpoint', token='token') as client:
                with mock.patch.object(BaseUnstructuredSolver, 'upload_problem', mock_upload):
                    solver = UnstructuredSolver(client, unstructured_solver_data())

                    # direct bqm sampling
                    ss = dimod.ExactSolver().sample(bqm)
                    session.post = lambda path, _: choose_reply(
                        path, {'problems/': complete_reply_bq(ss)})

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
        def mock_upload(self, bqm_file):
            return Present(exception=mock_upload_exc)

        # construct a functional solver by mocking client and api response data
        with mock.patch.object(Client, 'create_session', lambda self: session):
            with Client(endpoint='endpoint', token='token') as client:
                with mock.patch.object(BaseUnstructuredSolver, 'upload_problem', mock_upload):
                    solver = BQMSolver(client, unstructured_solver_data())

                    futs = [solver.sample_bqm(bqm) for _ in range(100)]

                    for fut in futs:
                        with self.assertRaises(type(mock_upload_exc)):
                            fut.result()


class TestProblemLabel(unittest.TestCase):

    class PrimaryAssertionSatisfied(Exception):
        """Raised by `on_submit_label_verifier` to signal correct label."""

    def on_submit_label_verifier(self, expected_label):
        """Factory for mock Client._submit() that will verify existence, and
        optionally validate label value.
        """

        # replacement for Client._submit()
        def _submit(client, body_data, computation):
            body = json.loads(body_data.result())

            if 'label' not in body:
                if expected_label is None:
                    raise TestProblemLabel.PrimaryAssertionSatisfied
                else:
                    raise AssertionError("label field missing")

            label = body['label']
            if label != expected_label:
                raise AssertionError(
                    "unexpected label value: {!r} != {!r}".format(label, expected_label))

            raise TestProblemLabel.PrimaryAssertionSatisfied

        return _submit

    @parameterized.expand([
        ("undefined", None),
        ("empty", ""),
        ("string", "text label")
    ])
    @mock.patch.object(Client, 'create_session', lambda client: mock.Mock())
    def test_label_is_sent(self, name, label):
        """Problem label is set on problem submit."""

        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, bqm_file):
            bqm_file.close()
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.multiple(Client, create_session=lambda self: session,
                                 upload_problem_encoded=mock_upload):
            with Client(endpoint='endpoint', token='token') as client:
                solver = BQMSolver(client, unstructured_solver_data())

                problems = [("sample_ising", (bqm.linear, bqm.quadratic)),
                            ("sample_qubo", (bqm.quadratic,)),
                            ("sample_bqm", (bqm,))]

                for method_name, problem_args in problems:
                    with self.subTest(method_name=method_name):
                        sample = getattr(solver, method_name)

                        with mock.patch.object(Client, '_submit', self.on_submit_label_verifier(label)):

                            with self.assertRaises(self.PrimaryAssertionSatisfied):
                                sample(*problem_args, label=label).result()

    @parameterized.expand([
        ("undefined", None),
        ("empty", ""),
        ("string", "text label")
    ])
    def test_label_is_received(self, name, label):
        """Problem label is set from response in result/sampleset."""

        # build a test problem
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, bqm_file):
            bqm_file.close()
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.multiple(Client, create_session=lambda self: session,
                                 upload_problem_encoded=mock_upload):
            with Client(endpoint='endpoint', token='token') as client:
                solver = BQMSolver(client, unstructured_solver_data())

                # construct mock response
                ss = dimod.ExactSolver().sample(bqm)
                session.post = lambda path, _: choose_reply(
                    path, {'problems/': complete_reply_bq(ss, id_=mock_problem_id, label=label)})

                # sample and verify label
                fut = solver.sample_bqm(bqm, label=label)
                info = fut.sampleset.info
                self.assertIn('problem_id', info)
                if label is not None:
                    self.assertIn('problem_label', info)
                    self.assertEqual(info['problem_label'], label)

                # verify sampleset is cached via weakref
                self.assertIsInstance(fut._sampleset(), dimod.SampleSet)


class TestAnswerDownloadFromBinaryRef(unittest.TestCase):

    def test_binary_ref_answer_download(self):
        # mock problem
        cqm = mock.Mock()
        cqm.to_file.return_value = io.BytesIO(b'123')

        mock_answer_data = b'abc'

        problem_type = 'cqm'

        # use a global mocked session, so we can modify it on the fly
        session = mock.Mock()
        setattr(session, '__enter__', mock.Mock())
        setattr(session, '__exit__', mock.Mock())
        session.__enter__.return_value = session
        session.__exit__.return_value = None
        session.get.return_value.iter_content.return_value = iter([mock_answer_data])

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, cqm_file):
            cqm_file.close()
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.multiple(Client, create_session=lambda _: session,
                                 upload_problem_encoded=mock_upload):
            with Client(endpoint='endpoint', token='token') as client:
                solver = CQMSolver(client, unstructured_solver_data(problem_type=problem_type))

                # solver has to support binary-ref
                solver._handled_encoding_formats.add('binary-ref')

                # use mock answer data
                answer_url = f'/problems/{mock_problem_id}/answer/data/'
                session.post = lambda path, _: choose_reply(
                    path, {
                        'problems/': complete_reply_binary_ref(answer_url, id_=mock_problem_id, type_=problem_type),
                        answer_url: answer_data_reply(mock_answer_data),
                    })

                # verify decoding works
                fut = solver.sample_cqm(cqm)
                self.assertIsInstance(fut.answer_data, io.IOBase)
                self.assertEqual(fut.answer_data.read(), mock_answer_data)
                self.assertEqual(fut.problem_type, problem_type)


class TestSerialization(unittest.TestCase):

    class AssertionSatisfied(Exception):
        """Raised by `on_submit_data_verifier` to signal correct serialization."""

    def on_submit_data_verifier(self, expected_params):
        """Factory for mock Client._submit() that will validate parameter values."""

        # replacement for Client._submit(), called with exact network request data
        def _submit(client, body_data, computation):
            body = json.loads(body_data.result())

            params = body.get('params')
            if params != expected_params:
                raise AssertionError("params don't match")

            raise TestSerialization.AssertionSatisfied

        return _submit

    @parameterized.expand([
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
    ]))
    @mock.patch.object(Client, 'create_session', lambda client: mock.Mock())
    def test_params_are_serialized(self, np_val, py_val):
        """Parameters supplied as NumPy types are correctly serialized."""

        user_params = dict(num_reads=np_val)
        expected_params = dict(num_reads=py_val)

        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        # use a global mocked session, so we can modify it on-the-fly
        session = mock.Mock()

        # upload is now part of submit, so we need to mock it
        mock_problem_id = 'mock-problem-id'
        def mock_upload(self, bqm_file):
            bqm_file.close()
            return Present(result=mock_problem_id)

        # construct a functional solver by mocking client and api response data
        with mock.patch.multiple(Client, create_session=lambda self: session,
                                 upload_problem_encoded=mock_upload):
            with Client(endpoint='endpoint', token='token') as client:
                solver = BQMSolver(client, unstructured_solver_data())

                problems = [("sample_ising", (bqm.linear, bqm.quadratic)),
                            ("sample_qubo", (bqm.quadratic,)),
                            ("sample_bqm", (bqm,))]

                for method_name, problem_args in problems:
                    with self.subTest(method_name=method_name, np_val=np_val, py_val=py_val):
                        sample = getattr(solver, method_name)

                        with mock.patch.object(
                                Client, '_submit', self.on_submit_data_verifier(expected_params)):

                            with self.assertRaises(self.AssertionSatisfied):
                                sample(*problem_args, **user_params).result()

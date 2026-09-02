# Copyright 2025 D-Wave
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

import unittest

from enum import Enum

from pydantic import ValidationError
from parameterized import parameterized

from dwave.cloud.api import constants, models
from dwave.cloud.testing.mocks import structured_solver_data, unstructured_solver_data

from tests.api.mocks import StructuredSapiMockResponses


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    @parameterized.expand([
        (False, ),
        (True, 'unknown_top_level_field', 'value'),
    ])
    def test_solver_models(self, include_extra, extra_field_name=None, extra_field_value=None):
        with self.subTest('structured solver'):
            name = 'qpu-solver'
            graph_id = '01abcd1234'
            data = structured_solver_data(name, graph_id)
            if include_extra:
                data[extra_field_name] = extra_field_value
            solver = models.SolverConfiguration(**data)
            self.assertIsNotNone(solver.get('identity'))
            self.assertEqual(solver.identity.name, name)
            self.assertEqual(solver.identity.version.graph_id, graph_id)
            self.assertIsNotNone(solver.get('properties'))
            self.assertEqual(solver.properties['category'], 'qpu')
            if include_extra:
                self.assertEqual(getattr(solver, extra_field_name), extra_field_value)

        with self.subTest('unstructured solver'):
            name = 'hybrid-solver'
            data = unstructured_solver_data(name)
            if include_extra:
                data[extra_field_name] = extra_field_value
            solver = models.SolverConfiguration(**data)
            self.assertEqual(solver.identity.name, name)
            self.assertIsNone(solver.identity.version)
            self.assertEqual(solver.properties['category'], 'hybrid')
            if include_extra:
                self.assertEqual(getattr(solver, extra_field_name), extra_field_value)

        with self.subTest('filtered configuration contains identity'):
            name = 'qpu-solver'
            graph_id = '01abcd1234'
            data = structured_solver_data(name, graph_id)
            filtered_data = dict(identity=data['identity'])
            if include_extra:
                filtered_data[extra_field_name] = extra_field_value
            solver = models.SolverConfiguration(**filtered_data)
            self.assertEqual(solver.identity.name, name)
            self.assertEqual(solver.identity.version.graph_id, graph_id)
            self.assertIsNone(solver.get('properties'))
            if include_extra:
                self.assertEqual(getattr(solver, extra_field_name), extra_field_value)

    def test_solver_identity_model(self):
        # test validation, construction and basic serialization with `.dict()`/`str()`
        name = 'qpu-solver'
        graph_id = '01abcd1234'

        # name required
        with self.assertRaises(ValidationError):
            models.SolverIdentity()

        with self.subTest('minimal solver identity'):
            self.assertEqual(models.SolverIdentity(name=name).dict(), dict(name=name))
            self.assertEqual(models.SolverIdentity(name=name, version=None).dict(), dict(name=name))

        with self.subTest('allow empty version'):
            data = {"name": name, "version": {}}
            identity = models.SolverIdentity.model_validate(data)
            self.assertEqual(identity.dict(), data)
            self.assertEqual(str(identity), f"{name}")

        with self.subTest('minimal qpu solver identity'):
            data = {"name": name, "version": {"graph_id": graph_id}}
            identity = models.SolverIdentity.model_validate(data)
            self.assertEqual(identity.dict(), data)
            self.assertEqual(str(identity), f"{name};graph_id={graph_id}")

        with self.subTest('allow additional version specs'):
            extra = "1.0"
            data = {"name": name, "version": {"graph_id": graph_id, "param_id": extra}}
            identity = models.SolverIdentity.model_validate(data)
            self.assertEqual(identity.dict(), data)
            self.assertEqual(str(identity), f"{name};graph_id={graph_id};param_id={extra}")

        with self.subTest('identity equality'):
            # compares to dict
            data = {"name": name, "version": {"graph_id": graph_id}}
            identity = models.SolverIdentity.model_validate(data)
            self.assertEqual(identity, data)

            # compares to SolverIdentity
            identity2 = models.SolverIdentity.model_validate(data)
            self.assertEqual(identity, identity2)

            # compares with empty version as well
            data = {"name": name}
            identity = models.SolverIdentity.model_validate(data)
            self.assertEqual(identity, data)

        with self.subTest('version equality'):
            # compares to dict
            data = {"name": name, "version": {"graph_id": graph_id}}
            identity = models.SolverIdentity.model_validate(data)
            self.assertEqual(identity.version.dict(), data['version'])
            self.assertEqual(identity.version, data['version'])

            # compares to SolverVersion
            version = models.SolverVersion.model_validate(data['version'])
            self.assertEqual(identity.version, version)

    @parameterized.expand([
        (dict(name='hss'), 'hss'),
        (dict(name='qpu', version=dict(graph_id='123')), 'qpu;graph_id=123'),
        (dict(name='qpu', version=dict(a='a', b='b')), 'qpu;a=a;b=b'),
        (dict(name='qpu;a=b', version=dict(a=';', b='=')), 'qpu%3Ba%3Db;a=%3B;b=%3D'),
        (dict(name='|_%.:', version={";": '"'}), '%7C_%25.%3A;%3B=%22'),
    ])
    def test_solver_identity_serialization(self, identity_dict, id_string):
        # test advanced serialization and deserialization with `.to_id()`/`.from_id()`

        identity = models.SolverIdentity.model_validate(identity_dict)

        with self.subTest('serialization'):
            self.assertEqual(str(identity), id_string)
            self.assertEqual(identity.to_id(), id_string)

        with self.subTest('deserialization'):
            self.assertEqual(models.SolverIdentity.from_id(id_string).dict(), identity_dict)

        with self.subTest('from id to id via model'):
            self.assertEqual(models.SolverIdentity.from_id(id_string).to_id(), id_string)

        with self.subTest('from model to model via string'):
            self.assertEqual(models.SolverIdentity.from_id(identity.to_id()), identity)

    @parameterized.expand([
        (dict(), ),
        (dict(extra_field='value'), ),
    ])
    def test_problem_models(self, extras):
        def _validate_extras(model):
            if not extras:
                return
            for key, val in extras.items():
                self.assertEqual(getattr(model, key), val)

        with self.subTest('ProblemStatus'):
            status = models.ProblemStatus(**self.sapi.complete_no_answer_reply(**extras))
            _validate_extras(status)

        with self.subTest('ProblemStatusWithAnswer'):
            status = models.ProblemStatusWithAnswer(**self.sapi.complete_reply(**extras))
            _validate_extras(status)

        with self.subTest('ProblemAnswer'):
            answer = models.ProblemAnswer(**self.sapi.answer)

        with self.subTest('ProblemStatusMaybeWithAnswer'):
            s1 = models.ProblemStatusMaybeWithAnswer(**self.sapi.complete_no_answer_reply(**extras))
            s2 = models.ProblemStatusMaybeWithAnswer(**self.sapi.complete_reply(**extras))
            self.assertEqual(s1.id, s2.id)
            self.assertIsNone(s1.answer)
            self.assertEqual(s2.answer, answer)
            _validate_extras(s1)
            _validate_extras(s2)

        with self.subTest('ProblemData'):
            data = models.ProblemData(**self.sapi.problem_data(**extras))
            _validate_extras(data)

        with self.subTest('ProblemMetadata'):
            metadata = models.ProblemMetadata(**self.sapi.problem_metadata(**extras))
            self.assertEqual(metadata.label, status.label)
            self.assertEqual(metadata.status, status.status)
            _validate_extras(metadata)

        with self.subTest('ProblemInfo'):
            info = models.ProblemInfo(**self.sapi.problem_info(**extras))
            _validate_extras(info)
            info = models.ProblemInfo(**self.sapi.problem_info(answer=None))

        with self.subTest('ProblemJob'):
            job = models.ProblemJob.from_info(info)

            self.assertEqual(job.data, info.data)
            self.assertEqual(job.params, info.params)
            self.assertEqual(job.solver, status.solver)
            self.assertEqual(job.type, status.type)
            self.assertEqual(job.label, status.label)


class TestConstants(unittest.TestCase):

    def test_open_enum(self):
        class Color(constants._OpenStrEnumMixin, str, Enum):
            RED = "RED"

        with self.subTest("standard string enum"):
            red = Color("RED")
            self.assertEqual(red.value, "RED")
            self.assertEqual(red.name, "RED")
            self.assertEqual(red, "RED")

        with self.subTest("unknown value added"):
            blue = Color("BLUE")
            self.assertEqual(blue.value, "BLUE")
            self.assertEqual(blue.name, "BLUE")
            self.assertEqual(blue, "BLUE")

        with self.subTest("comparison of unknown values"):
            another = Color("BLUE")
            self.assertEqual(blue, another)
            self.assertIs(blue, another)

    def test_closed_enum(self):
        class Color(str, Enum):
            RED = "RED"

        self.assertEqual(Color("RED").value, "RED")

        with self.assertRaises(ValueError):
            Color("BLUE")

    @parameterized.expand([
        (constants.ProblemStatus, "PENDING"),
        (constants.ProblemEncodingFormat, "qp"),
        (constants.AnswerEncodingFormat, "binary-ref"),
        (constants.BinaryRefAuthMethod, "sapi-token"),
        (constants.ProblemType, "ising"),
        (constants.DeprecationContext, "api"),
    ])
    def test_sapi_enums_are_open(self, cls, known):
        self.assertEqual(cls(known), known)
        self.assertEqual(cls("UNKNOWN"), "UNKNOWN")

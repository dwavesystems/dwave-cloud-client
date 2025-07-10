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

from dwave.cloud.api import models
from dwave.cloud.testing.mocks import structured_solver_data, unstructured_solver_data

from tests.api.mocks import StructuredSapiMockResponses


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    def test_solver_models(self):
        with self.subTest('structured solver'):
            name = 'qpu-solver'
            graph_id = '01abcd1234'
            solver = models.SolverConfiguration(**structured_solver_data(name, graph_id))
            self.assertIsNotNone(solver.get('identity'))
            self.assertEqual(solver.identity.name, name)
            self.assertEqual(solver.identity.version.graph_id, graph_id)
            self.assertIsNotNone(solver.get('properties'))
            self.assertEqual(solver.properties['category'], 'qpu')

        with self.subTest('unstructured solver'):
            name = 'hybrid-solver'
            solver = models.SolverConfiguration(**unstructured_solver_data(name))
            self.assertEqual(solver.identity.name, name)
            self.assertIsNone(solver.identity.version)
            self.assertEqual(solver.properties['category'], 'hybrid')

        with self.subTest('filtered configuration contains identity'):
            name = 'qpu-solver'
            graph_id = '01abcd1234'
            data = structured_solver_data(name, graph_id)
            filtered_data = dict(identity=data['identity'])
            solver = models.SolverConfiguration(**filtered_data)
            self.assertEqual(solver.identity.name, name)
            self.assertEqual(solver.identity.version.graph_id, graph_id)
            self.assertIsNone(solver.get('properties'))

    def test_problem_models(self):
        with self.subTest('ProblemStatus'):
            status = models.ProblemStatus(**self.sapi.complete_no_answer_reply())

        with self.subTest('ProblemStatusWithAnswer'):
            status = models.ProblemStatusWithAnswer(**self.sapi.complete_reply())

        with self.subTest('ProblemAnswer'):
            answer = models.ProblemAnswer(**self.sapi.answer)

        with self.subTest('ProblemStatusMaybeWithAnswer'):
            s1 = models.ProblemStatusMaybeWithAnswer(**self.sapi.complete_no_answer_reply())
            s2 = models.ProblemStatusMaybeWithAnswer(**self.sapi.complete_reply())
            self.assertEqual(s1.id, s2.id)
            self.assertIsNone(s1.answer)
            self.assertEqual(s2.answer, answer)

        with self.subTest('ProblemData'):
            data = models.ProblemData(**self.sapi.problem_data())

        with self.subTest('ProblemMetadata'):
            metadata = models.ProblemMetadata(**self.sapi.problem_metadata())

            self.assertEqual(metadata.label, status.label)
            self.assertEqual(metadata.status, status.status)

        with self.subTest('ProblemInfo'):
            info = models.ProblemInfo(**self.sapi.problem_info())
            info = models.ProblemInfo(**self.sapi.problem_info(answer=None))

        with self.subTest('ProblemJob'):
            job = models.ProblemJob.from_info(info)

            self.assertEqual(job.data, data)
            self.assertEqual(job.params, info.params)
            self.assertEqual(job.solver, status.solver)
            self.assertEqual(job.type, status.type)
            self.assertEqual(job.label, status.label)

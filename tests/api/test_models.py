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

from tests.api.mocks import StructuredSapiMockResponses


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sapi = StructuredSapiMockResponses()

    def test_construction(self):
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

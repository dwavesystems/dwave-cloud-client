# Copyright 2017 D-Wave Systems Inc.
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

"""
Check the interface to the solver object.

Note:
    These are all aggregate tests, not really testing single units.
"""

import random
import unittest
import warnings
from datetime import datetime

import numpy
from parameterized import parameterized

try:
    import dimod
except ImportError:
    dimod = None

import dwave.cloud.computation
from dwave.cloud.client import Client
from dwave.cloud.exceptions import (
    CanceledFutureError, SolverFailureError, InvalidProblemError)
from dwave.cloud.utils.qubo import evaluate_ising, generate_random_ising_problem


from tests import config


@unittest.skipUnless(config, "No live server configuration available.")
class PropertyLoading(unittest.TestCase):
    """Ensure that the properties of solvers can be retrieved."""

    def test_load_properties(self):
        """Ensure that the propreties are populated."""
        with Client(**config) as client:
            solver = client.get_solver()
            self.assertTrue(len(solver.properties) > 0)

    def test_load_parameters(self):
        """Make sure the parameters are populated."""
        with Client(**config) as client:
            solver = client.get_solver()
            self.assertTrue(len(solver.parameters) > 0)

    def test_submit_invalid_parameter(self):
        """Ensure that the parameters are populated."""
        with Client(**config) as client:
            solver = client.get_solver()
            self.assertNotIn('not_a_parameter', solver.parameters)
            with self.assertRaises(KeyError):
                solver.sample_ising({}, {}, not_a_parameter=True)

    def test_submit_experimental_parameter(self):
        """Ensure that the experimental parameters are populated."""
        with Client(**config) as client:
            solver = client.get_solver()
            self.assertNotIn('x_test', solver.parameters)
            with self.assertRaises(SolverFailureError):
                self.assertTrue(solver.sample_ising([0], {}, x_test=123).result())

    def test_read_connectivity(self):
        """Ensure that the edge set is populated."""
        with Client(**config) as client:
            solver = client.get_solver()
            self.assertTrue(len(solver.edges) > 0)

    def test_max_num_reads(self):
        with Client(**config) as client:
            solver = client.get_solver()

            if solver.qpu:
                # for lower anneal time num_reads is bounded by num_reads_range
                anneal_time = 10 * solver.properties['default_annealing_time']
                num_reads = solver.max_num_reads(annealing_time=anneal_time)
                # doubling the anneal_time, num_reads halves
                self.assertEqual(num_reads // 2, solver.max_num_reads(annealing_time=2*anneal_time))
            else:
                self.assertEqual(solver.max_num_reads(),
                                 solver.properties['num_reads_range'][1])


class _QueryTest(unittest.TestCase):
    def _submit_and_check(self, solver, linear, quad, **kwargs):
        results = solver.sample_ising(linear, quad, num_reads=100, **kwargs)

        # Did we get the right number of samples?
        self.assertEqual(100, sum(results.num_occurrences))

        # verify .occurrences property still works, although is deprecated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(100, sum(results.num_occurrences))

        # offset is optional
        offset = kwargs.get('offset', 0)

        # Make sure the number of occurrences and energies are all correct
        for energy, state in zip(results.energies, results.samples):
            self.assertAlmostEqual(
                energy, evaluate_ising(linear, quad, state, offset=offset))

        # label is optional
        label = kwargs.get('label')
        if label is not None:
            self.assertEqual(results.label, label)

        return results

@unittest.skipUnless(config, "No live server configuration available.")
class Submission(_QueryTest):
    """Submit some sample problems."""

    def test_result_structure(self):
        with Client(**config) as client:
            solver = client.get_solver()
            h = {next(iter(solver.nodes)): 0}
            computation = solver.sample_ising(h, {}, answer_mode='histogram')
            result = computation.result()
            self.assertIn('solutions', result)
            self.assertIn('energies', result)
            self.assertIn('num_occurrences', result)
            self.assertIn('timing', result)

    def test_future_structure(self):
        with Client(**config) as client:
            solver = client.get_solver()
            h = {next(iter(solver.nodes)): 0}
            computation = solver.sample_ising(h, {})
            _ = computation.result()
            self.assertIsInstance(computation.id, str)
            self.assertEqual(computation.remote_status, Client.STATUS_COMPLETE)
            self.assertEqual(computation.solver, solver)
            self.assertIsInstance(computation.time_received, datetime)
            self.assertIsInstance(computation.time_solved, datetime)

    def test_submit_extra_qubit(self):
        """Submit a defective problem with an unsupported variable."""

        with Client(**config) as client:
            solver = client.get_solver()

            # Build a linear problem and add a variable that shouldn't exist
            linear, quad = generate_random_ising_problem(solver)
            linear[max(solver.nodes) + 1] = 1

            with self.assertRaises(InvalidProblemError):
                results = solver.sample_ising(linear, quad)
                results.samples

    def test_submit_linear_problem(self):
        """Submit a problem with all the linear terms populated."""

        with Client(**config) as client:
            solver = client.get_solver()
            linear, quad = generate_random_ising_problem(solver)
            self._submit_and_check(solver, linear, {})

    def test_submit_full_problem(self):
        """Submit a problem with all supported coefficients set."""

        with Client(**config) as client:
            solver = client.get_solver()
            linear, quad = generate_random_ising_problem(solver)
            self._submit_and_check(solver, linear, quad)

    def test_submit_full_ising_problem_with_offset(self):
        """Submit a problem with all supported coefficients set, including energy offset."""

        with Client(**config) as client:
            solver = client.get_solver()
            linear, quad = generate_random_ising_problem(solver)
            self._submit_and_check(solver, linear, quad, offset=3)

    def test_submit_list_problem(self):
        """Submit a problem using a list for the linear terms."""

        with Client(**config) as client:
            solver = client.get_solver()

            linear = [1 if qubit in solver.nodes else 0 for qubit in range(0, max(solver.nodes)+1)]
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            self._submit_and_check(solver, linear, quad)

    def test_submit_partial_problem(self):
        """Submit a problem with only some of the terms set."""

        with Client(**config) as client:
            solver = client.get_solver()

            # Build a linear problem, then remove half the qubits
            linear, quad = generate_random_ising_problem(solver)
            nodes = list(solver.nodes)
            for index in nodes[0:len(nodes)//2]:
                del linear[index]
                quad = {key: value for key, value in quad.items() if index not in key}

            self._submit_and_check(solver, linear, quad)

    def test_problem_label(self):
        """Problem label is set."""

        with Client(**config) as client:
            solver = client.get_solver()
            linear, quad = generate_random_ising_problem(solver)
            self._submit_and_check(solver, linear, quad, label="test")

    @unittest.skipUnless(dimod, "dimod required for 'Solver.sample_bqm'")
    def test_submit_bqm_ising_problem(self):
        """Submit an Ising BQM with all supported coefficients set."""

        with Client(**config) as client:
            solver = client.get_solver()

            linear, quad = generate_random_ising_problem(solver)
            offset = 3

            # sample ising as bqm
            bqm = dimod.BinaryQuadraticModel.from_ising(linear, quad, offset)
            response = solver.sample_bqm(bqm, num_reads=100)
            sampleset = response.sampleset

            self.assertEqual(sampleset.wait_id(), sampleset.info['problem_id'])
            self.assertEqual(response.id, sampleset.info['problem_id'])

            # Did we get the right number of samples?
            self.assertEqual(100, sum(response.num_occurrences))

            # Make sure the number of occurrences and energies are all correct
            numpy.testing.assert_array_almost_equal(
                bqm.energies(sampleset), sampleset.record.energy)

    @unittest.skipUnless(dimod, "dimod required for 'Solver.sample_bqm'")
    def test_submit_bqm_qubo_problem(self):
        """Submit a QUBO BQM with all supported coefficients set."""

        with Client(**config) as client:
            solver = client.get_solver()

            _, quad = generate_random_ising_problem(solver)
            offset = 5

            # sample qubo as bqm
            bqm = dimod.BinaryQuadraticModel.from_qubo(quad, offset)
            response = solver.sample_bqm(bqm, num_reads=100)
            sampleset = response.sampleset

            self.assertEqual(sampleset.wait_id(), sampleset.info['problem_id'])
            self.assertEqual(response.id, sampleset.info['problem_id'])

            # Did we get the right number of samples?
            self.assertEqual(100, sum(response.num_occurrences))

            # Make sure the number of occurrences and energies are all correct
            numpy.testing.assert_array_almost_equal(
                bqm.energies(sampleset), sampleset.record.energy)

    @unittest.skipUnless(dimod, "dimod required for 'Solver.sample_bqm'")
    def test_all_sampling_methods_are_consistent(self):
        """Submit Ising/QUBO/BQM and verify the results are consistent."""

        with Client(**config) as client:
            solver = client.get_solver()

            # simple problem with a large energy gap
            # (ground state: [-1, -1] @ -2.0)
            n1, n2 = next(iter(solver.edges))
            h = {n1: 1, n2: 1}
            J = {(n1, n2): -1}
            offset = 1.0

            bqm = dimod.BinaryQuadraticModel.from_ising(h, J, offset)
            params = dict(num_reads=100)

            # sample_ising
            response = solver.sample_ising(h, J, offset, **params)
            ss_ising = response.sampleset

            # sample_qubo
            qubo = bqm.to_qubo()
            response = solver.sample_qubo(*qubo, **params)
            ss_qubo = response.sampleset

            # sample_bqm
            response = solver.sample_bqm(bqm, **params)
            ss_bqm = response.sampleset

            # this simple problem should always be solved to optimality
            self.assertTrue(len(ss_ising) == len(ss_qubo) == len(ss_bqm) == 1)

            # make sure all energies are correct
            numpy.testing.assert_array_almost_equal(
                bqm.energies(ss_ising), ss_ising.record.energy)
            numpy.testing.assert_array_almost_equal(
                ss_ising.record.energy, ss_qubo.record.energy)
            numpy.testing.assert_array_almost_equal(
                ss_qubo.record.energy, ss_bqm.record.energy)

    def test_reverse_annealing(self):
        with Client(**config) as client:
            solver = client.get_solver()

            # skip if we don't have access to the initial_state parameter for reverse annealing
            if 'initial_state' not in solver.parameters:
                raise unittest.SkipTest

            anneal_schedule = [(0, 1), (55.00000000000001, 0.45), (155.0, 0.45), (210.0, 1)]

            # make some subset of the qubits active
            active = [v for v in solver.properties['qubits'] if v % 2]

            # make an initial state, ising problem
            initial_state = {v: 2*bool(v % 3)-1 for v in active}

            h = {v: 0.0 for v in active}
            J = {(u, v): -1 for (u, v) in solver.properties['couplers'] if v in h and u in h}

            # doesn't catch fire
            solver.sample_ising(h, J, anneal_schedule=anneal_schedule, initial_state=initial_state).samples

    def test_reverse_annealing_off_vartype(self):
        with Client(**config) as client:
            solver = client.get_solver()

            # skip if we don't have access to the initial_state parameter for reverse annealing
            if 'initial_state' not in solver.parameters:
                raise unittest.SkipTest

            anneal_schedule = [(0, 1), (55.00000000000001, 0.45), (155.0, 0.45), (210.0, 1)]

            # make some subset of the qubits active
            active = [v for v in solver.properties['qubits'] if v % 2]

            # ising state, qubo problem
            initial_state_ising = {v: 2*bool(v % 3)-1 for v in active}
            Q = {(v, v): 1 for v in active}
            fq = solver.sample_qubo(Q, anneal_schedule=anneal_schedule, initial_state=initial_state_ising)

            # qubo state, ising problem
            initial_state_qubo = {v: bool(v % 3) for v in active}
            h = {v: 1 for v in active}
            J = {}
            fi = solver.sample_ising(h, J, anneal_schedule=anneal_schedule, initial_state=initial_state_qubo)

            fq.samples
            fi.samples

    def test_submit_batch(self):
        """Submit batch of problems."""

        with Client(**config) as client:
            solver = client.get_solver()

            result_list = []
            for _ in range(100):
                linear, quad = generate_random_ising_problem(solver)
                results = solver.sample_ising(linear, quad, num_reads=10)
                result_list.append([results, linear, quad])

            for results, linear, quad in result_list:
                # Did we get the right number of samples?
                self.assertEqual(10, sum(results.num_occurrences))

                # Make sure the number of occurrences and energies are all correct
                for energy, state in zip(results.energies, results.samples):
                    self.assertAlmostEqual(energy, evaluate_ising(linear, quad, state))

    def test_cancel_batch(self):
        """Submit batch of problems, then cancel them."""

        with Client(**config) as client:
            solver = client.get_solver()

            linear, quad = generate_random_ising_problem(solver)

            max_num_reads = max(solver.properties.get('num_reads_range', [1, 100]))
            num_reads = min(max_num_reads, 50)

            result_list = []
            for _ in range(100):
                results = solver.sample_ising(linear, quad, num_reads=num_reads)
                result_list.append([results, linear, quad])

            [r[0].cancel() for r in result_list]

            for results, linear, quad in result_list:
                # Responses must be canceled or correct
                try:
                    # Did we get the right number of samples?
                    self.assertEqual(num_reads, sum(results.num_occurrences))

                    # Make sure the number of occurrences and energies are all correct
                    for energy, state in zip(results.energies, results.samples):
                        self.assertAlmostEqual(energy, evaluate_ising(linear, quad, state))

                except CanceledFutureError:
                    pass

    def test_wait_many(self):
        """Submit a batch of problems then use `wait_multiple` to wait on all of them."""

        with Client(**config) as client:
            solver = client.get_solver()

            linear, quad = generate_random_ising_problem(solver)

            result_list = []
            for _ in range(100):
                results = solver.sample_ising(linear, quad, num_reads=40)
                result_list.append([results, linear, quad])

            dwave.cloud.computation.Future.wait_multiple([f[0] for f in result_list])

            for results, _, _ in result_list:
                self.assertTrue(results.done())

            for results, linear, quad in result_list:
                # Did we get the right number of samples?
                self.assertEqual(40, sum(results.num_occurrences))

                # Make sure the number of occurrences and energies are all correct
                for energy, state in zip(results.energies, results.samples):
                    self.assertAlmostEqual(energy, evaluate_ising(linear, quad, state))

    def test_as_completed(self):
        """Submit a batch of problems then use `as_completed` to iterate over
        all of them."""

        with Client(**config) as client:
            solver = client.get_solver()

            linear, quad = generate_random_ising_problem(solver)

            # Sample the solution 100x40 times
            computations = [solver.sample_ising(linear, quad, num_reads=40) for _ in range(100)]

            # Go over computations, one by one, as they're done and check they're OK
            for computation in dwave.cloud.computation.Future.as_completed(computations):
                self.assertTrue(computation.done())
                self.assertEqual(40, sum(computation.num_occurrences))
                for energy, state in zip(computation.energies, computation.samples):
                    self.assertAlmostEqual(energy, evaluate_ising(linear, quad, state))


@unittest.skipUnless(config, "No live server configuration available.")
class UnstructuredSubmission(unittest.TestCase):
    """Smoke test for hybrid solver live submit."""

    @parameterized.expand([
        ("bqm", "sample_bqm", 3, lambda: dimod.BQM.from_qubo({})),
        ("cqm", "sample_cqm", 5, lambda: dimod.CQM.from_bqm(dimod.BQM.from_qubo({'ab': 1}))),
        ("dqm", "sample_dqm", 5, lambda: dimod.DQM.from_numpy_vectors([0], [0], ([], [], []))),
        ("nl", "sample_bqm", 1, lambda: dimod.BQM.from_ising({0: 1}, {})),
    ])
    @unittest.skipUnless(dimod, "dimod required to submit problems to Leap hybrid solvers")
    def test_sample(self, problem_type, sample_meth, time_limit, problem_gen):

        # use default config; skip loading config file, assume token in env
        with Client.from_config(config_file=False) as client:
            solver = client.get_solver(supported_problem_types__contains=problem_type)
            problem = problem_gen()

            f = getattr(solver, sample_meth)(problem, time_limit=time_limit)
            f.result()

            self.assertEqual(f.problem_type, problem_type)
            if f.answer_data is None:
                # bqm/cqm/dqm
                self.assertIsInstance(f.sampleset, dimod.SampleSet)
                self.assertEqual(f.sampleset.info.get('problem_id'), f.wait_id())
            else:
                # nlm
                self.assertGreater(len(f.answer_data.read()), 0)

    def test_sample_nl(self):
        # test model states in addition to a simple smoke test above
        try:
            from dwave.optimization import Model
        except ImportError:
            self.skipTest("dwave-optimization required to submit NL problems to Leap hybrid solvers")

        problem_type = "nl"
        time_limit = 1

        def model_gen():
            # create a simple model
            model = Model()
            x = model.list(5)
            W = model.constant(numpy.arange(25).reshape((5, 5)))
            model.minimize(W[x, :][:, x].sum())
            return model

        # use default config; skip loading config file, assume token in env
        with Client.from_config(config_file=False) as client:
            solver = client.get_solver(supported_problem_types__contains=problem_type)
            model = model_gen()

            f = solver.sample_nlm(model, time_limit=time_limit)
            f.result()

            model.states.from_file(f.answer_data)
            self.assertGreater(model.states.size(), 0)


@unittest.skipUnless(config, "No live server configuration available.")
class DecodingMethod(_QueryTest):
    """Test different decoding behaviors.

    For now, we will just set the _numpy flag in the future module to control
    how the module acts without it.
    """

    def setUp(self):
        """Reload the future module to undo any changes."""
        from importlib import reload
        reload(dwave.cloud.computation)

    @classmethod
    def tearDownClass(cls):
        """Reload the future module to undo any changes."""
        from importlib import reload
        reload(dwave.cloud.computation)

    def test_request_matrix_with_no_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            dwave.cloud.computation._numpy = False
            solver = client.get_solver()
            solver.return_matrix = True

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            with self.assertRaises(ValueError):
                self._submit_and_check(solver, linear, quad)

    def test_request_matrix_with_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            assert dwave.cloud.computation._numpy
            solver = client.get_solver()
            solver.return_matrix = True

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            result = self._submit_and_check(solver, linear, quad)
            self.assertIsInstance(result.samples, numpy.ndarray)
            self.assertIsInstance(result.energies, numpy.ndarray)
            self.assertIsInstance(result.num_occurrences, numpy.ndarray)

    def test_request_list_with_no_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            dwave.cloud.computation._numpy = False
            solver = client.get_solver()
            solver.return_matrix = False

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            self._submit_and_check(solver, linear, quad)

    def test_request_list_with_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            assert dwave.cloud.computation._numpy
            solver = client.get_solver()
            solver.return_matrix = False

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            self._submit_and_check(solver, linear, quad)

    def test_request_raw_matrix_with_no_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            dwave.cloud.computation._numpy = False
            solver = client.get_solver()
            solver.return_matrix = True

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            with self.assertRaises(ValueError):
                self._submit_and_check(solver, linear, quad, answer_mode='raw')

    def test_request_raw_matrix_with_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            assert dwave.cloud.computation._numpy
            solver = client.get_solver()
            solver.return_matrix = True

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            result = self._submit_and_check(solver, linear, quad, answer_mode='raw')
            self.assertIsInstance(result.samples, numpy.ndarray)
            self.assertIsInstance(result.energies, numpy.ndarray)
            self.assertIsInstance(result.num_occurrences, numpy.ndarray)

    def test_request_raw_list_with_no_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            dwave.cloud.computation._numpy = False
            solver = client.get_solver()
            solver.return_matrix = False

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            self._submit_and_check(solver, linear, quad, answer_mode='raw')

    def test_request_raw_list_with_numpy(self):
        """Submit a problem using a dict for the linear terms."""
        # Connect
        with Client(**config) as client:
            assert dwave.cloud.computation._numpy
            solver = client.get_solver()
            solver.return_matrix = False

            # Build a problem
            linear = {index: random.choice([-1, 1]) for index in solver.nodes}
            quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

            # Solve the problem
            self._submit_and_check(solver, linear, quad)


if __name__ == '__main__':
    unittest.main()

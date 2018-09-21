"""
Check the interface to the solver object.

Note:
    These are all aggregate tests, not really testing single units.
"""
from __future__ import absolute_import, division

import unittest
import random
from datetime import datetime

import numpy

from dwave.cloud.utils import evaluate_ising, generate_random_ising_problem
from dwave.cloud.qpu import Client
from dwave.cloud.exceptions import CanceledFutureError, SolverFailureError
import dwave.cloud.computation

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


class _QueryTest(unittest.TestCase):
    def _submit_and_check(self, solver, linear, quad, **param):
        results = solver.sample_ising(linear, quad, num_reads=100, **param)

        # Did we get the right number of samples?
        self.assertEqual(100, sum(results.occurrences))

        # Make sure the number of occurrences and energies are all correct
        for energy, state in zip(results.energies, results.samples):
            self.assertAlmostEqual(energy, evaluate_ising(linear, quad, state))

        return results


@unittest.skipUnless(config, "No live server configuration available.")
class Submission(_QueryTest):
    """Submit some sample problems."""

    def test_result_structure(self):
        with Client(**config) as client:
            solver = client.get_solver()
            computation = solver.sample_ising({}, {})
            result = computation.result()
            self.assertIn('samples', result)
            self.assertIn('energies', result)
            self.assertIn('occurrences', result)
            self.assertIn('timing', result)

    def test_future_structure(self):
        with Client(**config) as client:
            solver = client.get_solver()
            computation = solver.sample_ising({}, {})
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

            with self.assertRaises(ValueError):
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
                self.assertEqual(10, sum(results.occurrences))

                # Make sure the number of occurrences and energies are all correct
                for energy, state in zip(results.energies, results.samples):
                    self.assertAlmostEqual(energy, evaluate_ising(linear, quad, state))

    def test_cancel_batch(self):
        """Submit batch of problems, then cancel them."""

        with Client(**config) as client:
            solver = client.get_solver()

            linear, quad = generate_random_ising_problem(solver)

            max_num_reads = max(solver.properties.get('num_reads_range', [1, 100]))

            result_list = []
            for _ in range(1000):
                results = solver.sample_ising(linear, quad, num_reads=max_num_reads)
                result_list.append([results, linear, quad])

            [r[0].cancel() for r in result_list]

            for results, linear, quad in result_list:
                # Responses must be canceled or correct
                try:
                    # Did we get the right number of samples?
                    self.assertEqual(max_num_reads, sum(results.occurrences))

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
                self.assertEqual(40, sum(results.occurrences))

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
                self.assertEqual(40, sum(computation.occurrences))
                for energy, state in zip(computation.energies, computation.samples):
                    self.assertAlmostEqual(energy, evaluate_ising(linear, quad, state))


@unittest.skipUnless(config, "No live server configuration available.")
class DecodingMethod(_QueryTest):
    """Test different decoding behaviors.

    For now, we will just set the _numpy flag in the future module to control
    how the module acts without it.
    """

    def setUp(self):
        """Reload the future module to undo any changes."""
        from six.moves import reload_module
        reload_module(dwave.cloud.computation)

    @classmethod
    def tearDownClass(cls):
        """Reload the future module to undo any changes."""
        from six.moves import reload_module
        reload_module(dwave.cloud.computation)

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
            self.assertIsInstance(result.occurrences, numpy.ndarray)

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
            self.assertIsInstance(result.occurrences, numpy.ndarray)

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

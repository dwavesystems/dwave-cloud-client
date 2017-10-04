"""Make sure the dwave_sapi2 interface still works.

Currently only tests by running the examples from the developer guide
"Connecting to the Solver" chapter.
"""
from __future__ import absolute_import, division

import unittest
import mock

import dwave_micro_client
import dwave_micro_client as dwave_sapi2
import dwave_micro_client.remote as dwave_sapi2_remote
import dwave_micro_client.core as dwave_sapi2_core

try:
    url, token, proxy_url, solver_name = dwave_micro_client.load_configuration()
    if None in [url, token, solver_name]:
        raise ValueError()
    skip_live = False
except:
    skip_live = True


class Dwave2DevGuideExamples(unittest.TestCase):
    """Ensure backwords compatability with dwave_sapi2 package where possible."""

    def setUp(self):
        """Replace `dwave_sapi2` with `dwave_micro_client` for some imports."""
        self.module_patcher = mock.patch.dict('sys.modules', {
            'dwave_sapi2': dwave_sapi2,
            'dwave_sapi2.remote': dwave_sapi2_remote,
            'dwave_sapi2.core': dwave_sapi2_core,
        })
        self.module_patcher.start()

    def tearDown(self):
        """Let normal import behaviour resume."""
        self.module_patcher.stop()

    def is_answer(self, data):
        """Check if an answer has the expectd format."""
        self.assertIsInstance(data['energies'], (list, tuple))
        self.assertIsInstance(data['num_occurrences'], (list, tuple))
        self.assertIsInstance(data['solutions'], (list, tuple))

    @unittest.skipIf(skip_live, "No live server available.")
    def test_remote_connection(self):
        from dwave_sapi2.remote import RemoteConnection
        remote_connection = RemoteConnection(url, token)
        remote_connection = RemoteConnection(url, token, proxy_url)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_remote_connection_example2(self):
        from dwave_sapi2.remote import RemoteConnection

        # define the url and a valid token
        # url = "http://myURL"
        # token = "myToken001"

        # solver_name = "solver_name"

        # create a remote connection using url and token
        remote_connection = RemoteConnection(url, token)
        # get a solver
        solver = remote_connection.get_solver(solver_name)

        # get solver's properties
        self.assertIsInstance(solver.properties, dict)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_solve_ising_example(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import solve_ising

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        # solve ising problem
        h = [1, -1, 1, 1, -1, 1, 1]
        J = {(0, 6): -10}

        params = {"num_reads": 10, "num_spin_reversal_transforms": 2}
        answer_1 = solve_ising(solver, h, J, **params)
        self.is_answer(answer_1)

        answer_2 = solve_ising(solver, h, J, num_reads=10)
        self.is_answer(answer_2)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_solve_qubo_example(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import solve_qubo

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        # solve qubo problem
        Q = {(0, 5): -10}

        params = {"num_reads": 10}
        answer_1 = solve_qubo(solver, Q, **params)
        self.is_answer(answer_1)

        answer_2 = solve_qubo(solver, Q, num_reads=10)
        self.is_answer(answer_2)

    @unittest.skipIf(skip_live, "No live server available.")
    def test_async_solve_ising_example(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import async_solve_ising, await_completion

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        h = [1, -1, 1, 1, -1, 1, 1]
        J = {(0, 6): -10}

        submitted_problem = async_solve_ising(solver, h, J, num_reads=10)

        # Wait until solved
        await_completion([submitted_problem], 1, float('inf'))

        # display result
        self.is_answer(submitted_problem.result())

    @unittest.skipIf(skip_live, "No live server available.")
    def test_async_solve_qubo_example(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import async_solve_qubo, await_completion

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        Q = {(0, 5): -10}

        submitted_problem = async_solve_qubo(solver, Q, num_reads=10)

        # Wait until solved
        await_completion([submitted_problem], 1, float('inf'))

        # display result
        self.is_answer(submitted_problem.result())

    @unittest.skipIf(skip_live, "No live server available.")
    def test_await_completion_example(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import async_solve_ising, await_completion

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        h = [1, -1, 1, 1, -1, 1, 1]
        J = {(0, 6): -10}

        p1 = async_solve_ising(solver, h, J, num_reads=10)
        p2 = async_solve_ising(solver, h, J, num_reads=20)

        min_done = 2
        timeout = 1.0
        done = await_completion([p1, p2], min_done, timeout)

        if done:
            self.is_answer(p1.result())
            self.is_answer(p2.result())


class NonExampleTests(unittest.TestCase):

    def setUp(self):
        """Replace `dwave_sapi2` with `dwave_micro_client` for some imports."""
        self.module_patcher = mock.patch.dict('sys.modules', {
            'dwave_sapi2': dwave_sapi2,
            'dwave_sapi2.remote': dwave_sapi2_remote,
            'dwave_sapi2.core': dwave_sapi2_core,
        })
        self.module_patcher.start()

    def tearDown(self):
        """Let normal import behaviour resume."""
        self.module_patcher.stop()

    def is_answer(self, data):
        """Check if an answer has the expectd format."""
        self.assertIsInstance(data['energies'], (list, tuple))
        self.assertIsInstance(data['num_occurrences'], (list, tuple))
        self.assertIsInstance(data['solutions'], (list, tuple))

    @unittest.skipIf(skip_live, "No live server available.")
    def test_async_status(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import async_solve_qubo, await_completion

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        Q = {(0, 5): -10}

        submitted_problem = async_solve_qubo(solver, Q, num_reads=10)
        self.assertEqual(submitted_problem.status()['remote_status'], None)
        self.assertEqual(submitted_problem.status()['state'], 'SUBMITTING')

        # Wait until solved
        await_completion([submitted_problem], 1, float('inf'))

        # display result
        self.is_answer(submitted_problem.result())
        self.assertEqual(submitted_problem.status()['remote_status'], RemoteConnection.STATUS_COMPLETE)
        self.assertEqual(submitted_problem.status()['state'], 'DONE')

    @unittest.skipIf(skip_live, "No live server available.")
    def test_async_bad_status(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import async_solve_qubo, await_completion

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        Q = {(0, 5): -float('inf')}

        submitted_problem = async_solve_qubo(solver, Q, num_reads=10)

        #
        await_completion([submitted_problem], 1, float('inf'))
        self.assertEqual(submitted_problem.status()['remote_status'], RemoteConnection.STATUS_FAILED)
        self.assertEqual(submitted_problem.status()['state'], 'DONE')
        self.assertEqual(submitted_problem.status()['error_type'], 'SOLVE')

    @unittest.skipIf(skip_live, "No live server available.")
    def test_async_retry(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import async_solve_qubo, await_completion

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        Q = {(0, 5): -10}

        submitted_problem = async_solve_qubo(solver, Q, num_reads=10)
        self.assertEqual(submitted_problem.status()['remote_status'], None)
        self.assertEqual(submitted_problem.status()['state'], 'SUBMITTING')

        # Wait until solved
        await_completion([submitted_problem], 1, float('inf'))

        # display result
        self.is_answer(submitted_problem.result())
        self.assertEqual(submitted_problem.status()['remote_status'], RemoteConnection.STATUS_COMPLETE)
        self.assertEqual(submitted_problem.status()['state'], 'DONE')

        submitted_problem.retry()
        self.assertEqual(submitted_problem.status()['remote_status'], None)
        self.assertEqual(submitted_problem.status()['state'], 'SUBMITTING')

        # Wait until solved
        await_completion([submitted_problem], 1, float('inf'))

        # display result
        self.is_answer(submitted_problem.result())
        self.assertEqual(submitted_problem.status()['remote_status'], RemoteConnection.STATUS_COMPLETE)
        self.assertEqual(submitted_problem.status()['state'], 'DONE')

    @unittest.skipIf(skip_live, "No live server available.")
    def test_async_bad_retry(self):
        from dwave_sapi2.remote import RemoteConnection
        from dwave_sapi2.core import async_solve_qubo, await_completion

        # get a solver
        solver = RemoteConnection().get_solver(solver_name)

        Q = {(0, 5): -float('inf')}

        submitted_problem = async_solve_qubo(solver, Q, num_reads=10)

        #
        await_completion([submitted_problem], 1, float('inf'))
        self.assertEqual(submitted_problem.status()['remote_status'], RemoteConnection.STATUS_FAILED)
        self.assertEqual(submitted_problem.status()['state'], 'DONE')
        self.assertEqual(submitted_problem.status()['error_type'], 'SOLVE')

        #
        submitted_problem.retry()
        await_completion([submitted_problem], 1, float('inf'))
        self.assertEqual(submitted_problem.status()['remote_status'], RemoteConnection.STATUS_FAILED)
        self.assertEqual(submitted_problem.status()['state'], 'DONE')
        self.assertEqual(submitted_problem.status()['error_type'], 'SOLVE')

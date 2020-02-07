.. _solver_cloud:

======
Solver
======

.. currentmodule:: dwave.cloud

.. automodule:: dwave.cloud.solver


Class
-----

.. autoclass:: Solver
.. autoclass:: BaseSolver
.. autoclass:: StructuredSolver
.. autoclass:: UnstructuredSolver


Methods
-------

.. autosummary::
   :toctree: generated

   StructuredSolver.check_problem
   StructuredSolver.sample_ising
   StructuredSolver.sample_qubo
   StructuredSolver.max_num_reads

   UnstructuredSolver.sample_ising
   UnstructuredSolver.sample_qubo
   UnstructuredSolver.sample_bqm


Properties
----------

.. autosummary::
   :toctree: generated

   BaseSolver.name
   BaseSolver.online
   BaseSolver.avg_load
   BaseSolver.qpu
   BaseSolver.software

   StructuredSolver.num_active_qubits
   StructuredSolver.num_qubits
   StructuredSolver.is_vfyc
   StructuredSolver.has_flux_biases
   StructuredSolver.has_anneal_schedule
   StructuredSolver.lower_noise

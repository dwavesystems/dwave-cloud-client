.. _cloud_solver:

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
   StructuredSolver.estimate_qpu_access_time
   StructuredSolver.max_num_reads
   StructuredSolver.reformat_parameters
   StructuredSolver.sample_bqm
   StructuredSolver.sample_ising
   StructuredSolver.sample_qubo



   UnstructuredSolver.sample_ising
   UnstructuredSolver.sample_qubo
   UnstructuredSolver.sample_bqm
   UnstructuredSolver.upload_bqm

   BQMSolver.sample_ising
   BQMSolver.sample_qubo
   BQMSolver.sample_bqm
   BQMSolver.upload_bqm

   CQMSolver.sample_cqm
   CQMSolver.upload_cqm

   DQMSolver.sample_dqm
   DQMSolver.upload_dqm

Properties
----------

.. autosummary::
   :toctree: generated

   BaseSolver.name
   BaseSolver.avg_load
   BaseSolver.online
   BaseSolver.qpu
   BaseSolver.hybrid
   BaseSolver.software

   StructuredSolver.num_active_qubits
   StructuredSolver.num_qubits
   StructuredSolver.is_vfyc
   StructuredSolver.has_flux_biases
   StructuredSolver.has_anneal_schedule
   StructuredSolver.lower_noise

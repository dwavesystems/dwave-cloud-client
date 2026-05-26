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
.. autoclass:: BaseUnstructuredSolver
.. autoclass:: UnstructuredSolver


Methods
-------

.. autosummary::
   :toctree: generated

   StructuredSolver.check_problem
   StructuredSolver.estimate_qpu_access_time
   StructuredSolver.max_num_reads
   StructuredSolver.reformat_parameters
   StructuredSolver.sample_ising
   StructuredSolver.sample_qubo
   StructuredSolver.sample_bqm
   StructuredSolver.sample_problem

   UnstructuredSolver.sample_ising
   UnstructuredSolver.sample_qubo
   UnstructuredSolver.sample_bqm
   UnstructuredSolver.sample_problem
   UnstructuredSolver.upload_bqm

   BQMSolver.sample_ising
   BQMSolver.sample_qubo
   BQMSolver.sample_bqm
   BQMSolver.sample_problem
   BQMSolver.upload_bqm

   CQMSolver.sample_cqm
   CQMSolver.sample_problem
   CQMSolver.upload_cqm

   DQMSolver.sample_dqm
   DQMSolver.sample_problem
   DQMSolver.upload_dqm

   NLSolver.sample_nlm
   NLSolver.sample_problem
   NLSolver.upload_nlm

   QCDLSolver.sample_qcdl
   QCDLSolver.sample_problem
   QCDLSolver.upload_qcdl

Properties
----------

.. autosummary::
   :toctree: generated

   BaseSolver.identity
   BaseSolver.name
   BaseSolver.avg_load
   BaseSolver.online
   BaseSolver.qpu
   BaseSolver.hybrid
   BaseSolver.software
   BaseSolver.minimal_problem

   StructuredSolver.nodes
   StructuredSolver.variables
   StructuredSolver.edges
   StructuredSolver.couplers
   StructuredSolver.undirected_edges

   StructuredSolver.graph_id
   StructuredSolver.version
   StructuredSolver.num_active_qubits
   StructuredSolver.num_qubits
   StructuredSolver.is_vfyc
   StructuredSolver.has_flux_biases
   StructuredSolver.has_anneal_schedule
   StructuredSolver.lower_noise

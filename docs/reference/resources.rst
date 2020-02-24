.. _resources_cloud:

=======
Clients
=======

The :term:`solver`\ s that provide sampling for solving :term:`Ising` and :term:`QUBO` problems, such
as a D-Wave 2000Q QPU or a software :term:`sampler` such as the `dimod <https://github.com/dwavesystems/dimod>`_
simulated annealing sampler, are typically remote resources. The D-Wave Cloud Client
:class:`~dwave.cloud.client.Client` class manages such remote solver resources.

Preferred use is with a context manager---a :code:`with Client.from_config(...) as`
construct---to ensure proper closure of all resources. The following example snippet
creates a client based on an auto-detected configuration file and instantiates
a solver.

>>> with Client.from_config() as client:   # doctest: +SKIP
...     solver = client.get_solver(num_qubits__gt=2000)

Alternatively, the following example snippet creates a client for software resources
that it later explicitly closes.

>>> client = Client.from_config(software=True)   # doctest: +SKIP
>>> # code that uses client
>>> client.close()    # doctest: +SKIP

Typically you use the :class:`~dwave.cloud.client.Client` class. By default, it instantiates
a QPU client. You can also use the specialized QPU and CPU/GPU clients directly.

Client (Base Client)
====================

.. automodule:: dwave.cloud.client

Class
-----

.. autoclass:: dwave.cloud.client.Client

Methods
-------

.. currentmodule:: dwave.cloud

.. autosummary::
   :toctree: generated

   client.Client.from_config
   client.Client.solvers
   client.Client.retrieve_answer
   client.Client.get_solver
   client.Client.get_solvers
   client.Client.is_solver_handled
   client.Client.close

Specialized Clients
===================

Typically you use the :class:`~dwave.cloud.client.Client` class. By default, it instantiates
a QPU client. You can also instantiate a QPU or CPU/GPU client directly.

QPU Client
----------

.. currentmodule:: dwave.cloud.qpu

.. automodule:: dwave.cloud.qpu

Class
-----

.. autoclass:: dwave.cloud.qpu.Client

Methods
-------

.. currentmodule:: dwave.cloud

.. autosummary::
   :toctree: generated

   qpu.Client.is_solver_handled

Software-Samplers Client
------------------------

.. currentmodule:: dwave.cloud.sw


Class
-----

.. autoclass:: dwave.cloud.sw.Client

Methods
-------

.. currentmodule:: dwave.cloud

.. autosummary::
   :toctree: generated

   sw.Client.is_solver_handled

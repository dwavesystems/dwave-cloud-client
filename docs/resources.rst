.. _cloud_resources:

=======
Clients
=======

The :term:`solver`\ s that provide sampling for a :term:`binary quadratic model` 
problem, such as an Advantage quantum computer, or a quantum-classical hybrid 
:term:`sampler`, such as Leap's :class:`~dwave.system.samplers.LeapHybridCQMSampler`
hybrid constrained quadratic model (CQM) sampler, are typically remote resources. 
The D-Wave Cloud Client :class:`~dwave.cloud.client.Client` class manages such 
remote solver resources.

Preferred use is with a context manager---a :code:`with Client.from_config(...) as`
construct---to ensure proper closure of all resources. The following example snippet
creates a client based on an auto-detected configuration file and instantiates
a solver.

>>> with Client.from_config() as client:   # doctest: +SKIP
...     solver = client.get_solver(num_qubits__gt=5000)

Alternatively, the following example snippet creates a client for hybrid resources
that it later explicitly closes.

>>> client = Client.from_config(client="hybrid")   # doctest: +SKIP
>>> # code that uses client
>>> client.close()    # doctest: +SKIP

Typically you use the :class:`~dwave.cloud.client.Client` class. You can also 
instantiate specialized QPU, hybrid, and CPU clients directly.

Client (Base Client)
====================

.. automodule:: dwave.cloud.client
.. currentmodule:: dwave.cloud.client

Class
-----

.. autoclass:: Client

Properties
----------

.. autosummary::
   :toctree: generated

   Client.DEFAULTS

Methods
-------

.. autosummary::
   :toctree: generated

   Client.from_config
   Client.get_regions
   Client.get_solver
   Client.get_solvers
   Client.is_solver_handled
   Client.retrieve_answer
   Client.close


Specialized Clients
===================

Typically you use the :class:`~dwave.cloud.client.Client` class. You can also 
instantiate a QPU, hybrid, or CPU client directly.

QPU Client
----------

.. automodule:: dwave.cloud.qpu
.. currentmodule:: dwave.cloud.qpu

Class
~~~~~

.. autoclass:: dwave.cloud.qpu.Client


Hybrid Client
-------------

.. automodule:: dwave.cloud.hybrid
.. currentmodule:: dwave.cloud.hybrid

Class
~~~~~

.. autoclass:: dwave.cloud.hybrid.Client


Software Client
---------------

.. automodule:: dwave.cloud.sw
.. currentmodule:: dwave.cloud.sw

Class
~~~~~

.. autoclass:: dwave.cloud.sw.Client







PayloadCompressingSession
=========================

.. currentmodule:: dwave.cloud.api.client

Class
-----

.. autoclass:: PayloadCompressingSession

.. autoclass:: VersionedAPISession

Methods
-------

.. autosummary::
   :toctree: generated

   PayloadCompressingSession.set_payload_compress



.. currentmodule:: dwave.cloud.api.resources

Class
-----

.. autoclass:: compress_if
.. _resources:

=========
Resources
=========

TODO: short description of term client and then how it's standardly used:
from dwave.cloud import Client, for simplicity. The default client (if `client=None`) is a QPU one
dwave.cloud.qpu.Client.from_config(client='sw') would return instance of dwave.cloud.sw.Client

Base Client
===========



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
   client.Client.get_solver
   client.Client.get_solvers
   client.Client.is_solver_handled
   client.Client.close

QPU Client
==========

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
========================

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

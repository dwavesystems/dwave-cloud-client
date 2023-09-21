.. _computation:

===========
Computation
===========


.. currentmodule:: dwave.cloud.computation

.. automodule:: dwave.cloud.computation

Class
-----

.. autoclass:: dwave.cloud.computation.Future

Methods
-------

.. autosummary::
   :toctree: generated

   Future.result
   Future.exception
   Future.as_completed
   Future.wait
   Future.wait_id
   Future.wait_sampleset
   Future.wait_multiple
   Future.done
   Future.cancel

Properties
----------

.. autosummary::
   :toctree: generated

   Future.samples
   Future.variables
   Future.energies
   Future.num_occurrences
   Future.sampleset

   Future.id
   Future.problem_type
   Future.timing

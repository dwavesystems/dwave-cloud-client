.. _reference:

Reference Documentation
***********************

   :Release: |release|
   :Date: |today|


Configuration
-------------

.. automodule:: dwave.cloud.config
    :members: load_config_from_file, load_config, legacy_load_config,
        load_profile, detect_configfile_path


Resources
---------

Base Client
^^^^^^^^^^^

.. autoclass:: dwave.cloud.client.Client
    :members:

QPU Client
^^^^^^^^^^

.. autoclass:: dwave.cloud.qpu.Client
    :members:

Software Samplers Client
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dwave.cloud.sw.Client
    :members:


Solver
------

.. autoclass:: dwave.cloud.solver.Solver
    :members:


Computation
-----------

.. autoclass:: dwave.cloud.computation.Future
    :members:


Exceptions
----------

.. automodule:: dwave.cloud.exceptions
    :members:

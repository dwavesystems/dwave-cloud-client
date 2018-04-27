.. _configuration:

=============
Configuration
=============

.. currentmodule:: dwave.cloud.config
.. automodule:: dwave.cloud.config

Loading Configuration
=====================

These functions deploy D-Wave Cloud Client settings from a configuration file.

.. currentmodule:: dwave.cloud.config

.. autosummary::
   :toctree: generated

   load_config
   legacy_load_config

Managing Files
==============

These functions manage your D-Wave Cloud Client configuration files.

.. autosummary::
   :toctree: generated

   get_configfile_paths
   get_configfile_path
   get_default_configfile_path

Configuration Utilities
=======================

These functions provide non-standard options to deploy D-Wave Cloud Client settings
from configuration files. **Most users do not need to use these methods.**

.. currentmodule:: dwave.cloud.config

.. autosummary::
   :toctree: generated

   load_config_from_files
   load_profile_from_files

.. _intro:

============
Introduction
============

TODO: some words about solvers, work with SAPI, etc

Configuration
=============

It's recommended you set up a configuration file through the interactive CLI utility.

Configuration Files
-------------------

Candidates paths for configuration files are set by the D-Wave homebase_ package.

For example, on a Unix system, depending on its flavor, these might include (in order)::

          /usr/share/dwave/dwave.conf
          /usr/local/share/dwave/dwave.conf
          ~/.config/dwave/dwave.conf
          ./dwave.conf

while on Windows 7+, configuration files are expected to be located under::

      C:\\Users\\<username>\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf

and on Mac OS X under::

     ~/Library/Application Support/dwave/dwave.conf

For details on user/system config paths see homebase_.

.. _homebase: https://github.com/dwavesystems/homebase

Interactive CLI Configuration
-----------------------------

TODO

Work Flow
=========

TODO: describe the general use of cloud client including problem submission and
monitoring (computation)

Solvers
=======

TODO: add short description of available resources for sampling


Terminology
===========

.. glossary::

    model
        A collection of variables with associated linear and
        quadratic biases.

    sampler
        A process that samples from low energy states of a problem’s objective function.
        A binary quadratic model (BQM) sampler samples from low energy states in models such
        as those defined by an Ising equation or a Quadratic Unconstrained Binary Optimization
        (QUBO) problem and returns an iterable of samples, in order of increasing energy. A dimod
        sampler provides ‘sample_qubo’ and ‘sample_ising’ methods as well as the generic
        BQM sampler method.

    Solver
        A resource that runs a problem. Some solvers interface to the QPU; others leverage CPU
        and GPU resources.

.. _intro:

============
Introduction
============

TODO: some words about solvers, work with SAPI, etc

Configuration
=============

TODO: add description  and some examples

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

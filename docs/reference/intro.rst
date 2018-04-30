.. _intro:

============
Introduction
============

D-Wave Cloud Client is a minimal implementation of the REST interface used to communicate with
D-Wave Sampler API (SAPI) servers.

SAPI is an application layer built to provide resource discovery, permissions,
and scheduling for quantum annealing resources at D-Wave Systems.
This package provides a minimal Python interface to that layer without
compromising the quality of interactions and workflow.

Configuration
=============

It's recommended you set up a configuration file through the interactive CLI utility.

Configuration Files
-------------------

** THE FOLLOWING IS JUST DRAFT CONTENT **

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


One config file can contain multiple profiles, each defining a separate
      (endpoint, token, solver, etc.) combination. Since config file conforms to a
      standard Windows INI-style format, profiles are defined by sections like:
      ``[profile-a]`` and ``[profile-b]``.

      Default values for undefined profile keys are taken from the ``[defaults]``
      section.

      For example, assuming ``~/.config/dwave/dwave.conf`` contains::

          [defaults]
          endpoint = https://cloud.dwavesys.com/sapi
          client = qpu

          [dw2000]
          solver = DW_2000Q_1
          token = ...

          [software]
          client = sw
          solver = c4-sw_sample
          token = ...

          [alpha]
          endpoint = https://url.to.alpha/api
          proxy = http://user:pass@myproxy.com:8080/
          token = ...

      We can instantiate a client for D-Wave 2000Q QPU endpoint with

      >>> from dwave.cloud import Client
      >>> client = Client.from_config(profile='dw2000')

      and a client for remote software solver with::

      >>> client = Client.from_config(profile='software')

      ``alpha`` profile will connect to a pre-release API endpoint via defined HTTP
      proxy server.


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

    Ising
         Traditionally used in statistical mechanics. Variables are "spin up"
         (:math:`\uparrow`) and "spin down" (:math:`\downarrow`), states that
         correspond to :math:`+1` and :math:`-1` values. Relationships between
         the spins, represented by couplings, are correlations or anti-correlations.
         The objective function expressed as an Ising model is as follows:

         .. math::

                  \begin{equation}
                       \text{E}_{ising}(\pmb{s}) = \sum_{i=1}^N h_i s_i + \sum_{i=1}^N \sum_{j=i+1}^N J_{i,j} s_i s_j
                  \end{equation}

         where the linear coefficients corresponding to qubit biases
         are :math:`h_i`, and the quadratic coefficients corresponding to coupling
         strengths are :math:`J_{i,j}`.

    model
        A collection of variables with associated linear and
        quadratic biases.

    QUBO
         Quadratic unconstrained binary optimization.
         QUBO problems are traditionally used in computer science. Variables
         are TRUE and FALSE, states that correspond to 1 and 0 values.
         A QUBO problem is defined using an upper-diagonal matrix :math:`Q`,
         which is an :math:`N` x :math:`N` upper-triangular matrix of real weights,
         and :math:`x`, a vector of binary variables, as minimizing the function

         .. math::

            \begin{equation}
              f(x) = \sum_{i} {Q_{i,i}}{x_i} + \sum_{i<j} {Q_{i,j}}{x_i}{x_j}
            \end{equation}

         where the diagonal terms :math:`Q_{i,i}` are the linear coefficients and
         the nonzero off-diagonal terms are the quadratic coefficients
         :math:`Q_{i,j}`.
         This can be expressed more concisely as

         .. math::

            \begin{equation}
              \min_{{x} \in {\{0,1\}^n}} {x}^{T} {Q}{x}.
            \end{equation}

         In scalar notation, the objective function expressed as a QUBO
         is as follows:

         .. math::

            \begin{equation}
                        \text{E}_{qubo}(a_i, b_{i,j}; q_i) = \sum_{i} a_i q_i + \sum_{i<j} b_{i,j} q_i q_j.
            \end{equation}

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

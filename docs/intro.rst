.. _intro:

============
Introduction
============

D-Wave Cloud Client is a minimal implementation of the REST interface used to communicate
with D-Wave Sampler API (SAPI) servers.

SAPI is an application layer built to provide resource discovery, permissions,
and scheduling for quantum annealing resources at D-Wave Systems.
This package provides a minimal Python interface to that layer without
compromising the quality of interactions and workflow.

The D-Wave Cloud Client :class:`~dwave.cloud.solver.Solver` class enables low-level control of problem
submission. It is used, for example, by the :std:doc:`dwave-system <system:index>`
:class:`~dwave.system.samplers.DWaveSampler`, which enables quick incorporation
of the D-Wave system as a sampler in your code.


Configuration
=============

It's recommended you set up your D-Wave Cloud Client configuration through the
:ref:`interactive CLI utility <interactiveCliConfiguration>`.

As described in the :std:doc:`Using a D-Wave System <oceandocs:overview/dwavesys>` section
of Ocean Documentation, for your code to access remote D-Wave compute resources, you must
configure communication through SAPI; for example, your code needs your API
token for authentication. D-Wave Cloud Client provides multiple options for configuring
the required information:

* One or more locally saved :ref:`configuration files <configurationFiles>`
* :ref:`Environment variables <environmentVariables>`
* Direct setting of key values in functions

These options can be flexibly used together.

.. _configurationFiles:

Configuration Files
-------------------

If a D-Wave Cloud Client configuration file is not explicitly specified when instantiating a
client or solver, auto-detection searches for candidate files in a number of standard
directories, depending on your local system's operating system. You can see the standard
locations with the :func:`~dwave.cloud.config.get_configfile_paths` method.

For example, on a Unix system, depending on its flavor, these might include (in order)::

          /usr/share/dwave/dwave.conf
          /usr/local/share/dwave/dwave.conf
          ~/.config/dwave/dwave.conf
          ./dwave.conf

On Windows 7+, configuration files are expected to be located under::

      C:\\Users\\<username>\\AppData\\Local\\dwavesystem\\dwave\\dwave.conf

On Mac OS X, configuration files are expected to be located under::

     ~/Library/Application Support/dwave/dwave.conf

(For details on the D-Wave API for determining platform-independent paths to user
data and configuration folders see the homebase_ tool.)

.. _homebase: https://github.com/dwavesystems/homebase

You can check the directories searched by :func:`~dwave.cloud.config.get_configfile_paths`
from a console using the :ref:`interactive CLI utility <interactiveCliConfiguration>`;
for example::

  $ dwave config ls -m
  /etc/xdg/xdg-ubuntu/dwave/dwave.conf
  /usr/share/upstart/xdg/dwave/dwave.conf
  /etc/xdg/dwave/dwave.conf
  /home/jane/.config/dwave/dwave.conf
  ./dwave.conf

A single D-Wave Cloud Client configuration file can contain multiple profiles, each
defining a separate combination of communication parameters such as the URL to the
remote resource, authentication token, solver, etc.
Configuration files conform to a standard Windows INI-style format:
profiles are defined by sections such as, `[profile-a]` and `[profile-b]`.
Default values for undefined profile keys are taken from the `[defaults]` section.

For example, if the configuration file, `~/.config/dwave/dwave.conf`, selected
through auto-detection as the default configuration, contains the following
profiles::

          [defaults]
          token = ABC-123456789123456789123456789

          [first-available-qpu]
          solver = {"qpu": true}

          [software]
          client = sw
          solver = c4-sw_sample
          token = DEF-987654321987654321987654321
          proxy = http://user:pass@myproxy.com:8080/

          [backup-dwave2000q]
          endpoint = https://url.of.my.backup.dwavesystem.com/sapi
          solver = {"num_qubits__gt": 2000}

You can instantiate clients for a D-Wave system and a CPU with::

      >>> from dwave.cloud import Client
      >>> client_qpu = Client.from_config()   # doctest: +SKIP
      >>> client_cpu = Client.from_config(profile='software')   # doctest: +SKIP

.. _environmentVariables:

Environment Variables
---------------------

In addition to files, you can set configuration information through environment
variables; for example:

* ``DWAVE_CONFIG_FILE`` may select the configuration file path.
* ``DWAVE_PROFILE`` may select the name of a profile (section).
* ``DWAVE_API_TOKEN`` may select the API token.

For details on supported environment variables and prioritizing between these and
values set explicitly or through a configuration file, see :mod:`dwave.cloud.config`.

.. _interactiveCliConfiguration:

Interactive CLI Configuration
-----------------------------

As part of the installation of the D-Wave Cloud Client package, a `dwave` executable
is installed; for example, in a virtual environment it might be installed as
`<virtual_environment>\\Scripts\\dwave.exe`. Running this file from your system's
console opens an interactive command line interface (CLI) that guides you through
setting up a D-Wave Cloud Client configuration file. It also provides additional helpful
functionality; for example:

* List and update existing configuration files on your system
* Establish a connection to (ping) a solver and return timing information
* Show information on configured solvers

Run *dwave* -\\-\ *help* for information on all the CLI options.

.. note:: If you work in a Bash shell and want command completion for `dwave`, add

          .. code-block:: bash

             eval "$(_DWAVE_COMPLETE=source <path>/dwave)"

          to your shell's `.bashrc` configuration file, where `<path>` is the absolute
          path to the installed `dwave` executable, for example `/home/Mary/my-quantum-app/env/bin`.

Work Flow
=========

A typical workflow may include the following steps:

1. Instantiate a :class:`~dwave.cloud.client.Client` to manage communication
   with remote :term:`solver` resources, selecting and authenticating access to
   available solvers; for example, you can list all solvers available to a client with its
   :func:`~dwave.cloud.client.Client.get_solvers` method and select and return one with its
   :func:`~dwave.cloud.client.Client.get_solver` method.

   Preferred use is with a context manager---a :code:`with Client.from_config(...) as`
   construct---to ensure proper closure of all resources. The following example snippet
   creates a client based on an auto-detected configuration file and instantiates
   a solver.

   >>> with Client.from_config() as client:   # doctest: +SKIP
   ...     solver = client.get_solver(qpu=True)

   Alternatively, the following example snippet creates a client for software resources
   that it later explicitly closes.

   >>> client = Client.from_config(client='sw')   # doctest: +SKIP
   >>> # code that uses client
   >>> client.close()    # doctest: +SKIP

2. Instantiate a selected :class:`~dwave.cloud.solver.Solver`, a resource for solving problems.
   Solvers are responsible for:

      - Encoding submitted problems
      - Checking submitted parameters
      - Adding problems to a client's submission queue

   Solvers that provide sampling for solving :term:`Ising` and :term:`QUBO` problems,
   such as a D-Wave 2000Q :term:`sampler` :class:`~dwave.system.samplers.DWaveSampler`
   or software sampler :class:`~neal.sampler.SimulatedAnnealingSampler`, might be remote
   resources.

3. Submit your problem, using your solver, and then process the returned
   :class:`~dwave.cloud.computation.Future`, instantiated by your solver to handle
   remotely executed problem solving.

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

    solver
        A resource that runs a problem. Some solvers interface to the QPU; others leverage CPU
        and GPU resources.

.. image:: https://badge.fury.io/py/dwave-cloud-client.svg
    :target: https://badge.fury.io/py/dwave-cloud-client
    :alt: Latest version on PyPI

.. image:: https://circleci.com/gh/dwavesystems/dwave-cloud-client.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-cloud-client
    :alt: Linux/MacOS build status

.. image:: https://ci.appveyor.com/api/projects/status/6a2wjq9xtgtr2t2c/branch/master?svg=true
    :target: https://ci.appveyor.com/project/dwave-adtt/dwave-cloud-client/branch/master
    :alt: Windows build status

.. image:: https://codecov.io/gh/dwavesystems/dwave-cloud-client/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-cloud-client
    :alt: Coverage report

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-cloud-client/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/cloud-client/en/latest/?badge=latest
    :alt: Documentation Status

.. index-start-marker

==================
dwave-cloud-client
==================

D-Wave Cloud Client is a minimal implementation of the REST interface used to
communicate with D-Wave Sampler API (SAPI) servers.

SAPI is an application layer built to provide resource discovery, permissions,
and scheduling for quantum annealing resources at D-Wave Systems.
This package provides a minimal Python interface to that layer without
compromising the quality of interactions and workflow.

The example below instantiates a D-Wave Cloud Client and solver based on the local
system's auto-detected default configuration file and samples a random Ising problem
tailored to fit the solver's graph.

.. code-block:: python

    import random
    from dwave.cloud import Client

    # Connect using the default or environment connection information
    with Client.from_config() as client:

        # Load the default solver
        solver = client.get_solver()

        # Build a random Ising model to exactly fit the graph the solver supports
        linear = {index: random.choice([-1, 1]) for index in solver.nodes}
        quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

        # Send the problem for sampling, include solver-specific parameter 'num_reads'
        computation = solver.sample_ising(linear, quad, num_reads=100)

        # Print the first sample out of a hundred
        print(computation.samples[0])

.. index-end-marker


Installation
------------

.. installation-start-marker

Requires Python 3.7+:

.. code-block:: bash

    pip install dwave-cloud-client

To install from source (available on GitHub in `dwavesystems/dwave-cloud-client`_ repo):

.. code-block:: bash

    pip install -r requirements.txt
    python setup.py install

.. _`dwavesystems/dwave-cloud-client`: https://github.com/dwavesystems/dwave-cloud-client

.. installation-end-marker


License
-------

Released under the Apache License 2.0. See `<LICENSE>`_ file.


Contributing
------------

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
~~~~~~~~~~~~~

D-Wave Cloud Client uses `reno <https://docs.openstack.org/reno/>`_ to manage
its release notes.

When making a contribution to D-Wave Cloud Client that will affect users, create
a new release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.

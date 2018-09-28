.. image:: https://badge.fury.io/py/dwave-cloud-client.svg
    :target: https://badge.fury.io/py/dwave-cloud-client
    :alt: Last version on PyPI

.. image:: https://travis-ci.org/dwavesystems/dwave-cloud-client.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/dwave-cloud-client
    :alt: Linux/Mac build status

.. image:: https://ci.appveyor.com/api/projects/status/6a2wjq9xtgtr2t2c/branch/master?svg=true
    :target: https://ci.appveyor.com/project/dwave-adtt/dwave-cloud-client/branch/master
    :alt: Windows build status

.. image:: https://coveralls.io/repos/github/dwavesystems/dwave-cloud-client/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/dwave-cloud-client?branch=master
    :alt: Coverage report

.. image:: https://readthedocs.com/projects/d-wave-systems-dwave-cloud-client/badge/?version=latest
    :target: https://docs.ocean.dwavesys.com/projects/cloud-client/en/latest/?badge=latest
    :alt: Documentation Status

.. index-start-marker

D-Wave Cloud Client
===================

D-Wave Cloud Client is a minimal implementation of the REST interface used to
communicate with D-Wave Sampler API (SAPI) servers.

SAPI is an application layer built to provide resource discovery, permissions,
and scheduling for quantum annealing resources at D-Wave Systems.
This package provides a minimal Python interface to that layer without
compromising the quality of interactions and workflow.

Example
-------

This example instantiates a D-Wave Cloud Client and solver based on the local
system`s auto-detected default configuration file and samples a random Ising problem
tailored to fit the solver`s graph.

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

Compatible with Python 2 and 3:

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


Contribution
------------

See `<CONTRIBUTING.rst>`_ file.

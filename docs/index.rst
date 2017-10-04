dwave\_micro\_client
===========================
.. py:currentmodule:: dwave_micro_client

.. automodule:: dwave_micro_client

Configuration
-------------

To store your connection information in a configuration file create a text file named ``.dwrc`` in your home directory.
The default configuration format is a modified CSV where the first comma is replaced with a bar character '|'.

The columns are:

``connection label | server url, token, proxy url, default solver name``

Where everything after the token is optional.

Once the configuration file is created, a connection object created without any
parameters will use the first row of the configuration file.

For the example, with the configuration file:

::

    connection-a|https://one.com,token-one
    connection-b|https://two.com,token-two

>>> con = dwave_micro_client.Connection()
# Will try to connect with the url `https://one.com` and the token `token-one`.

It is also possible to create the connection using only the label or url, and the token will be retrived from the configuration.

>>> con = dwave_micro_client.Connection('connection-a')
# Will try to connect with the url `https://one.com` and the token `token-one`.

>>> con = dwave_micro_client.Connection('https://two.com')
# Will try to connect with the url `https://two.com` and the token `token-two`.

Classes
-------

.. autoclass:: Connection
    :members:

.. autoclass:: Solver
    :members:

.. autoclass:: Future
    :members:

Error Classes
-------------

.. autoclass :: SolverFailureError
    :show-inheritance:

.. autoclass :: CanceledFutureError
    :show-inheritance:

``dwave_sapi2`` Compatibility
-----------------------------

Some classes introduced to allow some limited compatibility with code that makes
simple use of features from ``dwave_sapi2.core`` and ``dwave_sapi2.remote``.
Not all the features of those modules are exposed, only those that have
a direct analog in this package.

.. code-block:: python

    #from dwave_sapi2.remote import RemoteConnection
    from dwave_micro_client.remote import RemoteConnection
    #from dwave_sapi2.core import solve_ising
    from dwave_micro_client.core import solve_ising

    # get a solver
    connection = RemoteConnection(sapi_url, sapi_token)
    solver = connection.get_solver(solver_name)

    # solve ising problem
    h = [1, -1, 1, 1, -1, 1, 1]
    J = {(0, 6): -10}

    params = {"num_reads": 10, "num_spin_reversal_transforms": 2}
    answer = solve_ising(solver, h, J, **params)
    print(answer['energies'][0])
    print(answer['solutions'][0])


.. autoclass :: core
    :show-inheritance:
    :members:

.. autoclass :: remote
    :show-inheritance:
    :members:

.. autoclass :: AsyncInterfaceWrapper
    :members:

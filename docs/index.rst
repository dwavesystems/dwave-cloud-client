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

>>> conn = dwave_micro_client.Connection()
# Will try to connect with the url `https://one.com` and the token `token-one`.

It is also possible to create the connection using only the label or url, and the token will be retrived from the configuration.

>>> conn = dwave_micro_client.Connection('connection-a')
# Will try to connect with the url `https://one.com` and the token `token-one`.

>>> conn = dwave_micro_client.Connection('https://two.com')
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

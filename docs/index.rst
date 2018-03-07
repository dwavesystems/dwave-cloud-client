dwave.cloud
===========
.. py:currentmodule:: dwave.cloud

.. automodule:: dwave.cloud

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

>>> client = dwave.cloud.qpu.Client()
# Will try to connect with the url `https://one.com` and the token `token-one`.

It is also possible to create the connection using only the label or url, and the token will be retrieved from the configuration.

>>> client = dwave.cloud.qpu.Client('connection-a')
# Will try to connect with the url `https://one.com` and the token `token-one`.

>>> client = dwave.cloud.qpu.Client('https://two.com')
# Will try to connect with the url `https://two.com` and the token `token-two`.

Classes
-------

.. autoclass:: dwave.cloud.client.BaseClient
    :members:

.. autoclass:: dwave.cloud.qpu.Client
    :members:

.. autoclass:: dwave.cloud.sw.Client
    :members:

.. autoclass:: dwave.cloud.solver.Solver
    :members:

.. autoclass:: dwave.cloud.computation.Future
    :members:

Error Classes
-------------

.. automodule:: dwave.cloud.exceptions

from __future__ import division, absolute_import

import struct
import base64

from dwave.cloud.utils import uniform_iterator, uniform_get


def encode_bqm_as_qp(solver, linear, quadratic):
    """Encode the problem for submission to a given solver.

    Args:
        solver: solver requested.
        lin: linear terms of the model.
        quad: Quadratic terms of the model.

    Returns:
        encoded submission dictionary

    """

    # Encode linear terms. The coefficients of the linear terms of the objective
    # are encoded as an array of little endian 64 bit doubles.
    # This array is then base64 encoded into a string safe for json.
    # The order of the terms is determined by the _encoding_qubits property
    # specified by the server.
    lin = [uniform_get(linear, qubit, 0) for qubit in solver._encoding_qubits]
    lin = base64.b64encode(struct.pack('<' + ('d' * len(lin)), *lin))

    # Encode the coefficients of the quadratic terms of the objective
    # in the same manner as the linear terms, in the order given by the
    # _encoding_couplers property
    quad = [quadratic.get(edge, 0) + quadratic.get((edge[1], edge[0]), 0)
            for edge in solver._encoding_couplers]
    quad = base64.b64encode(struct.pack('<' + ('d' * len(quad)), *quad))

    # The name for this encoding is 'qp' and is explicitly included in the
    # message for easier extension in the future.
    return {
        'format': 'qp',
        'lin': lin.decode('utf-8'),
        'quad': quad.decode('utf-8')
    }

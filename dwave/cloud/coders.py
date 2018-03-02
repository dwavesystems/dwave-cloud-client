from __future__ import division, absolute_import

import struct
import base64

from dwave.cloud.utils import uniform_iterator, uniform_get


def encode_bqm_as_qp(solver, linear, quadratic):
    """Encode the binary quadratic problem for submission to a given solver,
    using the `qp` format for data.

    Args:
        solver (:class:`dwave.cloud.qpu.solver.Solver`):
            The solver used.

        linear (dict[variable, bias]/list[variable, bias]):
            Linear terms of the model.

        quadratic (dict[(variable, variable), bias]):
            Quadratic terms of the model.

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


def decode_qp(msg):
    """Decode SAPI response that uses `qp` format, without numpy.

    The 'qp' format is the current encoding used for problems and samples.
    In this encoding the reply is generally json, but the samples, energy,
    and histogram data (the occurrence count of each solution), are all
    base64 encoded arrays.
    """
    # Decode the simple buffers
    result = msg['answer']
    result['active_variables'] = _decode_ints(result['active_variables'])
    active_variables = result['active_variables']
    if 'num_occurrences' in result:
        result['num_occurrences'] = _decode_ints(result['num_occurrences'])
    result['energies'] = _decode_doubles(result['energies'])

    # Measure out the size of the binary solution data
    num_solutions = len(result['energies'])
    num_variables = len(result['active_variables'])
    solution_bytes = -(-num_variables // 8)  # equivalent to int(math.ceil(num_variables / 8.))
    total_variables = result['num_variables']

    # Figure out the null value for output
    default = 3 if msg['type'] == 'qubo' else 0

    # Decode the solutions, which will be byte aligned in binary format
    binary = base64.b64decode(result['solutions'])
    solutions = []
    for solution_index in range(num_solutions):
        # Grab the section of the buffer related to the current
        buffer_index = solution_index * solution_bytes
        solution_buffer = binary[buffer_index:buffer_index + solution_bytes]
        bytes = struct.unpack('B' * solution_bytes, solution_buffer)

        # Assume None values
        solution = [default] * total_variables
        index = 0
        for byte in bytes:
            # Parse each byte and read how ever many bits can be
            values = _decode_byte(byte)
            for _ in range(min(8, len(active_variables) - index)):
                i = active_variables[index]
                index += 1
                solution[i] = values.pop()

        # Switch to the right variable space
        if msg['type'] == 'ising':
            values = {0: -1, 1: 1}
            solution = [values.get(v, default) for v in solution]
        solutions.append(solution)

    result['solutions'] = solutions
    return result


def _decode_byte(byte):
    """Helper for decode_qp, turns a single byte into a list of bits.

    Args:
        byte: byte to be decoded

    Returns:
        list of bits corresponding to byte
    """
    bits = []
    for _ in range(8):
        bits.append(byte & 1)
        byte >>= 1
    return bits


def _decode_ints(message):
    """Helper for decode_qp, decodes an int array.

    The int array is stored as little endian 32 bit integers.
    The array has then been base64 encoded. Since we are decoding we do these
    steps in reverse.
    """
    binary = base64.b64decode(message)
    return struct.unpack('<' + ('i' * (len(binary) // 4)), binary)


def _decode_doubles(message):
    """Helper for decode_qp, decodes a double array.

    The double array is stored as little endian 64 bit doubles.
    The array has then been base64 encoded. Since we are decoding we do these
    steps in reverse.

    Args:
        message: the double array

    Returns:
        decoded double array
    """
    binary = base64.b64decode(message)
    return struct.unpack('<' + ('d' * (len(binary) // 8)), binary)


def decode_qp_numpy(msg, return_matrix=True):
    """Decode SAPI response, results in a `qp` format, explicitly using numpy.
    If numpy is not installed, the method will fail.

    To use numpy for decoding, but return the results a lists (instead of
    numpy matrices), set `return_matrix=False`.
    """
    import numpy as np

    result = msg['answer']

    # Build some little endian type encodings
    double_type = np.dtype(np.double)
    double_type = double_type.newbyteorder('<')
    int_type = np.dtype(np.int32)
    int_type = int_type.newbyteorder('<')

    # Decode the simple buffers
    result['energies'] = np.frombuffer(base64.b64decode(result['energies']),
                                       dtype=double_type)

    if 'num_occurrences' in result:
        result['num_occurrences'] = \
            np.frombuffer(base64.b64decode(result['num_occurrences']),
                        dtype=int_type)

    result['active_variables'] = \
        np.frombuffer(base64.b64decode(result['active_variables']),
                      dtype=int_type)

    # Measure out the binary data size
    num_solutions = len(result['energies'])
    active_variables = result['active_variables']
    num_variables = len(active_variables)
    total_variables = result['num_variables']

    # Decode the solutions, which will be a continuous run of bits
    byte_type = np.dtype(np.uint8)
    byte_type = byte_type.newbyteorder('<')
    bits = np.unpackbits(np.frombuffer(base64.b64decode(result['solutions']),
                         dtype=byte_type))

    # Clip off the extra bits from encoding
    bits = np.reshape(bits, (num_solutions, bits.size // num_solutions))
    bits = np.delete(bits, range(num_variables, bits.shape[1]), 1)

    # Switch from bits to spins
    default = 3
    if msg['type'] == 'ising':
        bits = bits.astype(np.int8)
        bits *= 2
        bits -= 1
        default = 0

    # Fill in the missing variables
    solutions = np.full((num_solutions, total_variables), default, dtype=np.int8)
    solutions[:, active_variables] = bits
    result['solutions'] = solutions

    # If the final result shouldn't be numpy formats switch back to python objects
    if not return_matrix:
        result['energies'] = result['energies'].tolist()
        if 'num_occurrences' in result:
            result['num_occurrences'] = result['num_occurrences'].tolist()
        result['active_variables'] = result['active_variables'].tolist()
        result['solutions'] = result['solutions'].tolist()

    return result

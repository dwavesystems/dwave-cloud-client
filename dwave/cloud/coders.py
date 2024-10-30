# Copyright 2017 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
import base64
import warnings
from collections import abc
from typing import Union

try:
    # note: TypedDict is available in py38+, but NotRequired only in py311+
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict

from dwave.cloud.utils.qubo import uniform_get, active_qubits

__all__ = [
    'encode_problem_as_qp', 'decode_qp', 'decode_qp_numpy', 'decode_qp_problem',
    'encode_problem_as_bq', 'decode_bq',
    'encode_problem_as_ref', 'decode_binary_ref',
    'bqm_as_file',
]


# a plain dict version of `dwave.cloud.api.models.StructuredProblemData`
class EncodedQP(TypedDict):
    format: str
    lin: str
    quad: str
    offset: NotRequired[float]

class QuadraticProblem(TypedDict):
    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]
    offset: NotRequired[float]


def encode_problem_as_qp(solver: 'dwave.cloud.solver.StructuredSolver',
                         linear: Union[list[float], dict[int, float]],
                         quadratic: dict[tuple[int, int], float],
                         offset: float = 0,
                         undirected_biases: bool = False
                         ) -> EncodedQP:
    """Encode the binary quadratic problem for submission to a given solver,
    using the `qp` format for data.

    Args:
        solver (:class:`dwave.cloud.solver.Solver`):
            The solver used.

        linear (dict[variable, bias]/list[variable, bias]):
            Linear terms of the model.

        quadratic (dict[(variable, variable), bias]):
            Quadratic terms of the model.

        offset (number, default=0):
            Constant offset applied to the model.

        undirected_biases (boolean, default=False):
            Are (quadratic) biases specified on undirected edges?

    Returns:
        Encoded submission dictionary.
    """
    # convert legacy format (list) to dict for performance
    if isinstance(linear, abc.Sequence):
        linear = dict(enumerate(linear))

    active = active_qubits(linear, quadratic)

    # Encode linear terms. The coefficients of the linear terms of the objective
    # are encoded as an array of little endian 64 bit doubles.
    # This array is then base64 encoded into a string safe for json.
    # The order of the terms is determined by the _encoding_qubits property
    # specified by the server.
    # Note: only active qubits are coded with double, inactive with NaN
    nan = float('nan')
    lin = [linear.get(qubit, 0 if qubit in active else nan)
           for qubit in solver._encoding_qubits]

    lin = base64.b64encode(struct.pack('<' + ('d' * len(lin)), *lin))

    # Encode the coefficients of the quadratic terms of the objective
    # in the same manner as the linear terms, in the order given by the
    # _encoding_couplers property, discarding tailing zero couplings
    if undirected_biases:
        # quadratic biases are given in a triangular or symmetric matrix
        quad = [quadratic.get((q1,q2), quadratic.get((q2,q1), 0))
                for (q1,q2) in solver._encoding_couplers
                if q1 in active and q2 in active]
    else:
        # quadratic biases are defined on directed edges, conflate with sum
        quad = [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                for (q1,q2) in solver._encoding_couplers
                if q1 in active and q2 in active]

    quad = base64.b64encode(struct.pack('<' + ('d' * len(quad)), *quad))

    # The name for this encoding is 'qp' and is explicitly included in the
    # message for easier extension in the future.
    return {
        'format': 'qp',
        'lin': lin.decode('utf-8'),
        'quad': quad.decode('utf-8'),
        'offset': offset
    }


def decode_qp_problem(solver: 'dwave.cloud.solver.StructuredSolver',
                      qp: EncodedQP,
                      undirected_edges: bool = True
                      ) -> QuadraticProblem:
    """Decode quadratic unconstrained binary problem encoded using SAPI's
    ``qp`` format."""

    encoding_qubits = solver._encoding_qubits
    encoding_couplers = solver._encoding_couplers

    lin = base64.decodebytes(qp['lin'].encode('ascii'))
    lin = struct.unpack('<' + ('d' * int(len(lin)/8)), lin)
    linear = {qubit: lin[idx] for idx, qubit in enumerate(encoding_qubits)}

    quad = base64.decodebytes(qp['quad'].encode('ascii'))
    quad = struct.unpack('<' + ('d' * int(len(quad)/8)), quad)
    quadratic = {(q1,q2): quad[idx] for idx, (q1,q2) in enumerate(encoding_couplers)}

    # return quadratic as upper triangular (regardless of encoding couplers)
    if undirected_edges:
        quadratic = {(q1,q2): quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
                     for (q1,q2) in solver.undirected_edges}

    offset = qp.get('offset', 0.0)

    return QuadraticProblem(linear=linear, quadratic=quadratic, offset=offset)


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

    # adjust energies by offset (in future this might be handled by SAPI)
    offset = result.setdefault('offset', 0)
    if offset:
        result['energies'] = [en + offset for en in result['energies']]

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

    # include problem type
    if 'type' in msg:
        result['problem_type'] = msg['type']

    return result


def _decode_byte(byte):
    """Helper for decode_qp, turns a single byte into a list of bits.

    Args:
        byte (int):
            Byte to be decoded.

    Returns:
        List of bits corresponding to byte.
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

    Args:
        message (str):
            The int array, base-64 encoded.

    Returns:
        Decoded double array.
    """
    binary = base64.b64decode(message)
    return struct.unpack('<' + ('i' * (len(binary) // 4)), binary)


def _decode_doubles(message):
    """Helper for decode_qp, decodes a double array.

    The double array is stored as little endian 64 bit doubles.
    The array has then been base64 encoded. Since we are decoding we do these
    steps in reverse.

    Args:
        message (str):
            The double array, base-64 encoded.

    Returns:
        Decoded double array.
    """
    binary = base64.b64decode(message)
    return struct.unpack('<' + ('d' * (len(binary) // 8)), binary)


def decode_qp_numpy(msg, return_matrix=True):
    """Decode SAPI response, results in a `qp` format, explicitly using numpy.
    If numpy is not installed, the method will fail.

    To use numpy for decoding, but return the results as lists (instead of
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

    # adjust energies by offset (in future this might be handled by SAPI)
    offset = result.setdefault('offset', 0)
    if offset:
        # we need to make a copy because frombuffer returns read-only array
        result['energies'] = result['energies'] + offset

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
    if num_solutions:
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

    # include problem type
    if 'type' in msg:
        result['problem_type'] = msg['type']

    return result


def encode_problem_as_ref(problem):
    """Encode the problem given via reference for submission in the `ref` data
    format.

    Args:
        problem (str):
            A reference to an uploaded problem (problem ID).

    Returns:
        Encoded submission dictionary.
    """

    if not isinstance(problem, str):
        raise TypeError("unsupported problem reference type")

    return {
        'format': 'ref',
        'data': problem
    }


def encode_problem_as_bq(problem):
    """Encode the binary quadratic problem for submission in the `bq` data
    format.

    Args:
        problem (:class:`~dimod.BinaryQuadraticModel`):
            A binary quadratic model.

    Returns:
        Encoded submission dictionary.

    Note:
        The `bq` format assumes the complete BQM is sent embedded in the sample
        job submit data, something none of the production solvers currently
        support.
    """
    # NOTE: semi-deprecated format; `bqm.to_serializable()` is still supported
    # (as of dimod 0.12.18), but `bqm.to_file()` is preferred.

    if not hasattr(problem, 'to_serializable'):
        raise TypeError("unsupported problem type")

    return {
        'format': 'bq',
        'data': problem.to_serializable(use_bytes=False)
    }


def decode_bq(msg):
    """Decode answer for problem submitted in the `bq` data format."""
    try:
        import dimod
    except ImportError:     # pragma: no cover
        raise RuntimeError("Can't decode BQMs without dimod. "
                           "Re-install the library with 'bqm' support.")

    answer = msg['answer']
    if answer['format'] != 'bq':
        raise ValueError(f"Unsupported answer format: {answer['format']}")
    if 'data' not in answer:
        raise ValueError("Incomplete answer")

    result = {}

    # sampleset is encoded in data field
    result['sampleset'] = dimod.SampleSet.from_serializable(answer['data'])

    # include problem type
    result['problem_type'] = msg['type']

    return result


def decode_binary_ref(msg: dict, ref_resolver: abc.Callable) -> dict:
    """Decode binary-ref answer."""

    answer = msg['answer']
    if answer['format'] != 'binary-ref':
        raise ValueError(f"Unsupported answer format: {answer['format']}")
    if 'auth_method' not in answer or 'url' not in answer:
        raise ValueError("Incomplete binary-ref answer")

    result = {
        'problem_type': msg['type'],
        'timing': answer.get('timing', {}),
        'shape': answer.get('shape', {}),
        'answer': ref_resolver(auth_method=answer['auth_method'], url=answer['url'])
    }
    return result


def bqm_as_file(bqm, **options):
    """Deprecated. Use :func:`~dimod.binary.BinaryQuadraticModel.to_file` instead.

    Encode in-memory BQM as DIMODBQM binary file format.

    Args:
        bqm (:class:`~dimod.BQM`):
            Binary quadratic model.

        **options (dict):
            :class:`~dimod.serialization.fileview.FileView` options.

    Returns:
        file-like:
            Binary stream with BQM encoded in DIMODBQM format.

    .. deprecated:: 0.13.2

        This function will be removed in dwave-cloud-client 0.15.0.
    """
    warnings.warn("`bqm_as_file` is deprecated since dwave-cloud-client 0.13.2 and "
                  "will be removed in 0.15.0. Use `bqm.to_file` instead",
                  DeprecationWarning, stacklevel=2)

    # NOTE: to_file() added in dimod 0.10.0
    return bqm.to_file(**options)

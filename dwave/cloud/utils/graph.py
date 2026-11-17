# Copyright 2026 D-Wave
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

"""QPU working graph utilities.

.. versionadded:: 0.14.10
   Added ``dwave.cloud.utils.graph`` module.
"""

from collections.abc import Iterable
from itertools import chain

__all__ = [
    "graph_digest",
]


_GRAPH_DIGEST_V1_PRIME_BASE = 1526414577
_GRAPH_DIGEST_V1_MODULO = 4294967291
_GRAPH_DIGEST_V1_LENGTH = 8


def _polynomial_rolling_hash(graph: Iterable[int], prime_base: int, modulo: int) -> int:
    # note: for current production solvers, the code below runs in order of 10ms;
    # 100x speed-up should be possible with cpp/cython
    value = 0
    for node in graph:
        value = (value * prime_base + node) % modulo
    return value


def graph_digest(qubits: Iterable[int],
                 couplers: Iterable[tuple[int, int]],
                 version: int = 1) -> str:
    """QPU working graph identifier (digest).

    Args:
        qubits: Working graph qubits.
        couplers: Working graph couplers.
        version: Digest algorithm version. Only ``version=1`` is supported.

    Returns:
        Graph digest as a string (8 hex chars), with a 2-char zero-padded version prefix.
    """

    if version != 1:
        raise ValueError(f"Unknown graph digest {version=}")

    flattened = chain(sorted(qubits), *sorted(couplers))

    value = _polynomial_rolling_hash(
        flattened, prime_base=_GRAPH_DIGEST_V1_PRIME_BASE, modulo=_GRAPH_DIGEST_V1_MODULO)

    return f"{version:02}{value:0{_GRAPH_DIGEST_V1_LENGTH}x}"

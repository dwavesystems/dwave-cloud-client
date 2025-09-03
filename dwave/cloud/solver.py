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

"""
A :term:`solver` is a resource for solving problems.

Solvers are responsible for:

- Encoding submitted problems
- Checking submitted parameters
- Decoding answers
- Adding problems to a client's submission queue

You can list all solvers available to a :class:`~dwave.cloud.client.Client` with its
:func:`~dwave.cloud.client.Client.get_solvers` method and select and return one with its
:func:`~dwave.cloud.client.Client.get_solver` method.

"""

import io
import concurrent.futures
import copy
import logging
import orjson
import warnings
import weakref
from collections import abc
from functools import partial, cached_property
from tempfile import SpooledTemporaryFile
from typing import Any, Literal, Optional, Union, TYPE_CHECKING

from dwave.cloud.api.models import (
    SolverConfiguration, SolverIdentity, SolverVersion)
from dwave.cloud.exceptions import (
    SolverPropertyMissingError, UnsupportedSolverError, ProblemStructureError)
from dwave.cloud.coders import (
    encode_problem_as_qp, encode_problem_as_ref, decode_binary_ref,
    decode_qp_numpy, decode_qp, decode_bq)
from dwave.cloud.computation import Future
from dwave.cloud.concurrency import Present
from dwave.cloud.events import dispatches_events
from dwave.cloud.utils.qubo import reformat_qubo_as_ising

# Use numpy if available for fast encoding/decoding
try:
    import numpy
    _numpy = True
except ImportError:
    _numpy = False

try:
    import dimod
except ImportError:
    dimod = None

__all__ = [
    'BaseSolver', 'StructuredSolver',
    'BaseUnstructuredSolver', 'UnstructuredSolver',
    'Solver', 'BQMSolver', 'CQMSolver', 'DQMSolver', 'NLSolver',
]

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # do a bit of fiddling
    try:
        _Type = Literal['qubo', 'ising']
    except ImportError:  # Python < 3.8
        _Type = str

    try:
        _Vartype = Union[_Type, dimod.typing.VartypeLike]
    except AttributeError:  # dimod not installed or too old
        _Vartype = _Type


class BaseSolver:
    """Base class for a general D-Wave solver.

    This class provides :term:`Ising`, :term:`QUBO` and :term:`BQM` sampling
    methods and encapsulates the solver description returned from the D-Wave
    cloud API.

    Args:
        client (:class:`Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.

    Examples:
        This example creates a client using the local system's default D-Wave Cloud
        Client configuration file and checks the identity of its default solver.

        >>> from dwave.cloud import Client
        >>> with Client.from_config() as client:
        ...     solver = client.get_solver()
        ...     solver.name     # doctest: +SKIP
        'Advantage_system4'
    """

    # Classes of problems the remote solver has to support (at least one of these)
    # in order for `Solver` to be able to abstract, or use, that solver
    _handled_problem_types = {}
    _handled_encoding_formats = {}

    _client_ref = None

    #: Parsed solver configuration
    data: SolverConfiguration

    #: Solver identity/specification (name, version, etc)
    identity: SolverIdentity

    @property
    def client(self):
        """Returns a reference to client if it still exists, or None otherwise."""

        client = None
        if self._client_ref is not None:
            client = self._client_ref()
            if client is None:
                raise RuntimeError('Client unavailable / closed')

        return client

    def __init__(self, client: Optional['dwave.cloud.Client'],
                 data: Union[dict, SolverConfiguration]):
        # note: we use a weakref so that client can be closed and gc'ed
        #       while keeping solver instances alive
        # dev note: allow None value for testing
        self._client_ref = weakref.ref(client) if client is not None else None

        # note: conversion to `SolverConfiguration` is cheap (order of ~10us)
        if not isinstance(data, SolverConfiguration):
            data = SolverConfiguration.model_validate(data)
        self.data = data

        self.identity = data.identity
        if self.identity is None:
            raise ValueError("Missing solver identity field")

        # Properties of this solver the server presents: dict
        try:
            self.properties = data.properties
        except AttributeError:
            raise SolverPropertyMissingError("Missing solver property: 'properties'")

        # The set of extra parameters this solver will accept in sample_ising or sample_qubo: dict
        self.parameters = self.properties.get('parameters', {})

        # Ensure this remote solver supports at least one of the problem types we know how to handle
        try:
            self.supported_problem_types = set(self.properties['supported_problem_types'])
        except KeyError:
            raise SolverPropertyMissingError(
                "Missing solver property: 'properties.supported_problem_types'")

        if self.supported_problem_types.isdisjoint(self._handled_problem_types):
            raise UnsupportedSolverError(
                ("Remote solver {name!r} supports {supports} problems, "
                 "but {cls!r} class of solvers handles only {handled}").format(
                    name=self.name,
                    supports=list(self.supported_problem_types),
                    cls=type(self).__name__,
                    handled=list(self._handled_problem_types)))

        # When True the solution data will be returned as numpy matrices: False
        # TODO: deprecate
        self.return_matrix = False

        # Derived solver properties (not present in solver data properties dict)
        self.derived_properties = {
            'id', 'name', 'online', 'avg_load', 'qpu', 'hybrid', 'software',
        }

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"

    def check_problem(self, *args, **kwargs):
        return True

    def _decode_qp(self, msg):
        if _numpy:
            return decode_qp_numpy(msg, return_matrix=self.return_matrix)
        else:
            return decode_qp(msg)

    def _download_binary_ref(self, *, auth_method: str, url: str,
                             output: Optional[io.IOBase] = None) -> Union[bytes, io.IOBase]:
        return self.client._download_answer_binary_ref(
            auth_method=auth_method, url=url, output=output).result()

    def decode_response(self, msg, answer_data: Optional[io.IOBase] = None):
        if msg['type'] not in self._handled_problem_types:
            raise ValueError('Unknown problem type received.')

        fmt = msg.get('answer', {}).get('format')
        if fmt not in self._handled_encoding_formats:
            raise ValueError(f'Unhandled answer encoding format received: {fmt!r}.')

        if fmt == 'qp':
            return self._decode_qp(msg)
        elif fmt == 'bq':
            return decode_bq(msg)
        elif fmt == 'binary-ref':
            ref_resolver = partial(self._download_binary_ref, output=answer_data)
            return decode_binary_ref(msg, ref_resolver=ref_resolver)
        else:
            raise ValueError("Don't know how to decode %r answer format" % fmt)

    # Sampling methods
    def sample_ising(self, linear, quadratic, **params):
        raise NotImplementedError

    def sample_qubo(self, qubo, **params):
        raise NotImplementedError

    def sample_bqm(self, bqm, **params):
        raise NotImplementedError

    def upload_bqm(self, bqm):
        raise NotImplementedError

    # Derived properties

    @property
    def id(self) -> str:
        """Unique solver string identifier, derived from :attr:`.identity`."""
        # keep `Solver.id` for backwards-compat, but derive it from `Solver.identity`
        warnings.warn(
            "`Solver.id` attribute meaning has changed in dwave-cloud-client 0.14.0, "
            "due to upstream API changes. Use of `Solver.identity` is now preferred.",
            DeprecationWarning, stacklevel=2)
        return str(self.identity)

    @property
    def name(self) -> str:
        """Solver name."""
        return self.identity.name

    @property
    def online(self) -> bool:
        "Is this solver online (or offline)?"
        return self.data.get('status', 'online').lower() == 'online'

    @property
    def avg_load(self) -> Optional[float]:
        "Solver's average load, at the time of description fetch."
        return self.data.get('avg_load')

    @property
    def qpu(self) -> bool:
        "Is this a QPU-based solver?"
        category = self.properties.get('category', '').lower()
        if category:
            return category == 'qpu'
        else:
            # fallback for legacy solvers without the `category` property
            # TODO: remove when all production solvers are updated
            return not (self.software or self.hybrid)

    @property
    def software(self) -> bool:
        "Is this a software-based solver?"
        category = self.properties.get('category', '').lower()
        if category:
            return category == 'software'
        else:
            # fallback for legacy solvers without the `category` property
            # TODO: remove when all production solvers are updated
            return self.name.startswith('c4-sw_')

    @property
    def hybrid(self) -> bool:
        "Is this a hybrid quantum-classical solver?"
        category = self.properties.get('category', '').lower()
        if category:
            return category == 'hybrid'
        else:
            # fallback for legacy solvers without the `category` property
            # TODO: remove when all production solvers are updated
            return self.name.startswith('hybrid')


class BaseUnstructuredSolver(BaseSolver):
    """Base class for D-Wave unstructured solvers.

    This class provides :term:`Ising`, :term:`QUBO` and :term:`BQM` sampling
    methods and encapsulates the solver description returned from the D-Wave
    cloud API.

    Args:
        client (:class:`~dwave.cloud.client.Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.

    Note:
        Events are not yet dispatched from unstructured solvers.
    """

    def sample_ising(self, linear, quadratic, offset=0, label=None, **params):
        """Sample from the specified :term:`Ising` model.

        Args:
            linear (dict/list):
                Linear biases of the Ising problem. If a dict, should be of the
                form `{v: bias, ...}` where v is a spin-valued variable and `bias`
                is its associated bias. If a list, it is treated as a list of
                biases where the indices are the variable labels.

            quadratic (dict[(variable, variable), bias]):
                Quadratic terms of the model (J), stored in a dict. With keys
                that are 2-tuples of variables and values are quadratic biases
                associated with the pair of variables (the interaction).

            offset (optional, default=0):
                Constant offset applied to the model.

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dimod package has to be installed.
        """
        try:
            import dimod
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't use unstructured solver without dimod. "
                               "Re-install the library with bqm/cqm/dqm support.")

        bqm = dimod.BinaryQuadraticModel.from_ising(linear, quadratic, offset)
        return self.sample_bqm(bqm, label=label, **params)

    def sample_qubo(self, qubo, offset=0, label=None, **params):
        """Sample from the specified :term:`QUBO`.

        Args:
            qubo (dict):
                Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem. Should be a dict of the form `{(u, v): bias, ...}`
                where `u`, `v`, are binary-valued variables and `bias` is their
                associated coefficient.

            offset (optional, default=0):
                Constant offset applied to the model.

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dimod package has to be installed.
        """
        try:
            import dimod
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't use unstructured solver without dimod. "
                               "Re-install the library with bqm/cqm/dqm support.")

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset)
        return self.sample_bqm(bqm, label=label, **params)

    def _encode_problem_for_upload(self, problem, **kwargs):
        """Encode problem for upload to solver.

        Args:
            problem (dimod-model-like):
                Problem of type `._handled_problem_types`.

            **kwargs:
                Optional problem encoding and upload parameters.

        Returns:
            file-like:
                Binary stream ready for reading and seeking. Closing the stream
                is left as an exercise to the caller.
        """
        raise NotImplementedError

    def upload_problem(self, problem, **kwargs):
        r"""Encode and upload the problem.

        Args:
            problem (model-like/file-like):
                A quadratic model or a nonlinear model handled by the solver,
                for example :class:`~dimod.BQM`, :class:`~dimod.CQM` or
                :class:`~dwave.optimization.Model`. Alternatively, encoded
                problem is given in a file-like.

            **kwargs:
                Optional problem encoding and upload parameters.

        Returns:
            :class:`concurrent.futures.Future`\ [str]:
                Problem ID in a Future. Problem ID can be used to submit
                problems by reference.
        """
        try:
            data = self._encode_problem_for_upload(problem, **kwargs)
        except Exception as e:
            data = problem
            logger.debug("Problem encoding failed with: %r, "
                         "assuming it's already encoded.", e)
        else:
            logger.debug("Problem encoded using params=%r for upload as %r",
                         kwargs, data)

        return self.client.upload_problem_encoded(data, **kwargs)

    def _encode_problem_for_submission(self, *, problem, problem_type,
                                       upload_params=None, sample_params=None,
                                       label=None, on_uploaded=None):
        """Encode `problem` for submitting in `ref` format. Upload the
        problem if it's not already uploaded.

        Args:
            problem (dimod-model-like/str):
                A quadratic model, or a reference to one (Problem ID).

            problem_type (str):
                Problem type, one of the handled problem types by the solver.

            upload_params (dict):
                Upload parameters, solver-specific.

            sample_params (dict):
                Sampling parameters, solver-specific.

            label (str, optional):
                Problem label.

            on_uploaded (callable, optional):
                Callback that's called when problem is uploaded.

        Returns:
            str:
                JSON-encoded problem submit body
        """

        if upload_params is None:
            upload_params = {}
        if sample_params is None:
            sample_params = {}

        if isinstance(problem, str):
            problem_id = problem
        else:
            logger.debug("To encode the problem for submit in the 'ref' format, "
                         "we need to upload it first.")
            problem_id = self.upload_problem(problem, **upload_params).result()

        if on_uploaded is not None:
            on_uploaded(problem_data_id=problem_id)

        body = {
            'solver': self.identity.dict(),
            'data': encode_problem_as_ref(problem_id),
            'type': problem_type,
            'params': sample_params
        }
        if label is not None:
            body['label'] = label
        body_data = orjson.dumps(body, option=orjson.OPT_SERIALIZE_NUMPY)
        logger.trace("Sampling request encoded as: %r", body_data)

        return body_data

    @dispatches_events('sample')
    def sample_problem(self, problem, problem_type=None, label=None,
                       upload_params=None, **sample_params):
        """Sample from the specified problem.

        Args:
            problem (model-like/str):
                A quadratic model (e.g. :class:`~dimod.BQM`, :class:`~dimod.CQM`,
                :class:`~dimod.DQM`), a nonlinear model (:class:`~dwave.optimization.Model`)
                or a reference to one (Problem ID returned by :meth:`.upload_problem` method).

            problem_type (str, optional):
                Problem type, one of the handled problem types by the solver.
                If not specified, the first handled problem type is used.

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            upload_params (dict):
                Optional upload/encode parameters, solver specific.

            **sample_params:
                Sampling parameters, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`
        """

        # infer problem_type; for now just the the first handled (always just one)
        if problem_type is None:
            problem_type = next(iter(self._handled_problem_types))

        # computation future holds a reference to the remote job
        computation = Future(
            solver=self, id_=None, return_matrix=self.return_matrix)

        # encode the request (body as future)
        body = self.client._encode_problem_executor.submit(
            self._encode_problem_for_submission,
            problem=problem, problem_type=problem_type, label=label,
            upload_params=upload_params, sample_params=sample_params,
            on_uploaded=computation._notify_uploaded)

        logger.debug("Submitting new problem to: %r", self.identity)
        self.client._submit(body, computation)

        return computation


class BQMSolver(BaseUnstructuredSolver):
    """Class for D-Wave unstructured binary quadratic model solvers.

    This class provides :term:`Ising`, :term:`QUBO` and :term:`BQM` sampling
    methods and encapsulates the solver description returned from the D-Wave
    cloud API.

    Args:
        client (:class:`~dwave.cloud.client.Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.
    """

    _handled_problem_types = {"bqm"}
    _handled_encoding_formats = {"bq"}

    def _encode_problem_for_upload(self, bqm, **kwargs):
        return bqm.to_file()

    def sample_bqm(self, bqm, label=None, **params):
        """Sample from the specified :term:`BQM`.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`/str):
                A binary quadratic model, or a reference to one
                (Problem ID returned by `.upload_bqm` method).

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.sample_problem(bqm, label=label, **params)

    def upload_bqm(self, bqm):
        r"""Upload the specified :term:`BQM` to SAPI, returning a Problem ID
        that can be used to submit the BQM to this solver (i.e. call the
        :meth:`~BQMSolver.sample_bqm` method).

        Args:
            bqm (:class:`~dimod.binary.BinaryQuadraticModel`\ /bytes-like/file-like):
                A binary quadratic model given either as an in-memory
                :class:`~dimod.binary.BinaryQuadraticModel` object, or as raw data
                (encoded serialized model) in either a file-like or a bytes-like
                object.

        Returns:
            :class:`concurrent.futures.Future`\ [str]:
                Problem ID in a Future. Problem ID can be used to submit
                problems by reference.

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.upload_problem(bqm)


class DQMSolver(BaseUnstructuredSolver):
    """Class for D-Wave unstructured discrete quadratic model solvers.

    This class provides a :term:`DQM` sampling
    method and encapsulates the solver description returned from the D-Wave
    cloud API.

    Args:
        client (:class:`~dwave.cloud.client.Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.
    """

    _handled_problem_types = {"dqm"}
    _handled_encoding_formats = {"bq"}

    @staticmethod
    def _bqm_to_dqm(bqm):
        """Represent a :class:`dimod.BQM` as a :class:`dimod.DQM`."""
        try:
            from dimod import DiscreteQuadraticModel
        except ImportError: # pragma: no cover
            raise RuntimeError(
                "dimod package with support for DiscreteQuadraticModel required."
                "Re-install the library with 'dqm' support.")

        dqm = DiscreteQuadraticModel()

        ising = bqm.spin

        for v, bias in ising.linear.items():
            dqm.add_variable(2, label=v)
            dqm.set_linear(v, [-bias, bias])

        for (u, v), bias in ising.quadratic.items():
            biases = numpy.array([[bias, -bias], [-bias, bias]], dtype=numpy.float64)
            dqm.set_quadratic(u, v, biases)

        return dqm

    def _encode_problem_for_upload(self, dqm, **kwargs):
        return dqm.to_file()

    def sample_bqm(self, bqm, label=None, **params):
        """Use for testing only."""

        # to sample BQM problems, we need to convert them to DQM
        if isinstance(bqm, str):
            # unless bqm already uploaded
            dqm = bqm
        else:
            dqm = self._bqm_to_dqm(bqm)

        # TODO: convert sampleset back
        return self.sample_dqm(dqm, label=label, **params)

    def sample_dqm(self, dqm, label=None, **params):
        """Sample from the specified :term:`DQM`.

        Args:
            dqm (:class:`~dimod.DiscreteQuadraticModel`/str):
                A discrete quadratic model, or a reference to one
                (Problem ID returned by `.upload_dqm` method).

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.sample_problem(dqm, label=label, **params)

    def upload_dqm(self, dqm):
        r"""Upload the specified :term:`DQM` to SAPI, returning a Problem ID
        that can be used to submit the DQM to this solver (i.e. call the
        `.sample_dqm` method).

        Args:
            dqm (:class:`~dimod.DiscreteQuadraticModel`/bytes-like/file-like):
                A discrete quadratic model given either as an in-memory
                :class:`~dimod.DiscreteQuadraticModel` object, or as raw data
                (encoded serialized model) in either a file-like or a bytes-like
                object.

        Returns:
            :class:`concurrent.futures.Future`\ [str]:
                Problem ID in a Future. Problem ID can be used to submit
                problems by reference.

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.upload_problem(dqm)


class CQMSolver(BaseUnstructuredSolver):
    """Class for D-Wave unstructured constrained quadratic model solvers.

    This class provides a :term:`CQM` sampling method and encapsulates the
    solver description returned from the D-Wave cloud API.

    Args:
        client (:class:`~dwave.cloud.client.Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.
    """

    _handled_problem_types = {"cqm"}
    _handled_encoding_formats = {"bq"}

    def _encode_problem_for_upload(self, cqm, **kwargs):
        return cqm.to_file()

    def sample_bqm(self, bqm, label=None, **params):
        """Use for testing."""
        try:
            import dimod
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't sample from 'cqm' without dimod. "
                               "Re-install the library with 'cqm' support.")

        # to sample BQM problems, we need to set them as objective of a CQM
        cqm = dimod.CQM.from_bqm(bqm)

        return self.sample_cqm(cqm, label=label, **params)

    def sample_cqm(self, cqm, label=None, **params):
        """Sample from the specified :term:`CQM`.

        Args:
            cqm (:class:`~dimod.ConstrainedQuadraticModel`/str):
                A constrained quadratic model, or a reference to one
                (Problem ID returned by `.upload_cqm` method).

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.sample_problem(cqm, label=label, **params)

    def upload_cqm(self, cqm):
        r"""Upload the specified :term:`CQM` to SAPI, returning a Problem ID
        that can be used to submit the CQM to this solver (i.e. call the
        `.sample_cqm` method).

        Args:
            cqm (:class:`~dimod.ConstrainedQuadraticModel`/bytes-like/file-like):
                A constrained quadratic model given either as an in-memory
                :class:`~dimod.ConstrainedQuadraticModel` object, or as raw data
                (encoded serialized model) in either a file-like or a bytes-like
                object.

        Returns:
            :class:`concurrent.futures.Future`\ [str]:
                Problem ID in a Future. Problem ID can be used to submit
                problems by reference.

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.upload_problem(cqm)


class NLSolver(BaseUnstructuredSolver):
    """NL solver interface.

    This class provides an :term:`NL model` sampling method and encapsulates
    the solver description returned from the D-Wave cloud API.

    Args:
        client (:class:`~dwave.cloud.client.Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.
    """

    _handled_problem_types = {"nl"}
    _handled_encoding_formats = {"binary-ref"}

    def _encode_problem_for_upload(self,
                                   model: Union['dwave.optimization.Model', io.IOBase],
                                   **kwargs
                                   ) -> io.IOBase:
        encode_params = {k: v for k, v in kwargs.items()
                         if k in {'max_num_states', 'only_decision'}}
        return model.to_file(**encode_params)

    def sample_bqm(self,
                   bqm: 'dimod.BQM',
                   label: Optional[str] = None,
                   **params) -> Future:
        """Use just for testing."""
        try:
            import dimod
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't sample from 'bqm' without dimod. "
                               "Re-install the library with 'bqm' support.")
        try:
            from dwave.optimization.generators import _from_constrained_quadratic_model
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't sample from nonlinear model without dwave-optimization. "
                               "Re-install the library with 'nlm' support.")

        # cqm to model generator requires binary bqm
        bqm.change_vartype(dimod.BINARY, inplace=True)

        # TODO: simplify when a public BQM -> Model generator is added to dwave-optimization
        cqm = dimod.CQM.from_bqm(bqm)
        nlm = _from_constrained_quadratic_model(cqm)

        return self.sample_nlm(nlm, label=label, **params)

    def sample_nlm(self,
                   model: Union['dwave.optimization.Model', io.IOBase, str],
                   label: Optional[str] = None,
                   upload_params: Optional[dict] = None,
                   **sample_params
                   ) -> Future:
        """Sample from the specified :term:`NL model`.

        Args:
            model (:class:`~dwave.optimization.Model`/bytes/str):
                A nonlinear model, serialized model, or a reference to uploaded
                model (Problem ID returned by `.upload_nlm` method).

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            upload_params (dict):
                Model encoding and upload parameters, for example parameters
                passed down to ``:meth:~dwave.optimization.model.Model.to_file``,
                like ``max_num_states``.

            **sample_params:
                Sampling parameters, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dwave-optimization package has to be installed.
        """
        try:
            import dwave.optimization
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't sample from nonlinear model without dwave-optimization. "
                               "Re-install the library with 'nlm' support.")

        return self.sample_problem(
            model, label=label, upload_params=upload_params, **sample_params)

    def sample_problem(self, *args, **kwargs):
        sf = SpooledTemporaryFile(max_size=1e8, mode='w+b')
        # backport a fix for bpo-26175, https://github.com/python/cpython/pull/29560.
        # to make sure `zipfile` works with `SpooledTemporaryFile` in Python < 3.11
        if not hasattr(sf, 'seekable'):
            # of all the methods fixed in the above PR, only seekable is actually
            # ever used in `zipfile`.
            sf.seekable = lambda: sf._file.seekable()

        future = super().sample_problem(*args, **kwargs)
        future._answer_data = sf

        return future

    def upload_nlm(self,
                   model: Union['dwave.optimization.Model', io.IOBase, bytes],
                   **upload_params,
                   ) -> concurrent.futures.Future:
        r"""Upload the specified :term:`NL model` to SAPI, returning a Problem ID
        that can be used to submit the NL model to this solver (i.e. call the
        :meth:`.sample_nlm` method).

        Args:
            model (:class:`~dwave.optimization.Model`/bytes-like/file-like):
                A nonlinear model given either as an in-memory
                :class:`~dwave.optimization.Model` object, or as raw data
                (encoded serialized model) in either a file-like or a bytes-like
                object.

            **upload_params:
                Model encoding and upload parameters, for example parameters
                passed down to ``:meth:~dwave.optimization.model.Model.to_file``,
                like ``max_num_states``.

        Returns:
            :class:`concurrent.futures.Future`\ [str]:
                Problem ID in a Future. Problem ID can be used to submit
                problems by reference.

        Note:
            To use this method, dwave-optimization package has to be installed.
        """
        return self.upload_problem(model, **upload_params)


class StructuredSolver(BaseSolver):
    """Class for D-Wave structured solvers.

    This class provides :term:`Ising`, :term:`QUBO` and :term:`BQM` sampling
    methods and encapsulates the solver description returned from the D-Wave
    cloud API.

    Args:
        client (:class:`~dwave.cloud.client.Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.
    """

    _handled_problem_types = {"ising", "qubo"}
    _handled_encoding_formats = {"qp"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The exact sequence of nodes/edges is used in encoding problems and must be preserved
        try:
            self._encoding_qubits = self.properties['qubits']
        except KeyError:
            raise SolverPropertyMissingError("Missing solver property: 'properties.qubits'")

        if 'couplers' not in self.properties:
            raise SolverPropertyMissingError("Missing solver property: 'properties.couplers'")

        # Create a set of default parameters for the queries
        self._params = {}

        # Add derived properties specific for this solver
        self.derived_properties.update({'lower_noise', 'num_active_qubits', 'version', 'graph_id'})

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, graph_id={self.graph_id!r})"

    # Derived properties

    @property
    def version(self) -> dict:
        """QPU solver version dict (contains at least ``graph_id``). Returns
        an empty dict for non-QPU solvers."""
        return v.model_dump() if (v := self.identity.version) else {}

    @property
    def graph_id(self) -> Optional[str]:
        """QPU solver working graph id. Returns ``None`` for non-QPU solvers."""
        return self.version.get('graph_id')

    @property
    def num_active_qubits(self):
        "The number of active (encoding) qubits."
        return len(self.nodes)

    @property
    def is_vfyc(self):
        "Is this a virtual full-yield chip?"
        return self.properties.get('vfyc') == True

    @property
    def has_flux_biases(self):
        "Solver supports/accepts ``flux_biases``."
        return 'flux_biases' in self.parameters

    @property
    def has_anneal_schedule(self):
        "Solver supports/accepts ``anneal_schedule``."
        return 'anneal_schedule' in self.parameters

    @property
    def num_qubits(self):
        "Nominal number of qubits on chip (includes active AND inactive)."
        return self.properties.get('num_qubits')

    @property
    def lower_noise(self):
        return "lower_noise" in self.properties.get("tags", [])

    # lazy-loaded solver attributes
    @cached_property
    def nodes(self) -> set[int]:
        """The nodes in this solver's graph."""
        return set(self._encoding_qubits)

    @property
    def variables(self) -> set[int]:
        """Alias for :attr:`.nodes`."""
        return self.nodes

    @cached_property
    def _encoding_couplers(self) -> list[tuple[int, int]]:
        # solver couplers converted to list of tuples
        return [tuple(edge) for edge in self.properties['couplers']]

    @cached_property
    def edges(self) -> set[tuple[int, int]]:
        """The edges in this solver's graph, including both directions: (a,b) and (b,a)."""
        return set(self._encoding_couplers) | \
            set((edge[1], edge[0]) for edge in self._encoding_couplers)

    @property
    def couplers(self) -> set[tuple[int, int]]:
        """Alias for :attr:`.edges`."""
        return self.edges

    @cached_property
    def undirected_edges(self) -> set[tuple[int, int]]:
        """The edges in this solver's graph, with each edge represented only once."""
        return {edge for edge in self.edges if edge[0] < edge[1]}

    def max_num_reads(self, **params):
        """Returns the maximum number of reads for the given solver parameters.

        Args:
            **params:
                Parameters for the sampling method. Relevant to num_reads:

                - annealing_time
                - readout_thermalization
                - num_reads
                - programming_thermalization

        Returns:
            int: The maximum number of reads.

        """
        # dev note: in the future it would be good to have a way of doing this
        # server-side, as we are duplicating logic here.

        properties = self.properties

        if self.software or not params:
            # software solvers don't use any of the above parameters
            return properties['num_reads_range'][1]

        # qpu

        _, duration = properties['problem_run_duration_range']

        annealing_time = params.get('annealing_time',
                                    properties['default_annealing_time'])

        readout_thermalization = params.get('readout_thermalization',
                                            properties['default_readout_thermalization'])

        programming_thermalization = params.get('programming_thermalization',
                                                properties['default_programming_thermalization'])

        return min(properties['num_reads_range'][1],
                   int((duration - programming_thermalization)
                       / (annealing_time + readout_thermalization)))

    # Sampling methods

    def sample_ising(self, linear, quadratic, offset=0, label=None, **params):
        """Sample from the specified :term:`Ising` model.

        Args:
            linear (dict/list):
                Linear biases of the Ising problem. If a dict, should be of the
                form `{v: bias, ...}` where v is a spin-valued variable and `bias`
                is its associated bias. If a list, it is treated as a list of
                biases where the indices are the variable labels.

            quadratic (dict[(int, int), float]):
                Quadratic terms of the model (J), stored in a dict. With keys
                that are 2-tuples of variables and values are quadratic biases
                associated with the pair of variables (the interaction).

            offset (float, optional, default=0):
                Constant offset applied to the model.

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Examples:
            This example creates a client using the local system's default D-Wave
            Cloud Client configuration file, which is configured to access an 
            Advantage QPU, submits a simple :term:`Ising` problem (opposite 
            linear biases on two coupled qubits), and samples 5 times.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     computation = solver.sample_ising({u: -1, v: 1}, {}, num_reads=5)   # doctest: +SKIP
            ...     for i in range(5):
            ...         print(computation.samples[i][u], computation.samples[i][v])
            ...
            ...
            (1, -1)
            (1, -1)
            (1, -1)
            (1, -1)
            (1, -1)

        """
        # Our linear and quadratic objective terms are already separated in an
        # ising model so we can just directly call `_sample`.
        return self._sample('ising', linear, quadratic, offset, params, label=label)

    def sample_qubo(self, qubo, offset=0, label=None, **params):
        """Sample from the specified :term:`QUBO`.

        Args:
            qubo (dict[(int, int), float]):
                Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem. Should be a dict of the form `{(u, v): bias, ...}`
                where `u`, `v`, are binary-valued variables and `bias` is their
                associated coefficient.

            offset (optional, default=0):
                Constant offset applied to the model.

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Examples:
            This example creates a client using the local system's default D-Wave 
            Cloud Client configuration file, which is configured to access an 
            Advantage QPU, submits a :term:`QUBO` problem (a Boolean NOT gate 
            represented by a penalty model), and samples 5 times.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     u, v = next(iter(solver.edges))
            ...     Q = {(u, u): -1, (u, v): 0, (v, u): 2, (v, v): -1}
            ...     computation = solver.sample_qubo(Q, num_reads=5)
            ...     for i in range(5):
            ...         print(computation.samples[i][u], computation.samples[i][v])
            ...
            ...
            (0, 1)
            (1, 0)
            (1, 0)
            (0, 1)
            (1, 0)

        """
        linear, quadratic = reformat_qubo_as_ising(qubo)
        return self._sample('qubo', linear, quadratic, offset, params, label=label)

    def sample_bqm(self, bqm, label=None, **params):
        """Sample from the specified :term:`BQM`.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                A binary quadratic model.

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dimod package has to be installed.
        """
        try:
            import dimod
        except ImportError: # pragma: no cover
            raise RuntimeError("Can't sample from 'bqm' without dimod. "
                               "Re-install the library with 'bqm' support.")

        if bqm.vartype is dimod.SPIN:
            problem_type = 'ising'
        elif bqm.vartype is dimod.BINARY:
            problem_type = 'qubo'
        else:
            raise TypeError("unknown/unsupported vartype")

        # convert to dicts once, to save multiple conversions later on
        linear = dict(bqm.linear)
        quadratic = dict(bqm.quadratic)

        return self._sample(problem_type, linear, quadratic, bqm.offset,
                            params, label=label, undirected_biases=True)

    @dispatches_events('sample')
    def _sample(self, type_, linear, quadratic, offset, params,
                label=None, undirected_biases=False):
        """Internal method for `sample_ising`, `sample_qubo` and `sample_bqm`.

        Args:
            linear (list/dict):
                Linear terms of the model.

            quadratic (dict[(int, int), float]):
                Quadratic terms of the model.

            offset (number):
                Constant offset applied to the model.

            params (dict):
                Parameters for the sampling method, solver-specific.

            label (str, optional):
                Problem label.

            undirected_biases (boolean, default=False):
                Are (quadratic) biases specified on undirected edges? For
                triangular or symmetric matrix of quadratic biases set it to
                ``True``.

        Returns:
            :class:`~dwave.cloud.computation.Future`
        """

        # Check the problem
        if not self.check_problem(linear, quadratic):
            raise ProblemStructureError(
                f"Problem graph incompatible with {self!r}")

        # Mix the new parameters with the default parameters
        combined_params = dict(self._params)
        combined_params.update(params)

        # Check the parameters before submitting
        for key in combined_params:
            if key not in self.parameters and not key.startswith('x_'):
                raise KeyError("{} is not a parameter of this solver.".format(key))

        # transform some of the parameters in-place
        self._format_params(type_, combined_params)

        body_dict = {
            'solver': self.identity.dict(),
            'data': encode_problem_as_qp(self, linear, quadratic, offset,
                                         undirected_biases=undirected_biases),
            'type': type_,
            'params': combined_params
        }
        if label is not None:
            body_dict['label'] = label
        body_data = orjson.dumps(body_dict, option=orjson.OPT_SERIALIZE_NUMPY)
        logger.trace("Encoded sample request: %r", body_data)

        body = Present(result=body_data)
        computation = Future(solver=self, id_=None, return_matrix=self.return_matrix)

        # XXX: offset is carried on Future until implemented in SAPI
        computation._offset = offset

        logger.debug("Submitting new problem to: %r", self.identity)
        self.client._submit(body, computation)

        return computation

    # kept for internal backwards compatibility and in case it's being
    # used externally anywhere.
    def _format_params(self, type_, params):
        """Reformat some of the parameters for sapi."""
        self.reformat_parameters(type_, params, self.properties, inplace=True)

    @staticmethod
    def reformat_parameters(vartype: '_Vartype',
                            parameters: abc.MutableMapping[str, Any],
                            properties: abc.Mapping[str, Any],
                            inplace: bool = False,
                            ) -> abc.MutableMapping[str, Any]:
        """Reformat some solver parameters for SAPI.

        Currently the only reformatted parameter is ``initial_state``. This
        method allows ``initial_state`` to be submitted as a dictionary
        mapping the qubits to their initial value.

        Args:
            vartype: One of ``'ising'`` or ``'qubo'``. If :mod:`dimod` is
                installed, this can also be any
                :class:`~dimod.typing.VartypeLike`.
            parameters: The parameters to submit to this solver.
            properties: The solver's properties. Either
                :attr:`StructuredSolver.properties` or
                :attr:`dwave.systems.DWaveSampler.properties` can be
                provided.
            inplace: Whether to modify the ``parameters`` in-place or return
                a copy.

        Returns:
            The reformatted solver parameters.
            If ``inplace`` is true the modified ``parameters`` is returned.
            If ``inplace`` is false then a deep copy of ``parameters``
            with the relevant fields updated is returned.

        """
        # whether to copy or not
        parameters = parameters if inplace else copy.deepcopy(parameters)

        # handle the vartype
        if vartype not in ('ising', 'qubo'):
            try:
                vartype = 'ising' if dimod.as_vartype(vartype) is dimod.SPIN else 'qubo'
            except (TypeError, AttributeError):
                msg = "expected vartype to be one of: 'ising', 'qubo'"
                if dimod:
                    msg += ", 'BINARY', 'SPIN', dimod.BINARY, dimod.SPIN"
                msg += f"; {vartype!r} provided"
                raise ValueError(msg) from None

        # update the parameters
        if 'initial_state' in parameters:
            initial_state = parameters['initial_state']
            if isinstance(initial_state, abc.Mapping):

                initial_state_list = [3]*properties['num_qubits']

                low = -1 if vartype == 'ising' else 0

                for v, val in initial_state.items():
                    if val == 3:
                        continue
                    if val <= 0:
                        initial_state_list[v] = low
                    else:
                        initial_state_list[v] = 1

                parameters['initial_state'] = initial_state_list
            # else: support old format

        return parameters

    def check_problem(self, linear, quadratic):
        """Test if an Ising model matches the graph provided by the solver.

        Args:
            linear (list/dict):
                Linear terms of the model (h).

            quadratic (dict[(int, int), float]):
                Quadratic terms of the model (J).

        Returns:
            bool

        Examples:
            This example creates a client using the local system's default D-Wave 
            Cloud Client configuration file, which is configured to access an 
            Advantage QPU, and tests a simple :term:`Ising` model for two target 
            minor embeddings (that is, representations of the model's graph by 
            coupled qubits on the QPU's sparsely connected graph), where only 
            the second is valid.

            >>> from dwave.cloud import Client
            >>> with Client.from_config() as client:  # doctest: +SKIP
            ...     solver = client.get_solver()
            ...     print(solver.check_problem({0: -1, 1: 1},{(0, 1):0.5}))
            ...     print(solver.check_problem({30: -1, 31: 1},{(30, 31):0.5}))
            ...
            False
            True
        """
        # handle legacy format
        if not isinstance(linear, abc.Mapping):
             linear = {idx: val for idx, val in enumerate(linear) if val != 0}

        return self.nodes.issuperset(linear) and self.edges.issuperset(quadratic)

    def estimate_qpu_access_time(self,
                                 num_qubits: int,
                                 num_reads: int = 1,
                                 annealing_time: Optional[float] = None,
                                 anneal_schedule: Optional[list[tuple[float, float]]] = None,
                                 initial_state:  Optional[list[tuple[float, float]]] = None,
                                 reverse_anneal: bool = False,
                                 reinitialize_state: bool = False,
                                 programming_thermalization: Optional[float] = None,
                                 readout_thermalization: Optional[float] = None,
                                 reduce_intersample_correlation: bool = False,
                                 **kwargs) -> float:
        """Estimates QPU access time for a submission to the selected solver.

        Estimates a problem’s quantum processing unit (QPU) access time from the
        parameter values you specify, timing data provided in the ``problem_timing_data``
        :ref:`solver property <qpu_solver_properties_all>`,
        and the number of qubits used to embed the problem on the selected QPU, as
        described in the :ref:`qpu_runtime_estimating` section.

        Requires that you provide the number of qubits to be used for your
        problem submission. :term:`Embedding` is typically heuristic and the number
        of required qubits can vary between executions.

        Args:
            num_qubits:
                Number of qubits required to represent your binary quadratic model
                on the selected solver.
            num_reads:
                Number of reads. Provide this value if you explicitly set ``num_reads``
                in your submission.
            annealing_time:
                Annealing duration. Provide this value of if you set
                ``annealing_time`` in your submission.
            anneal_schedule:
                Anneal schedule. Provide the ``anneal_schedule`` if you set it in
                your submission.
            initial_state:
                Initial state. Provide the ``initial_state`` if your submission
                uses reverse annealing.
            reinitialize_state:
                Set to ``True`` if your submission sets ``reinitialize_state``.
            programming_thermalization:
                programming thermalization time. Provide this value if you explicitly
                set a value for ``programming_thermalization`` in your submission.
            readout_thermalization:
                Set to ``True`` if your submission sets ``readout_thermalization``.
            reduce_intersample_correlation:
                Set to ``True`` if your submission sets ``reduce_intersample_correlation``.

        Returns:
            Estimated QPU access time, in microseconds.

        Raises:
            KeyError: If a solver property, or a field in the ``problem_timing_data``
                solver property, required by the timing model is missing for the
                selected solver.

            ValueError: If conflicting parameters are set or the selected solver
                uses an unsupported timing model.

        Examples:

            This example estimates the QPU access time for a ferromagnetic problem
            using all the selected QPU's qubits before deciding whether to submit
            the problem.

            .. testsetup::

                estimated_runtime = 42657   # to test solver.sample_bqm()

            >>> from dimod import BinaryQuadraticModel
            >>> from dwave.cloud import Client
            >>> reads = 100
            >>> max_time = 100000
            >>> with Client.from_config() as client:
            ...    solver = client.get_solver(qpu=True)
            ...    bqm = BinaryQuadraticModel({}, {edge: -1 for edge in solver.edges}, "BINARY")
            ...    num_qubits = len(solver.nodes)
            ...    estimated_runtime = solver.estimate_qpu_access_time(num_qubits, num_reads=reads) # doctest: +SKIP
            ...    print("Estimate of {:.0f}us on {}".format(estimated_runtime, solver.name)) # doctest: +SKIP
            ...    if estimated_runtime < max_time:
            ...       computation = solver.sample_bqm(bqm, num_reads=reads)
            Estimate of 42657us on Advantage_system4.1
            >>> print("QPU access time: {:.0f}us".format(computation.timing["qpu_access_time"]))  # doctest: +SKIP
            QPU access time: 42640us

        """
        if anneal_schedule and annealing_time:
            raise ValueError("set only one of ``anneal_schedule`` or ``annealing_time``")

        if anneal_schedule and anneal_schedule[0][1] == 1 and not initial_state:
            raise ValueError("reverse annealing requested (``anneal_schedule`` "
                             "starts with s==1) but ``initial_state`` not provided")

        try:
            problem_timing_data = self.properties['problem_timing_data']
        except:
            raise KeyError("selected solver is missing required property ``problem_timing_data``")

        try:
            version_timing_model = problem_timing_data['version']
        except:
            raise KeyError("selected solver is missing ``problem_timing_data`` field ``version``")

        try:
            typical_programming_time = problem_timing_data['typical_programming_time']
            ra_with_reinit_prog_time_delta = problem_timing_data['reverse_annealing_with_reinit_prog_time_delta']
            ra_without_reinit_prog_time_delta = problem_timing_data['reverse_annealing_without_reinit_prog_time_delta']
            default_programming_thermalization = problem_timing_data['default_programming_thermalization']
            default_annealing_time = problem_timing_data['default_annealing_time']
            readout_time_model = problem_timing_data['readout_time_model']
            readout_time_model_parameters = problem_timing_data['readout_time_model_parameters']
            qpu_delay_time_per_sample = problem_timing_data['qpu_delay_time_per_sample']
            ra_with_reinit_delay_time_delta = problem_timing_data['reverse_annealing_with_reinit_delay_time_delta']
            ra_without_reinit_delay_time_delta = problem_timing_data['reverse_annealing_without_reinit_delay_time_delta']
            decorrelation_max_nominal_anneal_time = problem_timing_data['decorrelation_max_nominal_anneal_time']
            decorrelation_time_range = problem_timing_data['decorrelation_time_range']
            default_readout_thermalization = problem_timing_data['default_readout_thermalization']

        except KeyError as err:
            err_msg = f"selected solver is missing ``problem_timing_data`` field " + \
                      f"{err} required by timing model version {version_timing_model}"
            raise ValueError(err_msg)

        # Support for sapi timing model versions: 1.0.x
        if not version_timing_model.startswith("1.0."):
            raise ValueError(f"``estimate_qpu_access_time`` does not currently "
                             f"support timing model {version_timing_model} "
                             "used by the selected solver")

        ra_programming_time = 0
        ra_delay_time = 0
        if anneal_schedule and anneal_schedule[0][1] == 1:
            if reinitialize_state:
                ra_programming_time = ra_with_reinit_prog_time_delta
                ra_delay_time = ra_with_reinit_delay_time_delta
            else:
                ra_programming_time = ra_without_reinit_prog_time_delta
                ra_delay_time = ra_without_reinit_delay_time_delta

        if programming_thermalization:
            programming_thermalization_time = programming_thermalization
        else:
            programming_thermalization_time = default_programming_thermalization

        programming_time = typical_programming_time + ra_programming_time + \
                           programming_thermalization_time

        anneal_time = default_annealing_time
        if annealing_time:
            anneal_time = annealing_time
        elif anneal_schedule:
            anneal_time = anneal_schedule[-1][0]

        n = len(readout_time_model_parameters)
        if n % 2:
             raise ValueError(f"for the selected solver, ``problem_timing_data`` "
                              f"property field ``readout_time_model_parameters`` "
                              f"is not of an even length as required by timing "
                              f"model version {version_timing_model}")

        q = readout_time_model_parameters[:n//2]
        t = readout_time_model_parameters[n//2:]
        if readout_time_model == 'pwl_log_log':
            readout_time = pow(10, numpy.interp(numpy.emath.log10(num_qubits), q, t))
        elif readout_time_model == 'pwl_linear':
            readout_time = numpy.interp(num_qubits, q, t)
        else:
            raise ValueError("``estimate_qpu_access_time`` does not support "
                             f"``readout_time_model`` value {readout_time_model} "
                             f"in version {version_timing_model}")

        if readout_thermalization:
            readout_thermalization_time = readout_thermalization
        else:
            readout_thermalization_time = default_readout_thermalization

        decorrelation_time = 0
        if reduce_intersample_correlation:
            r_min = decorrelation_time_range[0]
            r_max = decorrelation_time_range[1]
            decorrelation_time = anneal_time/decorrelation_max_nominal_anneal_time * (r_max - r_min) + r_min

        sampling_time_per_read = anneal_time + readout_time + qpu_delay_time_per_sample + \
            ra_delay_time + max(readout_thermalization_time, decorrelation_time)

        sampling_time = num_reads * sampling_time_per_read

        return sampling_time + programming_time


# for backwards compatibility:
Solver = StructuredSolver
UnstructuredSolver = BQMSolver

# list of all available solvers, ordered according to loading attempt priority
# (more specific first)
available_solvers = [StructuredSolver, BQMSolver, CQMSolver, DQMSolver, NLSolver]

from __future__ import division, absolute_import

import json
import threading
import base64
import struct
import time
import sys
import os
import posixpath
import types
import logging
import requests
import collections
import datetime

from dwave.cloud.exceptions import *


class Solver:
    """
    A solver enables sampling from an Ising model.

    Get solver objects by calling get_solver(name) on a connection object.

    The solver has responsibility for:
    - Encoding problems submitted
    - Checking the submitted parameters
    - Add problems to the Connection's submission queue

    Args:
        connection (`Connection`): Connection through which the solver is accessed.
        data: Data from the server describing this solver.
    """

    # Special flag to notify the system a solver needs access to special hardware
    _PARAMETER_ENABLE_HARDWARE = 'use_hardware'

    # Classes of problems the remote solver has to support (at least one of these)
    # in order for `Solver` to be able to abstract, or use, that solver
    _HANDLED_PROBLEM_TYPES = {"ising", "qubo"}

    def __init__(self, connection, data):
        self.connection = connection

        # data for each solver includes at least: id, description, and properties
        self.data = data

        # Each solver has an ID field
        try:
            self.id = data['id']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'id'")

        # Properties of this solver the server presents: dict
        try:
            self.properties = data['properties']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'properties'")

        # Ensure this remote solver supports at least one of the problem types we know how to handle
        try:
            self.supported_problem_types = set(self.properties['supported_problem_types'])
        except KeyError:
            raise InvalidAPIResponseError(
                "Missing solver property: 'properties.supported_problem_types'")

        if self.supported_problem_types.isdisjoint(self._HANDLED_PROBLEM_TYPES):
            raise UnsupportedSolverError(
                "Remote solver {!r} supports {} problems, but Solver() handles only {}".format(
                    self.id,
                    list(self.supported_problem_types),
                    list(self._HANDLED_PROBLEM_TYPES)))

        # The set of extra parameters this solver will accept in sample_ising or sample_qubo: dict
        try:
            self.parameters = self.properties['parameters']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'parameters'")

        # When True the solution data will be returned as numpy matrices: False
        self.return_matrix = False

        # The exact sequence of nodes/edges is used in encoding problems and must be preserved
        try:
            self._encoding_qubits = self.properties['qubits']
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'properties.qubits'")

        try:
            self._encoding_couplers = [tuple(edge) for edge in self.properties['couplers']]
        except KeyError:
            raise InvalidAPIResponseError("Missing solver property: 'properties.couplers'")

        # The nodes in this solver's graph: set(int)
        self.nodes = self.variables = set(self._encoding_qubits)

        # The edges in this solver's graph, every edge will be present as (a, b) and (b, a): set(tuple(int, int))
        self.edges = self.couplers = set(tuple(edge) for edge in self._encoding_couplers) | \
            set((edge[1], edge[0]) for edge in self._encoding_couplers)

        # The edges in this solver's graph, each edge will only be represented once: set(tuple(int, int))
        self.undirected_edges = {edge for edge in self.edges if edge[0] < edge[1]}

        # Create a set of default parameters for the queries
        self._params = {}

        # As a heuristic to guess if this is a hardware sampler check if
        # the 'annealing_time_range' property is set.
        if 'annealing_time_range' in self.properties:
            self._params[self._PARAMETER_ENABLE_HARDWARE] = True

    def __str__(self):
        return "Solver(id={!r})".format(self.id)

    def sample_ising(self, linear, quadratic, **params):
        """Draw samples from the provided Ising model.

        To submit a problem: ``POST /problems/``

        Args:
            linear (list/dict): Linear terms of the model (h).
            quadratic (dict of (int, int):float): Quadratic terms of the model (J).
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj:`Future`
        """
        # Our linear and quadratic objective terms are already separated in an
        # ising model so we can just directly call `_sample`.
        return self._sample('ising', linear, quadratic, params)

    def sample_qubo(self, qubo, **params):
        """Draw samples from the provided QUBO.

        To submit a problem: ``POST /problems/``

        Args:
            qubo (dict of (int, int):float): Terms of the model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj:`Future`
        """
        # In a QUBO the linear and quadratic terms in the objective are mixed into
        # a matrix. For the sake of encoding, we will separate them before calling `_sample`
        linear = {i1: v for (i1, i2), v in _uniform_iterator(qubo) if i1 == i2}
        quadratic = {(i1, i2): v for (i1, i2), v in _uniform_iterator(qubo) if i1 != i2}
        return self._sample('qubo', linear, quadratic, params)

    def _sample(self, type_, linear, quadratic, params, reuse_future=None):
        """Internal method for both sample_ising and sample_qubo.

        Args:
            linear (list/dict): Linear terms of the model.
            quadratic (dict of (int, int):float): Quadratic terms of the model.
            **params: Parameters for the sampling method, specified per solver.

        Returns:
            :obj: `Future`
        """
        # Check the problem
        if not self.check_problem(linear, quadratic):
            raise ValueError("Problem graph incompatible with solver.")

        # Mix the new parameters with the default parameters
        combined_params = dict(self._params)
        combined_params.update(params)

        # Check the parameters before submitting
        for key in combined_params:
            if key not in self.parameters and key != self._PARAMETER_ENABLE_HARDWARE:
                raise KeyError("{} is not a parameter of this solver.".format(key))

        # Encode the problem, use the newer format
        data = self._base64_format(self, linear, quadratic)

        body = json.dumps({
            'solver': self.id,
            'data': data,
            'type': type_,
            'params': params
        })

        # Construct where we will put the result when we finish, submit the query
        if reuse_future is not None:
            future = reuse_future
            future.__init__(self, None, self.return_matrix, (type_, linear, quadratic, params))
        else:
            future = Future(self, None, self.return_matrix, (type_, linear, quadratic, params))

        log.debug("Submitting new problem to: %s", self.id)
        self.connection._submit(body, future)
        return future

    def check_problem(self, linear, quadratic):
        """Test if an Ising model matches the graph provided by the solver.

        Args:
            linear (list/dict): Linear terms of the model (h).
            quadratic (dict of (int, int):float): Quadratic terms of the model (J).

        Returns:
            boolean
        """
        for key, value in _uniform_iterator(linear):
            if value != 0 and key not in self.nodes:
                return False
        for key, value in _uniform_iterator(quadratic):
            if value != 0 and tuple(key) not in self.edges:
                return False
        return True

    def retrieve_problem(self, id_):
        """Resume polling for a problem previously submitted.

        Args:
            id_: Identification of the query.

        Returns:
            :obj: `Future`
        """
        future = Future(self, id_, self.return_matrix, None)
        self.connection._poll(future)
        return future

    def _base64_format(self, solver, lin, quad):
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
        lin = [_uniform_get(lin, qubit, 0) for qubit in solver._encoding_qubits]
        lin = base64.b64encode(struct.pack('<' + ('d' * len(lin)), *lin))

        # Encode the coefficients of the quadratic terms of the objective
        # in the same manner as the linear terms, in the order given by the
        # _encoding_couplers property
        quad = [quad.get(edge, 0) + quad.get((edge[1], edge[0]), 0)
                for edge in solver._encoding_couplers]
        quad = base64.b64encode(struct.pack('<' + ('d' * len(quad)), *quad))

        # The name for this encoding is 'qp' and is explicitly included in the
        # message for easier extension in the future.
        return {
            'format': 'qp',
            'lin': lin.decode('utf-8'),
            'quad': quad.decode('utf-8')
        }

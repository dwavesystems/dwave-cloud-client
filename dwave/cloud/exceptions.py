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

import requests


class ConfigFileError(Exception):
    """Base exception for all config file processing errors."""

class ConfigFileReadError(ConfigFileError):
    """Non-existing or unreadable config file specified or implied."""

class ConfigFileParseError(ConfigFileError):
    """Invalid format of config file."""


class SAPIRequestError(requests.exceptions.RequestException):
    """Generic SAPI request error base class."""

    def __init__(self, *args, **kwargs):
        self.error_msg = kwargs.pop('error_msg', None)
        self.error_code = kwargs.pop('error_code', None)
        if len(args) < 1:
            args = (self.error_msg, )
        super().__init__(*args, **kwargs)

    def __str__(self):
        return super().__str__() or self.error_msg or ''

# alias for backward compatibility
SAPIError = SAPIRequestError

# 400
class BadRequestError(SAPIRequestError):
    pass

# 401
class UnauthorizedRequestError(SAPIRequestError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.error_msg:
            self.error_msg = "Invalid token or access denied"

# 403
class ForbiddenRequestError(SAPIRequestError):
    pass

# 404
class NotFoundError(SAPIRequestError):
    pass

# 409
class ConflictedRequestError(SAPIRequestError):
    pass

# 429
class TooManyRequestsError(SAPIRequestError):
    pass

# 5xx
class InternalServerError(SAPIRequestError):
    pass


class ResourceAuthenticationError(UnauthorizedRequestError):
    """Access to resource not authorized."""

class ResourceNotFoundError(NotFoundError):
    """Resource not found."""

class ResourceConflictError(ConflictedRequestError):
    """Conflict in the current state of the resource."""

class APILimitExceeded(TooManyRequestsError):
    """Number of API requests exceeds the permitted limit."""


class SolverError(SAPIRequestError):
    """Generic base class for all solver-related errors."""

class ProblemError(SAPIRequestError):
    """Generic base class for all problem-related errors."""


class SolverFailureError(SolverError):
    """An exception raised when there is a remote failure calling a solver."""

class SolverNotFoundError(ResourceNotFoundError, SolverError):
    """Solver with matching feature set not found / not available."""

class ProblemNotFoundError(ResourceNotFoundError, ProblemError):
    """Problem not found."""

class SolverOfflineError(SolverError):
    """Action attempted on an offline solver."""

class SolverAuthenticationError(ResourceAuthenticationError, SolverError):
    """An exception raised when there is an authentication error."""

class UnsupportedSolverError(SolverError):
    """The solver we received from the API is not supported by the client."""

class SolverPropertyMissingError(UnsupportedSolverError):
    """The solver we received from the API does not have required properties."""


class Timeout(SAPIRequestError):
    """General timeout error."""

class RequestTimeout(Timeout):
    """REST API request timed out."""

class PollingTimeout(Timeout):
    """Problem polling timed out."""


class CanceledFutureError(Exception):
    """An exception raised when code tries to read from a canceled future."""

    def __init__(self):
        super().__init__("An error occurred reading results from a canceled request")


class InvalidAPIResponseError(SAPIError):
    """Raised when an invalid/unexpected response from D-Wave Solver API is received."""


class InvalidProblemError(ValueError):
    """Solver cannot handle the given binary quadratic model."""


class ProblemUploadError(Exception):
    """Problem multipart upload failed."""

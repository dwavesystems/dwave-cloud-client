# Copyright 2021 D-Wave Systems Inc.
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


class RequestError(requests.exceptions.RequestException):
    """Generic SAPI request error"""

    error_msg = None
    error_code = None

    def __init__(self, *args, **kwargs):
        # exception message populated from, in order of precedence: `args`,
        # `error_msg` kwarg, exception docstring
        self.error_msg = kwargs.pop('error_msg', self.error_msg)
        self.error_code = kwargs.pop('error_code', self.error_code)
        if len(args) < 1 and self.error_msg is not None:
            args = (self.error_msg, )
        if len(args) < 1:
            args = (self.__doc__, )
        super().__init__(*args, **kwargs)


class ResourceBadRequestError(RequestError):
    """Resource failed to parse the request"""

class ResourceAuthenticationError(RequestError):
    """Access to resource not authorized: token is invalid or missing"""

class ResourceAccessForbiddenError(RequestError):
    """Access to resource forbidden"""

class ResourceNotFoundError(RequestError):
    """Resource not found"""

class ResourceConflictError(RequestError):
    """Conflict in the current state of the resource"""

class ResourceLimitsExceededError(RequestError):
    """Number of resource requests exceed the permitted limit"""

class ResourceBadResponseError(RequestError):
    """Unexpected resource response"""

class InternalServerError(RequestError):
    """internal server error occurred while request handling."""

class RequestTimeout(RequestError):
    """API request timed out"""

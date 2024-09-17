# Copyright 2024 D-Wave Inc.
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

"""Config-related exceptions.

.. versionchanged:: 0.13.1
   These exceptions previously lived under ``dwave.cloud.exceptions``.

"""

__all__ = ['ConfigFileError', 'ConfigFileReadError', 'ConfigFileParseError']


class ConfigFileError(Exception):
    """Base exception for all config file processing errors."""

class ConfigFileReadError(ConfigFileError):
    """Non-existing or unreadable config file specified or implied."""

class ConfigFileParseError(ConfigFileError):
    """Invalid format of config file."""

# Copyright 2020 D-Wave Systems Inc.
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
Early prototype of an event subsystem that allows external actors to hook into
data processing pipeline and access otherwise private data for closer
inspection.
"""

import logging

__all__ = ['add_handler']

logger = logging.getLogger(__name__)


# package-global event hooks registry
# Dict[event_name: str, handler_functions: list]
_client_event_hooks_registry = {
    'before_client_init': [],
    'after_client_init': [],
    'before_get_solvers': [],
    'after_get_solvers': [],
    'before_sample': [],
    'after_sample': [],
}


# TODO: rewrite as decorator that automatically captures function input/output
def add_handler(name, handler):
    """Register a `handler` function to be called on event `name`.

    Handler's signature are::

        def before_event_handler(event_name, obj, args):
            pass

        def after_event_handler(event_name, obj, args, return_value):
            pass

    """

    if name not in _client_event_hooks_registry:
        raise ValueError('invalid hook name')
    if not callable(handler):
        raise TypeError('callable handler required')

    _client_event_hooks_registry[name].append(handler)


def dispatch_event(name, *args, **kwargs):
    """Call the complete chain of event handlers attached to event `name`."""

    if name not in _client_event_hooks_registry:
        raise ValueError('invalid event name')

    for handler in _client_event_hooks_registry[name]:
        try:
            handler(name, *args, **kwargs)
        except Exception as e:
            logger.debug("Exception in {!r} event handler {!r}: {!r}".format(
                name, handler, e))

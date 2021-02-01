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
import inspect
from functools import wraps

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

    Handler signatures are::

        def before_event_handler(event_name, obj, args):
            # called just before `obj.method(**args)` executes

        def after_event_handler(event_name, obj, args, return_value):
            # function succeeded with `return_value`

        def after_event_handler(event_name, obj, args, exception):
            # function failed with `exception` raised
            # after event handler invocation, exception is re-raised

    """

    if name not in _client_event_hooks_registry:
        raise ValueError('invalid hook name')
    if not callable(handler):
        raise TypeError('callable handler required')

    _client_event_hooks_registry[name].append(handler)


def dispatch_event(name, *args, **kwargs):
    """Call the complete chain of event handlers attached to event `name`."""

    logger.trace("dispatch_event(%r, *%r, **%r)", name, args, kwargs)

    if name not in _client_event_hooks_registry:
        raise ValueError('invalid event name')

    for handler in _client_event_hooks_registry[name]:
        try:
            handler(name, *args, **kwargs)
        except Exception as e:
            logger.debug("Exception in {!r} event handler {!r}: {!r}".format(
                name, handler, e))


class dispatches_events:
    """Decorate function to :func:`.dispatch_event` on entry and exit."""

    def __init__(self, basename):
        self.before_eventname = 'before_' + basename
        self.after_eventname = 'after_' + basename

    def __call__(self, fn):
        if not callable(fn):
            raise TypeError("decorated object must be callable")

        @wraps(fn)
        def wrapped(*pargs, **kwargs):
            sig = inspect.signature(fn)
            bound = sig.bind(*pargs, **kwargs)
            bound.apply_defaults()
            args = bound.arguments
            obj = args.pop('self', None)

            dispatch_event(self.before_eventname, obj=obj, args=args)
            try:
                rval = fn(*pargs, **kwargs)
            except Exception as exc:
                dispatch_event(self.after_eventname, obj=obj, args=args, exception=exc)
                raise
            else:
                dispatch_event(self.after_eventname, obj=obj, args=args, return_value=rval)
                return rval

        return wrapped

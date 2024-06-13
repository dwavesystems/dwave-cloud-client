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

"""Command-line interface utilities for private and Ocean-internal use.

.. versionchanged:: 0.12.0
   These functions previously lived under ``dwave.cloud.utils``.
"""

from functools import partial
from typing import Any, Callable, Optional, Union, Sequence

__all__ = []


def default_text_input(prompt: str, default: Optional[Any] = None, *,
                       optional: bool = True,
                       choices: Optional[Sequence[Any]] = None) -> Union[str, None]:
    # CLI util; defer click import until actually needed (see #473)
    import click

    _skip = 'skip'
    kwargs = dict(text=prompt)
    if default:
        kwargs.update(default=default)
    else:
        # make click print [skip] next to prompt
        if optional:
            kwargs.update(default=_skip)
    if choices:
        _type = click.Choice(choices)
        kwargs.update(type=_type)
        # a special case to skip user input instead of forcing input
        if optional:
            def allow_skip(value):
                if value == _skip:
                    return value
                return click.types.convert_type(_type)(value)

            kwargs.update(value_proc=allow_skip)

    value = click.prompt(**kwargs)
    if optional and value == _skip:
        value = None
    return value


def strtrunc(s: Any, maxlen: int = 60) -> str:
    s = str(s)
    return s[:(maxlen-3)]+'...' if len(s) > maxlen else s


class CLIError(Exception):
    """CLI command error that includes the error code in addition to the
    standard error message."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def deprecated_option(msg: Optional[str] = None, update: Optional[str] = None) -> Callable:
    """Generate a Click callback function that will print a deprecation notice
    to stderr with a customized message and copy option value to new option.

    Note: if you provide the ``update`` option name, make sure that option is
    processed before the deprecated one (set ``is_eager``).

    Example::

        @click.option('--config-file', '-f', default=None, is_eager=True)
        @click.option(
            '-c', default=None, expose_value=False,
            help="[Deprecated in favor of '-f']",
            callback=deprecated_option(DEPRECATION_MSG, update='config_file'))
        ...
        def ping(config_file, ...):
            pass

    """
    # CLI util; defer click import until actually needed (see #473)
    import click

    def _print_deprecation(ctx, param, value, msg=None, update=None):
        if msg is None:
            msg = "DeprecationWarning: The following options are deprecated: {opts!r}."
        if value and not ctx.resilient_parsing:
            click.echo(click.style(msg.format(opts=param.opts), fg="red"), err=True)
            if update:
                ctx.params[update] = value

    # click seems to strip closure variables in calls to `callback`,
    # so we pass `msg` and `update` via partial application
    return partial(_print_deprecation, msg=msg, update=update)

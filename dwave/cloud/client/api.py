import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Union

import requests
import urllib3

from dwave.cloud.package_info import __packagename__, __version__
from dwave.cloud.exceptions import (
    # standard http errors returned by sapi
    BadRequestError, UnauthorizedRequestError, ForbiddenRequestError,
    NotFoundError, ConflictedRequestError, TooManyRequestsError, InternalServerError)
from dwave.cloud.utils import (
    TimeoutingHTTPAdapter, BaseUrlSession, user_agent, is_caused_by)

__all__ = ['SAPIClient']

logger = logging.getLogger(__name__)


class LazyUserAgentClassProperty:
    # roughly equivalent to ``classmethod(property(cached_user_agent))``, but it
    # doesn't require chained decorators support available in py39+
    _user_agent = None

    def __get__(self, obj, objtype=None):
        # Note: The only tags that might change are platform tags, as returned
        # by `dwave.common.platform.tags` entry points, and `platform.platform()`
        # (like linux kernel version). Assuming OS/machine won't change during
        # client's lifespan, and typical platform tags defined via entry points
        # depend on process environment variables (which rarely change), it's
        # pretty safe to cache the user-agent per-class (or even globally).
        if self._user_agent is None:
            self._user_agent = user_agent(__packagename__, __version__)
        return self._user_agent


class SAPISession(BaseUrlSession):
    """:class:`.BaseUrlSession` extended to unify timeout exceptions and log
    requests.
    """

    def request(self, method, *args, **kwargs):
        # log args with and a caller from a correct stack level
        caller = inspect.stack()[1].function
        callee = type(self).__name__
        logger.trace("[%s] %s.request(%r, *%r, **%r)",
                    caller, callee, method, args, kwargs)

        # unify timeout exceptions
        try:
            response = super().request(method, *args, **kwargs)

        except Exception as exc:
            logger.trace("[%s] %s.request failed with %r", caller, callee, exc)

            if is_caused_by(exc, (requests.exceptions.Timeout,
                                urllib3.exceptions.TimeoutError)):
                raise RequestTimeout from exc
            else:
                raise

        logger.trace("[%s] %s.request response=(code=%r, body=%r)",
                    caller, callee, response.status_code, response.text)

        return response


class SAPIClient:
    """Low-level SAPI client, as a thin wrapper around `requests.Session`,
    that handles SAPI specifics like authentication and response parsing.
    """

    DEFAULTS = {
        'endpoint': None,
        'token': None,
        'cert': None,

        # (connect, read) timeout in sec
        'timeout': (60, 120),

        # urllib3.Retry options
        'retry': dict(total=10, backoff_factor=0.01, backoff_max=60),

        # optional additional headers
        'headers': None,

        # ssl verify
        'verify': True,

        # proxy urls, see :attr:`requests.Session.proxies`
        'proxies': None,
    }

    # User-Agent string used in SAPI requests, as returned by
    # :meth:`~dwave.cloud.utils.user_agent`, computed on first access and
    # cached for the lifespan of the class.
    # TODO: consider exposing "user_agent" config parameter
    user_agent = LazyUserAgentClassProperty()

    def __init__(self, **kwargs):
        # populate .config with defaults overridden with supplied kwargs
        self.config = {}
        for opt, default in self.DEFAULTS.items():
            self.config[opt] = kwargs.get(opt, default)

        self.session = self._create_session()

    @classmethod
    def from_client_config(cls, client):
        """Create SAPI client instance configured from a
        :class:`~dwave.cloud.client.base.Client' instance.
        """

        headers = client.headers.copy()
        if client.connection_close:
            headers.update({'Connection': 'close'})

        opts = dict(
            endpoint=client.endpoint,
            token=client.token,
            cert=client.client_cert,
            timeout=client.request_timeout,
            proxies=dict(
                http=client.proxy,
                https=client.proxy,
            ),
            retry=dict(
                total=client.http_retry_total,
                connect=client.http_retry_connect,
                read=client.http_retry_read,
                redirect=client.http_retry_redirect,
                status=client.http_retry_status,
                raise_on_redirect=True,
                raise_on_status=True,
                respect_retry_after_header=True,
                backoff_factor=client.http_retry_backoff_factor,
                backoff_max=client.http_retry_backoff_max,
            ),
            headers=client.headers,
            verify=not client.permissive_ssl,
        )

        return cls(**opts)

    @staticmethod
    def _retry(backoff_max=None, **kwargs):
        """Create http idempotent urllib3.Retry config."""

        retry = urllib3.Retry(**kwargs)

        # note: `Retry.BACKOFF_MAX` can't be set on construction
        if backoff_max is not None:
            retry.BACKOFF_MAX = backoff_max

        return retry

    def _create_session(self):
        # allow endpoint path to not end with /
        # (handle incorrect user input when merging paths, see rfc3986, sec 5.2.3)
        endpoint = self.config['endpoint']
        if not endpoint.endswith('/'):
            endpoint += '/'

        # configure request timeout and retries
        session = SAPISession(base_url=endpoint)
        timeout = self.config['timeout']
        retry = self.config['retry']
        session.mount('http://',
            TimeoutingHTTPAdapter(
                timeout=timeout, max_retries=self._retry(**retry)))
        session.mount('https://',
            TimeoutingHTTPAdapter(
                timeout=timeout, max_retries=self._retry(**retry)))

        # configure headers
        session.headers.update({'User-Agent': self.user_agent})
        if self.config['headers']:
            session.headers.update(self.config['headers'])

        # auth
        if self.config['token']:
            session.headers.update({'X-Auth-Token': self.config['token']})
        if self.config['cert']:
            session.cert = self.config['cert']

        if self.config['proxies']:
            session.proxies = self.config['proxies']

        # raise all response errors as exceptions automatically
        session.hooks['response'].append(self._raise_for_status)

        # debug log
        logger.debug("create_session from config={!r}".format(self.config))

        return session

    @staticmethod
    def _raise_for_status(response, **kwargs):
        """Raises :class:`~dwave.cloud.exceptions.SAPIRequestError`, if one
        occurred, with message populated from SAPI error response.

        See:
            :meth:`requests.Response.raise_for_status`.

        Raises:
            :class:`dwave.cloud.exceptions.SAPIRequestError` subclass
        """
        # NOTE: the expected behavior is for SAPI to return JSON error on
        # failure. However, that is currently not the case. We need to work
        # around this until it's fixed.

        # no error -> body is json
        # error -> body can be json or plain text error message
        if response.ok:
            try:
                response.json()
            except:
                raise InvalidAPIResponseError("JSON response expected")

        else:
            try:
                msg = response.json()
                error_msg = msg['error_msg']
                error_code = msg['error_code']
            except:
                # TODO: better message when error body blank
                error_msg = response.text or response.reason
                error_code = response.status_code

            kw = dict(error_msg=error_msg,
                      error_code=error_code,
                      response=response)

            # map known SAPI error codes to exceptions
            exception_map = {
                400: BadRequestError,
                401: UnauthorizedRequestError,
                403: ForbiddenRequestError,
                404: NotFoundError,
                409: ConflictedRequestError,
                429: TooManyRequestsError,
            }
            if error_code in exception_map:
                raise exception_map[error_code](**kw)
            elif 500 <= error_code < 600:
                raise InternalServerError(**kw)
            else:
                raise SAPIRequestError(**kw)


@dataclass
class SolverDescription:
    id: str
    status: str
    description: str
    properties: dict
    avg_load: float


class Solvers(SAPIClient):

    def list_solvers(self) -> List[SolverDescription]:
        path = 'solvers/remote/'
        response = self.session.get(path)
        solvers = response.json()
        return [SolverDescription(**s) for s in solvers]

    def get_solver(self, solver_id: str) -> SolverDescription:
        path = 'solvers/remote/{}'.format(solver_id)
        response = self.session.get(path)
        solver = response.json()
        return SolverDescription(**solver)


@dataclass
class ProblemStatus:
    id: str
    solver: str
    type: str
    label: str
    submitted_on: str   # TODO: convert to datetime?
    solved_on: str      # TODO: convert to datetime?
    status: str         # TODO: make enum?

@dataclass
class ProblemAnswer:
    format: str
    active_variables: str
    energies: str
    solutions: str
    timing: dict
    num_occurrences: int
    num_variables: int

@dataclass
class ProblemStatusWithAnswer(ProblemStatus):
    answer: ProblemAnswer

@dataclass
class ProblemData:
    format: str
    # qp format fields
    lin: str = None
    quad: str = None
    offset: float = 0
    # ref format fields
    data: str = None

@dataclass
class ProblemMetadata:
    solver: str
    type: str
    label: str
    submitted_by: str
    submitted_on: str   # TODO: convert to datetime?
    solved_on: str      # TODO: convert to datetime?
    status: str         # TODO: make enum?
    messages: list = None

@dataclass
class ProblemInfo:
    id: str
    data: ProblemData
    params: dict
    metadata: ProblemMetadata
    answer: ProblemAnswer

    # helper to unpack nested fields
    def __init__(self, **info):
        self.id = info['id']
        self.data = ProblemData(**info['data'])
        self.params = info['params']
        self.metadata = ProblemMetadata(**info['metadata'])
        self.answer = ProblemAnswer(**info['answer'])


class Problems(SAPIClient):

    def list_problems(self, **params) -> List[ProblemStatus]:
        # available params: id, label, max_results, status, solver
        path = 'problems'
        response = self.session.get(path, params=params)
        statuses = response.json()
        return [ProblemStatus(**s) for s in statuses]

    def get_problem(self, problem_id: str) -> Union[ProblemStatus,
                                                    ProblemStatusWithAnswer]:
        """Retrieve problem short status and answer if answer is available."""
        path = 'problems/{}'.format(problem_id)
        response = self.session.get(path)
        status = response.json()
        answer = status.pop('answer', None)
        if answer is not None:
            return ProblemStatusWithAnswer(answer=ProblemAnswer(**answer), **status)
        else:
            return ProblemStatus(**status)

    def get_problem_status(self, problem_id: str) -> ProblemStatus:
        """Retrieve short status of a single problem."""
        path = 'problems/'
        params = dict(id=problem_id)
        response = self.session.get(path, params=params)
        status = response.json()[0]
        return ProblemStatus(**status)

    def get_problem_statuses(self, problem_ids: List[str]) -> List[ProblemStatus]:
        """Retrieve short problem statuses for a list of problems."""
        if len(problem_ids) > 1000:
            raise ValueError('number of problem ids is limited to 1000')

        path = 'problems/'
        params = dict(id=','.join(problem_ids))
        response = self.session.get(path, params=params)
        statuses = response.json()
        return [ProblemStatus(**s) for s in statuses]

    def get_problem_info(self, problem_id: str) -> ProblemInfo:
        """Retrieve complete problem info."""
        path = 'problems/{}/info'.format(problem_id)
        response = self.session.get(path)
        info = response.json()
        return ProblemInfo(**info)

    def get_problem_answer(self, problem_id: str) -> ProblemAnswer:
        """Retrieve problem answer."""
        path = 'problems/{}/answer'.format(problem_id)
        response = self.session.get(path)
        answer = response.json()
        return ProblemAnswer(**answer)

    def get_problem_messages(self, problem_id: str) -> List[dict]:
        """Retrieve list of problem messages."""
        path = 'problems/{}/messages'.format(problem_id)
        response = self.session.get(path)
        return response.json()

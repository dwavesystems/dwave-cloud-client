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

from __future__ import annotations

import io
import orjson
from collections import abc
from typing import Callable, Union, Optional, get_type_hints, TYPE_CHECKING
from functools import wraps

from pydantic import TypeAdapter

from dwave.cloud.api import constants, models
from dwave.cloud.api.client import (
    DWaveAPIClient, SolverAPIClient, MetadataAPIClient, LeapAPIClient)

if TYPE_CHECKING:
    from dwave.cloud.config.models import ClientConfig

__all__ = ['Solvers', 'Problems', 'Regions']


class accepts:
    """Decorate :class:`ResourceBase` methods to enforce API response type
    validation.
    """

    def __init__(self,
                 media_type: Optional[str] = constants.DEFAULT_API_MEDIA_TYPE,
                 version: Optional[str] = None,
                 **kwargs):

        self.ctx = dict(media_type=media_type, accept_version=version)
        self.ctx.update(kwargs)

    def __call__(self, fn: abc.Callable):
        @wraps(fn)
        def wrapper(obj: 'ResourceBase', *args, **kwargs):
            try:
                obj.session.set_accept(**self.ctx)
                return fn(obj, *args, **kwargs)

            finally:
                obj.session.unset_accept()

        return wrapper


class compress_if:
    """Decorate :class:`ResourceBase` methods to enable compression on outbound
    requests if predicate evaluates to truthy.
    """

    def __init__(self, compress: Union[bool, Callable] = True, **kwargs):
        if not callable(compress):
            compress = lambda *args, **kwargs: compress
        self.compress = compress

    def __call__(self, fn: abc.Callable):
        @wraps(fn)
        def wrapper(obj: 'ResourceBase', *args, **kwargs):
            restore_to = False
            try:
                compress = self.compress(obj, *args, **kwargs)
                restore_to = obj.session.set_payload_compress(compress=compress)
                return fn(obj, *args, **kwargs)

            finally:
                obj.session.set_payload_compress(compress=restore_to)

        return wrapper


class ResourceBase:
    """A class for interacting with a Leap resource."""

    # api client used by the resource class
    client_class: DWaveAPIClient = DWaveAPIClient

    # endpoint path prefix (base path) specific to all methods on the resource
    resource_path: Optional[str] = None

    def __init__(self, client: Optional[DWaveAPIClient] = None, **config):
        if client is not None:
            self.client = client
        else:
            self.client = self.client_class(**config)

    @property
    def session(self):
        session = getattr(self, '_session', None)

        # set path prefix on first access only
        if session is None:
            session = self._session = self.client.session
            if self.resource_path:
                session.base_url = session.create_url(self.resource_path)

        return session

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def from_config(cls, config: Optional[Union[ClientConfig, str]] = None, **kwargs):
        client = cls.client_class.from_config(config, **kwargs)
        return cls(client=client)


class Regions(ResourceBase):

    resource_path = 'regions/'
    client_class = MetadataAPIClient

    @accepts(media_type='application/vnd.dwave.metadata.regions+json', version='~=1.0')
    def list_regions(self, **kwargs) -> list[models.Region]:
        path = ''
        response = self.session.get(path, **kwargs)
        regions = orjson.loads(response.content)
        return TypeAdapter(list[models.Region]).validate_python(regions)

    @accepts(media_type='application/vnd.dwave.metadata.region+json', version='~=1.0')
    def get_region(self, code: str, **kwargs) -> models.Region:
        path = '{}'.format(code)
        response = self.session.get(path, **kwargs)
        region = orjson.loads(response.content)
        return TypeAdapter(models.Region).validate_python(region)


class Solvers(ResourceBase):

    resource_path = 'solvers/'
    client_class = SolverAPIClient

    @accepts(media_type='application/vnd.dwave.sapi.solver-definition-list+json', version='~=2.0')
    def list_solvers(self, filter: Optional[str] = None, **kwargs) -> list[models.SolverConfiguration]:
        path = 'remote/'
        params = {'filter': filter} if filter is not None else None
        response = self.session.get(path, params=params, **kwargs)
        # see asv benchmarks `JSONResponseDecode` for json -> model performance
        # tl;dr: model_validate() on orjson-decoded content is much faster then
        #        model_validate_json() and faster than model_validate on json-decoded
        solvers = orjson.loads(response.content)
        return TypeAdapter(list[models.SolverConfiguration]).validate_python(solvers)

    @accepts(media_type='application/vnd.dwave.sapi.solver-definition+json', version='~=2.0')
    def get_solver(self, solver_id: str, filter: Optional[str] = None, **kwargs) -> models.SolverConfiguration:
        path = 'remote/{}'.format(solver_id)
        params = {'filter': filter} if filter is not None else None
        response = self.session.get(path, params=params, **kwargs)
        solver = orjson.loads(response.content)
        return models.SolverConfiguration.model_validate(solver)


class Problems(ResourceBase):

    resource_path = 'problems/'
    client_class = SolverAPIClient

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def list_problems(self, *,
                      id: str = None,
                      label: str = None,
                      max_results: int = None,
                      status: Union[str, constants.ProblemStatus] = None,
                      solver: str = None,
                      **params) -> list[models.ProblemStatus]:
        """Retrieve a filtered list of submitted problems."""

        # build query
        if id is not None:
            params.setdefault('id', id)
        if label is not None:
            params.setdefault('label', label)
        if max_results is not None:
            params.setdefault('max_results', max_results)
        if isinstance(status, constants.ProblemStatus):
            params.setdefault('status', status.value)
        elif status is not None:
            params.setdefault('status', status)
        if solver is not None:
            params.setdefault('solver', solver)

        path = ''
        response = self.session.get(path, params=params)
        statuses = orjson.loads(response.content)
        return TypeAdapter(list[models.ProblemStatus]).validate_python(statuses)

    @accepts(media_type='application/vnd.dwave.sapi.problem+json', version='>=2.1,<3')
    def get_problem(self,
                    problem_id: str,
                    timeout: Optional[int] = None,
                    ) -> models.ProblemStatusMaybeWithAnswer:
        """Retrieve problem short status and answer if answer is available."""
        path = '{}'.format(problem_id)
        params = {}
        if timeout is not None:
            params.setdefault('timeout', timeout)
        response = self.session.get(path, params=params)
        status = orjson.loads(response.content)
        return models.ProblemStatusMaybeWithAnswer.model_validate(status)

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def get_problem_status(self,
                           problem_id: str,
                           timeout: Optional[int] = None,
                           ) -> models.ProblemStatus:
        """Retrieve short status of a single problem."""
        path = ''
        params = dict(id=problem_id)
        if timeout is not None:
            params.setdefault('timeout', timeout)
        response = self.session.get(path, params=params)
        status = orjson.loads(response.content)[0]
        return models.ProblemStatus.model_validate(status)

    # XXX: @pydantic.validate_arguments
    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def get_problem_statuses(self,
                             problem_ids: list[str],
                             timeout: Optional[int] = None,
                             ) -> list[models.ProblemStatus]:
        """Retrieve short problem statuses for a list of problems."""
        if not isinstance(problem_ids, list):
            raise TypeError('a list of problem ids expected')
        if len(problem_ids) > 1000:
            raise ValueError('number of problem ids is limited to 1000')

        path = ''
        params = dict(id=','.join(problem_ids))
        if timeout is not None:
            params.setdefault('timeout', timeout)
        response = self.session.get(path, params=params)
        statuses = orjson.loads(response.content)
        return TypeAdapter(list[models.ProblemStatus]).validate_python(statuses)

    @accepts(media_type='application/vnd.dwave.sapi.problem-data+json', version='>=2.1,<3')
    def get_problem_info(self, problem_id: str) -> models.ProblemInfo:
        """Retrieve complete problem info."""
        path = '{}/info'.format(problem_id)
        response = self.session.get(path)
        info = orjson.loads(response.content)
        return models.ProblemInfo.model_validate(info)

    @accepts(media_type='application/vnd.dwave.sapi.problem-answer+json', version='>=2.1,<3')
    def get_problem_answer(self, problem_id: str) -> models.ProblemAnswer:
        """Retrieve problem answer."""
        path = '{}/answer'.format(problem_id)
        response = self.session.get(path)
        answer = orjson.loads(response.content)['answer']
        return models.ProblemAnswer.model_validate(answer)

    @accepts(media_type='application/octet-stream')
    def get_answer_data(self, answer: models.UnstructuredProblemAnswerBinaryRef,
                        output: Optional[io.IOBase] = None) -> io.IOBase:
        """Retrieve binary-ref answer data."""
        if answer.auth_method != constants.BinaryRefAuthMethod.SAPI_TOKEN:
            raise ValueError(f"Authentication method {answer.auth_method!r} not supported.")

        if output is None:
            output = io.BytesIO()

        for chunk in self.session.get(answer.url, stream=True).iter_content(chunk_size=8192):
            output.write(chunk)

        output.seek(0)

        return output

    @accepts(media_type='application/vnd.dwave.sapi.problem-message+json', version='>=2.1,<3')
    def get_problem_messages(self, problem_id: str) -> list[dict]:
        """Retrieve list of problem messages."""
        path = '{}/messages'.format(problem_id)
        response = self.session.get(path)
        return orjson.loads(response.content)

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    @compress_if(lambda obj, **_: obj.client.config.get('compress_qpu_problem_data'))
    def submit_problem(self, *,
                       data: models.ProblemData,
                       params: dict,
                       solver: str,
                       type: constants.ProblemType,
                       label: Optional[str] = None) -> \
            Union[models.ProblemStatusMaybeWithAnswer, models.ProblemSubmitError]:
        """Blocking problem submit with timeout, returning final status and
        answer, if problem is solved within the (undisclosed) time limit.
        """
        path = ''
        # note: orjson.dumps(model.model_dump()) is about 6-7x faster then
        # model.model_dump_json() - 45us vs 300us (including model construction)
        job_dict = dict(data=data.model_dump(), params=params, solver=solver,
                        type=type, label=label)
        data = orjson.dumps(job_dict, option=orjson.OPT_SERIALIZE_NUMPY)
        headers = {'Content-Type': 'application/json'}
        response = self.session.post(path, data=data, headers=headers)
        rtype = get_type_hints(self.submit_problem)['return']
        return TypeAdapter(rtype).validate_python(orjson.loads(response.content))

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    @compress_if(lambda obj, *_, **__: obj.client.config.get('compress_qpu_problem_data'))
    def submit_problems(self, problems: list[models.ProblemJob]) -> \
            list[Union[models.ProblemInitialStatus, models.ProblemSubmitError]]:
        """Asynchronous multi-problem submit, returning initial statuses."""
        path = ''
        # note: TypeAdapter().dump_json() is 2ms vs 50us by orjson.dumps(model.model_dump())
        data = orjson.dumps([job.model_dump() for job in problems])
        headers = {'Content-Type': 'application/json'}
        response = self.session.post(path, data=data, headers=headers)
        rtype = get_type_hints(self.submit_problems)['return']
        return TypeAdapter(rtype).validate_python(orjson.loads(response.content))

    @accepts(media_type='application/vnd.dwave.sapi.problem+json', version='>=2.1,<3')
    def cancel_problem(self, problem_id: str) -> \
            Union[models.ProblemStatus, models.ProblemCancelError]:
        """Initiate problem cancel by problem id."""
        path = '{}'.format(problem_id)
        response = self.session.delete(path)
        rtype = get_type_hints(self.cancel_problem)['return']
        return TypeAdapter(rtype).validate_python(orjson.loads(response.content))

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def cancel_problems(self, problem_ids: list[str]) -> \
            list[Union[models.ProblemStatus, models.ProblemCancelError]]:
        """Initiate problem cancel for a list of problems."""
        path = ''
        response = self.session.delete(path, json=problem_ids)
        rtype = get_type_hints(self.cancel_problems)['return']
        return TypeAdapter(rtype).validate_python(orjson.loads(response.content))


class LeapAccount(ResourceBase):

    resource_path = 'account/'
    client_class = LeapAPIClient

    # TODO: constrain accepted resource/response version, when
    # media type defined for Leap API responses
    def get_active_project(self) -> models.LeapProject:
        """Retrieve user's active Leap project, as selected in Leap dashboard."""
        path = 'active_project/oauth/'
        response = self.session.get(path)
        parsed = models._LeapActiveProjectResponse.model_validate(
            orjson.loads(response.content))
        return parsed.data.project

    def list_projects(self) -> list[models.LeapProject]:
        """Retrieve list of Leap projects accessible to the authenticated user."""
        path = 'projects/oauth/'
        response = self.session.get(path)
        parsed = models._LeapProjectsResponse.model_validate(
            orjson.loads(response.content))
        return [p.project for p in parsed.data.projects]

    def get_project_token(self, *,
                          project: Optional[models.LeapProject] = None,
                          project_id: Optional[str] = None) -> Optional[str]:
        """Retrieve SAPI token for a given Leap project."""
        if project is not None:
            project_id = project.id
        if project_id is None:
            raise ValueError("'project' or 'project_id' must be given.")
        path = 'token/oauth/'
        query = {'project_id': project_id}
        response = self.session.get(path, params=query)
        parsed = models._LeapProjectTokenResponse.model_validate(
            orjson.loads(response.content))
        return parsed.data.token

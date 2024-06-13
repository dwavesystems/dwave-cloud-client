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

import io
import json
from typing import List, Union, Optional, Callable, get_type_hints
from functools import wraps

from pydantic import TypeAdapter

from dwave.cloud.api.client import (
    DWaveAPIClient, SolverAPIClient, MetadataAPIClient, LeapAPIClient)
from dwave.cloud.api import constants, models
from dwave.cloud.config.models import ClientConfig
from dwave.cloud.utils.coders import NumpyEncoder

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

    def __call__(self, fn: Callable):
        @wraps(fn)
        def wrapper(obj: 'ResourceBase', *args, **kwargs):
            try:
                obj.session.set_accept(**self.ctx)
                return fn(obj, *args, **kwargs)

            finally:
                obj.session.unset_accept()

        return wrapper


class ResourceBase:
    """A class for interacting with a Leap resource."""

    # api client used by the resource class
    client_class: DWaveAPIClient = DWaveAPIClient

    # endpoint path prefix (base path) specific to all methods on the resource
    resource_path: Optional[str] = None

    def __init__(self, client: Optional[DWaveAPIClient] = None, **config):
        if isinstance(client, DWaveAPIClient):
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
    def list_regions(self) -> List[models.Region]:
        path = ''
        response = self.session.get(path)
        regions = response.json()
        return TypeAdapter(List[models.Region]).validate_python(regions)

    @accepts(media_type='application/vnd.dwave.metadata.region+json', version='~=1.0')
    def get_region(self, code: str) -> models.Region:
        path = '{}'.format(code)
        response = self.session.get(path)
        region = response.json()
        return TypeAdapter(models.Region).validate_python(region)


class Solvers(ResourceBase):

    resource_path = 'solvers/'
    client_class = SolverAPIClient

    @accepts(media_type='application/vnd.dwave.sapi.solver-definition-list+json', version='~=2.0')
    def list_solvers(self) -> List[models.SolverConfiguration]:
        path = 'remote/'
        response = self.session.get(path)
        solvers = response.json()
        return TypeAdapter(List[models.SolverConfiguration]).validate_python(solvers)

    @accepts(media_type='application/vnd.dwave.sapi.solver-definition+json', version='~=2.0')
    def get_solver(self, solver_id: str) -> models.SolverConfiguration:
        path = 'remote/{}'.format(solver_id)
        response = self.session.get(path)
        solver = response.json()
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
                      **params) -> List[models.ProblemStatus]:
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
        statuses = response.json()
        return TypeAdapter(List[models.ProblemStatus]).validate_python(statuses)

    @accepts(media_type='application/vnd.dwave.sapi.problem+json', version='>=2.1,<3')
    def get_problem(self, problem_id: str) -> models.ProblemStatusMaybeWithAnswer:
        """Retrieve problem short status and answer if answer is available."""
        path = '{}'.format(problem_id)
        response = self.session.get(path)
        status = response.json()
        return models.ProblemStatusMaybeWithAnswer.model_validate(status)

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def get_problem_status(self, problem_id: str) -> models.ProblemStatus:
        """Retrieve short status of a single problem."""
        path = ''
        params = dict(id=problem_id)
        response = self.session.get(path, params=params)
        status = response.json()[0]
        return models.ProblemStatus.model_validate(status)

    # XXX: @pydantic.validate_arguments
    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def get_problem_statuses(self, problem_ids: List[str]) -> List[models.ProblemStatus]:
        """Retrieve short problem statuses for a list of problems."""
        if not isinstance(problem_ids, list):
            raise TypeError('a list of problem ids expected')
        if len(problem_ids) > 1000:
            raise ValueError('number of problem ids is limited to 1000')

        path = ''
        params = dict(id=','.join(problem_ids))
        response = self.session.get(path, params=params)
        statuses = response.json()
        return TypeAdapter(List[models.ProblemStatus]).validate_python(statuses)

    @accepts(media_type='application/vnd.dwave.sapi.problem-data+json', version='>=2.1,<3')
    def get_problem_info(self, problem_id: str) -> models.ProblemInfo:
        """Retrieve complete problem info."""
        path = '{}/info'.format(problem_id)
        response = self.session.get(path)
        info = response.json()
        return models.ProblemInfo.model_validate(info)

    @accepts(media_type='application/vnd.dwave.sapi.problem-answer+json', version='>=2.1,<3')
    def get_problem_answer(self, problem_id: str) -> models.ProblemAnswer:
        """Retrieve problem answer."""
        path = '{}/answer'.format(problem_id)
        response = self.session.get(path)
        answer = response.json()['answer']
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
    def get_problem_messages(self, problem_id: str) -> List[dict]:
        """Retrieve list of problem messages."""
        path = '{}/messages'.format(problem_id)
        response = self.session.get(path)
        return response.json()

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def submit_problem(self, *,
                       data: models.ProblemData,
                       params: dict,
                       solver: str,
                       type: constants.ProblemType,
                       label: str = None) -> \
            Union[models.ProblemStatusMaybeWithAnswer, models.ProblemSubmitError]:
        """Blocking problem submit with timeout, returning final status and
        answer, if problem is solved within the (undisclosed) time limit.
        """
        path = ''
        body = dict(data=data.model_dump(), params=params, solver=solver,
                    type=type, label=label)
        data = json.dumps(body, cls=NumpyEncoder)
        response = self.session.post(
            path, data=data, headers={'Content-Type': 'application/json'})
        rtype = get_type_hints(self.submit_problem)['return']
        return TypeAdapter(rtype).validate_python(response.json())

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def submit_problems(self, problems: List[models.ProblemJob]) -> \
            List[Union[models.ProblemInitialStatus, models.ProblemSubmitError]]:
        """Asynchronous multi-problem submit, returning initial statuses."""
        path = ''
        # encode iteratively so that timestamps are serialized (via pydantic json encoder)
        body = '[%s]' % ','.join(p.model_dump_json() for p in problems)
        response = self.session.post(
            path, data=body, headers={'Content-Type': 'application/json'})
        rtype = get_type_hints(self.submit_problems)['return']
        return TypeAdapter(rtype).validate_python(response.json())

    @accepts(media_type='application/vnd.dwave.sapi.problem+json', version='>=2.1,<3')
    def cancel_problem(self, problem_id: str) -> \
            Union[models.ProblemStatus, models.ProblemCancelError]:
        """Initiate problem cancel by problem id."""
        path = '{}'.format(problem_id)
        response = self.session.delete(path)
        rtype = get_type_hints(self.cancel_problem)['return']
        return TypeAdapter(rtype).validate_python(response.json())

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def cancel_problems(self, problem_ids: List[str]) -> \
            List[Union[models.ProblemStatus, models.ProblemCancelError]]:
        """Initiate problem cancel for a list of problems."""
        path = ''
        response = self.session.delete(path, json=problem_ids)
        rtype = get_type_hints(self.cancel_problems)['return']
        return TypeAdapter(rtype).validate_python(response.json())


class LeapAccount(ResourceBase):

    resource_path = 'account/'
    client_class = LeapAPIClient

    # TODO: constrain accepted resource/response version, when
    # media type defined for Leap API responses
    def get_active_project(self) -> models.LeapProject:
        """Retrieve user's active Leap project, as selected in Leap dashboard."""
        path = 'active_project/oauth/'
        response = self.session.get(path)
        parsed = models._LeapActiveProjectResponse.model_validate(response.json())
        return parsed.data.project

    def list_projects(self) -> List[models.LeapProject]:
        """Retrieve list of Leap projects accessible to the authenticated user."""
        path = 'projects/oauth/'
        response = self.session.get(path)
        parsed = models._LeapProjectsResponse.model_validate(response.json())
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
        parsed = models._LeapProjectTokenResponse.model_validate(response.json())
        return parsed.data.token

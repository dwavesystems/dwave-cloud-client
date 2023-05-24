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

import json
from typing import List, Union, Optional, Callable, get_type_hints
from functools import wraps

from pydantic import parse_obj_as

from dwave.cloud.api.client import DWaveAPIClient, SolverAPIClient, MetadataAPIClient
from dwave.cloud.api import constants, models
from dwave.cloud.utils import NumpyEncoder

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
    """A class for interacting with a SAPI resource."""

    # api client used by the resource class
    client_class: DWaveAPIClient = DWaveAPIClient

    # endpoint path prefix (base path) specific to all methods on the resource
    resource_path: str = None

    def __init__(self, **config):
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
    def from_client_config(cls, client: Union[DWaveAPIClient, 'dwave.cloud.client.base.Client']):
        """Create Resource instance configured from a
        :class:`~dwave.cloud.client.base.Client' instance.
        """
        # TODO: also accept configuration dict/dataclass when config/client refactored
        if isinstance(client, DWaveAPIClient):
            return cls(**client.config)
        else: # assume isinstance(client, dwave.cloud.Client), without importing
            sapiclient = SolverAPIClient.from_client_config(client)
            return cls(**sapiclient.config)


class Regions(ResourceBase):

    resource_path = 'regions/'
    client_class = MetadataAPIClient

    @accepts(media_type='application/vnd.dwave.metadata.regions+json', version='~=1.0')
    def list_regions(self) -> List[models.Region]:
        path = ''
        response = self.session.get(path)
        regions = response.json()
        return parse_obj_as(List[models.Region], regions)

    @accepts(media_type='application/vnd.dwave.metadata.region+json', version='~=1.0')
    def get_region(self, code: str) -> models.Region:
        path = '{}'.format(code)
        response = self.session.get(path)
        region = response.json()
        return parse_obj_as(models.Region, region)


class Solvers(ResourceBase):

    resource_path = 'solvers/'
    client_class = SolverAPIClient

    @accepts(media_type='application/vnd.dwave.sapi.solver-definition-list+json', version='~=2.0')
    def list_solvers(self) -> List[models.SolverConfiguration]:
        path = 'remote/'
        response = self.session.get(path)
        solvers = response.json()
        return parse_obj_as(List[models.SolverConfiguration], solvers)

    @accepts(media_type='application/vnd.dwave.sapi.solver-definition+json', version='~=2.0')
    def get_solver(self, solver_id: str) -> models.SolverConfiguration:
        path = 'remote/{}'.format(solver_id)
        response = self.session.get(path)
        solver = response.json()
        return models.SolverConfiguration.parse_obj(solver)


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
        return parse_obj_as(List[models.ProblemStatus], statuses)

    @accepts(media_type='application/vnd.dwave.sapi.problem+json', version='>=2.1,<3')
    def get_problem(self, problem_id: str) -> models.ProblemStatusMaybeWithAnswer:
        """Retrieve problem short status and answer if answer is available."""
        path = '{}'.format(problem_id)
        response = self.session.get(path)
        status = response.json()
        return models.ProblemStatusMaybeWithAnswer.parse_obj(status)

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def get_problem_status(self, problem_id: str) -> models.ProblemStatus:
        """Retrieve short status of a single problem."""
        path = ''
        params = dict(id=problem_id)
        response = self.session.get(path, params=params)
        status = response.json()[0]
        return models.ProblemStatus.parse_obj(status)

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
        return parse_obj_as(List[models.ProblemStatus], statuses)

    @accepts(media_type='application/vnd.dwave.sapi.problem-data+json', version='>=2.1,<3')
    def get_problem_info(self, problem_id: str) -> models.ProblemInfo:
        """Retrieve complete problem info."""
        path = '{}/info'.format(problem_id)
        response = self.session.get(path)
        info = response.json()
        return models.ProblemInfo.parse_obj(info)

    @accepts(media_type='application/vnd.dwave.sapi.problem-answer+json', version='>=2.1,<3')
    def get_problem_answer(self, problem_id: str) -> models.ProblemAnswer:
        """Retrieve problem answer."""
        path = '{}/answer'.format(problem_id)
        response = self.session.get(path)
        answer = response.json()['answer']
        return models.ProblemAnswer.parse_obj(answer)

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
        body = dict(data=data.dict(), params=params, solver=solver,
                    type=type, label=label)
        data = json.dumps(body, cls=NumpyEncoder)
        response = self.session.post(
            path, data=data, headers={'Content-Type': 'application/json'})
        rtype = get_type_hints(self.submit_problem)['return']
        return parse_obj_as(rtype, response.json())

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def submit_problems(self, problems: List[models.ProblemJob]) -> \
            List[Union[models.ProblemInitialStatus, models.ProblemSubmitError]]:
        """Asynchronous multi-problem submit, returning initial statuses."""
        path = ''
        # encode iteratively so that timestamps are serialized (via pydantic json encoder)
        body = '[%s]' % ','.join(p.json() for p in problems)
        response = self.session.post(
            path, data=body, headers={'Content-Type': 'application/json'})
        rtype = get_type_hints(self.submit_problems)['return']
        return parse_obj_as(rtype, response.json())

    @accepts(media_type='application/vnd.dwave.sapi.problem+json', version='>=2.1,<3')
    def cancel_problem(self, problem_id: str) -> \
            Union[models.ProblemStatus, models.ProblemCancelError]:
        """Initiate problem cancel by problem id."""
        path = '{}'.format(problem_id)
        response = self.session.delete(path)
        rtype = get_type_hints(self.cancel_problem)['return']
        return parse_obj_as(rtype, response.json())

    @accepts(media_type='application/vnd.dwave.sapi.problems+json', version='>=2.1,<3')
    def cancel_problems(self, problem_ids: List[str]) -> \
            List[Union[models.ProblemStatus, models.ProblemCancelError]]:
        """Initiate problem cancel for a list of problems."""
        path = ''
        response = self.session.delete(path, json=problem_ids)
        rtype = get_type_hints(self.cancel_problems)['return']
        return parse_obj_as(rtype, response.json())

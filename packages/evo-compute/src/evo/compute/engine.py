#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Discovery-driven generic compute engine (async).

:class:`ComputeClient` is an instance-bound entry point to every compute task an
organization can run. It synthesises a topic/task namespace on the fly and, when
a task is run, reads that task's schema from the live discovery catalogue to shape
and submit the job::

    client = ComputeClient(context)
    result = await client.geostatistics.kriging.run(source=..., target=..., ...)

A call is checked, resolved, checked again and submitted: required parameters first
(before anything reaches the network), then :mod:`~evo.compute.resolution` turns the
caller's objects and attributes into the references the schema declares, then optional
deep validation runs on that resolved payload -- the form the platform actually receives.
What comes back is hydrated against the task's ``results`` schema by
:mod:`~evo.compute.outputs`, so the objects a task wrote can be loaded straight back.

Topic, task and parameter names are the platform's names with hyphens replaced by
underscores, so ``normal-score`` is reached as ``client.geostatistics.normal_score``.
The catalogue is fetched the first time a task is ``run(...)``. It is then held by the
underlying :class:`~evo.compute.discovery.DiscoveryClient`, so repeated runs are served
from there until its cache expires.
"""

from __future__ import annotations

import inspect
import keyword
from typing import Any, Literal, Optional, Union
from uuid import UUID

from evo.common import APIConnector, IContext

from .client import JobClient
from .discovery import DEFAULT_CACHE_TTL_SECONDS, DiscoveryClient
from .endpoints.models import TaskResource
from .exceptions import ParameterValidationError
from .outputs import TaskResult
from .resolution import ReferenceResolver
from .validation import validate_parameters

__all__ = [
    "ComputeClient",
]


_JSON_SCHEMA_TO_PYTHON: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _normalise(name: str) -> str:
    """Map a platform name to a Python identifier (``normal-score`` -> ``normal_score``)."""
    return name.replace("-", "_")


def _python_annotation(prop: dict[str, Any]) -> Any:
    """Best-effort Python annotation for a JSON-Schema property."""
    if "enum" in prop:
        return Literal[tuple(prop["enum"])]  # type: ignore[misc]
    json_type = prop.get("type", "")
    if isinstance(json_type, list):
        members = [_JSON_SCHEMA_TO_PYTHON[t] for t in json_type if t != "null" and t in _JSON_SCHEMA_TO_PYTHON]
        base = members[0] if len(members) == 1 else (Union[tuple(members)] if members else Any)
        return Optional[base] if "null" in json_type else base
    return _JSON_SCHEMA_TO_PYTHON.get(json_type, Any)


def _wire_names(spec: TaskResource) -> dict[str, str]:
    """Map each parameter's Python name to the name the task's schema uses.

    Empty when a parameter cannot have a keyword argument of its own: its normalised name
    is not an identifier, is a reserved word, or collides with another parameter's.
    """
    names: dict[str, str] = {}
    for wire_name in (spec.parameters or {}).get("properties", {}):
        python_name = _normalise(wire_name)
        if not python_name.isidentifier() or keyword.iskeyword(python_name) or python_name in names:
            return {}
        names[python_name] = wire_name
    return names


def _signature_from_schema(spec: TaskResource) -> inspect.Signature:
    """Synthesise a keyword-only ``run(...)`` signature from a task's parameter schema.

    Required parameters come first, then optional ones (defaulting to their schema
    default or ``None``), then the engine's own ``preview`` flag. Parameters that cannot
    all be named as keyword arguments are accepted as ``**parameters`` instead.
    """
    schema = spec.parameters or {}
    properties: dict[str, Any] = schema.get("properties", {})
    required = list(schema.get("required", []))
    names = _wire_names(spec)
    python_names = {wire_name: python_name for python_name, wire_name in names.items()}

    parameters: list[inspect.Parameter] = []
    if names or not properties:
        for name in required:
            parameters.append(
                inspect.Parameter(
                    python_names.get(name, name),
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=_python_annotation(properties.get(name, {})),
                )
            )
        for name, prop in properties.items():
            if name in required:
                continue
            parameters.append(
                inspect.Parameter(
                    python_names[name],
                    inspect.Parameter.KEYWORD_ONLY,
                    default=prop.get("default", None),
                    annotation=_python_annotation(prop),
                )
            )
    # Preview tasks are an explicit opt-in, whether or not the task is behind a feature flag.
    parameters.append(inspect.Parameter("preview", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool))
    if properties and not names:
        parameters.append(inspect.Parameter("parameters", inspect.Parameter.VAR_KEYWORD, annotation=Any))
    return inspect.Signature(parameters, return_annotation=TaskResult)


def _check_fallback_parameters(spec: TaskResource, parameters: dict[str, Any], label: str) -> None:
    """Enforce required and declared parameter names when ``bind`` uses ``**parameters``."""
    schema = spec.parameters or {}
    properties: dict[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in parameters:
            detail = f"missing a required keyword-only argument: {name!r}"
            raise ParameterValidationError(f"{label}.run(): {detail}", task=label, errors=[detail])
    for name in parameters:
        if name not in properties:
            detail = f"got an unexpected keyword argument {name!r}"
            raise ParameterValidationError(f"{label}.run(): {detail}", task=label, errors=[detail])


class ComputeClient:
    """Instance-bound async entry point to the compute task catalogue.

    Reach a task as ``client.<topic>.<task>`` and call ``await client.<topic>.<task>.run(...)``
    to submit it and wait for its results. ``arun`` does the same for names held as strings.

    :param context: An authenticated Evo context.
    :param cache_ttl_seconds: How long a discovered task catalogue is cached.
    :param validate: Check the parameters against the task's JSON Schema before submitting
        (required-field presence, and, under ``deep_validation``, the whole payload).
        Defaults to ``True``. ``False`` turns off deep validation too, whatever
        ``deep_validation`` says. It does *not* make the call unchecked: the parameters are
        still bound to the signature synthesised from the schema, so an unknown or missing
        required argument is still rejected. That is how the payload is built, not a
        validation pass, and there is no useful call to make without it.
    :param deep_validation: Additionally run full JSON Schema Draft 2020-12 validation.
        Defaults to ``False``. Only consulted when ``validate`` is ``True``.
    :param check_schemas: Check that each referenced geoscience object is of a schema the
        task declares support for. Defaults to following ``validate``. Independent of it
        because the cost and the question differ: schema validation is local and free,
        while this loads each referenced object's metadata. Pass ``False`` to keep
        validation but skip the requests, or ``True`` alongside ``validate=False`` to keep
        the guard that catches an object the task cannot read.
    """

    def __init__(
        self,
        context: IContext,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        validate: bool = True,
        deep_validation: bool = False,
        check_schemas: bool | None = None,
    ) -> None:
        self._context = context
        self._org_id: UUID = context.get_org_id()
        self._connector: APIConnector = context.get_connector()
        self._discovery = DiscoveryClient(self._connector, self._org_id, cache_ttl_seconds=cache_ttl_seconds)
        self._resolver = ReferenceResolver(context)
        self._validate = validate
        self._deep_validation = deep_validation
        self._check_schemas = check_schemas

    # -- dynamic namespace ------------------------------------------------- #

    def __getattr__(self, name: str) -> _TopicProxy:
        # Only fires for names not found normally. Private/dunder probes must raise.
        if name.startswith("_"):
            raise AttributeError(name)
        return _TopicProxy(self, name)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__()) | {_normalise(resource.topic) for resource in self._discovery.peek_tasks()}
        )

    def __repr__(self) -> str:
        return f"ComputeClient(org_id={str(self._org_id)!r})"

    # -- non-blocking reads of the discovery cache -------------------------- #

    def _peek_spec(self, topic: str, task: str) -> TaskResource | None:
        """Return an already-discovered spec, or ``None`` while the catalogue is unfetched or stale."""
        for resource in self._discovery.peek_tasks():
            if _normalise(resource.topic) == _normalise(topic) and _normalise(resource.name) == _normalise(task):
                return resource
        return None

    # -- execution --------------------------------------------------------- #

    async def arun(
        self,
        topic: str,
        task: str,
        parameters: dict[str, Any],
        *,
        validate: bool | None = None,
        deep_validation: bool | None = None,
        check_schemas: bool | None = None,
    ) -> TaskResult:
        """Discover the task (cached), resolve and validate the parameters, submit, and hydrate the results.

        :param topic: The task topic.
        :param task: The task name.
        :param parameters: The task parameters, plus ``preview=True`` to run a preview task.
        :param validate: Override the client's schema-validation setting for this call.
            ``False`` skips deep validation too, but never the signature binding that
            builds the payload.
        :param deep_validation: Override the client's deep-validation setting for this call.
            Only consulted when validation is enabled.
        :param check_schemas: Override the client's supported-schema setting for this call.
            Falls back to ``validate`` when neither is set.

        :return: The hydrated results of the finished job.

        :raises AttributeError: If the topic does not advertise the task.
        :raises ParameterValidationError: If the parameters or referenced objects fail validation.
        """
        spec = await self._resolve_spec(topic, task)
        label = f"{topic}.{_normalise(task)}"
        signature = _signature_from_schema(spec)
        try:
            bound = signature.bind(**parameters)
        except TypeError as error:
            raise ParameterValidationError(f"{label}.run(): {error}", task=label, errors=[str(error)]) from None

        # Forward only what the caller actually passed, so unset optionals fall back to the
        # platform's own defaults while an explicit ``None`` still reaches the wire.
        wire_parameters = dict(bound.arguments)
        preview = bool(wire_parameters.pop("preview", signature.parameters["preview"].default))
        names = _wire_names(spec)
        if names:
            wire_parameters = {names.get(name, name): value for name, value in wire_parameters.items()}
        else:
            wire_parameters.update(wire_parameters.pop("parameters", {}))
            if (spec.parameters or {}).get("properties"):
                _check_fallback_parameters(spec, wire_parameters, label)

        if validate is None:
            validate = self._validate
        if deep_validation is None:
            deep_validation = self._deep_validation
        if check_schemas is None:
            check_schemas = self._check_schemas if self._check_schemas is not None else validate
        # ``validate`` masters the schema checks; ``check_schemas`` stands on its own.
        if validate:
            # Required fields first: a missing parameter is worth reporting before resolution
            # spends a request loading the objects the other parameters name.
            validate_parameters(spec, wire_parameters, task_label=label)
        wire_parameters = await self._resolver.resolve(
            spec, wire_parameters, check_schemas=check_schemas, task_label=label
        )
        if validate and deep_validation:
            validate_parameters(spec, wire_parameters, deep=True, task_label=label)

        job: JobClient[dict] = await JobClient.submit(
            connector=self._connector,
            org_id=self._org_id,
            topic=spec.topic,
            task=spec.name,
            parameters=wire_parameters,
            result_type=dict,
            preview=preview,
        )
        return TaskResult(await job.wait_for_results(), spec.results, self._context)

    async def _resolve_spec(self, topic: str, task: str) -> TaskResource:
        """Return the discovery spec for ``topic``/``task``, whether spelled with hyphens or underscores.

        The catalogue lives in :class:`~evo.compute.discovery.DiscoveryClient`, which fetches
        it once and serves it from there until its TTL expires.
        """
        topic_tasks = [
            resource
            for resource in await self._discovery.list_tasks()
            if _normalise(resource.topic) == _normalise(topic)
        ]
        for resource in topic_tasks:
            if _normalise(resource.name) == _normalise(task):
                return resource

        available = ", ".join(sorted(_normalise(resource.name) for resource in topic_tasks)) or "(none)"
        raise AttributeError(f"no task {task!r} in topic {topic!r}. Available: {available}")


class _TopicProxy:
    """A single topic within the catalogue; resolves attribute access to task proxies."""

    def __init__(self, client: ComputeClient, topic: str) -> None:
        self._client = client
        self._topic = topic

    def __getattr__(self, name: str) -> _TaskProxy:
        if name.startswith("_"):
            raise AttributeError(name)
        return _TaskProxy(self._client, self._topic, name)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__())
            | {
                _normalise(resource.name)
                for resource in self._client._discovery.peek_tasks()
                if _normalise(resource.topic) == _normalise(self._topic)
            }
        )

    def __repr__(self) -> str:
        return f"<compute topic {self._topic!r}>"


class _TaskProxy:
    """A single task; exposes a schema-shaped async ``run(...)``."""

    def __init__(self, client: ComputeClient, topic: str, task: str) -> None:
        self._client = client
        self._topic = topic
        self._task = task
        self.run = _make_run(client, topic, task)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | {"run"})

    def __repr__(self) -> str:
        return f"<compute task {self._topic!r}.{_normalise(self._task)!r}>"


def _make_run(client: ComputeClient, topic: str, task: str):
    """Build the async ``run`` callable for a task proxy.

    If the task's schema has already been discovered, the callable advertises a synthesised
    signature for editor tab-completion. Otherwise it accepts generic keyword arguments and the
    schema is fetched on first call, after which the callable re-describes itself. Either way
    discovery is never triggered by attribute access alone.
    """

    async def run(**parameters: Any) -> TaskResult:
        try:
            return await client.arun(topic, task, parameters)
        finally:
            # The first call populates the catalogue, so this callable can now describe itself.
            # Safe to repeat: it recomputes the same shape, or picks up a newer spec after a refresh.
            _describe_from_schema(run, client, topic, task)

    run.__name__ = "run"
    run.__qualname__ = f"{_normalise(task)}.run"
    _describe_from_schema(run, client, topic, task)
    return run


def _describe_from_schema(run: Any, client: ComputeClient, topic: str, task: str) -> None:
    """Shape ``run``'s signature and docstring from the task schema, if it has been discovered."""
    if (spec := client._peek_spec(topic, task)) is not None:
        run.__signature__ = _signature_from_schema(spec)
        run.__doc__ = spec.description or run.__doc__

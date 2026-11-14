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

``run(...)`` submits and waits for hydrated results. ``submit(...)`` returns a
:class:`~evo.compute.jobs.TaskJob` without waiting, exposing status and cancellation.
Both paths share the same discovery, parameter preparation and result hydration.

A task can opt into a hand-written surface through :mod:`evo.compute.overrides`.
Named calls through ``arun`` and ``asubmit`` use the same override as attribute-based
calls, including its parameter checks and native result conversion.
"""

from __future__ import annotations

import inspect
import keyword
from typing import Any, Literal, Optional, Union
from uuid import UUID

from evo.common import APIConnector, IContext
from evo.common.interfaces import IFeedback
from evo.common.utils import NoFeedback

from .client import JobClient
from .discovery import DEFAULT_CACHE_TTL_SECONDS, DiscoveryClient
from .endpoints.models import TaskResource
from .exceptions import ParameterValidationError
from .jobs import TaskJob
from .outputs import TaskResult
from .overrides import load_override
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

# Shared by the runtime ``run`` docstring and the generated stub.
_RUN_FEEDBACK_NOTE = (
    "Progress feedback is not a ``run`` argument, so it never clashes with a task parameter.\n"
    "Set it with ``ComputeClient(context, fb=...)``, or call ``submit`` and pass it to ``job.results(fb=...)``."
)


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
    """Map unambiguous Python parameter names back to the catalogue's wire names."""
    override = load_override(_normalise(spec.topic), _normalise(spec.name))
    aliases = getattr(override, "PARAMETER_ALIASES", {}) if override is not None else {}
    names: dict[str, str] = {}
    for wire_name in (spec.parameters or {}).get("properties", {}):
        python_name = aliases.get(wire_name, _normalise(wire_name))
        if not python_name.isidentifier() or keyword.iskeyword(python_name) or python_name in names:
            return {}
        names[python_name] = wire_name
    return names


def _signature_from_schema(spec: TaskResource, returns: type = TaskResult) -> inspect.Signature:
    """Synthesise a keyword-only ``run(...)`` signature from a task's parameter schema.

    Required parameters come first, then optional ones (defaulting to their schema
    default or ``None``), then the engine's own ``preview`` flag. Parameters that cannot
    all be named as keyword arguments are accepted as ``**parameters`` instead.

    :param returns: ``TaskResult`` for execution or ``TaskJob`` for submission.
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
    return inspect.Signature(parameters, return_annotation=returns)


def _check_fallback_required(
    spec: TaskResource, parameters: dict[str, Any], label: str, operation: str = "run"
) -> None:
    """Enforce required parameter names when ``bind`` uses ``**parameters``."""
    for name in (spec.parameters or {}).get("required", []):
        if name not in parameters:
            detail = f"missing a required keyword-only argument: {name!r}"
            raise ParameterValidationError(f"{label}.{operation}(): {detail}", task=label, errors=[detail])


class ComputeClient:
    """Instance-bound async entry point to the compute task catalogue.

    Reach a task as ``client.<topic>.<task>`` and call ``await client.<topic>.<task>.run(...)``
    to submit it and wait for its results. ``arun`` does the same for names held as strings.

    :param context: An authenticated Evo context.
    :param cache_ttl_seconds: How long a discovered task catalogue is cached.
    :param validate: Check the parameters against the task's JSON Schema before submitting
        (required-field presence, and, under ``deep_validation``, known value constraints).
        Defaults to ``True``. ``False`` turns off deep validation too, whatever
        ``deep_validation`` says. It does *not* make the call unchecked: the parameters are
        still bound to the task's required arguments, and explicit extra-field rules
        still apply. Custom runners also retain their own parameter checks.
    :param deep_validation: Additionally check the full JSON Schema Draft 2020-12 constraints.
        Defaults to ``False``. Only consulted when ``validate`` is ``True``.
    :param check_schemas: Check that each referenced geoscience object is of a schema the
        task declares support for. Defaults to following ``validate``. Independent of it
        because the cost and the question differ: schema validation is local and free,
        while this loads each referenced object's metadata. Pass ``False`` to keep
        validation but skip the requests, or ``True`` alongside ``validate=False`` to keep
        the guard that catches an object the task cannot read.
    :param fb: Feedback used while waiting for any job submitted by this client.
        Set it here so engine options do not occupy task parameter names in the namespace.
        ``TaskJob.results`` can override it for one wait.
    """

    def __init__(
        self,
        context: IContext,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        validate: bool = True,
        deep_validation: bool = False,
        check_schemas: bool | None = None,
        fb: IFeedback = NoFeedback,
    ) -> None:
        self._context = context
        self._org_id: UUID = context.get_org_id()
        self._connector: APIConnector = context.get_connector()
        self._discovery = DiscoveryClient.from_context(context, cache_ttl_seconds=cache_ttl_seconds)
        self._resolver = ReferenceResolver(context)
        self._validate = validate
        self._deep_validation = deep_validation
        self._check_schemas = check_schemas
        self._fb = fb

    @property
    def context(self) -> IContext:
        """The context used to submit tasks and load their results."""
        return self._context

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
    ) -> Any:
        """Run a task by name, using its custom runner when one is registered.

        Parameters use the same names as ``client.<topic>.<task>.run(...)``.
        Custom tasks return their native result; other tasks return ``TaskResult``.
        Validation options override the client's settings for this call.
        Preview is off unless ``parameters`` explicitly includes ``preview=True``.
        Feedback comes from the client; for one call, use :meth:`asubmit` and pass ``fb`` to
        ``job.results``.
        """
        options = dict(validate=validate, deep_validation=deep_validation, check_schemas=check_schemas)
        if (override := load_override(_normalise(topic), _normalise(task))) is not None:
            return await override.bind(self, topic, task).run_parameters(parameters, **options)
        return await self._run_generic(topic, task, parameters, **options)

    async def _run_generic(
        self,
        topic: str,
        task: str,
        parameters: dict[str, Any],
        *,
        validate: bool | None = None,
        deep_validation: bool | None = None,
        check_schemas: bool | None = None,
    ) -> TaskResult:
        """Submit through the shared preparation path and wait for hydrated results.

        :param validate: Override the client's schema-validation setting for this call.
            ``False`` skips deep validation too, but never the signature binding that
            builds the payload.
        :param deep_validation: Override the client's deep-validation setting for this call.
            Only consulted when validation is enabled.
        :param check_schemas: Override the client's supported-schema setting for this call.
            Falls back to ``validate`` when neither is set.
        """
        job = await self._submit(
            topic,
            task,
            parameters,
            "run",
            validate=validate,
            deep_validation=deep_validation,
            check_schemas=check_schemas,
        )
        return await job.results()

    async def asubmit(
        self,
        topic: str,
        task: str,
        parameters: dict[str, Any],
        *,
        validate: bool | None = None,
        deep_validation: bool | None = None,
        check_schemas: bool | None = None,
    ) -> TaskJob[Any]:
        """Submit by name, using the same custom runner as attribute-based submission.

        Validation options have the same meaning as on ``arun``. The accepted job carries
        its result converter and client feedback for the caller's eventual wait.
        """
        options = dict(validate=validate, deep_validation=deep_validation, check_schemas=check_schemas)
        if (override := load_override(_normalise(topic), _normalise(task))) is not None:
            return await override.bind(self, topic, task).submit_parameters(parameters, **options)
        return await self._submit(
            topic,
            task,
            parameters,
            "submit",
            validate=validate,
            deep_validation=deep_validation,
            check_schemas=check_schemas,
        )

    async def _submit(
        self,
        topic: str,
        task: str,
        parameters: dict[str, Any],
        operation: str,
        *,
        validate: bool | None,
        deep_validation: bool | None,
        check_schemas: bool | None,
    ) -> TaskJob[TaskResult]:
        """Prepare and submit once; every error names the operation the caller used."""
        spec = await self._resolve_spec(topic, task)
        label = f"{topic}.{_normalise(task)}"
        signature = _signature_from_schema(spec)
        accepts_extra = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        extra_parameters = (
            {}
            if accepts_extra
            else {name: value for name, value in parameters.items() if name not in signature.parameters}
        )
        try:
            bound = signature.bind(
                **{name: value for name, value in parameters.items() if name not in extra_parameters}
            )
        except TypeError as error:
            raise ParameterValidationError(f"{label}.{operation}(): {error}", task=label, errors=[str(error)]) from None

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
                _check_fallback_required(spec, wire_parameters, label, operation)
        wire_parameters.update(extra_parameters)

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
            validate_parameters(spec, wire_parameters, task_label=label, operation=operation)
        wire_parameters = await self._resolver.resolve(
            spec, wire_parameters, check_schemas=check_schemas, task_label=label, operation=operation
        )
        if validate and deep_validation:
            validate_parameters(spec, wire_parameters, deep=True, task_label=label, operation=operation)

        job: JobClient[dict] = await JobClient.submit(
            connector=self._connector,
            org_id=self._org_id,
            topic=spec.topic,
            task=spec.name,
            parameters=wire_parameters,
            result_type=dict,
            preview=preview,
        )
        return self._hydrate_job(job, spec)

    def _hydrate_job(self, job: JobClient[dict], spec: TaskResource) -> TaskJob[TaskResult]:
        async def hydrate(payload: dict[str, Any]) -> TaskResult:
            return TaskResult(payload, spec.results, self._context)

        return TaskJob(job, hydrate, self._fb)

    async def _restore_job(self, url: str, *, preview: bool) -> TaskJob[Any]:
        backend = JobClient.from_url(self._connector, url, result_type=dict, preview=preview)
        if backend.org_id != self._org_id:
            raise ValueError("The job URL does not match the context organization.")
        spec = await self._resolve_spec(backend.topic, backend.task)
        job = self._hydrate_job(backend, spec)
        if (override := load_override(_normalise(spec.topic), _normalise(spec.name))) is not None:
            return job.map(override.bind(self, spec.topic, spec.name).convert_result)
        return job

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

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if (override := load_override(_normalise(self._topic), _normalise(name))) is not None:
            return override.bind(self._client, self._topic, name)
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
    """A single task; exposes schema-shaped async ``run(...)`` and ``submit(...)``."""

    def __init__(self, client: ComputeClient, topic: str, task: str) -> None:
        self._client = client
        self._topic = topic
        self._task = task
        self.run = _make_run(client, topic, task)
        self.submit = _make_submit(client, topic, task)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | {"run", "submit"})

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
            _describe_from_schema(run, client, topic, task, note=_RUN_FEEDBACK_NOTE)

    run.__name__ = "run"
    run.__qualname__ = f"{_normalise(task)}.run"
    run.__doc__ = _RUN_FEEDBACK_NOTE
    _describe_from_schema(run, client, topic, task, note=_RUN_FEEDBACK_NOTE)
    return run


def _make_submit(client: ComputeClient, topic: str, task: str):
    """Build the async submit callable without discovery on attribute access."""

    async def submit(**parameters: Any) -> TaskJob[TaskResult]:
        try:
            return await client.asubmit(topic, task, parameters)
        finally:
            _describe_from_schema(submit, client, topic, task, TaskJob)

    submit.__name__ = "submit"
    submit.__qualname__ = f"{_normalise(task)}.submit"
    _describe_from_schema(submit, client, topic, task, TaskJob)
    return submit


def _describe_from_schema(
    run: Any, client: ComputeClient, topic: str, task: str, returns: type = TaskResult, note: str = ""
) -> None:
    """Shape ``run``'s signature and docstring from the task schema, if it has been discovered."""
    if (spec := client._peek_spec(topic, task)) is not None:
        run.__signature__ = _signature_from_schema(spec, returns)
        run.__doc__ = "\n\n".join(part for part in ((spec.description or "").strip(), note) if part) or run.__doc__

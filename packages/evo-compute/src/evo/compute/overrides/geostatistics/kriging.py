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

"""Hand-written runner for ``geostatistics/kriging`` -- the worked example of an override.

Kriging is the task with the most behind it: a parameter model that has been through
review, a result type with helpers a synthesised one could not have, and two mistakes the
published schema has no way to describe. So it is the one that earns a hand-written
surface, and ``client.geostatistics.kriging`` is served from here instead of from the
generic proxy.

What the override adds, and nothing more:

* **Checks the JSON Schema cannot express.** The schema bounds each parameter on its own --
  ``min_samples`` and ``max_samples`` are both integers of at least 1 -- but it cannot say
  that one must not exceed the other, and it cannot say that a task must not estimate an
  attribute from itself. Both are silent failures otherwise: the first finds no
  neighbourhood anywhere, the second overwrites its own input.
* **The typed parameter model.** :class:`~evo.compute.tasks.geostatistics.kriging.KrigingParameters`
  accepts typed objects and attributes, folds ``source_filter`` and ``target_filter`` into
  the source and target the wire expects, and names the arguments as the SDK names them
  rather than as the wire spells them.
* **The typed result.** :class:`~evo.compute.tasks.geostatistics.kriging.KrigingResult` --
  ``target_name``, ``attribute_name``, ``schema``, ``get_target_object()``,
  ``to_dataframe()`` -- in place of the generic :class:`~evo.compute.outputs.TaskResult`.

Attribute-based calls, :meth:`~evo.compute.engine.ComputeClient.arun` and
:meth:`~evo.compute.engine.ComputeClient.asubmit` all use this runner.
The shared engine still discovers, validates, resolves and submits the job.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any

from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference
from evo.objects.typed import Attribute, BlockModelAttribute
from evo.objects.typed.base import BaseObject
from pydantic import BaseModel

from ...exceptions import ParameterValidationError
from ...jobs import TaskJob
from ...outputs import TaskResult
from ...tasks.common import Filter, SearchNeighborhood, Source, Target
from ...tasks.common.source_target import AnyTypedAttribute, UpdateAttribute
from ...tasks.geostatistics.kriging import (
    BlockDiscretization,
    KrigingDiagnostics,
    KrigingParameters,
    KrigingResult,
    KrigingResultModel,
    OrdinaryKriging,
    SimpleKriging,
)

if TYPE_CHECKING:
    from ...engine import ComputeClient

__all__ = [
    "KrigingOverride",
    "bind",
]

# The published catalogue uses the British spelling; the SDK keeps the current public name.
PARAMETER_ALIASES = {"block_discretisation": "block_discretization"}

# What each reference parameter takes. The model fields are annotated with their *validated*
# type, which a checker reads as the narrow one; a signature has to say what it accepts.
_SourceInput = Source | Attribute | BlockModelAttribute
_TargetInput = Target | AnyTypedAttribute
_ObjectInput = str | ObjectReference | BaseObject | DownloadedObject | ObjectMetadata

# Arguments the model folds into another field, and where the wire carries each one.
_FOLDED_ARGUMENTS = {
    "source_filter": ("source", "filter"),
    "target_filter": ("target", "filter"),
    "diagnostics": ("target", "diagnostics"),
}


def _declared_fields(model: Any) -> dict[str, str]:
    """The keys ``model`` reads, by field name or alias, mapped to the attribute each one fills."""
    if isinstance(model, BaseModel):
        keys: dict[str, str] = {}
        for name, field in type(model).model_fields.items():
            keys[name] = name
            for alias in (field.alias, field.validation_alias):
                if isinstance(alias, str):
                    keys[alias] = name
        return keys
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        return {field.name: field.name for field in dataclasses.fields(model)}
    return {}


def _preserve_extra_fields(payload: Any, supplied: Any, model: Any = None) -> None:
    """Put back the fields the typed models dropped, so the engine can apply the schema's rule.

    Pydantic ignores keys a model does not declare, so an unknown field would otherwise vanish
    before the engine could reject, validate, forward or warn about it. A key the model does
    read is not an extra, even when the payload carries it under another name or leaves it out.
    """
    if isinstance(supplied, list) and isinstance(payload, list) and len(supplied) == len(payload):
        models = model if isinstance(model, list) and len(model) == len(supplied) else [None] * len(supplied)
        for payload_item, supplied_item, model_item in zip(payload, supplied, models, strict=True):
            _preserve_extra_fields(payload_item, supplied_item, model_item)
        return
    if not isinstance(supplied, dict) or not isinstance(payload, dict):
        return
    declared = _declared_fields(model)
    for name, value in supplied.items():
        if name in payload:
            field = declared.get(name)
            _preserve_extra_fields(payload[name], value, getattr(model, field) if field else None)
        elif name not in declared:
            payload[name] = value


def _same_object(left: str, right: str) -> bool:
    """Whether two references name the same object, whatever version or spelling each uses.

    Comparing the URLs as text would miss it. :class:`~evo.objects.ObjectReference` is a ``str``
    subclass that keeps whatever the caller wrote, and its path pattern is case-insensitive, so
    an uppercase UUID on one side and a lowercase one on the other are one object written two
    ways. A trailing fragment does the same. The parsed components are what identify an object;
    ``version_id`` is not one of them, since a version names a moment in the object's history.

    A path reference and a UUID reference cannot be told apart without loading the object, so
    they never match here -- this guard would rather miss a case than invent one.
    """
    first, second = ObjectReference(left), ObjectReference(right)
    return (first.environment, first.object_id, first.object_path) == (
        second.environment,
        second.object_id,
        second.object_path,
    )


class KrigingOverride:
    """The typed ``run(...)`` and ``submit(...)`` surface for Kriging."""

    params_type = KrigingParameters

    def __init__(self, client: ComputeClient, topic: str, task: str) -> None:
        self._client = client
        self._topic = topic
        self._task = task

    async def run(
        self,
        *,
        source: _SourceInput,
        target: _TargetInput,
        variogram: _ObjectInput,
        neighborhood: SearchNeighborhood,
        kriging_method: SimpleKriging | OrdinaryKriging | None = None,
        source_filter: Filter | None = None,
        target_filter: Filter | None = None,
        block_discretization: BlockDiscretization | None = None,
        diagnostics: KrigingDiagnostics | None = None,
        preview: bool = False,
    ) -> KrigingResult:
        """Estimate an attribute on a target object by kriging from a source attribute.

        Progress feedback is not a ``run`` argument, so it never clashes with a task parameter.
        Set it with ``ComputeClient(context, fb=...)``, or call ``submit`` and pass it to ``job.results(fb=...)``.

        :param source: The object and attribute holding the known values, or the typed
            attribute itself.
        :param target: Where to write the estimate -- a :class:`~evo.compute.tasks.common.Target`,
            or the typed attribute to create or update.
        :param variogram: The variogram object, or a reference to it.
        :param neighborhood: The search neighbourhood: the ellipsoid to look in, and how many
            samples to use.
        :param kriging_method: Ordinary kriging by default; :meth:`KrigingMethod.simple` when the
            mean is known.
        :param source_filter: Restrict the estimate to a subset of the source data.
        :param target_filter: Restrict the estimate to a subset of the target object.
        :param block_discretization: Sub-block discretization for block kriging. Omitted
            means point kriging.
        :param diagnostics: Optional diagnostic attributes to write beside the estimate.
        :param preview: Opt in to the preview API. Defaults to ``False``.

        :raises ParameterValidationError: If the parameters are inconsistent in a way the
            task schema cannot describe, or fail the engine's own checks.
        """
        return await self.run_parameters(
            dict(
                source=source,
                target=target,
                variogram=variogram,
                neighborhood=neighborhood,
                kriging_method=kriging_method if kriging_method is not None else OrdinaryKriging(),
                source_filter=source_filter,
                target_filter=target_filter,
                block_discretization=block_discretization,
                diagnostics=diagnostics,
                preview=preview,
            )
        )

    async def submit(
        self,
        *,
        source: _SourceInput,
        target: _TargetInput,
        variogram: _ObjectInput,
        neighborhood: SearchNeighborhood,
        kriging_method: SimpleKriging | OrdinaryKriging | None = None,
        source_filter: Filter | None = None,
        target_filter: Filter | None = None,
        block_discretization: BlockDiscretization | None = None,
        diagnostics: KrigingDiagnostics | None = None,
        preview: bool = False,
    ) -> TaskJob[KrigingResult]:
        """Submit with the same arguments and checks as ``run``, without waiting.

        ``await job.results()`` returns ``KrigingResult``, including its object and
        dataframe helpers. Preview requires an explicit ``preview=True``.
        """
        return await self.submit_parameters(
            dict(
                source=source,
                target=target,
                variogram=variogram,
                neighborhood=neighborhood,
                kriging_method=kriging_method if kriging_method is not None else OrdinaryKriging(),
                source_filter=source_filter,
                target_filter=target_filter,
                block_discretization=block_discretization,
                diagnostics=diagnostics,
                preview=preview,
            )
        )

    def _prepare_parameters(self, supplied: dict[str, Any], operation: str) -> dict[str, Any]:
        parameters = dict(supplied)
        preview = parameters.pop("preview", False)
        label = f"{self._topic}.{self._task.replace('-', '_')}"
        for wire_name, argument in PARAMETER_ALIASES.items():
            if wire_name in parameters:
                if argument in parameters:
                    message = f"{argument} was supplied both as {argument} and {wire_name}"
                    raise ParameterValidationError(f"{label}.{operation}(): {message}", task=label, errors=[message])
                parameters[argument] = parameters.pop(wire_name)
        for argument, (name, field) in _FOLDED_ARGUMENTS.items():
            if isinstance(frame := parameters.get(name), dict) and field in frame:
                if parameters.get(argument) is not None:
                    message = f"{argument} was supplied both directly and inside {name}"
                    raise ParameterValidationError(f"{label}.{operation}(): {message}", task=label, errors=[message])
                frame = dict(frame)
                parameters[argument] = frame.pop(field)
                parameters[name] = frame
        signature = inspect.signature(self.run)
        extra = {name: value for name, value in parameters.items() if name not in signature.parameters}
        parameters = {name: value for name, value in parameters.items() if name not in extra}
        try:
            signature.bind(**parameters)
        except TypeError as error:
            raise ParameterValidationError(f"{label}.{operation}(): {error}", task=label, errors=[str(error)]) from None
        validated = parameters
        if isinstance(diagnostics := parameters.get("diagnostics"), dict):
            # The diagnostics model refuses unknown keys outright; the schema decides what happens to them.
            known = {name: value for name, value in diagnostics.items() if name in KrigingDiagnostics.model_fields}
            validated = {**parameters, "diagnostics": known}
        model = KrigingParameters.model_validate(validated)
        self._check_parameters(model, operation)
        payload = model.model_dump(mode="json", by_alias=True, exclude_none=True)
        _preserve_extra_fields(payload, parameters, model)
        for argument, (name, field) in _FOLDED_ARGUMENTS.items():
            _preserve_extra_fields(payload[name].get(field), parameters.get(argument), getattr(model, argument))
        payload.update(extra)
        payload["preview"] = preview
        return payload

    async def run_parameters(
        self,
        parameters: dict[str, Any],
        *,
        validate: bool | None = None,
        deep_validation: bool | None = None,
        check_schemas: bool | None = None,
    ) -> KrigingResult:
        """Run named or typed inputs through the same preparation and checks."""
        job = await self._submit_parameters(
            parameters,
            "run",
            validate=validate,
            deep_validation=deep_validation,
            check_schemas=check_schemas,
        )
        return await job.results()

    async def submit_parameters(
        self,
        parameters: dict[str, Any],
        *,
        validate: bool | None = None,
        deep_validation: bool | None = None,
        check_schemas: bool | None = None,
    ) -> TaskJob[KrigingResult]:
        """Submit named or typed inputs through the same preparation and checks."""
        return await self._submit_parameters(
            parameters,
            "submit",
            validate=validate,
            deep_validation=deep_validation,
            check_schemas=check_schemas,
        )

    async def _submit_parameters(
        self,
        parameters: dict[str, Any],
        operation: str,
        *,
        validate: bool | None,
        deep_validation: bool | None,
        check_schemas: bool | None,
    ) -> TaskJob[KrigingResult]:
        job = await self._client._submit(
            self._topic,
            self._task,
            self._prepare_parameters(parameters, operation),
            operation,
            validate=validate,
            deep_validation=deep_validation,
            check_schemas=check_schemas,
        )
        return job.map(self.convert_result)

    async def convert_result(self, result: TaskResult) -> KrigingResult:
        """Restore the native result for submitted and reopened jobs alike."""
        return KrigingResult(self._client.context, KrigingResultModel.model_validate(dict(result)))

    def _check_parameters(self, parameters: KrigingParameters, operation: str) -> None:
        """Reject the two inconsistencies the task's JSON Schema has no way to state."""
        label = f"{self._topic}.{self._task.replace('-', '_')}"
        errors: list[str] = []

        search = parameters.search
        if search.min_samples is not None and search.min_samples > search.max_samples:
            errors.append(
                f"neighborhood: min_samples ({search.min_samples}) exceeds max_samples "
                f"({search.max_samples}), so no location can ever be estimated"
            )

        attribute = parameters.target.attribute
        if (
            isinstance(attribute, UpdateAttribute)
            and _same_object(parameters.source.object, parameters.target.object)
            and attribute.reference == parameters.source.attribute
        ):
            errors.append(
                f"target: {attribute.reference!r} is also the source attribute, "
                "so the estimate would overwrite its own input"
            )

        if errors:
            raise ParameterValidationError(
                f"{label}.{operation}(): " + "; ".join(errors),
                task=label,
                errors=errors,
            )

    def __repr__(self) -> str:
        return f"<compute task {self._topic!r}.{self._task.replace('-', '_')!r} (override)>"


def bind(client: ComputeClient, topic: str, task: str) -> KrigingOverride:
    """Hand this task to the override. Called by :mod:`evo.compute.overrides`."""
    return KrigingOverride(client, topic, task)

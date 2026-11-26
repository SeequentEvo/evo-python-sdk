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
* **The typed parameter model.** ``KrigingParameters``
  accepts typed objects and attributes, folds ``source_filter`` and ``target_filter`` into
  the source and target the wire expects, and names the arguments as the SDK names them
  rather than as the wire spells them.
* **The typed result.** ``KrigingResult`` --
  ``target_name``, ``attribute_name``, ``schema``, ``get_target_object()``,
  ``to_dataframe()`` -- in place of the generic ``TaskResult``.

Attribute-based calls and ``ComputeClient.arun`` both use this
runner. The shared engine still discovers, validates, resolves and submits the job.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any
from uuid import UUID

from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference
from evo.objects.typed import Attribute, BlockModelAttribute
from evo.objects.typed.base import BaseObject
from pydantic import AliasChoices, BaseModel, ValidationError

from ...exceptions import ParameterValidationError
from ...resolution import _as_uuid
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
]

# The published catalogue uses the British spelling for this parameter.
PARAMETER_ALIASES = {"block_discretisation": "block_discretization"}

# What each reference parameter takes. The model fields are annotated with their *validated*
# type, which a checker reads as the narrow one; a signature has to say what it accepts.
_SourceInput = Source | Attribute | BlockModelAttribute
_TargetInput = Target | AnyTypedAttribute
_ObjectInput = str | UUID | ObjectReference | BaseObject | DownloadedObject | ObjectMetadata

# Arguments the model folds into another field, and where the wire carries each one.
_FOLDED_ARGUMENTS = {
    "source_filter": ("source", "filter"),
    "target_filter": ("target", "filter"),
    "diagnostics": ("target", "diagnostics"),
}


def _model_accepted_keys(model_cls: type[BaseModel]) -> set[str]:
    """Every input key ``model_cls`` accepts, including ``AliasChoices`` validation aliases."""
    keys: set[str] = set()
    for name, field in model_cls.model_fields.items():
        keys.add(name)
        for alias in (field.alias, field.validation_alias):
            if isinstance(alias, str):
                keys.add(alias)
            elif isinstance(alias, AliasChoices):
                keys.update(choice for choice in alias.choices if isinstance(choice, str))
    return keys


def _declared_fields(model: Any) -> dict[str, str]:
    """The keys ``model`` reads, by field name or alias, mapped to the attribute each one fills."""
    if isinstance(model, BaseModel):
        keys: dict[str, str] = {}
        for name, field in type(model).model_fields.items():
            keys[name] = name
            for alias in (field.alias, field.validation_alias):
                if isinstance(alias, str):
                    keys[alias] = name
                elif isinstance(alias, AliasChoices):
                    for choice in alias.choices:
                        if isinstance(choice, str):
                            keys[choice] = name
        return keys
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        return {field.name: field.name for field in dataclasses.fields(model)}
    return {}


def _clean_filter_expr(expr: Any) -> Any:
    if not isinstance(expr, dict):
        return expr
    kind = expr.get("type")
    if kind in ("all_of", "any_of"):
        cleaned = {k: v for k, v in expr.items() if k in {"type", "filters"}}
        if isinstance(filters := cleaned.get("filters"), list):
            cleaned["filters"] = [_clean_filter_expr(item) for item in filters]
        return cleaned
    if kind == "condition":
        return {k: v for k, v in expr.items() if k in {"type", "attribute", "operator", "values", "threshold"}}
    return expr


def _clean_known_fields(parameters: dict[str, Any]) -> dict[str, Any]:
    """Strip unknown nested dict keys so strict ``extra='forbid'`` models let schema rules judge extras."""
    cleaned = dict(parameters)
    if isinstance(source := cleaned.get("source"), dict):
        cleaned["source"] = {k: v for k, v in source.items() if k in _model_accepted_keys(Source)}
    if isinstance(target := cleaned.get("target"), dict):
        target_clean = {k: v for k, v in target.items() if k in _model_accepted_keys(Target)}
        if isinstance(attr := target_clean.get("attribute"), dict) and attr.get("operation") in ("create", "update"):
            allowed = {"operation", "name"} if attr["operation"] == "create" else {"operation", "reference"}
            target_clean["attribute"] = {k: v for k, v in attr.items() if k in allowed}
        cleaned["target"] = target_clean
    if isinstance(neighborhood := cleaned.get("neighborhood"), dict):
        nb_clean = {k: v for k, v in neighborhood.items() if k in _model_accepted_keys(SearchNeighborhood)}
        if isinstance(ellipsoid := nb_clean.get("ellipsoid"), dict):
            el_clean = {k: v for k, v in ellipsoid.items() if k in {"ellipsoid_ranges", "ranges", "rotation"}}
            for r_key in ("ellipsoid_ranges", "ranges"):
                if isinstance(r_val := el_clean.get(r_key), dict):
                    el_clean[r_key] = {k: v for k, v in r_val.items() if k in {"major", "semi_major", "minor"}}
            if isinstance(rot := el_clean.get("rotation"), dict):
                el_clean["rotation"] = {k: v for k, v in rot.items() if k in {"dip_azimuth", "dip", "pitch"}}
            nb_clean["ellipsoid"] = el_clean
        cleaned["neighborhood"] = nb_clean
    if isinstance(method := cleaned.get("kriging_method"), dict) and method.get("type") in ("ordinary", "simple"):
        allowed = {"type"} if method["type"] == "ordinary" else {"type", "mean"}
        cleaned["kriging_method"] = {k: v for k, v in method.items() if k in allowed}
    for filt_key in ("source_filter", "target_filter"):
        if isinstance(filt := cleaned.get(filt_key), dict):
            f_clean = {k: v for k, v in filt.items() if k in _model_accepted_keys(Filter)}
            if "where" in f_clean:
                f_clean["where"] = _clean_filter_expr(f_clean["where"])
            cleaned[filt_key] = f_clean
    if isinstance(disc := cleaned.get("block_discretization"), dict):
        cleaned["block_discretization"] = {
            k: v for k, v in disc.items() if k in _model_accepted_keys(BlockDiscretization)
        }
    if isinstance(diagnostics := cleaned.get("diagnostics"), dict):
        diag_clean: dict[str, Any] = {}
        for name, value in diagnostics.items():
            if name not in KrigingDiagnostics.model_fields:
                continue
            if isinstance(value, dict) and value.get("operation") in ("create", "update"):
                allowed = {"operation", "name"} if value["operation"] == "create" else {"operation", "reference"}
                diag_clean[name] = {k: v for k, v in value.items() if k in allowed}
            else:
                diag_clean[name] = value
        cleaned["diagnostics"] = diag_clean
    return cleaned


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

    Comparing the URLs as text would miss it. ``ObjectReference`` is a ``str``
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
    """The override's stand-in for the generic task proxy: same ``run(...)``, typed by hand."""

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

        :param source: The object and attribute holding the known values, or the typed
            attribute itself.
        :param target: Where to write the estimate -- a ``Target``,
            or the typed attribute to create or update.
        :param variogram: The variogram object, or a reference to it.
        :param neighborhood: The search neighbourhood: the ellipsoid to look in, and how many
            samples to use.
        :param kriging_method: Ordinary kriging by default; ``KrigingMethod.simple`` when the
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

    def _expand_uuid_handles(
        self, parameters: dict[str, Any], label: str, operation: str = "run"
    ) -> tuple[dict[str, Any], str | None]:
        """Expand bare UUID object handles using the client's environment, and note bare target attribute names."""
        updated = dict(parameters)
        raw_target_attr: str | None = None

        def expand_obj(val: Any, location: str) -> Any:
            if (object_id := _as_uuid(val)) is not None:
                try:
                    return str(ObjectReference.new(self._client.context.get_environment(), object_id=object_id))
                except (TypeError, ValueError) as error:
                    detail = f"{location}: {error}"
                    raise ParameterValidationError(
                        f"{label}.{operation}(): {detail}", task=label, errors=[detail]
                    ) from None
            return val

        if "variogram" in updated:
            updated["variogram"] = expand_obj(updated["variogram"], "variogram")
        if isinstance(source := updated.get("source"), dict) and "object" in source:
            updated["source"] = {**source, "object": expand_obj(source["object"], "source.object")}
        if isinstance(target := updated.get("target"), dict):
            target_copy = dict(target)
            if "object" in target_copy:
                target_copy["object"] = expand_obj(target_copy["object"], "target.object")
            if isinstance(target_copy.get("attribute"), str):
                raw_target_attr = target_copy["attribute"]
                target_copy["attribute"] = {"operation": "create", "name": raw_target_attr}
            updated["target"] = target_copy
        return updated, raw_target_attr

    def _prepare_parameters(self, supplied: dict[str, Any]) -> dict[str, Any]:
        parameters = dict(supplied)
        preview = parameters.pop("preview", False)
        label = f"{self._topic}.{self._task.replace('-', '_')}"
        for wire_name, argument in PARAMETER_ALIASES.items():
            if wire_name in parameters:
                if argument in parameters:
                    message = f"{argument} was supplied both as {argument} and {wire_name}"
                    raise ParameterValidationError(f"{label}.run(): {message}", task=label, errors=[message])
                parameters[argument] = parameters.pop(wire_name)
        for argument, (name, field) in _FOLDED_ARGUMENTS.items():
            if isinstance(frame := parameters.get(name), dict) and field in frame:
                if parameters.get(argument) is not None:
                    message = f"{argument} was supplied both directly and inside {name}"
                    raise ParameterValidationError(f"{label}.run(): {message}", task=label, errors=[message])
                frame = dict(frame)
                parameters[argument] = frame.pop(field)
                parameters[name] = frame
        signature = inspect.signature(self.run)
        extra = {name: value for name, value in parameters.items() if name not in signature.parameters}
        parameters = {name: value for name, value in parameters.items() if name not in extra}
        try:
            signature.bind(**parameters)
        except TypeError as error:
            raise ParameterValidationError(f"{label}.run(): {error}", task=label, errors=[str(error)]) from None
        expanded, raw_target_attr = self._expand_uuid_handles(parameters, label)
        validated = _clean_known_fields(expanded)
        try:
            model = KrigingParameters.model_validate(validated)
        except ValidationError as error:
            details = [
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" if item["loc"] else item["msg"]
                for item in error.errors()
            ]
            raise ParameterValidationError(
                f"{label}.run(): " + "; ".join(details),
                task=label,
                errors=details,
            ) from None
        self._check_parameters(model, raw_target_attr=raw_target_attr)
        payload = model.model_dump(mode="json", by_alias=True, exclude_none=True)
        if raw_target_attr is not None:
            payload["target"]["attribute"] = raw_target_attr
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
        result = await self._client._run_generic(
            self._topic,
            self._task,
            self._prepare_parameters(parameters),
            validate=validate,
            deep_validation=deep_validation,
            check_schemas=check_schemas,
        )
        return KrigingResult(self._client.context, KrigingResultModel.model_validate(dict(result)))

    def _check_parameters(self, parameters: KrigingParameters, *, raw_target_attr: str | None = None) -> None:
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
        same_obj = _same_object(parameters.source.object, parameters.target.object)
        if same_obj and (
            (isinstance(attribute, UpdateAttribute) and attribute.reference == parameters.source.attribute)
            or (raw_target_attr is not None and raw_target_attr == parameters.source.attribute)
        ):
            ref_label = raw_target_attr if raw_target_attr is not None else attribute.reference  # type: ignore[union-attr]
            errors.append(
                f"target: {ref_label!r} is also the source attribute, so the estimate would overwrite its own input"
            )

        if errors:
            raise ParameterValidationError(
                f"{label}.run(): " + "; ".join(errors),
                task=label,
                errors=errors,
            )

    def __repr__(self) -> str:
        return f"<compute task {self._topic!r}.{self._task.replace('-', '_')!r} (override)>"


def bind(client: ComputeClient, topic: str, task: str) -> KrigingOverride:
    """Hand this task to the override. Called by ``evo.compute.overrides``."""
    return KrigingOverride(client, topic, task)

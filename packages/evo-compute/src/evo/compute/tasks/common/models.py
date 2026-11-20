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

"""Shared base models, role validators, and output-collision helpers for compute tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeAlias
from urllib.parse import unquote
from uuid import UUID

from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference, ObjectSchema
from evo.objects.typed.attributes import (
    Attribute,
    BlockModelAttribute,
    BlockModelPendingAttribute,
    PendingAttribute,
)
from evo.objects.typed.base import BaseObject
from evo.objects.typed.downhole_intervals import DownholeIntervals
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from evo import jmespath

if TYPE_CHECKING:
    from .filters import Filter
    from .source_target import CreateAttribute, UpdateAttribute

__all__ = [
    "FiniteFloat",
    "OutputTarget",
    "QuantileFloat",
    "SourceTargetTaskParameters",
    "TaskInputModel",
    "TaskParameters",
    "canonicalize_attribute_selector",
    "check_batch_output_collisions",
    "deduplicate_aliases",
    "extract_typed_object",
    "object_identity",
    "object_matches_reference",
    "resolve_output_target",
    "validate_distribution_object",
    "validate_object_role",
]

FiniteFloat: TypeAlias = Annotated[float, Field(allow_inf_nan=False)]
"""A finite floating-point number (rejects ``NaN`` and ``±inf``)."""

QuantileFloat: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
"""A finite probability/quantile value in ``[0.0, 1.0]``."""

_PRIMARY_ATTRIBUTE_PATHS: frozenset[str] = frozenset({"attributes", "cell_attributes", "locations.attributes"})


def _normalize_path_str(raw: str) -> str | None:
    segments = [seg for seg in unquote(raw).split("/") if seg and seg != "."]
    return "/".join(segments) if segments else None


def _extract_metadata_path(meta: Any) -> str | None:
    raw_path = getattr(meta, "path", None)
    if isinstance(raw_path, str):
        return _normalize_path_str(raw_path)
    parent = getattr(meta, "parent", None)
    name = getattr(meta, "name", None)
    if isinstance(parent, str) and isinstance(name, str) and name:
        return _normalize_path_str(f"{parent}/{name}")
    return None


def _reference_identities(ref: ObjectReference, meta: Any = None) -> tuple[UUID | None, str | None]:
    obj_uuid: UUID | None = ref.object_id
    if obj_uuid is None and meta is not None:
        meta_id = getattr(meta, "id", None)
        if isinstance(meta_id, UUID):
            obj_uuid = meta_id

    obj_path: str | None = _normalize_path_str(ref.object_path) if ref.object_path is not None else None
    if obj_path is None and meta is not None:
        obj_path = _extract_metadata_path(meta)

    return obj_uuid, obj_path


def object_identity(reference: str) -> str:
    """Normalize an object reference URL by stripping any ``?version=...`` query string and trailing slash."""
    stripped = reference.split("?", 1)[0].rstrip("/")
    try:
        ref = ObjectReference(stripped)
    except ValueError:
        return stripped
    if ref.object_id is not None:
        return str(ObjectReference.new(ref.environment, object_id=ref.object_id))
    if ref.object_path is not None and (norm_path := _normalize_path_str(ref.object_path)):
        return str(ObjectReference.new(ref.environment, object_path=norm_path))
    return stripped


def object_matches_reference(
    candidate: Any,
    reference_url: str,
    *,
    supplied_obj: Any = None,
) -> bool:
    """Check whether *candidate* (an object or metadata) refers to the same object as *reference_url*.

    Handles versioned URLs as well as equivalent UUID-based and path-based :class:`ObjectReference` URLs
    when object metadata carries both ``id`` and ``path``.
    """
    if candidate is None:
        return True
    if supplied_obj is not None and (
        candidate is supplied_obj
        or candidate is getattr(supplied_obj, "_obj", None)
        or getattr(candidate, "_obj", None) is supplied_obj
    ):
        return True

    meta_a = candidate if isinstance(candidate, ObjectMetadata) else getattr(candidate, "metadata", None)
    if meta_a is None:
        return True
    raw_url_a = getattr(meta_a, "url", None)
    if not isinstance(raw_url_a, str):
        return True

    if object_identity(raw_url_a) == object_identity(reference_url):
        return True

    try:
        ref_a = ObjectReference(raw_url_a)
        ref_b = ObjectReference(reference_url)
    except ValueError:
        return False

    if (
        ref_a.hub_url.rstrip("/").lower() != ref_b.hub_url.rstrip("/").lower()
        or ref_a.org_id != ref_b.org_id
        or ref_a.workspace_id != ref_b.workspace_id
    ):
        return False

    meta_b = (
        supplied_obj
        if isinstance(supplied_obj, ObjectMetadata)
        else getattr(supplied_obj, "metadata", None)
        if supplied_obj is not None
        else None
    )
    uuid_a, path_a = _reference_identities(ref_a, meta_a)
    uuid_b, path_b = _reference_identities(ref_b, meta_b)

    if uuid_a is not None and uuid_b is not None:
        return uuid_a == uuid_b
    if path_a is not None and path_b is not None:
        return path_a == path_b
    return False


def _extract_field_chain(node: Mapping[str, Any]) -> list[str] | None:
    """Extract a dotted field-name chain from a JMESPath ``field`` or ``subexpression`` AST node."""
    node_type = node.get("type")
    if node_type == "field":
        return [str(node.get("value", ""))]
    if node_type == "subexpression":
        children = list(node.get("children", ()))
        if children and children[0].get("type") == "current":
            children = children[1:]
        if not children:
            return None
        fields: list[str] = []
        for child in children:
            sub = _extract_field_chain(child)
            if sub is None:
                return None
            fields.extend(sub)
        return fields
    return None


def _extract_index_selector(node: Mapping[str, Any]) -> tuple[list[str], int] | None:
    """Extract ``(base_field_chain, index)`` from a JMESPath index selector AST."""
    node_type = node.get("type")
    children = list(node.get("children", ()))
    if node_type == "index_expression" and len(children) == 2 and children[1].get("type") == "index":
        base_fields = _extract_field_chain(children[0])
        if base_fields:
            return base_fields, int(children[1]["value"])
        return None

    if node_type == "subexpression":
        if children and children[0].get("type") == "current":
            children = children[1:]
        if not children:
            return None
        if len(children) == 1:
            return _extract_index_selector(children[0])
        tail = _extract_index_selector(children[-1])
        if tail is not None:
            prefix_fields: list[str] = []
            for child in children[:-1]:
                sub = _extract_field_chain(child)
                if sub is None:
                    return None
                prefix_fields.extend(sub)
            return [*prefix_fields, *tail[0]], tail[1]
    return None


def _extract_filter_selector(node: Mapping[str, Any]) -> tuple[list[str], Mapping[str, Any]] | None:
    """Extract ``(base_field_chain, condition_ast)`` from a JMESPath filter projection AST."""
    node_type = node.get("type")
    children = list(node.get("children", ()))
    if node_type == "filter_projection" and len(children) == 3 and children[1].get("type") == "identity":
        base_fields = _extract_field_chain(children[0])
        if base_fields:
            return base_fields, children[2]
        return None

    if node_type == "subexpression":
        if children and children[0].get("type") == "current":
            children = children[1:]
        if len(children) == 1:
            return _extract_filter_selector(children[0])
        if len(children) >= 2:
            tail = _extract_filter_selector(children[-1])
            if tail is not None:
                prefix_fields: list[str] = []
                for child in children[:-1]:
                    sub = _extract_field_chain(child)
                    if sub is None:
                        return None
                    prefix_fields.extend(sub)
                return [*prefix_fields, *tail[0]], tail[1]
    return None


def _extract_eq_field_literal(node: Mapping[str, Any]) -> tuple[str, str] | None:
    """Extract ``('name' | 'key', literal_str)`` from a JMESPath equality comparator AST."""
    if node.get("type") != "comparator" or node.get("value") != "eq":
        return None
    children = node.get("children", ())
    if len(children) != 2:
        return None
    for field_node, val_node in ((children[0], children[1]), (children[1], children[0])):
        field_chain = _extract_field_chain(field_node)
        if (
            field_chain in (["name"], ["key"])
            and val_node.get("type") == "literal"
            and isinstance(val_node.get("value"), str)
        ):
            return field_chain[0], val_node["value"]
    return None


def _canonicalize_jmespath_ast(node: Mapping[str, Any]) -> str:
    """Produce a deterministic canonical string for a parsed JMESPath AST."""
    node_type = node.get("type", "")
    children = list(node.get("children", ()))
    value = node.get("value")

    if node_type == "subexpression":
        if children and children[0].get("type") == "current":
            children = children[1:]
        if len(children) == 1:
            return _canonicalize_jmespath_ast(children[0])

    if node_type == "comparator" and value == "eq" and len(children) == 2:
        parts = sorted(_canonicalize_jmespath_ast(c) for c in children)
        return f"eq({parts[0]},{parts[1]})"

    if node_type in ("and_expression", "or_expression"):
        parts = sorted(_canonicalize_jmespath_ast(c) for c in children)
        return f"{node_type}({','.join(parts)})"

    child_repr = ",".join(_canonicalize_jmespath_ast(c) for c in children)
    return f"{node_type}({value!r};[{child_repr}])"


def canonicalize_attribute_selector(expression: str) -> str:
    """Canonicalize a JMESPath attribute selector using parsed JMESPath AST semantics."""
    stripped = expression.strip()
    try:
        parsed = jmespath.compile(stripped).parsed
    except Exception:
        return f"raw:{''.join(stripped.split())}"

    if idx_sel := _extract_index_selector(parsed):
        base_fields, idx = idx_sel
        base_path = ".".join(base_fields)
        scope = "primary" if base_path in _PRIMARY_ATTRIBUTE_PATHS else base_path
        return f"{scope}:index:{idx}"

    if filt_sel := _extract_filter_selector(parsed):
        base_fields, cond_node = filt_sel
        base_path = ".".join(base_fields)
        scope = "primary" if base_path in _PRIMARY_ATTRIBUTE_PATHS else base_path
        if eq_pair := _extract_eq_field_literal(cond_node):
            field_name, literal_val = eq_pair
            return f"{scope}:{field_name}:{literal_val}"
        return f"ast:{scope}:filter:{_canonicalize_jmespath_ast(cond_node)}"

    return f"ast:{_canonicalize_jmespath_ast(parsed)}"


@dataclass(frozen=True)
class OutputTarget:
    """Describes an object (or specific attribute on an object) written by a compute task.

    When ``attribute_aliases`` is ``None``, the task writes whole-object or multi-attribute
    fixed outputs onto ``object_id`` and therefore conflicts with any other concurrent task
    targeting ``object_id``.
    """

    object_id: str
    """Normalized target object URL (without ``?version=...``)."""

    attribute_aliases: frozenset[str] | None = None
    """Canonical attribute identifiers for single-attribute outputs, or ``None`` for whole-object outputs."""

    display_attribute: str | None = None
    """Human-readable attribute name or selector for error reporting."""


def _existing_attribute_alias_sets(target_obj: BaseObject) -> list[frozenset[str]]:
    """Build full alias sets (name, key, positive/negative index) for each existing attribute on ``target_obj``."""
    try:
        attrs = list(getattr(target_obj, "attributes", ()))
    except Exception:
        return []
    total = len(attrs)
    alias_sets: list[frozenset[str]] = []
    for idx, attr in enumerate(attrs):
        aliases: set[str] = {
            f"primary:index:{idx}",
            f"primary:index:{idx - total}",
        }
        name = getattr(attr, "name", None)
        if isinstance(name, str) and name:
            aliases.add(f"primary:name:{name}")
        key = getattr(attr, "key", None)
        if isinstance(key, str) and key:
            aliases.add(f"primary:key:{key}")
        alias_sets.append(frozenset(aliases))
    return alias_sets


def resolve_output_target(
    object_url: str,
    spec: CreateAttribute | UpdateAttribute,
    *,
    target_obj: BaseObject | None = None,
    known_name: str | None = None,
) -> OutputTarget:
    """Build a canonical :class:`OutputTarget` for a ``CreateAttribute`` or ``UpdateAttribute`` spec."""
    from .source_target import CreateAttribute

    obj_id = object_identity(object_url)
    aliases: set[str] = set()

    if isinstance(spec, CreateAttribute):
        aliases.add(f"primary:name:{spec.name}")
        display = spec.name
    else:
        canonical = canonicalize_attribute_selector(spec.reference)
        aliases.add(canonical)
        if isinstance(known_name, str) and known_name:
            aliases.add(f"primary:name:{known_name}")
        display = known_name or spec.reference

        if target_obj is not None:
            try:
                matched = target_obj.search(spec.reference)
                if isinstance(matched, Sequence) and not isinstance(matched, (str, bytes)) and len(matched) == 1:
                    matched = matched[0]
                if isinstance(matched, Mapping):
                    matched_name = matched.get("name")
                    matched_key = matched.get("key")
                    if isinstance(matched_name, str) and matched_name:
                        aliases.add(f"primary:name:{matched_name}")
                    if isinstance(matched_key, str) and matched_key:
                        aliases.add(f"primary:key:{matched_key}")
            except Exception:
                pass

    if target_obj is not None:
        for existing_aliases in _existing_attribute_alias_sets(target_obj):
            if aliases & existing_aliases:
                aliases.update(existing_aliases)

    return OutputTarget(
        object_id=obj_id,
        attribute_aliases=frozenset(aliases),
        display_attribute=display,
    )


def _expand_target_with_object(target: OutputTarget, target_obj: BaseObject) -> OutputTarget:
    if target.attribute_aliases is None:
        return target
    aliases = set(target.attribute_aliases)
    for existing_aliases in _existing_attribute_alias_sets(target_obj):
        if aliases & existing_aliases:
            aliases.update(existing_aliases)
    return OutputTarget(
        object_id=target.object_id,
        attribute_aliases=frozenset(aliases),
        display_attribute=target.display_attribute,
    )


def check_batch_output_collisions(
    runner_classes: Sequence[type[Any]],
    parameters: Sequence[Any],
) -> None:
    """Verify that no two tasks in a ``run_tasks`` batch write to colliding outputs.

    Raises:
        ValueError: If two tasks write to the same target object/attribute, or if a
            whole-target task shares a target object with another task in the batch.
    """
    # Collect any typed BaseObject instances seen across the entire batch by normalized URL
    # so selectors using plain URLs can still be resolved against pre-loaded objects in the batch.
    known_objects: dict[str, BaseObject] = {}
    for params in parameters:
        typed_map = getattr(params, "_typed_objects", None)
        if isinstance(typed_map, Mapping):
            for obj_id, obj in typed_map.items():
                if isinstance(obj_id, str) and isinstance(obj, BaseObject):
                    known_objects.setdefault(obj_id, obj)

    whole_object_claims: dict[str, str] = {}
    attribute_claims_by_object: dict[str, list[tuple[frozenset[str], str, str | None]]] = {}

    for index, (runner_cls, params) in enumerate(zip(runner_classes, parameters, strict=True)):
        task_name = getattr(runner_cls, "task", type(params).__name__)
        task_label = f"task[{index}] ({task_name})"
        outputs: Sequence[OutputTarget] = runner_cls.output_targets(params)

        for output in outputs:
            if not isinstance(output.object_id, str) or not output.object_id:
                continue
            obj_id = object_identity(output.object_id)
            if obj_id in known_objects:
                matched_obj = known_objects[obj_id]
                output = _expand_target_with_object(output, matched_obj)
                meta_url = getattr(getattr(matched_obj, "metadata", None), "url", None)
                if isinstance(meta_url, str):
                    obj_id = object_identity(meta_url)

            if obj_id in whole_object_claims:
                other_label = whole_object_claims[obj_id]
                raise ValueError(
                    f"Output collision in batch: {other_label} and {task_label} both write to target object {obj_id!r}."
                )

            if output.attribute_aliases is None:
                if existing_attr_claims := attribute_claims_by_object.get(obj_id):
                    _, other_label, _ = existing_attr_claims[0]
                    raise ValueError(
                        f"Output collision in batch: {other_label} and {task_label} both write to target object "
                        f"{obj_id!r}."
                    )
                whole_object_claims[obj_id] = task_label
                continue

            existing_list = attribute_claims_by_object.setdefault(obj_id, [])
            for claimed_aliases, other_label, other_display in existing_list:
                if output.attribute_aliases & claimed_aliases:
                    attr_desc = output.display_attribute or other_display or next(iter(output.attribute_aliases))
                    raise ValueError(
                        f"Output collision in batch: {other_label} and {task_label} both write to attribute "
                        f"{attr_desc!r} on target object {obj_id!r}."
                    )
            existing_list.append((output.attribute_aliases, task_label, output.display_attribute))


def extract_typed_object(value: Any) -> BaseObject | None:
    """Recover a typed :class:`BaseObject` from a value, typed attribute, or wrapper model if present."""
    if isinstance(value, BaseObject):
        return value
    if isinstance(value, (BlockModelAttribute, BlockModelPendingAttribute)):
        obj = getattr(value, "_obj", None)
        return obj if isinstance(obj, BaseObject) else None
    if isinstance(value, Attribute):
        ctx = getattr(value, "_context", None)
        root = getattr(ctx, "root_model", None)
        if isinstance(root, BaseObject):
            return root
        obj = getattr(value, "_obj", None)
        return obj if isinstance(obj, BaseObject) else None
    if isinstance(value, PendingAttribute):
        parent = getattr(value, "_parent", None)
        ctx = getattr(parent, "_context", None)
        root = getattr(ctx, "root_model", None)
        if isinstance(root, BaseObject):
            return root
        obj = getattr(value, "_obj", None)
        return obj if isinstance(obj, BaseObject) else None
    saved = getattr(value, "_typed_object", None)
    if isinstance(saved, BaseObject):
        return saved
    return None


def _register_typed_object_urls(url_objects: dict[str, BaseObject], obj: BaseObject) -> None:
    meta = getattr(obj, "metadata", None)
    if meta is None:
        return
    raw_url = getattr(meta, "url", None)
    if isinstance(raw_url, str):
        url_objects[object_identity(raw_url)] = obj
    env = getattr(meta, "environment", None)
    meta_path = _extract_metadata_path(meta)
    if env is not None and meta_path is not None:
        try:
            path_ref = ObjectReference.new(env, object_path=meta_path)
            url_objects[object_identity(str(path_ref))] = obj
        except Exception:
            pass


def validate_object_role(
    field_name: str,
    value: Any,
    expected_types: tuple[type[BaseObject], ...],
) -> None:
    """Validate that *value*, if backed by a typed ``BaseObject``, is an instance of *expected_types*."""
    obj = extract_typed_object(value)
    if obj is None and isinstance(value, Mapping):
        obj = extract_typed_object(value.get("object"))
    if obj is not None and not isinstance(obj, expected_types):
        expected_names = " or ".join(cls.__name__ for cls in expected_types)
        raise ValueError(f"Expected {field_name!r} to be a {expected_names}, got {type(obj).__name__}.")


def validate_distribution_object(field_name: str, value: Any) -> None:
    """Validate that *value*, if a typed object or downloaded metadata, represents a distribution object."""
    obj = extract_typed_object(value)
    if obj is not None:
        sub = getattr(obj, "sub_classification", None)
        if not isinstance(sub, str) or "distribution" not in sub:
            raise ValueError(
                f"Expected {field_name!r} to reference a continuous distribution object, got {type(obj).__name__}."
            )
        return

    meta: ObjectMetadata | None = None
    if isinstance(value, DownloadedObject):
        meta = getattr(value, "metadata", None)
    elif isinstance(value, ObjectMetadata):
        meta = value
    if meta is not None:
        schema_id = getattr(meta, "schema_id", None)
        if isinstance(schema_id, ObjectSchema) and "distribution" not in schema_id.sub_classification:
            raise ValueError(
                f"Expected {field_name!r} to reference a continuous distribution object, "
                f"got schema {schema_id.sub_classification!r}."
            )


def deduplicate_aliases(
    data: Any,
    alias_pairs: Sequence[tuple[str, str]],
    validators: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """Deduplicate equal ``(canonical, legacy)`` alias pairs, raising ``ValueError`` when they conflict."""
    if not isinstance(data, Mapping) or not alias_pairs:
        return data

    updated: dict[str, Any] | None = None
    for canonical, legacy in alias_pairs:
        if canonical in data and legacy in data:
            c_val = data[canonical]
            l_val = data[legacy]
            equal = c_val == l_val
            if not equal and validators and canonical in validators:
                validator = validators[canonical]
                try:
                    equal = validator(c_val) == validator(l_val)
                except Exception:
                    equal = False
            if not equal:
                raise ValueError(f"Conflicting values for {canonical!r} ({c_val!r}) and {legacy!r} ({l_val!r}).")
            if updated is None:
                updated = dict(data)
            updated.pop(legacy, None)
    return updated if updated is not None else data


class TaskInputModel(BaseModel):
    """Base Pydantic model for all compute task input and parameter structures.

    Enforces ``extra="forbid"`` and ``allow_inf_nan=False`` across every field so
    typos, unknown parameters, ``NaN``, and ``±inf`` are caught locally before job
    submission.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        populate_by_name=True,
    )


class TaskParameters(TaskInputModel):
    """Base class for top-level compute task ``*Parameters`` models.

    Provides declarative hooks so existing and future compute tasks automatically inherit:
    - strict ``extra="forbid"`` and ``allow_inf_nan=False`` validation,
    - canonical/legacy alias deduplication (``_alias_pairs``),
    - typed ``BaseObject`` role validation (``_object_roles`` & ``_distribution_fields``),
    - typed attribute parent-object consistency checks (``_attribute_object_bindings``),
    - drillhole neighborhood source-object validation (``_drillhole_source_field``),
    - automatic ``Filter`` bare-attribute resolution against typed ``BaseObject`` inputs (``_filter_bindings``),
    - ``output_targets()`` extraction for ``run_tasks()`` batch output-collision checks.
    """

    _alias_pairs: ClassVar[tuple[tuple[str, str], ...]] = ()
    _alias_validators: ClassVar[Mapping[str, Callable[[Any], Any]]] = {}
    _object_roles: ClassVar[Mapping[str, tuple[type[BaseObject], ...]]] = {}
    _distribution_fields: ClassVar[tuple[str, ...]] = ()
    _attribute_object_bindings: ClassVar[tuple[tuple[str, str], ...]] = ()
    _filter_bindings: ClassVar[tuple[tuple[str, str], ...]] = ()
    _drillhole_source_field: ClassVar[str | None] = None

    _typed_objects: dict[str, BaseObject] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _deduplicate_task_aliases(cls, data: Any) -> Any:
        if cls._alias_pairs:
            return deduplicate_aliases(data, cls._alias_pairs, cls._alias_validators)
        return data

    @model_validator(mode="wrap")
    @classmethod
    def _validate_task_contracts(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        raw_map: Mapping[str, Any] = data if isinstance(data, Mapping) else {}

        # 1. Validate typed object roles before fields are reduced to URL strings.
        for field_name, expected_types in cls._object_roles.items():
            if field_name in raw_map and raw_map[field_name] is not None:
                validate_object_role(field_name, raw_map[field_name], expected_types)

        for field_name in cls._distribution_fields:
            if field_name in raw_map and raw_map[field_name] is not None:
                raw_val = raw_map[field_name]
                ref_val = getattr(raw_val, "reference", raw_val)
                if isinstance(raw_val, Mapping) and "reference" in raw_val:
                    ref_val = raw_val["reference"]
                validate_distribution_object(field_name, ref_val)

        # Capture typed BaseObjects keyed by field name and by normalized URL (UUID + path).
        field_objects: dict[str, BaseObject] = {}
        url_objects: dict[str, BaseObject] = {}
        if isinstance(data, TaskParameters):
            url_objects.update(data._typed_objects)
        for key, raw_val in raw_map.items():
            obj = extract_typed_object(raw_val)
            if obj is not None:
                field_objects[key] = obj
                _register_typed_object_urls(url_objects, obj)

        # Capture parent objects for typed attributes used in flat attribute fields.
        raw_attr_parents: dict[str, Any] = {}
        for attr_field, _ in cls._attribute_object_bindings:
            if attr_field in raw_map and isinstance(
                raw_map[attr_field],
                (Attribute, PendingAttribute, BlockModelAttribute, BlockModelPendingAttribute),
            ):
                attr_parent = getattr(raw_map[attr_field], "_obj", None)
                if attr_parent is not None:
                    raw_attr_parents[attr_field] = attr_parent

        instance: TaskParameters = handler(data)

        # Harvest _typed_object stored on validated sub-models (e.g. mapping-form Source, Target, DeclusteringSource).
        for field_name in type(instance).model_fields:
            field_val = getattr(instance, field_name, None)
            obj = extract_typed_object(field_val)
            if obj is not None:
                field_objects.setdefault(field_name, obj)
                _register_typed_object_urls(url_objects, obj)

        # Re-check object and distribution roles on harvested wrapper objects (e.g. mapping-form Source/Target).
        for field_name, expected_types in cls._object_roles.items():
            if field_name in field_objects:
                validate_object_role(field_name, field_objects[field_name], expected_types)

        for field_name in cls._distribution_fields:
            if field_name in field_objects:
                validate_distribution_object(field_name, field_objects[field_name])

        instance._typed_objects.update(url_objects)

        # 2. Verify typed attribute parent objects match their paired object field (supporting path/UUID equivalence).
        for attr_field, obj_field in cls._attribute_object_bindings:
            attr_parent = raw_attr_parents.get(attr_field)
            expected_url = getattr(instance, obj_field, None)
            if attr_parent is not None and isinstance(expected_url, str):
                if not object_matches_reference(
                    attr_parent,
                    expected_url,
                    supplied_obj=raw_map.get(obj_field),
                ):
                    raise ValueError(
                        f"{attr_field!r} belongs to a different object than {obj_field!r} "
                        f"({object_identity(expected_url)!r})."
                    )
                if obj_field not in field_objects and attr_field in field_objects:
                    field_objects[obj_field] = field_objects[attr_field]

        # 3. Drillhole neighborhood limits require a DownholeIntervals source when a typed object is supplied.
        if cls._drillhole_source_field is not None:
            from .search import _active_drillhole_limits

            neighborhood = getattr(instance, "neighborhood", None)
            source_obj = field_objects.get(cls._drillhole_source_field)
            if neighborhood is not None and source_obj is not None:
                limits = _active_drillhole_limits(neighborhood)
                if limits and not isinstance(source_obj, DownholeIntervals):
                    raise ValueError(
                        f"Active drillhole neighborhood limit(s) ({', '.join(limits)}) require a "
                        f"DownholeIntervals source object, got {type(source_obj).__name__}."
                    )

        # 4. Bind/resolve bare attribute names in Filter fields when a typed BaseObject is available,
        #    and verify typed filter attributes do not come from a different object.
        for filter_field, obj_field in cls._filter_bindings:
            filt: Filter | None = getattr(instance, filter_field, None)
            if filt is None:
                continue
            bound_obj = field_objects.get(obj_field)
            if bound_obj is not None:
                object.__setattr__(instance, filter_field, filt.resolve(bound_obj))
            else:
                # Even without a BaseObject, check if any typed attribute in the filter belonged to another URL.
                obj_val = getattr(instance, obj_field, None)
                bound_url = obj_val if isinstance(obj_val, str) else getattr(obj_val, "object", None)
                if isinstance(bound_url, str):
                    filt._check_object_url(bound_url)

        return instance

    def output_targets(self) -> list[OutputTarget]:
        """Return the :class:`OutputTarget` entries written by this parameter set."""
        from .source_target import Target

        target = getattr(self, "target", None)
        if isinstance(target, Target) and isinstance(target.object, str):
            target_obj = extract_typed_object(target) or self._typed_objects.get(object_identity(target.object))
            return [
                resolve_output_target(
                    target.object,
                    target.attribute,
                    target_obj=target_obj,
                    known_name=target._attribute_name,
                )
            ]

        if target is not None:
            if isinstance(target, str):
                return [OutputTarget(object_id=object_identity(target), attribute_aliases=None)]
            target_obj_url = getattr(target, "object", None) or getattr(target, "reference", None)
            if isinstance(target_obj_url, str):
                return [OutputTarget(object_id=object_identity(target_obj_url), attribute_aliases=None)]

        target_object = getattr(self, "target_object", None)
        if isinstance(target_object, str):
            return [OutputTarget(object_id=object_identity(target_object), attribute_aliases=None)]

        return []


def _lift_nested_field(
    outer: dict[str, Any],
    container_key: str,
    nested_key: str,
    top_key: str,
    validator: Callable[[Any], Any],
) -> None:
    """Lift ``outer[container_key][nested_key]`` to ``outer[top_key]``, rejecting conflicts."""
    container = outer.get(container_key)
    if not isinstance(container, Mapping) or nested_key not in container:
        return

    container_copy = dict(container)
    nested_val = container_copy.pop(nested_key)
    outer[container_key] = container_copy

    if top_key in outer:
        top_val = outer[top_key]
        if nested_val is None and top_val is None:
            return
        if nested_val is None or top_val is None:
            raise ValueError(f"Conflicting values for {top_key!r} and '{container_key}.{nested_key}'.")
        try:
            equal = validator(nested_val) == validator(top_val)
        except Exception:
            equal = False
        if not equal:
            raise ValueError(f"Conflicting values for {top_key!r} and '{container_key}.{nested_key}'.")
    else:
        outer[top_key] = nested_val


class SourceTargetTaskParameters(TaskParameters):
    """Shared base for tasks with ``source``/``target`` that nest ``filter`` (and optional ``diagnostics``) on the wire."""

    _nested_target_diagnostics_type: ClassVar[type[BaseModel] | None] = None
    _filter_bindings: ClassVar[tuple[tuple[str, str], ...]] = (
        ("source_filter", "source"),
        ("target_filter", "target"),
    )

    @model_validator(mode="before")
    @classmethod
    def _reshape_wire_filters_and_diagnostics(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        source = data.get("source")
        target = data.get("target")
        has_source_filter = isinstance(source, Mapping) and "filter" in source
        has_target_filter = isinstance(target, Mapping) and "filter" in target
        has_target_diag = (
            cls._nested_target_diagnostics_type is not None and isinstance(target, Mapping) and "diagnostics" in target
        )
        if not (has_source_filter or has_target_filter or has_target_diag):
            return data

        from .filters import Filter

        updated = dict(data)
        if has_source_filter:
            _lift_nested_field(updated, "source", "filter", "source_filter", Filter.model_validate)
        if has_target_filter:
            _lift_nested_field(updated, "target", "filter", "target_filter", Filter.model_validate)
        if has_target_diag and cls._nested_target_diagnostics_type is not None:
            _lift_nested_field(
                updated,
                "target",
                "diagnostics",
                "diagnostics",
                cls._nested_target_diagnostics_type.model_validate,
            )
        return updated

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict[str, Any]:
        result = handler(self)
        source_filter = getattr(self, "source_filter", None)
        if source_filter is not None:
            result["source"]["filter"] = source_filter.model_dump(
                mode=info.mode,
                by_alias=info.by_alias,
                exclude_none=info.exclude_none,
            )
        target_filter = getattr(self, "target_filter", None)
        if target_filter is not None:
            result["target"]["filter"] = target_filter.model_dump(
                mode=info.mode,
                by_alias=info.by_alias,
                exclude_none=info.exclude_none,
            )
        diagnostics = getattr(self, "diagnostics", None)
        if diagnostics is not None:
            result["target"]["diagnostics"] = diagnostics.model_dump(
                mode="json",
                by_alias=info.by_alias,
                exclude_none=True,
            )
        return result

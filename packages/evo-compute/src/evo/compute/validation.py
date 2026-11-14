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

"""Schema-driven validation of compute-task parameters before submission.

The generic engine shapes a task's ``run(...)`` call from its published JSON Schema.
This module validates the resolved payload against that schema before it is submitted,
so mistakes surface locally with actionable messages instead of as opaque platform
errors:

* **Shallow** (always on): required-field presence on the resolved payload. Signature
  binding in :mod:`evo.compute.engine` already rejects missing required keyword
  arguments; this adds the schema-driven guard that a required field wasn't supplied
  as ``None`` unless its schema permits null.
* **Deep** (opt-in): JSON Schema Draft 2020-12 constraints on known fields, including ``$ref`` /
  ``$defs`` resolution and discriminated unions, via :mod:`jsonschema`.

Extra fields follow each object's ``additionalProperties`` rule. Closed objects reject
them, schema-valued rules validate them, and explicitly open objects accept them silently.
When the rule is absent, reference resolution warns before forwarding them unchanged.

Reference leaves (properties carrying a ``reference_to`` annotation) are relaxed, because
the concrete object/attribute/file shape the schema declares is produced by the
reference-resolution layer rather than supplied by the caller. The relaxation is only as
wide as it has to be: a leaf transmitted as a scalar -- every one in the published
catalogue is a ``string`` URL -- keeps its declared type, so static hints and runtime
checks agree. A leaf declared as its resolved object or array form is relaxed to accept
any value, since the caller legitimately passes a bare reference instead. Everything else
in the schema is validated normally.

:func:`unknown_annotation_keys` backs the annotation-conformance test that fails CI if
the platform introduces a schema annotation the engine does not know how to handle.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from .endpoints.models import TaskResource
from .exceptions import ParameterValidationError

__all__ = [
    "KNOWN_SCHEMA_ANNOTATIONS",
    "unknown_annotation_keys",
    "validate_parameters",
]

logger = logging.getLogger("compute.validation")


KNOWN_SCHEMA_ANNOTATIONS: frozenset[str] = frozenset(
    {
        "reference_to",
        "output",
        "supported_schemas",
        "attribute_from",
        "attribute_path",
        "target",
        "discriminator",
        "composite",
    }
)
"""The closed set of non-JSON-Schema annotation keys the engine understands.

The annotation-conformance test fails if a schema uses any annotation outside this set,
flagging that the engine needs updating before the new task can be handled generically.

``composite`` names a sub-object the platform composes from several fields (for example,
the published ``kriging`` schema tags ``source.filter`` and ``target.filter`` with
``composite: filter``). It is descriptive only: the node it marks is an ordinary subschema
that resolution and validation already walk, so the engine records the key as understood
rather than reporting every discovery of a filtered task as vocabulary drift.
"""


# Standard JSON Schema Draft 2020-12 keywords. Any dict key at a schema position that is
# neither one of these nor a KNOWN_SCHEMA_ANNOTATIONS entry is an unrecognised annotation.
_JSON_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        # Core
        "$schema",
        "$id",
        "$ref",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "$defs",
        "definitions",
        # Applicators
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "prefixItems",
        "items",
        "additionalItems",
        "contains",
        "properties",
        "patternProperties",
        "additionalProperties",
        "propertyNames",
        "unevaluatedItems",
        "unevaluatedProperties",
        # Validation
        "type",
        "enum",
        "const",
        "multipleOf",
        "maximum",
        "exclusiveMaximum",
        "minimum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxContains",
        "minContains",
        "maxProperties",
        "minProperties",
        "required",
        "dependentRequired",
        # Meta-data
        "title",
        "description",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
        "examples",
        # Format and content
        "format",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
    }
)

# Keywords whose value is itself a subschema.
_SUBSCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "additionalItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    }
)
# Keywords whose value is a list of subschemas.
_SUBSCHEMA_LIST_KEYS: frozenset[str] = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
# Keywords whose value maps arbitrary names to subschemas (the names are not keywords).
_SUBSCHEMA_MAP_KEYS: frozenset[str] = frozenset(
    {"properties", "patternProperties", "dependentSchemas", "$defs", "definitions"}
)


def _iter_schema_nodes(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every dict that occupies a *schema* position within ``node``.

    Recurses only through keywords that hold subschemas, so arbitrary names (property
    names, ``$defs`` keys) and data values (``enum``, ``default``) are never mistaken
    for schema keywords.
    """
    if not isinstance(node, dict):
        return
    yield node
    for key in _SUBSCHEMA_KEYS:
        if key in node:
            yield from _iter_schema_nodes(node[key])
    items = node.get("items")
    if isinstance(items, list):
        for sub in items:
            yield from _iter_schema_nodes(sub)
    elif isinstance(items, dict):
        yield from _iter_schema_nodes(items)
    for key in _SUBSCHEMA_LIST_KEYS:
        value = node.get(key)
        if isinstance(value, list):
            for sub in value:
                yield from _iter_schema_nodes(sub)
    for key in _SUBSCHEMA_MAP_KEYS:
        value = node.get(key)
        if isinstance(value, dict):
            for sub in value.values():
                yield from _iter_schema_nodes(sub)


def unknown_annotation_keys(schema: dict[str, Any] | None) -> set[str]:
    """Return schema keys that are neither standard JSON Schema nor a known annotation.

    Two callers, for two audiences. :class:`~evo.compute.discovery.DiscoveryClient` calls it
    on every fetched catalogue and *warns*, so a caller on an SDK that predates an annotation
    the platform now publishes learns that the schema is only partly interpreted rather than
    being blocked by it. The conformance test calls it on bundled schemas and *fails*, so the
    same drift stops a release rather than reaching users as a warning.

    :param schema: A task ``parameters`` or ``results`` JSON Schema (or ``None``).

    :return: The set of unrecognised annotation keys; empty when the schema only uses
        standard keywords and the closed annotation vocabulary.
    """
    unknown: set[str] = set()
    for node in _iter_schema_nodes(schema):
        for key in node:
            if key in _JSON_SCHEMA_KEYWORDS or key in KNOWN_SCHEMA_ANNOTATIONS:
                continue
            unknown.add(key)
    return unknown


# Types a reference is actually transmitted as, so its declared type still holds when the
# caller passes the reference itself. ``null`` is not among them: it says the leaf may be
# omitted, not that a reference travels as one.
_TRANSMITTED_REFERENCE_TYPES: frozenset[str] = frozenset({"string", "integer", "number", "boolean"})


def _relaxed_reference(node: dict[str, Any]) -> dict[str, Any]:
    """The strongest constraint a ``reference_to`` leaf can still be held to.

    Keeps the declared type as soon as one of its members is a form the reference travels
    as, and drops every constraint only when the leaf describes nothing but the resolved
    object the caller does not pass. A nullable declaration is kept whole either way, so
    ``["string", "null"]`` still admits both while ``["object", "null"]`` relaxes: after
    resolution that leaf holds a URL, which its declared object type would reject.
    """
    declared = node.get("type")
    types = [declared] if isinstance(declared, str) else declared if isinstance(declared, list) else []
    if not any(entry in _TRANSMITTED_REFERENCE_TYPES for entry in types):
        return {}
    return {"type": types[0] if len(types) == 1 else types}


def _relax_reference_leaves(node: Any) -> Any:
    """Deep-copy ``node``, relaxing any subschema carrying ``reference_to``.

    A ``reference_to`` leaf describes the resolved object/attribute/file form the schema
    wants, which the reference-resolution layer produces. The caller passes a reference, so
    the leaf keeps only what still applies to the reference itself (see
    :func:`_relaxed_reference`); presence is still enforced by the parent's ``required``.

    Recurses through the same schema positions as :func:`_iter_schema_nodes`, so a literal
    inside ``const``/``enum``/``default`` is copied untouched rather than rewritten.
    """
    if not isinstance(node, dict):
        return node
    if "reference_to" in node:
        return _relaxed_reference(node)
    relaxed = dict(node)
    for key in (*_SUBSCHEMA_KEYS, *_SUBSCHEMA_LIST_KEYS, *_SUBSCHEMA_MAP_KEYS, "items"):
        if key not in relaxed:
            continue
        value = relaxed[key]
        if key in _SUBSCHEMA_MAP_KEYS and isinstance(value, dict):
            relaxed[key] = {name: _relax_reference_leaves(sub) for name, sub in value.items()}
        elif isinstance(value, list):
            relaxed[key] = [_relax_reference_leaves(sub) for sub in value]
        else:
            relaxed[key] = _relax_reference_leaves(value)
    return relaxed


def _location(error: ValidationError) -> str:
    return ".".join(str(part) for part in error.absolute_path) or "parameters"


def _describe_union(error: ValidationError) -> str | None:
    """Describe a *discriminated* ``anyOf``/``oneOf`` failure, or ``None`` if it isn't discriminated.

    :mod:`jsonschema` only reports that the value matched no branch, and its own
    best-match heuristic routinely picks the wrong one -- telling someone building a
    spherical variogram that ``'scale' is a required property``. A ``discriminator``
    makes the intended branch knowable: it is the one whose sub-errors do not fault the
    discriminating property, so report that branch's real errors instead.
    """
    schema = error.schema if isinstance(error.schema, dict) else {}
    discriminator = schema.get("discriminator")
    name = discriminator.get("propertyName") if isinstance(discriminator, dict) else None
    if not isinstance(name, str) or not isinstance(error.instance, dict) or name not in error.instance:
        return None

    tag_path = (*error.absolute_path, name)
    branches: dict[Any, list[ValidationError]] = {}
    for sub in error.context:
        branches.setdefault(sub.schema_path[0], []).append(sub)
    matched = [subs for subs in branches.values() if all(tuple(sub.absolute_path) != tag_path for sub in subs)]
    if len(matched) == 1:
        return "; ".join(_describe(sub) for sub in matched[0])
    if matched:
        # Two or more branches accept the discriminator value (the tags aren't mutually
        # exclusive), so the intended one can't be singled out. Decline, and let the generic
        # "not valid under any of the given schemas" message stand rather than guess.
        return None

    # No branch accepted the tag, so the tag itself is the problem.
    allowed: list[Any] = []
    for sub in error.context:
        if tuple(sub.absolute_path) != tag_path:
            continue
        if sub.validator == "const":
            allowed.append(sub.validator_value)
        elif sub.validator == "enum":
            allowed.extend(sub.validator_value)
    if not allowed:
        return None
    values = ", ".join(repr(value) for value in dict.fromkeys(allowed))
    return f"{'.'.join(str(part) for part in tag_path)}: must be one of [{values}], got {error.instance[name]!r}"


def _describe(error: ValidationError) -> str:
    """Map a :class:`jsonschema.ValidationError` to a short, actionable message."""
    location = _location(error)
    if error.validator == "required":
        instance = error.instance if isinstance(error.instance, dict) else {}
        missing = [name for name in (error.validator_value or []) if name not in instance]
        names = ", ".join(repr(name) for name in missing) or error.message
        # Unprefixed at the top level, where the location adds nothing.
        prefix = f"{location}: " if error.absolute_path else ""
        return f"{prefix}missing required parameter(s): {names}"
    if error.validator in {"anyOf", "oneOf"} and error.context:
        if (described := _describe_union(error)) is not None:
            return described
    if error.validator == "type":
        expected = error.validator_value
        expected_str = expected if isinstance(expected, str) else " or ".join(expected)
        return f"{location}: expected type {expected_str}, got {type(error.instance).__name__}"
    if error.validator == "enum":
        allowed = ", ".join(repr(value) for value in error.validator_value)
        return f"{location}: must be one of [{allowed}], got {error.instance!r}"
    if error.validator == "additionalProperties":
        return f"{location}: {error.message}"
    return f"{location}: {error.message}"


def _default_label(spec: TaskResource) -> str:
    return f"{spec.topic}.{spec.name.replace('-', '_')}"


def _permits_null(root: dict[str, Any], prop_schema: Any) -> bool:
    """Whether a property's subschema accepts ``None``.

    Evaluated in the root schema's scope so ``$ref``, ``allOf`` and bare schemas are
    judged exactly as deep validation would judge them. A required-but-nullable field
    (``Optional[X]`` with no default) has ``None`` as a real value, not a missing one.
    """
    return Draft202012Validator(root).evolve(schema=prop_schema).is_valid(None)


def _validate_required(spec: TaskResource, parameters: dict[str, Any], label: str, operation: str = "run") -> None:
    """Shallow check: every required field is supplied, and not as ``None`` unless nullable."""
    schema = spec.parameters or {}
    properties = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    missing = sorted(
        name
        for name in required
        if name not in parameters or (parameters[name] is None and not _permits_null(schema, properties.get(name, {})))
    )
    if missing:
        errors = [f"missing required parameter {name!r}" for name in missing]
        raise ParameterValidationError(
            f"{label}.{operation}(): missing required parameter(s): {', '.join(missing)}",
            task=label,
            errors=errors,
        )


def _warn_unknown_fields(
    node: dict[str, Any],
    value: dict[str, Any],
    path: list[str],
    label: str,
    *,
    root: dict[str, Any] | None = None,
    operation: str = "run",
) -> None:
    """Check extra fields and warn only when the schema leaves their policy unspecified."""
    properties = node.get("properties")
    additional = node.get("additionalProperties")
    patterns = node.get("patternProperties", {}) or {}
    unknown = sorted(
        (
            name
            for name in value
            if name not in (properties or {}) and not any(re.search(pattern, str(name)) for pattern in patterns)
        ),
        key=str,
    )
    if not unknown or additional is True:
        return
    location = ".".join(str(part) for part in path) or "parameters"
    if additional is False:
        message = f"{location}: unexpected parameter(s): {', '.join(repr(name) for name in unknown)}"
        raise ParameterValidationError(f"{label}.{operation}(): {message}", task=label, errors=[message])
    if isinstance(additional, dict):
        schema = _relax_reference_leaves(node if root is None else root)
        validator = Draft202012Validator(schema).evolve(
            schema={"type": "object", "additionalProperties": _relax_reference_leaves(additional)}
        )
        errors = sorted(
            validator.iter_errors({name: value[name] for name in unknown}),
            key=lambda error: [str(part) for part in error.absolute_path],
        )
        if errors:
            prefix = f"{location}." if path else ""
            messages = [f"{prefix}{_describe(error)}" for error in errors]
            raise ParameterValidationError(
                f"{label}.{operation}(): parameters do not match the task schema:\n  - " + "\n  - ".join(messages),
                task=label,
                errors=messages,
            )
        return
    logger.warning(
        "%s: the task catalogue does not describe these fields at %s: %s. "
        "Their values are to be submitted as is. Check the field names and supply values ready for the service.",
        label,
        location,
        ", ".join(repr(name) for name in unknown),
    )


def _validate_deep(spec: TaskResource, parameters: dict[str, Any], label: str, operation: str = "run") -> None:
    """Check the published constraints, relaxing only reference input shapes."""
    schema = _relax_reference_leaves(spec.parameters or {})
    validator = Draft202012Validator(schema)
    # Rendered parts keep the key uniformly typed: raw paths mix str and int and won't sort.
    errors = sorted(validator.iter_errors(parameters), key=lambda error: [str(part) for part in error.absolute_path])
    if errors:
        messages = [_describe(error) for error in errors]
        detail = "\n  - ".join(messages)
        raise ParameterValidationError(
            f"{label}.{operation}(): parameters do not match the task schema:\n  - {detail}",
            task=label,
            errors=messages,
        )


def validate_parameters(
    spec: TaskResource,
    parameters: dict[str, Any],
    *,
    deep: bool = False,
    task_label: str | None = None,
    operation: str = "run",
) -> None:
    """Validate a resolved parameter payload against a task's schema.

    :param spec: The discovered task, carrying its ``parameters`` JSON Schema.
    :param parameters: The resolved payload about to be submitted.
    :param deep: When ``True``, check the full JSON Schema Draft 2020-12 constraints.
    :param task_label: A ``topic.task`` label for error messages; derived from ``spec``
        when omitted.
    :param operation: The method named in error messages, such as ``run`` or ``submit``.

    :raises ParameterValidationError: If the payload is missing a required field or, when
        ``deep`` is set, fails schema validation.
    """
    label = task_label or _default_label(spec)
    _validate_required(spec, parameters, label, operation)
    if deep:
        _validate_deep(spec, parameters, label, operation)

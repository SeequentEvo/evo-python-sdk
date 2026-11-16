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

"""Reference resolution: friendly Python inputs in, wire-shaped payload out.

Compute tasks address data by reference -- an object URL, a JMESPath expression naming an
attribute, a File API URL. Asking callers to build those by hand is the ergonomic gap the
typed geostatistics tasks close with :class:`~evo.compute.tasks.Source` /
:class:`~evo.compute.tasks.Target`. :class:`ReferenceResolver` generalises that: it reads
the reference annotations off *any* task's published schema, so one walker serves every
task in the catalogue with no per-task code.

The annotations it acts on (the closed vocabulary the platform publishes):

.. code-block:: text

    reference_to: geoscience-object   -> a validated object URL
    reference_to: attribute           -> a JMESPath expression
    reference_to: file                -> a File API URL
    supported_schemas: [...]          -> object families/versions the leaf accepts
    attribute_from: <pointer>         -> which sibling object an attribute lives on
    attribute_path: {pattern: [...]}  -> per-object-family JMESPath container
    target: attribute                 -> a create/update attribute slot
    discriminator: <key>              -> tagged union; pick the branch by the tag

Resolution reuses the very helpers the hand-written runners use
(:mod:`evo.compute.tasks.common.source_target`), so the two paths agree on the wire by
construction rather than by convention. It is also idempotent: a value that is already in
its wire form -- a URL string, a ready JMESPath expression, a ``{"operation": ...}`` dict
-- passes through untouched, so there is nothing to switch off.

``supported_schemas`` is enforced as a hard error when the referenced object's schema can
be established, either from a typed object the caller already holds or from a metadata
download. When the object cannot be loaded (a cross-hub reference, a transient objects-API
failure) the check is downgraded to a logged warning: an availability problem must not
block a job the platform would happily accept, and the platform re-checks server side.
"""

from __future__ import annotations

import logging
import operator
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from evo.common import IContext
from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference, ObjectSchema, SchemaVersion
from evo.objects.typed import BaseObject

from .endpoints.models import TaskResource
from .exceptions import ParameterValidationError

# The engine deliberately shares the typed tasks' conversion helpers rather than
# reimplementing them, so a generic call and a hand-written runner cannot drift apart.
from .tasks.common.source_target import (
    AnyTypedAttribute,
    _convert_object_reference,
    _get_attribute_expression,
    _source_from_attribute,
    _validate_target_attribute,
)

__all__ = [
    "ReferenceResolver",
]

logger = logging.getLogger("compute.resolution")

# A JMESPath filter projection, e.g. ``locations.attributes[?name=='grade']``. Its presence
# is what separates a ready-made expression from a bare attribute name.
_PROJECTION = "[?"

# A File API v2 reference: an absolute URL, or the service-relative path form. Deliberately
# loose -- the aim is to catch a local filename, not to spell out a grammar this package has
# no File API client to check against.
_FILE_REFERENCE = re.compile(r"(?:https://|/file/)", re.IGNORECASE)

# What a caller may hand over in place of an object: anything ``_convert_object_reference``
# accepts, plus the bare object id ``_as_uuid`` expands.
_OBJECT_HANDLE = (str, UUID, BaseObject, DownloadedObject, ObjectMetadata)

# One bound of a ``supported_schemas`` version range, e.g. ``>=1.2`` or ``<2``.
_BOUND = re.compile(r"(?P<op>>=|<=|>|<|==|=)?\s*(?P<version>\d+(?:\.\d+)*)")

_COMPARISONS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
    "==": operator.eq,
}


@dataclass
class _Walk:
    """State shared by one :meth:`ReferenceResolver.resolve` pass.

    ``resolved`` is filled in as the walk proceeds so an ``attribute_from`` pointer can read
    a sibling object that was resolved moments earlier; ``supplied`` is the caller's original
    payload, consulted when a pointer names a parameter the walk has not reached yet.
    """

    schema: dict[str, Any]
    label: str
    supplied: dict[str, Any]
    check_schemas: bool = True
    resolved: dict[str, Any] = field(default_factory=dict)


class ReferenceResolver:
    """Resolves a task's caller-supplied parameters into the payload the platform expects.

    One instance serves every task. Object schemas are cached for the lifetime of the
    resolver, so an object referenced by several parameters is only fetched once.

    :param context: An authenticated Evo context, used to load referenced objects.
    """

    def __init__(self, context: IContext) -> None:
        self._context = context
        self._schemas: dict[str, ObjectSchema | None] = {}

    async def resolve(
        self,
        spec: TaskResource,
        parameters: dict[str, Any],
        *,
        check_schemas: bool = True,
        task_label: str | None = None,
    ) -> dict[str, Any]:
        """Resolve every reference in ``parameters`` against the task's parameter schema.

        :param spec: The discovered task, carrying its ``parameters`` JSON Schema.
        :param parameters: The parameters the caller supplied, keyed by wire name.
        :param check_schemas: Enforce ``supported_schemas`` on object references.
        :param task_label: A ``topic.task`` label for error messages; derived from ``spec``
            when omitted.

        :return: A new payload with every reference in the form the schema declares.

        :raises ParameterValidationError: If a value cannot be resolved to a reference, or
            names an object outside a leaf's ``supported_schemas``.
        """
        schema = spec.parameters or {}
        walk = _Walk(
            schema=schema,
            label=task_label or f"{spec.topic}.{spec.name.replace('-', '_')}",
            supplied=parameters,
            check_schemas=check_schemas,
        )
        properties = schema.get("properties", {}) or {}
        for name, value in parameters.items():
            node = _deref(properties.get(name, {}), schema)
            walk.resolved[name] = await self._resolve_node(node, value, [name], walk)
        return walk.resolved

    # -- the walk ---------------------------------------------------------- #

    async def _resolve_node(self, node: dict[str, Any], value: Any, path: list[str], walk: _Walk) -> Any:
        """Resolve one value against the schema node that describes it."""
        if value is None:
            return None
        node = _deref(node, walk.schema)

        # A tagged union describes several shapes; only the tagged branch's annotations apply.
        if (discriminator := _discriminator_of(node)) is not None:
            node = _select_branch(node, value, discriminator, walk.schema) or node

        match node.get("reference_to"):
            case "geoscience-object":
                return await self._resolve_object(node, value, path, walk)
            case "attribute":
                return await self._resolve_attribute(node, value, path, walk)
            case "file":
                return _resolve_file(value, path, walk)

        if node.get("target") == "attribute":
            return await self._resolve_target_attribute(node, value, path, walk)

        # One of the SDK's typed parameter models (Source, Target, a filter, a sub-block frame,
        # ...). Dumping it yields the shape the schema describes. This happens before the walk
        # looks for ``properties`` because a union or optional slot has none of its own, and a
        # model left alone there would reach the wire as a model.
        if not isinstance(value, (dict, AnyTypedAttribute)) and hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", by_alias=True)

        if properties := node.get("properties"):
            if (frame := self._frame_from_attribute(properties, value, path, walk)) is not None:
                # Carry on into the walk below rather than returning: the frame's object leaf
                # has to face ``supported_schemas`` like any other.
                value = frame
            elif (frame := await self._frame_from_object(properties, value, path, walk)) is not None:
                _graft(walk.resolved, path, frame)
                return frame
            if isinstance(value, dict):
                # Register the container before descending so a child's ``attribute_from``
                # can read a sibling object that this same pass has already resolved.
                resolved: dict[str, Any] = {}
                _graft(walk.resolved, path, resolved)
                for key, child in value.items():
                    child_node = _deref(properties.get(key, {}), walk.schema)
                    resolved[key] = await self._resolve_node(child_node, child, [*path, key], walk)
                return resolved

        if node.get("type") == "array" and isinstance(value, list):
            items = _deref(node.get("items", {}), walk.schema)
            return [
                await self._resolve_node(items, item, [*path, str(index)], walk) for index, item in enumerate(value)
            ]

        return value

    # -- leaves ------------------------------------------------------------ #

    async def _resolve_object(self, node: dict[str, Any], value: Any, path: list[str], walk: _Walk) -> str:
        """A ``reference_to: geoscience-object`` leaf: any object handle in, its URL out."""
        reference = self._object_reference(value, path, walk)
        # Remember what the caller already knew. An ``attribute_from`` pointer reads the
        # resolved URL back, and should not pay for a download to learn what the object in
        # hand had already said.
        if (metadata := _metadata_of(value)) is not None:
            self._schemas[reference] = metadata.schema_id

        supported = node.get("supported_schemas")
        if not walk.check_schemas or not supported:
            return reference

        schema = await self._object_schema(value, reference)
        if schema is None:
            logger.warning(
                "%s: could not load %s to check it against supported_schemas %s; "
                "submitting unchecked -- the platform will validate it.",
                _describe(path, walk),
                reference,
                supported,
            )
        elif not any(_schema_matches(schema, pattern) for pattern in supported):
            raise _invalid(path, walk, f"object schema {str(schema)!r} is not one of {supported}")
        return reference

    async def _resolve_attribute(self, node: dict[str, Any], value: Any, path: list[str], walk: _Walk) -> Any:
        """A ``reference_to: attribute`` leaf: a typed attribute or a name in, JMESPath out."""
        if isinstance(value, AnyTypedAttribute):
            try:
                return _get_attribute_expression(value)
            except ValueError as error:
                raise _invalid(path, walk, str(error)) from None
        if not isinstance(value, str):
            # Nothing downstream would catch this: an attribute reference is a string on the
            # wire, and deep validation relaxes reference leaves.
            raise _invalid(path, walk, f"expected an attribute name or expression, got {type(value).__name__}")
        if _PROJECTION in value:
            return value  # already an expression
        container = await self._attribute_container(node, path, walk)
        return f"{container}[?name=='{value}']"

    async def _resolve_target_attribute(self, node: dict[str, Any], value: Any, path: list[str], walk: _Walk) -> Any:
        """A ``target: attribute`` slot: which attribute to create, or which one to update."""
        if isinstance(value, AnyTypedAttribute):
            # The slot names the attribute only; the object it belongs to is its sibling.
            # Mirrors ``source_target._validate_target_attribute`` for that half alone.
            if value.exists:
                return {"operation": "update", "reference": _get_attribute_expression(value)}
            return {"operation": "create", "name": value.name}
        if isinstance(value, str):
            return {"operation": "create", "name": value}
        if hasattr(value, "model_dump"):  # a Source/Target-style model built by the caller
            value = value.model_dump(mode="json", by_alias=True)
        if isinstance(value, dict) and (branch := _select_branch(node, value, "operation", walk.schema)) is not None:
            return await self._resolve_node(branch, value, path, walk)
        return value

    # -- object identity --------------------------------------------------- #

    def _object_reference(self, value: Any, path: list[str], walk: _Walk) -> str:
        """The URL for an object handle: a UUID is expanded, everything else is delegated."""
        if (object_id := _as_uuid(value)) is not None:
            return str(ObjectReference.new(self._context.get_environment(), object_id=object_id))
        try:
            return _convert_object_reference(value)
        except (TypeError, ValueError) as error:
            raise _invalid(path, walk, str(error)) from None

    async def _object_schema(self, value: Any, reference: str | None = None) -> ObjectSchema | None:
        """The schema of a referenced object, from metadata in hand or a metadata download.

        Returns ``None`` when the object cannot be loaded, which callers treat as "unknown"
        rather than as a failure. Both the answer and a failure to get one are cached, so a
        repeated reference costs at most one request.
        """
        if (metadata := _metadata_of(value)) is not None:
            return metadata.schema_id
        if reference is None:
            if (object_id := _as_uuid(value)) is not None:
                reference = str(ObjectReference.new(self._context.get_environment(), object_id=object_id))
            elif isinstance(value, str):
                reference = value
            else:
                return None
        if reference in self._schemas:
            return self._schemas[reference]

        try:
            downloaded = await DownloadedObject.from_context(self._context, reference)
        except Exception as error:
            # Deliberately broad: any reason the object cannot be read -- a cross-hub
            # reference, a permissions gap, an outage -- leaves its schema unknown, which
            # the caller downgrades to a warning rather than a refusal to submit.
            logger.debug("could not load %s: %s", reference, error)
            schema = None
        else:
            schema = downloaded.metadata.schema_id
        self._schemas[reference] = schema
        return schema

    # -- attribute placement ----------------------------------------------- #

    async def _attribute_container(self, node: dict[str, Any], path: list[str], walk: _Walk) -> str:
        """The JMESPath container an attribute name sits in, chosen by its object's family.

        A pointset keeps attributes under ``locations.attributes`` and a block model under
        ``attributes``, so the container is only knowable once the owning object -- named by
        ``attribute_from`` -- is known. ``attributes`` is the fallback when the schema says
        nothing or the object cannot be identified.
        """
        containers = node.get("attribute_path")
        pointer = node.get("attribute_from")
        if not containers or not pointer:
            return "attributes"
        owner = _pointer_value(pointer, path, walk.resolved, walk.supplied)
        if owner is None or (schema := await self._object_schema(owner)) is None:
            return "attributes"
        if (container := _container_for(containers, schema)) is None:
            return "attributes"
        return container

    def _frame_from_attribute(
        self, properties: dict[str, Any], value: Any, path: list[str], walk: _Walk
    ) -> dict[str, Any] | None:
        """Expand a bare typed attribute into the ``{object, attribute}`` frame it implies.

        ``run(source=pointset.attributes["grade"])`` is how the typed tasks are called; the
        attribute already knows the object it belongs to, so an ``{object, attribute}``
        parameter can be filled from it alone. The frame is handed back unwalked so its object
        leaf goes through the ordinary path. Returns ``None`` when this is not that case.
        """
        if isinstance(value, dict) or not isinstance(value, AnyTypedAttribute):
            return None
        if not {"object", "attribute"} <= properties.keys():
            return None
        target = _deref(properties["attribute"], walk.schema).get("target") == "attribute"
        try:
            model = _validate_target_attribute(value) if target else _source_from_attribute(value)
        except ValueError as error:
            raise _invalid(path, walk, str(error)) from None

        frame: dict[str, Any] = model.model_dump(mode="json", by_alias=True)
        # The object came with the attribute, so its schema is already in hand; seed the cache
        # so the leaf can be held to ``supported_schemas`` without paying for a download.
        if (metadata := _metadata_of(value._obj)) is not None:
            self._schemas[frame["object"]] = metadata.schema_id
        return frame

    async def _frame_from_object(
        self, properties: dict[str, Any], value: Any, path: list[str], walk: _Walk
    ) -> dict[str, Any] | None:
        """Expand a bare object into the one-key frame a parameter wraps it in.

        Some parameters are an object and nothing else -- declustering's ``grid``, for one --
        and the typed tasks let the object be passed straight in rather than spelled out as
        ``{"object": ...}``. Returns ``None`` when this node is not such a frame.
        """
        if isinstance(value, dict) or not isinstance(value, _OBJECT_HANDLE) or len(properties) != 1:
            return None
        ((name, leaf),) = properties.items()
        if _deref(leaf, walk.schema).get("reference_to") != "geoscience-object":
            return None
        return {name: await self._resolve_node(leaf, value, [*path, name], walk)}


# --------------------------------------------------------------------------- #
# Schema helpers.
# --------------------------------------------------------------------------- #


def _sole_non_null_branch(node: dict[str, Any]) -> dict[str, Any] | None:
    """The one real branch of an optional wrapper such as ``anyOf: [X, {"type": "null"}]``.

    The catalogue inlines an optional sub-object and adds ``null`` to its ``type``, but the
    same schema is just as often written as a union with a null branch. Left uncollapsed,
    the node carries no ``properties`` of its own and the walk returns everything below it
    untouched -- a typed attribute inside an optional filter would reach the wire as an
    object. A tagged union has more than one real branch and is left for the discriminator.
    """
    branches = node.get("anyOf") or node.get("oneOf")
    if not isinstance(branches, list):
        return None
    real = [branch for branch in branches if isinstance(branch, dict) and branch.get("type") != "null"]
    return real[0] if len(real) == 1 and len(real) < len(branches) else None


def _deref(node: Any, root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``$ref`` and collapse an optional wrapper, keeping annotations alongside."""
    if not isinstance(node, dict):
        return {}
    if (branch := _sole_non_null_branch(node)) is not None:
        node = {**{key: value for key, value in node.items() if key not in ("anyOf", "oneOf")}, **branch}
    reference = node.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return node
    target: Any = root
    for part in reference[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            return node
        target = target[part]
    if not isinstance(target, dict):
        return node
    return {**target, **{key: value for key, value in node.items() if key != "$ref"}}


def _discriminator_of(node: dict[str, Any]) -> str | None:
    """The property a tagged union is keyed on, or ``None`` if this node is not one."""
    if not node.get("oneOf") and not node.get("anyOf"):
        return None
    discriminator = node.get("discriminator")
    if isinstance(discriminator, dict):
        discriminator = discriminator.get("propertyName")
    return discriminator if isinstance(discriminator, str) else None


def _select_branch(node: dict[str, Any], value: Any, key: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """The ``oneOf``/``anyOf`` branch whose ``key`` constant matches ``value[key]``."""
    if not isinstance(value, dict) or key not in value:
        return None
    for branch in node.get("oneOf") or node.get("anyOf") or []:
        resolved = _deref(branch, root)
        if resolved.get("properties", {}).get(key, {}).get("const") == value[key]:
            return resolved
    return None


def _schema_matches(schema: ObjectSchema, pattern: str) -> bool:
    """Whether an object schema satisfies a ``supported_schemas``/``attribute_path`` pattern.

    Patterns name an object family and an optional version range, ``pointset/[>=1.2,<2]``.
    An unparseable bound is treated as satisfied: the family is the part that decides which
    container an attribute lives in, and over-rejecting a legitimate object is the worse
    failure.
    """
    family, _, bounds = pattern.partition("/")
    if schema.sub_classification != family:
        return False
    return all(_version_satisfies(schema.version, bound) for bound in bounds.strip("[]").split(",") if bound.strip())


def _version_satisfies(version: SchemaVersion, bound: str) -> bool:
    match = _BOUND.fullmatch(bound.strip())
    if match is None:
        return True
    expected = tuple(int(part) for part in match.group("version").split("."))
    # Compare only as precisely as the bound is written, so ``<2`` admits ``1.9.9``.
    actual = (version.major, version.minor, version.patch)[: len(expected)]
    return _COMPARISONS[match.group("op") or ">="](actual, expected)


def _container_of(template: str) -> str:
    """The container an ``attribute_path`` template selects within.

    ``locations.attributes[?attribute_type=='scalar']`` -> ``locations.attributes``. The
    predicate narrows by attribute type, which a lookup by name does not need.
    """
    index = template.find("[")
    return template[:index] if index != -1 else template


def _container_for(containers: Any, schema: ObjectSchema) -> str | None:
    """The ``attribute_path`` container an object of this schema keeps its attributes in.

    ``None`` when the map says nothing about the object's family, which leaves the caller to
    decide whether to guess or to leave a reference alone.
    """
    if not isinstance(containers, dict):
        return None
    for pattern, templates in containers.items():
        if _schema_matches(schema, pattern):
            return _container_of(templates[0] if isinstance(templates, list) else templates)
    return None


# --------------------------------------------------------------------------- #
# Value helpers.
# --------------------------------------------------------------------------- #


def _as_uuid(value: Any) -> UUID | None:
    """``value`` as a bare object id, or ``None`` if it is not one."""
    if isinstance(value, UUID):
        return value
    if isinstance(value, str) and not value.startswith("http"):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def _metadata_of(value: Any) -> ObjectMetadata | None:
    """The metadata a caller-supplied object already carries, if it is one."""
    if isinstance(value, ObjectMetadata):
        return value
    if isinstance(value, (BaseObject, DownloadedObject)):
        return value.metadata
    return None


def _resolve_file(value: Any, path: list[str], walk: _Walk) -> str:
    """A ``reference_to: file`` leaf: a File API URL, or a handle that carries one.

    The reference is held to a File API shape because nothing downstream would: deep
    validation relaxes reference leaves, so a local filename would otherwise be submitted as
    if it were a URL. Building a URL from a local path, and uploading local bytes, are
    deferred until the import/converter tasks that need them are supported.
    """
    if isinstance(value, str):
        reference = value
    else:
        url = getattr(value, "url", None)
        if url is None:
            url = getattr(getattr(value, "metadata", None), "url", None)
        if url is None:
            raise _invalid(path, walk, f"cannot resolve a file reference from {type(value).__name__}")
        reference = str(url)
    if _FILE_REFERENCE.match(reference) is None:
        raise _invalid(path, walk, f"expected a File API URL, got {reference!r}")
    return reference


def _pointer_value(pointer: str, path: list[str], *roots: dict[str, Any]) -> Any:
    """Follow an ``attribute_from`` pointer to the object the attribute belongs to.

    Absolute (``/target/object``) indexes from the payload root. Relative (``1/object``)
    climbs at least that many levels from the attribute's own position -- at least, because
    the platform writes the same pointer whether the attribute sits directly under a frame
    or deeper inside that frame's filter, so the nearest enclosing frame that owns the named
    object is the intended one. ``roots`` are consulted in order, so a resolution pass can
    prefer the payload it has already resolved and fall back to the caller's original values
    for pointers the walk has not reached yet.
    """
    if pointer.startswith("/"):
        return _index([segment for segment in pointer.split("/") if segment], roots)

    head, _, rest = pointer.partition("/")
    try:
        climb = max(int(head), 1)
    except ValueError:
        return None
    tail = [segment for segment in rest.split("/") if segment]
    for levels in range(climb, len(path) + 1):
        if (value := _index(path[:-levels] + tail, roots)) is not None:
            return value
    return None


def _index(segments: list[str], roots: tuple[dict[str, Any], ...]) -> Any:
    for root in roots:
        current: Any = root
        for segment in segments:
            if not isinstance(current, dict) or segment not in current:
                current = None
                break
            current = current[segment]
        if current is not None:
            return current
    return None


def _graft(root: dict[str, Any], path: list[str], container: dict[str, Any]) -> None:
    """Attach ``container`` at ``path``, so pointers can read it before it is filled in."""
    current = root
    for segment in path[:-1]:
        nested = current.get(segment)
        if not isinstance(nested, dict):
            nested = {}
            current[segment] = nested
        current = nested
    if path:
        current[path[-1]] = container


def _describe(path: list[str], walk: _Walk) -> str:
    return f"{walk.label}.run(): {'.'.join(path)}"


def _invalid(path: list[str], walk: _Walk, message: str) -> ParameterValidationError:
    location = ".".join(path)
    detail = f"{location}: {message}"
    return ParameterValidationError(f"{walk.label}.run(): {detail}", task=walk.label, errors=[detail])

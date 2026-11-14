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

"""Offline generator for the compute engine's type stub.

:class:`~evo.compute.engine.ComputeClient` synthesises its topic/task namespace at
runtime from the live discovery catalogue, so a type checker has nothing to read. This
module closes that gap by emitting ``evo/compute/engine.pyi`` from a **snapshot** of the
catalogue that is checked into the repository, giving editors completion, signature help
and hover documentation for every snapshotted task::

    python -m evo.compute._stubgen generate          # rewrite the stub
    python -m evo.compute._stubgen generate --check  # fail if the stub is stale (CI)
    python -m evo.compute._stubgen capture           # refresh the snapshot (needs auth)

``generate`` never authenticates and never touches the network, so the artifact is
reproducible in CI. ``capture`` is the only online step, and it is run by hand when the
snapshot should be brought forward to a newer catalogue.

The runtime stays fully generic: a task published after the snapshot still runs, it is
just not statically known until the snapshot is refreshed. That is the deliberate trade —
total runtime breadth, point-in-time static breadth. :meth:`ComputeClient.arun` is the
typed escape hatch for anything the stub does not know about.

"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import keyword
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .endpoints.models import TaskResource
from .engine import _normalise, _wire_names
from .overrides import load_override, overridden_tasks

__all__ = [
    "capture_snapshot",
    "generate_stub",
    "load_snapshot",
]

_PACKAGE_DIR = Path(__file__).resolve().parent

DEFAULT_SNAPSHOT_DIR = _PACKAGE_DIR.parents[2] / "stubs" / "snapshot"
"""The catalogue snapshot checked into the repository, next to the package sources."""

DEFAULT_OUTPUT = _PACKAGE_DIR / "engine.pyi"
"""The generated stub, which shadows ``engine.py`` for type checkers."""

_MANIFEST_NAME = "manifest.json"

# Deployments accumulate throwaway topics -- `test-cbcfabfdabaf` and friends, one task each,
# published by whoever was exercising the platform that day. Capturing them would ship someone's
# scratch work as public typed API, so they are skipped and reported rather than silently kept.
_SCRATCH_TOPIC = re.compile(r"^test(-|$)")

_LINE_LENGTH = 120

_JSON_SCALAR_TO_PYTHON = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}

# A ``title`` that is already a bare identifier is a model name worth reusing (Pydantic
# emits those); anything else is a human-facing field label such as "Known Locations".
_MODEL_TITLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# What a caller may hand to a ``reference_to`` leaf. The schema declares the resolved wire
# form -- a URL string -- but ``resolution.ReferenceResolver`` runs before submission and
# accepts any handle it can derive that URL from, so the stub describes the input side.
_REFERENCE_INPUTS = {
    "geoscience-object": "ObjectInput",
    "attribute": "AttributeInput",
    "file": "FileInput",
}

_REFERENCE_ALIASES = {
    "ObjectInput": (
        "str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference",
        "Any handle reference resolution can derive a geoscience object URL from.",
        [],
    ),
    "AttributeInput": (
        "str | AnyTypedAttribute",
        "A typed attribute, a JMESPath expression, or a bare attribute name.",
        [],
    ),
    # ``evo-files`` is not a dependency, so resolution reads a file handle structurally and
    # the stub has to describe it the same way rather than naming a concrete class.
    "FileInput": (
        "str | _FileUrl | _FileWithMetadata",
        "A File API URL, or a handle carrying one.",
        [
            "class _FileUrl(Protocol):",
            '    """Anything exposing the URL directly, such as a file resource."""',
            "",
            "    url: Any",
            "",
            "class _FileWithMetadata(Protocol):",
            '    """Anything exposing it through metadata, such as an upload result."""',
            "",
            "    metadata: _FileUrl",
        ],
    ),
}


def _deref(node: Any, root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local ``$ref`` without generating a type for what it points at."""
    if not isinstance(node, dict):
        return {}
    pointer = node.get("$ref")
    if not isinstance(pointer, str) or not pointer.startswith("#/"):
        return node
    target: Any = root
    for token in pointer.lstrip("#/").split("/"):
        target = target.get(token.replace("~1", "/").replace("~0", "~"), {}) if isinstance(target, dict) else {}
    return target if isinstance(target, dict) else {}


def _camel(text: str) -> str:
    """``normal-score gcp`` -> ``NormalScoreGcp``."""
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[^A-Za-z0-9]+", text) if part)


def _identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _literal(value: Any) -> str:
    """Render a JSON literal as Python source, double-quoting strings like the formatter."""
    return json.dumps(value) if isinstance(value, str) else repr(value)


def _member_tag(member: Any, index: int) -> str:
    """Name a union member after its discriminating constant, falling back to its position."""
    if isinstance(member, dict):
        for prop in (member.get("properties") or {}).values():
            if isinstance(prop, dict) and isinstance(prop.get("const"), str):
                return prop["const"]
    return str(index)


def _is_single_literal(annotation: str) -> bool:
    """Whether ``annotation`` is one ``Literal[...]`` rather than a union containing one."""
    try:
        node = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return False
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Literal"


def _join_union(annotations: list[str]) -> str:
    """``A | B``, with any ``Literal`` members folded into a single ``Literal[...]``.

    A schema that spells its allowed values as sibling ``const`` branches would otherwise
    render as ``Literal[200] | Literal[201] | ...``, which says the same thing less legibly
    and cannot be line-split the way one ``Literal`` can.
    """
    values: list[str] = []
    others: list[str] = []
    for annotation in annotations:
        if _is_single_literal(annotation):
            values.append(annotation[len("Literal[") : -1])
        else:
            others.append(annotation)
    folded = [f"Literal[{', '.join(dict.fromkeys(values))}]"] if values else []
    return " | ".join([*folded, *others])


def _declaration(name: str, annotation: str, indent: str) -> list[str]:
    """Declare ``name: annotation``, splitting an over-long ``Literal`` as the formatter would."""
    line = f"{indent}{name}: {annotation}"
    if len(line) <= _LINE_LENGTH or not _is_single_literal(annotation):
        return [line]
    values = ast.literal_eval(f"[{annotation[len('Literal[') : -1]}]")
    return [f"{indent}{name}: Literal[", *(f"{indent}    {_literal(value)}," for value in values), f"{indent}]"]


def _docstring(text: str, indent: str) -> str:
    """Render ``text`` as a syntactically safe triple-quoted docstring."""
    escaped = text.strip().replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if escaped.endswith('"'):
        escaped = escaped[:-1] + '\\"'
    lines = [line.rstrip() for line in escaped.splitlines()] or [""]
    if len(lines) == 1:
        return f'{indent}"""{lines[0]}"""'
    body = "\n".join(f"{indent}{line}".rstrip() for line in lines)
    return f'{indent}"""\n{body}\n{indent}"""'


@dataclass
class _GeneratedType:
    name: str
    body: str


@dataclass
class _TaskStub:
    """Everything the emitter needs for one task."""

    topic: str
    attribute: str
    class_name: str
    description: str
    types: list[_GeneratedType]
    run_parameters: list[str]
    return_type: str


@dataclass
class _TaskRenderer:
    """Renders the ``TypedDict`` tree for a single task's parameter and result schemas.

    Every generated name is prefixed with the topic and task, so two tasks never fight over a
    name. Within a task, structurally identical objects collapse onto one type -- the published
    schemas inline the same shape repeatedly (a filter condition appears at four depths in
    the kriging schema), and one name per shape keeps the stub readable.

    """

    spec: TaskResource

    prefix: str = field(init=False)
    section: str = field(default="", init=False)
    types: list[_GeneratedType] = field(default_factory=list, init=False)
    _names: set[str] = field(default_factory=set, init=False)
    _by_body: dict[tuple[str, str], str] = field(default_factory=dict, init=False)
    _by_pointer: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Topic-scoped, not just task-scoped: `geotech` and `gtm` both publish `remesh`,
        # `fill-holes` and a dozen more, and their shapes are not interchangeable.
        self.prefix = f"{_camel(self.spec.topic)}{_camel(self.spec.name)}"

    # -- naming ------------------------------------------------------------ #

    def _reserve(self, node: dict[str, Any], path: tuple[str, ...]) -> str:
        """Claim a unique name for an object at ``path``, before its body is rendered."""
        title = node.get("title")
        base = _camel(title if isinstance(title, str) and _MODEL_TITLE.match(title) else "".join(map(_camel, path)))
        # A title that already names the task ('KrigingResult') must not be prefixed twice.
        candidate = base if base.startswith(self.prefix) else f"{self.prefix}{self.section}{base}"
        name, suffix = candidate, 2
        while name in self._names:
            name, suffix = f"{candidate}{suffix}", suffix + 1
        self._names.add(name)
        return name

    def _commit(self, name: str, header: str, body: str, base: str = "") -> str:
        """Record a rendered type, reusing an identical one already generated.

        Two shapes are only interchangeable if they share a base: an input ``TypedDict`` and a
        result ``ResultNode`` can have the same body, and collapsing them would hand back a
        plain dict where the caller was promised something that loads its own references.
        """
        if (existing := self._by_body.get((base, body))) is not None and not re.search(rf"\b{name}\b", body):
            self._names.discard(name)
            return existing
        self._by_body.setdefault((base, body), name)
        self.types.append(_GeneratedType(name, f"{header}\n{body}"))
        return name

    # -- schema -> annotation ---------------------------------------------- #

    def _typed_dict(
        self,
        node: dict[str, Any],
        root: dict[str, Any],
        path: tuple[str, ...],
        name_override: str | None = None,
        base: str | None = None,
    ) -> str:
        name = name_override or self._reserve(node, path)
        self._names.add(name)
        properties: dict[str, Any] = node.get("properties") or {}
        required = set(node.get("required") or [])
        # A result is what the platform sent back, so it arrives as a ``ResultNode`` -- a dict
        # that also loads what its references point at. Declaring the keys as attributes is
        # what makes ``result.target.attribute.name`` and ``.load()`` type-check together.
        base = base or ("ResultNode" if self.section == "Result" else "TypedDict")
        hydrated = base != "TypedDict"

        annotations: dict[str, str] = {}
        for prop_name, prop in properties.items():
            annotation = self._annotation(prop, root, (*path, prop_name))
            optional = not hydrated and prop_name not in required
            annotations[prop_name] = f"NotRequired[{annotation}]" if optional else annotation

        if not all(_identifier(prop_name) for prop_name in properties):
            if hydrated:
                # No class can declare these; the node still hydrates, just untyped.
                return self._commit(name, f"class {name}({base}):", "    ...", base)
            # A key that isn't a Python identifier can only be declared functionally. The
            # dict stays total, so requiredness still comes from NotRequired alone.
            entries = [f'        "{prop_name}": {annotation},' for prop_name, annotation in annotations.items()]
            body = "\n".join([f'    "{name}",', "    {", *entries, "    },", ")"])
            return self._commit(name, f"{name} = TypedDict(", body, base)

        lines: list[str] = []
        if description := node.get("description"):
            lines.append(_docstring(str(description), "    "))
            lines.append("")
        for prop_name, prop in properties.items():
            lines.extend(_declaration(prop_name, annotations[prop_name], "    "))
            if isinstance(prop, dict) and (prop_description := prop.get("description")):
                lines.append(_docstring(str(prop_description), "    "))
        return self._commit(name, f"class {name}({base}):", "\n".join(lines), base)

    def _pointer(self, pointer: str, root: dict[str, Any]) -> str:
        """Resolve a local ``$ref`` to its generated type, defining it on first sight."""
        if (known := self._by_pointer.get(pointer)) is not None:
            return known
        target: Any = root
        for token in pointer.lstrip("#/").split("/"):
            target = target.get(token.replace("~1", "/").replace("~0", "~"), {}) if isinstance(target, dict) else {}
        if not isinstance(target, dict) or not target:
            return "Any"

        path = (pointer.rsplit("/", 1)[-1],)
        if not target.get("properties"):
            return self._annotation(target, root, path)
        # Claim the name before rendering, so a schema that refers back to itself terminates
        # on the same name it will end up with.
        name = self._reserve(target, path)
        self._by_pointer[pointer] = name
        self._by_pointer[pointer] = self._typed_dict(target, root, path, name_override=name)
        return self._by_pointer[pointer]

    def _union(self, members: Any, root: dict[str, Any], path: tuple[str, ...]) -> str:
        if not isinstance(members, list) or not members:
            return "Any"
        annotations = [
            self._annotation(member, root, (*path, _member_tag(member, index)))
            for index, member in enumerate(members, 1)
        ]
        return _join_union(list(dict.fromkeys(annotations)))

    def _of_type(self, json_type: str, node: dict[str, Any], root: dict[str, Any], path: tuple[str, ...]) -> str:
        if json_type == "object":
            if node.get("properties"):
                return self._typed_dict(node, root, path)
            additional = node.get("additionalProperties")
            if isinstance(additional, dict) and additional:
                return f"dict[str, {self._annotation(additional, root, (*path, 'value'))}]"
            return "dict[str, Any]"
        if json_type == "array":
            items = node.get("items")
            if isinstance(items, dict):
                return f"list[{self._annotation(items, root, (*path, 'item'))}]"
            return "list[Any]"
        return _JSON_SCALAR_TO_PYTHON.get(json_type, "Any")

    def _annotation(self, node: Any, root: dict[str, Any], path: tuple[str, ...]) -> str:
        """What may be passed for ``node``: its declared shape, plus any shorthand for it."""
        declared = self._declared_annotation(node, root, path)
        if self.section == "Result" or not isinstance(node, dict):
            return declared
        shorthand = self._shorthand_for(node, root)
        if shorthand is not None and shorthand not in declared:
            declared = f"{declared} | {shorthand}"
        model = self._model_input_for(node, root)
        if model is not None and re.search(rf"\b{model}\b", declared) is None:
            declared = f"{declared} | {model}"
        return declared

    def _model_input_for(self, node: dict[str, Any], root: dict[str, Any]) -> str | None:
        """Recognise SDK input models by their serialized fields, not schema names."""
        node = _deref(node, root)
        if node.get("type") != "object" or not set(node.get("required") or []) <= {"ellipsoid", "max_samples"}:
            return None
        ellipsoid = _deref((node.get("properties") or {}).get("ellipsoid", {}), root)
        properties = ellipsoid.get("properties") or {}
        containers = (
            (ellipsoid, {"ellipsoid_ranges", "rotation"}),
            (_deref(properties.get("ellipsoid_ranges", {}), root), {"major", "semi_major", "minor"}),
            (_deref(properties.get("rotation", {}), root), {"dip_azimuth", "dip", "pitch"}),
        )
        if any(not set(container.get("required") or []) <= fields for container, fields in containers):
            return None
        fields = [(("max_samples",), "integer")]
        fields.extend((("ellipsoid", "ellipsoid_ranges", axis), "number") for axis in ("major", "semi_major", "minor"))
        fields.extend((("ellipsoid", "rotation", angle), "number") for angle in ("dip_azimuth", "dip", "pitch"))
        if "min_samples" in (node.get("properties") or {}):
            fields.append((("min_samples",), "integer"))
        for path, expected_type in fields:
            descendant = node
            for name in path:
                descendant = (_deref(descendant, root).get("properties") or {}).get(name, {})
            declared_type = _deref(descendant, root).get("type")
            declared_types = [declared_type] if isinstance(declared_type, str) else declared_type or []
            if expected_type not in declared_types:
                return None
        return "SearchNeighborhood"

    def _shorthand_for(self, node: dict[str, Any], root: dict[str, Any]) -> str | None:
        """The value ``ReferenceResolver`` will expand into ``node``'s declared shape, if any.

        Mirrors ``resolution._frame_from_object`` and ``_frame_from_attribute``: a parameter
        that wraps a single object reference takes the object itself, and one shaped
        ``{object, attribute}`` takes a typed attribute, which knows both. A ``target:
        attribute`` slot additionally takes a bare name for the attribute to create.
        """
        node = _deref(node, root)
        if node.get("target") == "attribute":
            return "AttributeInput"
        properties = node.get("properties") or {}
        if {"object", "attribute"} <= properties.keys():
            return "AnyTypedAttribute"
        if len(properties) == 1:
            ((leaf,),) = (tuple(properties.values()),)
            if _deref(leaf, root).get("reference_to") == "geoscience-object":
                return "ObjectInput"
        return None

    def _declared_annotation(self, node: Any, root: dict[str, Any], path: tuple[str, ...]) -> str:
        if not isinstance(node, dict):
            return "Any"
        # On the way in a reference is whatever the caller holds; on the way out it is the
        # URL the platform sent, so only the parameter side widens.
        if self.section != "Result" and (alias := _REFERENCE_INPUTS.get(str(node.get("reference_to")))) is not None:
            declared = node.get("type")
            nullable = declared == "null" or (isinstance(declared, list) and "null" in declared)
            return f"{alias} | None" if nullable else alias
        if (pointer := node.get("$ref")) is not None:
            return self._pointer(str(pointer), root)
        if "const" in node:
            return f"Literal[{_literal(node['const'])}]"
        if (values := node.get("enum")) is not None:
            named = [value for value in values if value is not None]
            literal = f"Literal[{', '.join(map(_literal, named))}]" if named else "None"
            return f"{literal} | None" if len(named) != len(values) else literal
        for combinator in ("oneOf", "anyOf"):
            if combinator in node:
                return self._union(node[combinator], root, path)
        if (members := node.get("allOf")) is not None:
            subschemas = [member for member in members if isinstance(member, dict)]
            return self._annotation(subschemas[0], root, path) if len(subschemas) == 1 else "Any"

        json_type = node.get("type")
        if isinstance(json_type, list):
            annotations = [self._of_type(str(entry), node, root, path) for entry in json_type]
            return " | ".join(dict.fromkeys(annotations))
        if isinstance(json_type, str):
            return self._of_type(json_type, node, root, path)
        return "Any"

    # -- task surface ------------------------------------------------------ #

    def _run_parameters(self) -> list[str]:
        schema = self.spec.parameters or {}
        properties: dict[str, Any] = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        ordered = [*required, *(name for name in properties if name not in required)]
        published = {wire: name for name, wire in _wire_names(self.spec).items()}
        if properties and not published:
            # The engine could not name these either, and takes them through ``**parameters``.
            return ["**parameters: Any"]

        lines = []
        for name in ordered:
            annotation = self._annotation(properties.get(name, {}), schema, (name,))
            default = "" if name in required else " = ..."
            lines.append(f"{published[name]}: {annotation}{default}")
        lines.append("preview: bool = False")
        return lines

    def _return_type(self) -> str:
        results = self.spec.results
        if not isinstance(results, dict) or not results.get("properties"):
            return "TaskResult"
        # Everything reachable from the results schema is named ``<Task>Result...``, so an
        # input and an output shape sharing a schema title do not collide.
        self.section = "Result"
        return self._typed_dict(results, results, ("result",), name_override=f"{self.prefix}Result", base="TaskResult")

    def render(self) -> _TaskStub:
        # Parameters first, so the shapes callers write get the unqualified names.
        run_parameters = self._run_parameters()
        return_type = self._return_type()
        return _TaskStub(
            topic=self.spec.topic,
            attribute=self.spec.name.replace("-", "_"),
            class_name=f"_{_camel(self.spec.topic)}{_camel(self.spec.name)}",
            description=(self.spec.description or "").strip(),
            types=self.types,
            run_parameters=run_parameters,
            return_type=return_type,
        )


def load_snapshot(snapshot_dir: Path) -> list[TaskResource]:
    """Load the catalogue snapshot from ``snapshot_dir``.

    Each task lives at ``<topic>/<task>.json`` and holds the discovery payload verbatim;
    the topic and task name are implied by the path.

    :param snapshot_dir: The directory holding the snapshot.

    :return: The snapshotted tasks, ordered by topic then name.
    """
    tasks: list[TaskResource] = []
    for path in sorted(snapshot_dir.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["topic"] = path.parent.name
        payload["name"] = path.stem
        tasks.append(TaskResource.model_validate(payload))
    return tasks


def _snapshot_payload(task: TaskResource) -> dict[str, Any]:
    """The contents of a snapshot file: the discovery payload, less what the path says.

    Deployment-specific fields go too. ``links`` carries the capturing deployment's hostname
    and org id, and ``key`` its record id for the task -- none of which describe the task's
    shape, and all of which would rewrite every file the next time someone captures from a
    different org.
    """
    payload = task.model_dump(mode="json", exclude_none=True)
    for implied_or_local in ("topic", "name", "key", "links"):
        payload.pop(implied_or_local, None)
    return payload


def _snapshot_json(payload: dict[str, Any]) -> str:
    """Serialise a snapshot file exactly as the committed snapshot is written.

    The format is pinned -- and checked by ``tests/test_stubgen.py`` -- so that refreshing
    the snapshot shows what changed in the catalogue instead of reformatting every line.
    """
    return json.dumps(payload, indent=2) + "\n"


def _header(snapshot_dir: Path) -> list[str]:
    manifest_path = snapshot_dir / _MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return [
        "#  Copyright © 2026 Bentley Systems, Incorporated",
        '#  Licensed under the Apache License, Version 2.0 (the "License");',
        "#  you may not use this file except in compliance with the License.",
        "#  You may obtain a copy of the License at",
        "#      http://www.apache.org/licenses/LICENSE-2.0",
        "#  Unless required by applicable law or agreed to in writing, software",
        '#  distributed under the License is distributed on an "AS IS" BASIS,',
        "#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "#  See the License for the specific language governing permissions and",
        "#  limitations under the License.",
        "",
        "# AUTO-GENERATED BY `python -m evo.compute._stubgen generate`. DO NOT EDIT.",
        "#",
        f"# Catalogue snapshot: {manifest.get('source', 'unknown')}",
        f"# Captured: {manifest.get('captured_at', 'unknown')}",
        "#",
        "# The runtime is always live, so tasks published after the snapshot still run --",
        "# they are simply not statically known until the snapshot is refreshed. Use",
        "# `ComputeClient.arun(topic, task, parameters)` to reach them.",
    ]


def _render_fallbacks() -> list[str]:
    """The classes an unlisted topic or task resolves to.

    The catalogue is live, scoped per organization, and moves between SDK releases, so a
    shipped stub can never be all of it. Declaring ``__getattr__`` keeps the design's
    division -- execution is live, hints are point-in-time -- instead of turning a task this
    snapshot has not seen into a static error. Naming one is never wrong; it is only untyped.
    """
    return [
        "\n".join(
            [
                "class _UnknownTask:",
                _docstring(
                    "A task this stub does not describe. It still runs, with generic arguments.",
                    "    ",
                ),
                "",
                "    async def run(self, **parameters: Any) -> TaskResult: ...",
            ]
        ),
        "\n".join(
            [
                "class _UnknownTopic:",
                _docstring(
                    "A topic this stub does not describe. Its tasks still resolve at run time.",
                    "    ",
                ),
                "",
                "    def __getattr__(self, name: str) -> _UnknownTask: ...",
            ]
        ),
    ]


def _render_client(topics: list[str]) -> list[str]:
    lines = [
        "class ComputeClient:",
        _docstring(
            "Instance-bound asynchronous entry point to the compute task catalogue.\n\n"
            "The topic and task attributes below come from a point-in-time snapshot of the\n"
            "discovery catalogue. The runtime resolves them live, so a task missing from this\n"
            "stub still runs -- reach it with :meth:`arun`.",
            "    ",
        ),
        "",
        "    def __init__(",
        "        self,",
        "        context: IContext,",
        "        *,",
        "        cache_ttl_seconds: float = ...,",
        "        validate: bool = ...,",
        "        deep_validation: bool = ...,",
        "        check_schemas: bool | None = ...,",
        "    ) -> None: ...",
        "    @property",
        "    def context(self) -> IContext: ...",
        "    async def arun(",
        "        self,",
        "        topic: str,",
        "        task: str,",
        "        parameters: dict[str, Any],",
        "        *,",
        "        validate: bool | None = ...,",
        "        deep_validation: bool | None = ...,",
        "        check_schemas: bool | None = ...,",
        "    ) -> Any:",
        _docstring(
            "Run any task by name, including one this stub does not know about.",
            "        ",
        ),
        "        ...",
        "    def __dir__(self) -> list[str]: ...",
        "    def __repr__(self) -> str: ...",
    ]
    lines.extend(f"    {_normalise(topic)}: _{_camel(topic)}Tasks" for topic in sorted(topics))
    lines.append("    def __getattr__(self, name: str) -> _UnknownTopic: ...")
    return lines


def _override_runner(topic: str, task: str) -> tuple[str, str]:
    module = load_override(topic, task)
    assert module is not None, "only called for a task an override claims"
    annotation = inspect.signature(module.bind).return_annotation
    return module.__name__.removeprefix("evo.compute"), getattr(annotation, "__name__", annotation)


def _render(tasks: list[TaskResource], snapshot_dir: Path) -> str:
    overridden = overridden_tasks()
    stubs: list[_TaskStub] = []
    override_imports: list[str] = []
    members: dict[str, list[tuple[str, str]]] = {}

    def remember(topic: str, attribute: str, class_name: str) -> None:
        members.setdefault(topic, []).append((attribute, class_name))

    for task in sorted(tasks, key=lambda task: (task.topic, task.name)):
        attribute = _normalise(task.name)
        if (_normalise(task.topic), attribute) in overridden:
            continue
        awaited = _TaskRenderer(task).render()
        stubs.append(awaited)
        remember(task.topic, attribute, awaited.class_name)

    for topic, task_name in overridden:
        class_name = f"_{_camel(topic)}{_camel(task_name)}"
        module_path, runner_name = _override_runner(topic, task_name)
        override_imports.append(f"from {module_path} import {runner_name} as {class_name}")
        remember(topic, task_name, class_name)

    blocks: list[str] = []
    for stub in stubs:
        blocks.extend(generated.body for generated in stub.types)

    for stub in stubs:
        lines = [f"class {stub.class_name}:"]
        if stub.description:
            lines.append(_docstring(stub.description, "    "))
            lines.append("")
        lines.append("    async def run(")
        lines.append("        self,")
        if stub.run_parameters != ["**parameters: Any"]:
            lines.append("        *,")
        lines.extend(f"        {parameter}," for parameter in stub.run_parameters)
        if stub.description:
            # Repeated on the method as well as the class: editors read the class docstring
            # when hovering the task, and the method docstring in signature help.
            lines.append(f"    ) -> {stub.return_type}:")
            lines.append(_docstring(stub.description, "        "))
            lines.append("        ...")
        else:
            lines.append(f"    ) -> {stub.return_type}: ...")
        blocks.append("\n".join(lines))

    blocks.extend(_render_fallbacks())

    for topic, entries in sorted(members.items()):
        entries = sorted(entries)
        lines = [
            f"class _{_camel(topic)}Tasks:",
            _docstring(f"Tasks published under the ``{topic}`` topic.", "    "),
            "",
        ]
        lines.extend(f"    {attribute}: {class_name}" for attribute, class_name in entries)
        lines.append("    def __getattr__(self, name: str) -> _UnknownTask: ...")
        blocks.append("\n".join(lines))

    blocks.append("\n".join(_render_client(sorted(members))))

    body = "\n\n".join(blocks)
    used = {name: alias for name, alias in _REFERENCE_ALIASES.items() if re.search(rf"\b{name}\b", body)}
    aliases: list[str] = []
    for name, (definition, description, support) in used.items():
        aliases.extend(["", *support] if support else [])
        aliases.extend(["", f"{name} = {definition}", f'"""{description}"""'])
    if aliases:
        aliases.append("")
    declared = body + "\n".join(aliases)
    typing_names = [
        "Any",
        *(["Literal"] if "Literal[" in body else []),
        *(["Protocol"] if "Protocol)" in declared else []),
    ]
    extensions = [name for name in ("NotRequired", "TypedDict") if name in body]
    outputs = [name for name in ("ResultNode", "TaskResult") if name in body]
    imports = [
        f"from typing import {', '.join(typing_names)}",
        *(["from uuid import UUID"] if "ObjectInput" in used else []),
        "",
        "from evo.common import IContext",
        *(["from evo.objects import ObjectMetadata, ObjectReference"] if "ObjectInput" in used else []),
        *(["from evo.objects.typed import BaseObject, DownloadedObject"] if "ObjectInput" in used else []),
        *([f"from typing_extensions import {', '.join(extensions)}"] if extensions else []),
        "",
        *sorted(
            [
                *([f"from .outputs import {', '.join(outputs)}"] if outputs else []),
                *(
                    ["from .tasks.common.search import SearchNeighborhood"]
                    if re.search(r"\bSearchNeighborhood\b", body)
                    else []
                ),
                *(
                    ["from .tasks.common.source_target import AnyTypedAttribute"]
                    if "AnyTypedAttribute" in declared
                    else []
                ),
                *override_imports,
            ]
        ),
    ]
    preamble = [
        *_header(snapshot_dir),
        "",
        *imports,
        "",
        '__all__ = ["ComputeClient"]',
        *(aliases or [""]),
    ]
    return "\n".join(preamble) + "\n" + body + "\n"


def _formatted(source: str, output: Path) -> str:
    """Hand the rendered stub to ``ruff format`` rather than imitating it.

    The generator used to emit formatter-ready text by construction, which held while the
    snapshot was three hand-picked tasks and stopped holding the moment it was not: a long
    ``Literal`` splits one way, an over-long union in a subscript another, an import a third.
    Matching that by hand is a losing game, and a stub that is not format-stable fails CI for
    a reason that has nothing to do with what it describes.

    ``--stdin-filename`` matters: ``.pyi`` files are formatted differently from ``.py``.
    """
    try:
        result = subprocess.run(
            ["ruff", "format", "--stdin-filename", str(output), "-"],
            input=source,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("ruff is needed to generate the stub; install the 'test' dependency group") from None
    return result.stdout


def generate_stub(snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR, output: Path = DEFAULT_OUTPUT) -> str:
    """Render the engine type stub from a catalogue snapshot.

    :param snapshot_dir: The snapshot to read. No network access is performed.
    :param output: The path the stub is written to, quoted in error messages.

    :return: The stub source.
    """
    tasks = load_snapshot(snapshot_dir)
    if not tasks:
        raise FileNotFoundError(f"no task schemas found in {snapshot_dir}")
    return _formatted(_render(tasks, snapshot_dir), output)


def capture_snapshot(snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR) -> list[TaskResource]:
    """Refresh the catalogue snapshot from live discovery.

    This is the one online step, run by hand when the stub should move to a newer
    catalogue. Credentials come from ``EVO_ACCESS_TOKEN``, ``EVO_HUB_URL`` and
    ``EVO_ORG_ID``.

    :param snapshot_dir: The directory to write the snapshot to.

    :return: The tasks written, which excludes any :data:`_SCRATCH_TOPIC` the deployment
        happens to be carrying.
    """
    import asyncio
    import os
    from datetime import date
    from uuid import UUID

    from evo.aio import AioTransport
    from evo.common import APIConnector
    from evo.oauth import AccessTokenAuthorizer

    from .discovery import DiscoveryClient

    async def _fetch() -> list[TaskResource]:
        transport = AioTransport(user_agent="evo-compute-stubgen")
        authorizer = AccessTokenAuthorizer(access_token=os.environ["EVO_ACCESS_TOKEN"])
        connector = APIConnector(base_url=os.environ["EVO_HUB_URL"], transport=transport, authorizer=authorizer)
        async with connector:
            return await DiscoveryClient(connector, UUID(os.environ["EVO_ORG_ID"])).list_tasks()

    discovered = sorted(asyncio.run(_fetch()), key=lambda task: (task.topic, task.name))
    tasks = [task for task in discovered if not _SCRATCH_TOPIC.match(task.topic)]
    skipped = sorted({task.topic for task in discovered} - {task.topic for task in tasks})
    if skipped:
        print(f"skipped scratch topics: {', '.join(skipped)}")

    for stale in snapshot_dir.glob("*/*.json"):
        stale.unlink()
    for task in tasks:
        path = snapshot_dir / task.topic / f"{task.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_snapshot_json(_snapshot_payload(task)), encoding="utf-8")
    for directory in snapshot_dir.iterdir():
        # A topic the catalogue dropped leaves its directory behind, and `load_snapshot`
        # would keep reading it as a topic with no tasks.
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    manifest = {
        "source": "GET /compute/orgs/{org_id}/tasks?details=true",
        "captured_at": date.today().isoformat(),
        "tasks": [{"topic": task.topic, "name": task.name, "version": task.version} for task in tasks],
    }
    (snapshot_dir / _MANIFEST_NAME).write_text(_snapshot_json(manifest), encoding="utf-8")
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evo.compute._stubgen", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="render the stub from the snapshot (offline)")
    generate.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    generate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--check", action="store_true", help="fail if the stub is out of date")

    capture = commands.add_parser("capture", help="refresh the snapshot from live discovery (needs auth)")
    capture.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_DIR)

    arguments = parser.parse_args(argv)
    if arguments.command == "capture":
        tasks = capture_snapshot(arguments.snapshot)
        print(f"captured {len(tasks)} tasks to {arguments.snapshot}")
        return 0

    source = generate_stub(arguments.snapshot, arguments.output)
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_text(encoding="utf-8") != source:
            print(f"{arguments.output} is out of date; run `python -m evo.compute._stubgen generate`", file=sys.stderr)
            return 1
        print("engine stub is up to date")
        return 0

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(source, encoding="utf-8")
    print(f"wrote {arguments.output} ({len(source.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

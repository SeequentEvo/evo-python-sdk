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

"""Offline Markdown reference generator for snapshotted compute tasks.

Generates one standalone reference page per topic under ``mkdocs/docs/packages/evo-compute/tasks/``
from the committed task JSON schema snapshot, sharing the same type and shorthand rules as
``evo.compute._stubgen`` while formatting headings and tables for MkDocs and the Seequent
Developer Portal importer.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from ._stubgen import DEFAULT_SNAPSHOT_DIR, _deref, _GeneratedField, _GeneratedType, _TaskRenderer, load_snapshot
from .endpoints.models import TaskResource
from .engine import _normalise, _wire_names
from .overrides import load_override, overridden_tasks

__all__ = [
    "render_topic_docs",
]

GITHUB_SNAPSHOT_BASE_URL = "https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot"
PORTAL_BASE_URL = "https://developer.seequent.com"

_TOPIC_TITLES: dict[str, str] = {
    "converter": "Converter Tasks",
    "cpt-to-borehole": "CPT to Borehole Tasks",
    "geology": "Geology Tasks",
    "geophysics": "Geophysics Tasks",
    "geostatistics": "Geostatistics Tasks",
    "geotech": "Geotechnical Tasks",
    "isosurfacing": "Isosurfacing Tasks",
    "rbf": "RBF Tasks",
    "utilities": "Utilities Tasks",
}

_ANCHOR_LINK_RE = re.compile(r"""<a\s+href=['"](/[^'"]+)['"]\s*>(.*?)</a>""", re.IGNORECASE | re.DOTALL)
_ANGLE_PLACEHOLDER_RE = re.compile(r"<([A-Za-z0-9_/][^<>]*)>")
_CODE_SPAN_SPLIT_RE = re.compile(r"(`+[^`]*`+)")


def _topic_title(topic: str) -> str:
    if topic in _TOPIC_TITLES:
        return _TOPIC_TITLES[topic]
    words = " ".join(part.capitalize() for part in topic.split("-") if part)
    return f"{words} Tasks"


def _sanitize_prose(text: str) -> str:
    """Normalise schema description text for MkDocs and Docusaurus MDX safety."""
    cleaned = _ANCHOR_LINK_RE.sub(lambda m: f"[{m.group(2).strip()}]({PORTAL_BASE_URL}{m.group(1)})", text.strip())
    parts = _CODE_SPAN_SPLIT_RE.split(cleaned)
    for idx in range(0, len(parts), 2):
        parts[idx] = _ANGLE_PLACEHOLDER_RE.sub(r"`<\1>`", parts[idx])
    return "".join(parts)


def _table_cell(text: str) -> str:
    """Format ``text`` for a single GFM table cell, escaping pipes only outside inline code."""
    collapsed = " ".join(line.strip() for line in _sanitize_prose(text).splitlines() if line.strip())
    parts = _CODE_SPAN_SPLIT_RE.split(collapsed)
    for idx in range(0, len(parts), 2):
        parts[idx] = parts[idx].replace("|", r"\|")
    return "".join(parts) or "—"


def _schema_notes(prop: dict[str, Any], root: dict[str, Any]) -> list[str]:
    """Collect constraint and reference notes for ``prop`` (and single-object wrappers)."""
    resolved = _deref(prop, root)
    notes: list[str] = []

    target_nodes = [resolved]
    sub_props = resolved.get("properties") or {}
    if len(sub_props) == 1:
        ((only_child,),) = (tuple(sub_props.values()),)
        if isinstance(only_child, dict):
            target_nodes.append(_deref(only_child, root))

    seen_schemas: list[str] = []
    seen_exts: list[str] = []
    for candidate in target_nodes:
        for entry in candidate.get("supported_schemas") or []:
            if isinstance(entry, str) and entry not in seen_schemas:
                seen_schemas.append(entry)
        for ext in candidate.get("file_extensions") or []:
            if isinstance(ext, str) and ext not in seen_exts:
                seen_exts.append(ext)

    if seen_schemas:
        formatted = ", ".join(f"`{entry}`" for entry in seen_schemas)
        notes.append(f"Supported schemas: {formatted}.")
    if seen_exts:
        formatted = ", ".join(f"`{ext}`" for ext in seen_exts)
        notes.append(f"Supported file extensions: {formatted}.")
    if resolved.get("target") == "attribute":
        notes.append(
            "Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), "
            'or `{"operation": "create" | "update", ...}`.'
        )

    bounds: list[str] = []
    for key, op in (
        ("minimum", ">="),
        ("exclusiveMinimum", ">"),
        ("maximum", "<="),
        ("exclusiveMaximum", "<"),
    ):
        if key in resolved and resolved[key] is not None:
            bounds.append(f"`{op} {resolved[key]}`")
    if bounds:
        notes.append(f"Bounds: {', '.join(bounds)}.")

    return notes


def _field_description(
    prop: dict[str, Any], root: dict[str, Any], *, wire_name: str | None = None, py_name: str | None = None
) -> str:
    resolved = _deref(prop, root)
    raw_desc = str(prop.get("description") or resolved.get("description") or "").strip()
    parts = [raw_desc] if raw_desc else []
    if wire_name and py_name and wire_name != py_name:
        parts.append(f"Wire key: `{wire_name}`.")
    parts.extend(_schema_notes(prop, root))
    return " ".join(parts)


def _hydrated_loader_cell(
    field_name: str, prop: dict[str, Any], root: dict[str, Any], *, prefix: str = "result"
) -> str:
    resolved = _deref(prop, root)
    output_kind = resolved.get("output")
    if output_kind == "geoscience-object":
        return f"`await {prefix}.{field_name}.load()`, `await {prefix}.{field_name}.to_dataframe(*keys)`"
    if output_kind == "attribute":
        return f"`await {prefix}.{field_name}.load()`, `await {prefix}.{field_name}.to_dataframe()`"
    if resolved.get("type") == "array" and isinstance(resolved.get("items"), dict):
        item_node = _deref(resolved["items"], root)
        if item_node.get("output") in ("geoscience-object", "attribute"):
            return f"`await {prefix}.{field_name}[i].load()`, `await {prefix}.{field_name}[i].to_dataframe()`"
    return "—"


def _parse_run_param_annotations(run_parameters: list[str]) -> dict[str, str]:
    """Extract ``{param_name: annotation}`` from ``_TaskStub.run_parameters``."""
    parsed: dict[str, str] = {}
    for entry in run_parameters:
        if entry.startswith("**"):
            continue
        lhs = entry[:-6] if entry.endswith(" = ...") else entry.split(" = ", 1)[0]
        name, _, annotation = lhs.partition(": ")
        parsed[name] = annotation
    return parsed


def _render_result_table(
    fields: tuple[_GeneratedField, ...], results_root: dict[str, Any], *, prefix: str = "result"
) -> list[str]:
    lines = [
        "| Field | Type | Hydrated Loader | Description |",
        "| --- | --- | --- | --- |",
    ]
    for f in fields:
        loader = _hydrated_loader_cell(f.name, f.schema, results_root, prefix=prefix)
        desc = _field_description(f.schema, results_root)
        lines.append(f"| `{f.name}` | `{f.annotation}` | {_table_cell(loader)} | {_table_cell(desc)} |")
    return lines


def _render_typed_dict_table(fields: tuple[_GeneratedField, ...], params_root: dict[str, Any]) -> list[str]:
    lines = [
        "| Field | Type | Required | Description |",
        "| --- | --- | --- | --- |",
    ]
    for f in fields:
        req = "Yes" if f.required else "No"
        desc = _field_description(f.schema, params_root)
        lines.append(f"| `{f.name}` | `{f.annotation}` | {req} | {_table_cell(desc)} |")
    return lines


def _render_generic_task(task: TaskResource) -> tuple[list[str], str]:
    """Render one generic task's Markdown section and return ``(lines, combined_type_text)``."""
    topic_attr = _normalise(task.topic)
    task_attr = _normalise(task.name)
    stub = _TaskRenderer(task).render()
    param_schema = task.parameters or {}
    results_schema = task.results if isinstance(task.results, dict) else {}

    badges = [f"`client.{topic_attr}.{task_attr}`", f"Task: `{task.topic}/{task.name}`"]
    if task.version:
        badges.append(f"Version: `{task.version}`")
    if task.feature_flag:
        badges.append("Preview (`preview=True` required)")

    lines: list[str] = [
        f"### {task_attr}",
        "",
        " · ".join(badges),
        "",
    ]
    if stub.description:
        lines.extend([_sanitize_prose(stub.description), ""])

    # Call signature
    sig_params = list(stub.run_parameters)
    if sig_params and sig_params[-1].startswith("preview: bool = "):
        sig_params[-1] = "preview: bool = False"
    lines.append("```python")
    lines.append(f"await client.{topic_attr}.{task_attr}.run(")
    if sig_params != ["**parameters: Any"]:
        lines.append("    *,")
    for p in sig_params:
        lines.append(f"    {p},")
    lines.append(f") -> {stub.return_type}")
    lines.append("```")
    lines.append("")

    # Parameters table
    properties: dict[str, Any] = param_schema.get("properties") or {}
    required = list(param_schema.get("required") or [])
    ordered = [*required, *(name for name in properties if name not in required)]
    published = {wire: py for py, wire in _wire_names(task).items()}
    annotations = _parse_run_param_annotations(stub.run_parameters)

    lines.extend(
        [
            "Parameters:",
            "",
            "| Name | Type | Description | Default |",
            "| --- | --- | --- | --- |",
        ]
    )
    if properties and not published:
        lines.append(
            "| `**parameters` | `Any` | Task parameters passed by wire key (required keys: "
            + (", ".join(f"`{k}`" for k in required) or "none")
            + "). | — |"
        )
    else:
        for wire_name in ordered:
            py_name = published[wire_name]
            prop = properties.get(wire_name) or {}
            ann = annotations.get(py_name, "Any")
            default = "_required_" if wire_name in required else f"`{prop.get('default', None)!r}`"
            desc = _field_description(prop, param_schema, wire_name=wire_name, py_name=py_name)
            lines.append(f"| `{py_name}` | `{ann}` | {_table_cell(desc)} | {default} |")

    preview_desc = (
        "Opt in to preview execution (required for this preview task: pass `preview=True`)."
        if task.feature_flag
        else "Whether to include the preview opt-in header."
    )
    lines.append(f"| `preview` | `bool` | {preview_desc} | `False` |")
    lines.append("")

    # Results table
    result_types = {gt.name: gt for gt in stub.types}
    top_result: _GeneratedType | None = result_types.get(stub.return_type)
    lines.append(f"Results (`{stub.return_type}`):")
    lines.append("")
    if top_result is not None and top_result.fields:
        lines.extend(_render_result_table(top_result.fields, results_schema, prefix="result"))
    else:
        lines.append("Returns a `TaskResult` (`dict` subclass) containing the raw result payload.")
    lines.append("")

    # Nested shapes (both input TypedDicts and nested ResultNodes)
    nested_shapes = [gt for gt in stub.types if gt.name != stub.return_type and gt.fields]
    if nested_shapes:
        lines.extend(["#### Nested Shapes", ""])
        for gt in nested_shapes:
            kind = "ResultNode" if gt.base != "TypedDict" else "TypedDict"
            lines.append(f"**`{gt.name}`** (`{kind}`)")
            lines.append("")
            if gt.description:
                lines.extend([_sanitize_prose(gt.description), ""])
            if kind == "ResultNode":
                lines.extend(_render_result_table(gt.fields, results_schema, prefix="node"))
            else:
                lines.extend(_render_typed_dict_table(gt.fields, param_schema))
            lines.append("")

    type_blob = "\n".join([*stub.run_parameters, *(gt.body for gt in stub.types)])
    return lines, type_blob


def _render_shorthands_section(type_blob: str) -> list[str]:
    rows: list[str] = []
    if re.search(r"\bObjectInput\b", type_blob):
        rows.append(
            "| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | "
            "Any handle `ReferenceResolver` can derive a geoscience object URL from. |"
        )
    if re.search(r"\bAttributeInput\b", type_blob):
        rows.append(
            "| `AttributeInput` | `str | AnyTypedAttribute` | "
            "A typed attribute, a JMESPath expression, or a bare attribute name. |"
        )
    if re.search(r"\bAnyTypedAttribute\b", type_blob):
        rows.append(
            "| `AnyTypedAttribute` | `Attribute | BlockModelAttribute | PendingAttribute | BlockModelPendingAttribute` | "
            "A typed attribute handle carrying both its parent geoscience object and attribute name. |"
        )
    if re.search(r"\bFileInput\b", type_blob):
        rows.append(
            "| `FileInput` | `str | FileHandle` | "
            "A File API URL or an SDK file/upload handle exposing `.url` or `.metadata.url`. |"
        )
    if re.search(r"\bSearchNeighborhood\b", type_blob):
        rows.append(
            "| `SearchNeighborhood` | `evo.compute.tasks.SearchNeighborhood` | "
            "Typed search ellipsoid and sample/octant/drillhole limit model accepted alongside raw dictionaries. |"
        )
    if not rows:
        return []
    return [
        "## Input Shorthands",
        "",
        "| Alias | Accepted Python Types | Description |",
        "| --- | --- | --- |",
        *rows,
        "",
    ]


_PARAM_DOC_RE = re.compile(
    r"^:param\s+(\w+):\s*(.+?)(?=^:(?:param|return|rtype|raises)\b|\Z)", re.MULTILINE | re.DOTALL
)


def _extract_param_docs(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Split a Sphinx-style docstring into ``(prose_intro, {param_name: description})``."""
    if not docstring:
        return "", {}
    params = {
        match.group(1): " ".join(line.strip() for line in match.group(2).splitlines() if line.strip())
        for match in _PARAM_DOC_RE.finditer(docstring)
    }
    intro_lines: list[str] = []
    for line in docstring.splitlines():
        if line.strip().startswith((":param ", ":return:", ":rtype:", ":raises ")):
            break
        intro_lines.append(line)
    return "\n".join(intro_lines).strip(), params


def _expand_private_alias(annotation: str, module: ModuleType) -> str:
    """Replace private ``_Alias`` tokens from ``module`` with readable public SDK type names."""
    if annotation == "_ObjectInput":
        return "ObjectInput"

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "_ObjectInput":
            return "ObjectInput"
        if not token.startswith("_"):
            return token
        val = getattr(module, token, None)
        if val is None:
            return token
        rendered = str(val)
        # Strip verbose internal package prefixes from runtime Union reprs.
        rendered = re.sub(r"\bevo\.[A-Za-z0-9_.]+\.([A-Z][A-Za-z0-9_]*)\b", r"\1", rendered)
        rendered = re.sub(r"\buuid\.UUID\b", "UUID", rendered)
        return rendered

    return re.sub(r"\b_[A-Za-z0-9_]+\b", repl, annotation)


def _render_override_result_members(result_cls: type) -> list[str]:
    """Render a summary table of public properties and helper methods on a custom result class."""
    rows: list[str] = []
    for name, member in inspect.getmembers(result_cls):
        if name.startswith("_"):
            continue
        if isinstance(member, property) and member.fget is not None:
            sig = inspect.signature(member.fget)
            ret = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else "Any"
            ret_str = ret if isinstance(ret, str) else getattr(ret, "__name__", str(ret))
            doc_first = (inspect.getdoc(member.fget) or "").splitlines()
            desc = doc_first[0] if doc_first else ""
            rows.append(f"| `{name}` | `{ret_str}` | `result.{name}` | {_table_cell(desc)} |")
        elif inspect.isfunction(member):
            sig = inspect.signature(member)
            ret = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else "Any"
            ret_str = ret if isinstance(ret, str) else getattr(ret, "__name__", str(ret))
            is_async = inspect.iscoroutinefunction(member)
            call_expr = f"{'await ' if is_async else ''}result.{name}()"
            doc_first = (inspect.getdoc(member) or "").splitlines()
            desc = doc_first[0] if doc_first else ""
            rows.append(f"| `{name}()` | `{ret_str}` | `{call_expr}` | {_table_cell(desc)} |")
    if not rows:
        return []
    return [
        "| Member | Type | Access | Description |",
        "| --- | --- | --- | --- |",
        *rows,
    ]


def _render_override_task(
    topic: str,
    task_name: str,
    override_mod: ModuleType,
    snapshot_task: TaskResource | None,
) -> tuple[list[str], str]:
    """Render a task section backed by a custom ``evo.compute.overrides`` runner."""
    topic_attr = _normalise(topic)
    task_attr = _normalise(task_name)
    bind_ret = inspect.signature(override_mod.bind).return_annotation
    runner_name = bind_ret if isinstance(bind_ret, str) else getattr(bind_ret, "__name__", str(bind_ret))
    runner_cls = getattr(override_mod, runner_name)

    run_sig = inspect.signature(runner_cls.run)
    raw_return = run_sig.return_annotation
    return_type = raw_return if isinstance(raw_return, str) else getattr(raw_return, "__name__", "Any")
    intro_doc, param_docs = _extract_param_docs(inspect.getdoc(runner_cls.run))
    description = (snapshot_task.description if snapshot_task and snapshot_task.description else intro_doc) or ""

    wire_topic = snapshot_task.topic if snapshot_task else topic
    wire_task = snapshot_task.name if snapshot_task else task_name
    badges = [
        f"`client.{topic_attr}.{task_attr}`",
        f"Task: `{wire_topic}/{wire_task}`",
        f"Custom runner: `{runner_name}`",
    ]
    if snapshot_task and snapshot_task.version:
        badges.append(f"Version: `{snapshot_task.version}`")
    if snapshot_task and snapshot_task.feature_flag:
        badges.append("Preview (`preview=True` required)")

    lines: list[str] = [
        f"### {task_attr}",
        "",
        " · ".join(badges),
        "",
    ]
    if description:
        lines.extend([_sanitize_prose(description), ""])

    param_entries: list[tuple[str, str, Any]] = []
    for pname, param in run_sig.parameters.items():
        if pname == "self":
            continue
        raw_ann = (
            param.annotation
            if isinstance(param.annotation, str)
            else getattr(param.annotation, "__name__", str(param.annotation))
        )
        clean_ann = _expand_private_alias(raw_ann, override_mod)
        param_entries.append((pname, clean_ann, param.default))

    lines.append("```python")
    lines.append(f"await client.{topic_attr}.{task_attr}.run(")
    lines.append("    *,")
    for pname, clean_ann, default in param_entries:
        suffix = "" if default is inspect.Parameter.empty else f" = {default!r}"
        lines.append(f"    {pname}: {clean_ann}{suffix},")
    lines.append(f") -> {return_type}")
    if hasattr(runner_cls, "submit"):
        lines.append("")
        lines.append(f"await client.{topic_attr}.{task_attr}.submit(")
        lines.append("    *,")
        for pname, clean_ann, default in param_entries:
            suffix = "" if default is inspect.Parameter.empty else f" = {default!r}"
            lines.append(f"    {pname}: {clean_ann}{suffix},")
        lines.append(f") -> TaskJob[{return_type}]")
    lines.append("```")
    lines.append("")

    lines.extend(
        [
            "Parameters:",
            "",
            "| Name | Type | Description | Default |",
            "| --- | --- | --- | --- |",
        ]
    )
    param_schema = (snapshot_task.parameters if snapshot_task else None) or {}
    schema_props = param_schema.get("properties") or {}
    for pname, clean_ann, default in param_entries:
        default_cell = "_required_" if default is inspect.Parameter.empty else f"`{default!r}`"
        desc = param_docs.get(pname, "")
        if pname in schema_props and isinstance(schema_props[pname], dict):
            notes = _schema_notes(schema_props[pname], param_schema)
            if notes:
                desc = " ".join(part for part in (desc, *notes) if part)
        lines.append(f"| `{pname}` | `{clean_ann}` | {_table_cell(desc)} | {default_cell} |")
    lines.append("")

    lines.append(f"Results (`{return_type}`):")
    lines.append("")
    result_cls = getattr(override_mod, return_type, None)
    member_table = _render_override_result_members(result_cls) if isinstance(result_cls, type) else []
    if member_table:
        lines.extend(member_table)
    else:
        lines.append(f"Returns `{return_type}`.")
    lines.append("")

    type_blob = "\n".join(clean_ann for _, clean_ann, _ in param_entries)
    return lines, type_blob


def render_topic_docs(snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR) -> dict[str, str]:
    """Render ``{topic_slug: markdown}`` reference pages from ``snapshot_dir`` and registered overrides."""
    tasks = load_snapshot(snapshot_dir)
    by_topic: dict[str, dict[str, TaskResource]] = {}
    topic_wire_names: dict[str, str] = {}
    for task in tasks:
        t_norm = _normalise(task.topic)
        topic_wire_names.setdefault(t_norm, task.topic)
        by_topic.setdefault(t_norm, {})[_normalise(task.name)] = task

    for ov_topic, ov_task in overridden_tasks():
        topic_wire_names.setdefault(ov_topic, ov_topic.replace("_", "-"))
        by_topic.setdefault(ov_topic, {}).setdefault(
            ov_task,
            TaskResource(topic=topic_wire_names[ov_topic], name=ov_task.replace("_", "-")),
        )

    pages: dict[str, str] = {}
    for topic_attr, task_map in sorted(by_topic.items()):
        topic = topic_wire_names[topic_attr]
        rendered_sections: list[str] = []
        blobs: list[str] = []
        for task_attr, task in sorted(task_map.items()):
            if (override_mod := load_override(topic_attr, task_attr)) is not None:
                section_lines, blob = _render_override_task(
                    topic, task_attr, override_mod, task if task.parameters else None
                )
            else:
                section_lines, blob = _render_generic_task(task)
            rendered_sections.extend(section_lines)
            blobs.append(blob)

        header = [
            f"[GitHub source]({GITHUB_SNAPSHOT_BASE_URL}/{topic})",
            "",
            f"# {_topic_title(topic)}",
            "",
            f"Discoverable compute tasks in the `{topic}` topic. Call any task below through `ComputeClient` as "
            f'`await client.{topic_attr}.<task>.run(...)`, or by name via `await client.arun("{topic}", "<task>", parameters)`.',
            "",
            *_render_shorthands_section("\n".join(blobs)),
            "## Tasks",
            "",
        ]
        pages[topic] = "\n".join([*header, *rendered_sections]).rstrip() + "\n"
    return pages

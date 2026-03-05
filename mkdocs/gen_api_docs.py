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

import importlib
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

GITHUB_BASE_URL = "https://github.com/SeequentEvo/evo-python-sdk/blob/main"

log = logging.getLogger("mkdocs.gen_api_docs")


class DocEntry(NamedTuple):
    class_name: str
    namespace: str  # Python import path for mkdocstrings
    github_url: str


def _read_lines(file_path: Path) -> list[str]:
    """Read non-empty, stripped lines from a text file."""
    return [line.strip() for line in file_path.read_text().splitlines() if line.strip()]


def _split_dot_path(dot_path: str) -> tuple[str, str, str]:
    """Split a ``packages.<pkg>.src.<rest>`` dot-path into (package, file_path, import_path).

    For example ``packages.evo-objects.src.evo.objects.typed`` returns
    ``("evo-objects", "packages/evo-objects/src/evo/objects/typed", "evo.objects.typed")``.
    """
    parts = dot_path.split(".")
    package = parts[1]
    file_path = "/".join(parts)
    src_idx = parts.index("src")
    import_path = ".".join(parts[src_idx + 1 :])
    return package, file_path, import_path


def _parse_api_client_entries(lines: list[str]) -> dict[str, list[DocEntry]]:
    """Parse ``packages.<pkg>.src.<module>.<ClassName>`` lines into DocEntry dicts keyed by package."""
    entries_by_package: dict[str, list[DocEntry]] = defaultdict(list)
    for dot_path in lines:
        parts = dot_path.rsplit(".", 1)
        class_name = parts[1]
        package, file_path, import_path = _split_dot_path(parts[0])
        github_url = f"{GITHUB_BASE_URL}/{file_path}.py"
        entries_by_package[package].append(DocEntry(class_name, f"{import_path}.{class_name}", github_url))
    return entries_by_package


def _load_api_client_entries(mkdocs_dir: Path) -> dict[str, list[DocEntry]]:
    """Read ``api_clients.txt`` and return parsed entries keyed by package."""
    api_clients_file = mkdocs_dir / "api_clients.txt"
    lines = _read_lines(api_clients_file)
    log.info(f"Loaded {len(lines)} API clients from {api_clients_file.name}")
    return _parse_api_client_entries(lines)


def _load_typed_modules(mkdocs_dir: Path) -> list[tuple[str, str, str]]:
    """Read ``typed_modules.txt`` and return (package, source_dir, import_prefix) tuples."""
    typed_modules_file = mkdocs_dir / "typed_modules.txt"
    lines = _read_lines(typed_modules_file)
    modules = [_split_dot_path(line) for line in lines]
    log.info(f"Loaded {len(modules)} typed module paths from {typed_modules_file.name}")
    return modules


def _discover_typed_entries(
    repo_root: Path,
    typed_modules: list[tuple[str, str, str]],
) -> dict[str, dict[str, list[DocEntry]]]:
    """Walk typed module directories and return ``{package: {subfolder: [DocEntry]}}``."""
    for _, src_dir, _ in typed_modules:
        src_root = str(repo_root / src_dir.split("/src/")[0] / "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)

    result: dict[str, dict[str, list[DocEntry]]] = defaultdict(lambda: defaultdict(list))
    for package, src_dir, import_prefix in typed_modules:
        abs_dir = repo_root / src_dir
        if not abs_dir.is_dir():
            log.warning(f"Typed source dir not found: {abs_dir}")
            continue
        for py_file in sorted(abs_dir.glob("*.py")):
            entries = _entries_from_file(py_file, src_dir, import_prefix)
            if entries:
                subfolder = py_file.stem.replace("_", "-")
                result[package][subfolder].extend(entries)
    return result


def _entries_from_file(
    py_file: Path,
    src_dir: str,
    import_prefix: str,
) -> list[DocEntry]:
    """Import a single source file and return a DocEntry per ``__all__`` symbol."""
    if py_file.stem.startswith("_"):
        return []

    module_name = f"{import_prefix}.{py_file.stem}"
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        log.warning(f"Could not import {module_name}: {exc}")
        return []

    all_names: list[str] | None = getattr(mod, "__all__", None)
    if not all_names:
        return []

    github_url = f"{GITHUB_BASE_URL}/{src_dir}/{py_file.name}"
    return [DocEntry(name, f"{module_name}.{name}", github_url) for name in all_names]


def _generate_doc(doc_path: Path, entry: DocEntry, mkdocs_dir: Path, *, show_labels: bool = True) -> None:
    """Write a single auto-generated doc file (GitHub link + mkdocstrings directive)."""
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    directive = f"::: {entry.namespace}\n"
    if not show_labels:
        directive = f"::: {entry.namespace}\n    options:\n      show_labels: false\n"
    doc_path.write_text(f"[GitHub source]({entry.github_url})\n{directive}")
    log.info(f"Generated doc: {doc_path.relative_to(mkdocs_dir)}")


_PRESERVED_FILENAMES = frozenset({"evo-python-sdk.md", "index.md"})


def _doc_path_for_entry(
    docs_packages_dir: Path,
    package: str,
    entry: DocEntry,
    subfolder: str | None = None,
) -> Path:
    """Return the output ``.md`` path for a single DocEntry."""
    if subfolder is None:
        return docs_packages_dir / package / f"{entry.class_name}.md"
    return docs_packages_dir / package / "typed-objects" / subfolder / f"{entry.class_name}.md"


def _collect_auto_generated_paths(
    docs_packages_dir: Path,
    api_entries: dict[str, list[DocEntry]],
    discovered: dict[str, dict[str, list[DocEntry]]],
) -> set[Path]:
    """Build the full set of expected auto-generated doc paths."""
    paths: set[Path] = set()
    for package, entries in api_entries.items():
        for entry in entries:
            paths.add(_doc_path_for_entry(docs_packages_dir, package, entry).resolve())
    for package, groups in discovered.items():
        for subfolder, entries in groups.items():
            for entry in entries:
                paths.add(_doc_path_for_entry(docs_packages_dir, package, entry, subfolder).resolve())
    return paths


def _clean_stale_docs(
    docs_packages_dir: Path,
    auto_generated_paths: set[Path],
    mkdocs_dir: Path,
) -> None:
    """Delete previously auto-generated docs so removed entries don't linger."""
    for old_md in docs_packages_dir.rglob("*.md"):
        if old_md.name in _PRESERVED_FILENAMES:
            continue
        if old_md.resolve() in auto_generated_paths:
            old_md.unlink()
            log.info(f"Deleted auto-generated doc: {old_md.relative_to(mkdocs_dir)}")
        else:
            log.info(f"Preserved manual doc: {old_md.relative_to(mkdocs_dir)}")


def _generate_all_docs(
    docs_packages_dir: Path,
    api_entries: dict[str, list[DocEntry]],
    discovered: dict[str, dict[str, list[DocEntry]]],
    mkdocs_dir: Path,
) -> None:
    """Write doc files for every API-client and auto-discovered entry."""
    for package, entries in api_entries.items():
        for entry in entries:
            _generate_doc(_doc_path_for_entry(docs_packages_dir, package, entry), entry, mkdocs_dir)
    for package, groups in discovered.items():
        for subfolder, entries in groups.items():
            for entry in entries:
                _generate_doc(
                    _doc_path_for_entry(docs_packages_dir, package, entry, subfolder),
                    entry,
                    mkdocs_dir,
                    show_labels=False,
                )


def on_startup(command: str, dirty: bool) -> None:
    mkdocs_dir = Path(__file__).parent
    repo_root = mkdocs_dir.parent
    docs_packages_dir = mkdocs_dir / "docs" / "packages"

    api_entries = _load_api_client_entries(mkdocs_dir)

    typed_modules = _load_typed_modules(mkdocs_dir)
    discovered = _discover_typed_entries(repo_root, typed_modules)
    total = sum(len(e) for groups in discovered.values() for e in groups.values())
    log.info(f"Auto-discovered {total} typed objects across {list(discovered.keys())}")

    auto_generated_paths = _collect_auto_generated_paths(docs_packages_dir, api_entries, discovered)
    _clean_stale_docs(docs_packages_dir, auto_generated_paths, mkdocs_dir)
    _generate_all_docs(docs_packages_dir, api_entries, discovered, mkdocs_dir)

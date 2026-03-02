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

import logging
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

GITHUB_BASE_URL = "https://github.com/SeequentEvo/evo-python-sdk/blob/main"

log = logging.getLogger("mkdocs.gen_api_docs")


class DocEntry(NamedTuple):
    class_name: str
    namespace: str
    github_url: str


def _parse_entries(lines: list[str]) -> dict[str, list[DocEntry]]:
    """Parse dotted module paths into :class:`DocEntry` instances grouped by package name.

    Works for both ``api_clients.txt`` and ``typed_objects.txt``.  Each line has the form
    ``packages.<package>.src.<module...>.<ClassName>`` and is resolved to a namespace
    (Python import path) and a GitHub source URL.
    """
    entries_by_package: dict[str, list[DocEntry]] = defaultdict(list)
    for module_path in lines:
        parts = module_path.split(".")
        package = parts[1]
        class_name = parts[-1]

        file_path = "/".join(parts[:-1]) + ".py"
        github_url = f"{GITHUB_BASE_URL}/{file_path}"

        src_idx = parts.index("src")
        namespace = ".".join(parts[src_idx + 1 :])

        entries_by_package[package].append(DocEntry(class_name, namespace, github_url))
    return entries_by_package


def on_startup(command: str, dirty: bool) -> None:
    mkdocs_dir = Path(__file__).parent
    docs_packages_dir = mkdocs_dir / "docs" / "packages"

    # --- Load API clients ---
    api_clients_file = mkdocs_dir / "api_clients.txt"
    api_clients = [line.strip() for line in api_clients_file.read_text().splitlines() if line.strip()]
    log.info(f"Loaded {len(api_clients)} API clients from {api_clients_file.relative_to(mkdocs_dir)}")
    api_entries = _parse_entries(api_clients)

    # --- Load typed objects ---
    typed_objects_file = mkdocs_dir / "typed_objects.txt"
    typed_objects: list[str] = []
    if typed_objects_file.exists():
        typed_objects = [line.strip() for line in typed_objects_file.read_text().splitlines() if line.strip()]
        log.info(f"Loaded {len(typed_objects)} typed objects from {typed_objects_file.relative_to(mkdocs_dir)}")
    typed_entries = _parse_entries(typed_objects)

    # --- Compute all auto-generated paths ---
    auto_generated_paths: set[Path] = set()

    for package, entries in api_entries.items():
        for entry in entries:
            doc_path = docs_packages_dir / f"{package}/{entry.class_name}.md"
            auto_generated_paths.add(doc_path.resolve())

    for package in typed_entries:
        doc_path = docs_packages_dir / f"{package}/TypedObjects.md"
        auto_generated_paths.add(doc_path.resolve())

    # --- Clean up only auto-generated files ---
    for old_md in docs_packages_dir.rglob("*.md"):
        if old_md.name in ("evo-python-sdk.md", "index.md"):
            continue
        if old_md.resolve() in auto_generated_paths:
            old_md.unlink()
            log.info(f"Deleted auto-generated doc: {old_md.relative_to(mkdocs_dir)}")
        else:
            log.info(f"Preserved manual doc: {old_md.relative_to(mkdocs_dir)}")

    # --- Generate API client docs ---
    for package, entries in api_entries.items():
        for entry in entries:
            doc_path = docs_packages_dir / f"{package}/{entry.class_name}.md"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"[GitHub source]({entry.github_url})\n::: {entry.namespace}\n")
            log.info(f"Generated API doc: {doc_path.relative_to(mkdocs_dir)}")

    # --- Generate typed object docs ---
    for package, entries in typed_entries.items():
        doc_path = docs_packages_dir / f"{package}/TypedObjects.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Typed Objects\n"]
        for entry in entries:
            lines.append(f"[GitHub source]({entry.github_url})\n")
            lines.append(f"::: {entry.namespace}\n")

        doc_path.write_text("\n".join(lines))
        log.info(f"Generated typed objects doc: {doc_path.relative_to(mkdocs_dir)}")

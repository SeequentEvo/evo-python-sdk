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

GITHUB_BASE_URL = "https://github.com/SeequentEvo/evo-python-sdk/blob/main"

log = logging.getLogger("mkdocs.gen_api_docs")


def _parse_api_entries(lines: list[str]) -> dict[str, list[tuple[str, str, str, str]]]:
    """Parse api_clients.txt lines into (class_name, module_path, github_url, namespace) grouped by doc_dir.

    The doc_dir determines the directory/file structure:
    - evo-sdk-common entries use ``<package>/<sub_package>`` (e.g. ``evo-sdk-common/discovery``)
    - All other entries use the package name (e.g. ``evo-objects``, ``evo-blockmodels``)
    """
    entries_by_dir: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for module_path in lines:
        module_parts = module_path.split(".")
        # packages.<package>.src.<...>.<ClassName>
        package = module_parts[1]  # e.g. "evo-objects", "evo-sdk-common"
        class_name = module_parts[-1]

        if package == "evo-sdk-common":
            # evo-sdk-common uses sub-package directories: evo-sdk-common/discovery, evo-sdk-common/workspaces
            sub_package = module_parts[4]  # packages.evo-sdk-common.src.evo.<sub_package>.*
            doc_dir = f"{package}/{sub_package}"
        else:
            doc_dir = package

        file_path_parts = module_parts[:-1]
        source_file_path = "/".join(file_path_parts) + ".py"
        github_url = f"{GITHUB_BASE_URL}/{source_file_path}"

        src_idx = module_parts.index("src")
        namespace = ".".join(module_parts[src_idx + 1 :])
        entries_by_dir[doc_dir].append((class_name, module_path, github_url, namespace))
    return entries_by_dir


def _parse_typed_entries(lines: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Parse typed_objects.txt lines into (class_name, namespace) grouped by package name.

    All typed object entries for a package are collected into a single TypedObjects.md.
    """
    entries_by_package: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for module_path in lines:
        module_parts = module_path.split(".")
        package = module_parts[1]  # e.g. "evo-objects", "evo-blockmodels"
        class_name = module_parts[-1]

        src_idx = module_parts.index("src")
        namespace = ".".join(module_parts[src_idx + 1 :])
        entries_by_package[package].append((class_name, namespace))
    return entries_by_package


def on_startup(command: str, dirty: bool) -> None:
    mkdocs_dir = Path(__file__).parent
    docs_packages_dir = mkdocs_dir / "docs" / "packages"

    # --- Load API clients ---
    api_clients_file = mkdocs_dir / "api_clients.txt"
    api_clients = [line.strip() for line in api_clients_file.read_text().splitlines() if line.strip()]
    log.info(f"Loaded {len(api_clients)} API clients from {api_clients_file.relative_to(mkdocs_dir)}")
    api_entries = _parse_api_entries(api_clients)

    # --- Load typed objects ---
    typed_objects_file = mkdocs_dir / "typed_objects.txt"
    typed_objects: list[str] = []
    if typed_objects_file.exists():
        typed_objects = [line.strip() for line in typed_objects_file.read_text().splitlines() if line.strip()]
        log.info(f"Loaded {len(typed_objects)} typed objects from {typed_objects_file.relative_to(mkdocs_dir)}")
    typed_entries = _parse_typed_entries(typed_objects)

    # --- Compute all auto-generated paths ---
    auto_generated_paths: set[Path] = set()

    # API client docs: always placed inside package directories as <ClassName>.md
    for doc_dir, entries in api_entries.items():
        for class_name, *_ in entries:
            doc_path = docs_packages_dir / f"{doc_dir}/{class_name}.md"
            auto_generated_paths.add(doc_path.resolve())

    # Typed object docs: one TypedObjects.md per package directory
    for package in typed_entries:
        doc_path = docs_packages_dir / f"{package}/TypedObjects.md"
        auto_generated_paths.add(doc_path.resolve())

    # --- Clean up only auto-generated files ---
    for old_md in docs_packages_dir.rglob("*.md"):
        if old_md.name == "evo-python-sdk.md":
            continue
        if old_md.resolve() in auto_generated_paths:
            old_md.unlink()
            log.info(f"Deleted auto-generated doc: {old_md.relative_to(mkdocs_dir)}")
        else:
            log.info(f"Preserved manual doc: {old_md.relative_to(mkdocs_dir)}")

    # --- Generate API client docs ---
    for doc_dir, entries in api_entries.items():
        for class_name, module_path, github_url, namespace in entries:
            doc_path = docs_packages_dir / f"{doc_dir}/{class_name}.md"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(f"[GitHub source]({github_url})\n::: {namespace}\n")
            log.info(f"Generated API doc: {doc_path.relative_to(mkdocs_dir)}")

    # --- Generate typed object docs ---
    for package, entries in typed_entries.items():
        doc_path = docs_packages_dir / f"{package}/TypedObjects.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Typed Objects\n"]
        for class_name, namespace in entries:
            lines.append(f"::: {namespace}")
            lines.append("    options:")
            lines.append("      show_root_heading: true")
            lines.append("      show_source: false")
            lines.append("")

        doc_path.write_text("\n".join(lines))
        log.info(f"Generated typed objects doc: {doc_path.relative_to(mkdocs_dir)}")

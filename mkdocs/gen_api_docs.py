#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ast
import logging
from pathlib import Path

log = logging.getLogger("mkdocs.gen_api_docs")


def on_startup(command: str, dirty: bool) -> None:
    mkdocs_dir = Path(__file__).parent
    docs_packages_dir = mkdocs_dir / "docs" / "packages"

    api_clients_file = mkdocs_dir / "api_clients.txt"
    api_clients = [line.strip() for line in api_clients_file.read_text().splitlines() if line.strip()]
    log.info(f"Loaded {len(api_clients)} API clients from {api_clients_file.relative_to(mkdocs_dir)}")

    for old_md in docs_packages_dir.rglob("*.md"):
        old_md.unlink()

    for module_path in api_clients:
        parts = module_path.split(".")
        package, sub_package, class_name = parts[1], parts[4], parts[-1]
        package_dir_name = f"{package}/{sub_package}" if package == "evo-sdk-common" else package

        package_dir = docs_packages_dir / package_dir_name
        package_dir.mkdir(parents=True, exist_ok=True)

        source_path = "/".join(parts[:-1]).replace(".", "/") + ".py"
        source = (mkdocs_dir.parent / source_path).read_text()
        methods = _get_class_methods(source, class_name)

        for method in methods:
            (package_dir / f"{method}.md").write_text(f"::: {module_path}.{method}\n")
        log.info(f"Generated {len(methods)} method pages for {class_name}")


def _get_class_methods(source: str, class_name: str) -> list[str]:
    """Extract public method names from a class, including __init__."""
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = item.name
                    if name == "__init__" or not name.startswith("_"):
                        methods.append(name)
            return sorted(methods)
    return []

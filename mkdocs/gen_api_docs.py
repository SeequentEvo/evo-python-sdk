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

import logging
from pathlib import Path

GITHUB_BASE_URL = "https://github.com/SeequentEvo/evo-python-sdk/blob/main"

log = logging.getLogger("mkdocs.gen_api_docs")


def on_startup(command: str, dirty: bool) -> None:
    mkdocs_dir = Path(__file__).parent
    docs_packages_dir = mkdocs_dir / "docs" / "packages"

    api_clients_file = mkdocs_dir / "api_clients.txt"
    api_clients = [line.strip() for line in api_clients_file.read_text().splitlines() if line.strip()]
    log.info(f"Loaded {len(api_clients)} API clients from {api_clients_file.relative_to(mkdocs_dir)}")

    for old_md in docs_packages_dir.rglob("*.md"):
        if old_md.name != "evo-python-sdk.md":
            old_md.unlink()
            log.info(f"Deleted old doc: {old_md.relative_to(mkdocs_dir)}")

    for module_path in api_clients:
        module_parts = module_path.split(".")
        _, package, _, _, sub_package, *rest = module_parts
        doc_name = f"{package}/{sub_package}" if package == "evo-sdk-common" else package

        file_path_parts = module_parts[:-1]
        source_file_path = "/".join(file_path_parts) + ".py"
        github_url = f"{GITHUB_BASE_URL}/{source_file_path}"

        doc_path = docs_packages_dir / f"{doc_name}.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"[GitHub source]({github_url})\n::: {module_path}\n"

        with doc_path.open("x") as f:
            f.write(content)
        log.info(f"Generated: {doc_path.relative_to(mkdocs_dir)}")

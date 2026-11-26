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

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "mkdocs" / "gen_api_docs.py"

spec = importlib.util.spec_from_file_location("gen_api_docs", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load gen_api_docs from {SCRIPT_PATH}")

gen_api_docs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gen_api_docs
spec.loader.exec_module(gen_api_docs)


def test_clean_stale_docs_removes_obsolete_generated_files_and_preserves_manual_docs(tmp_path: Path) -> None:
    mkdocs_dir = tmp_path / "mkdocs"
    pkg_dir = mkdocs_dir / "docs" / "packages" / "evo-compute"
    stale_dir = pkg_dir / "typed-objects" / "obsolete-task"
    stale_dir.mkdir(parents=True)

    index_md = pkg_dir / "index.md"
    index_md.write_text("# evo-compute\n", encoding="utf-8")

    manual_md = pkg_dir / "guide.md"
    manual_md.write_text("# Custom manual guide\n\nHand-written content.\n", encoding="utf-8")

    stale_md = stale_dir / "RemovedParams.md"
    stale_md.write_text(
        f"[GitHub source]({gen_api_docs.GITHUB_BASE_URL}/packages/evo-compute/src/evo/compute/old.py)\n"
        "::: evo.compute.old.RemovedParams\n",
        encoding="utf-8",
    )

    active_md = pkg_dir / "ComputeClient.md"
    active_md.write_text(
        f"[GitHub source]({gen_api_docs.GITHUB_BASE_URL}/packages/evo-compute/src/evo/compute/engine.py)\n"
        "::: evo.compute.engine.ComputeClient\n",
        encoding="utf-8",
    )

    gen_api_docs._clean_stale_docs(mkdocs_dir / "docs" / "packages", {active_md.resolve()}, mkdocs_dir)

    assert index_md.exists()
    assert manual_md.exists()
    assert not stale_md.exists()
    assert not stale_dir.exists()
    assert not active_md.exists()


def test_discover_typed_entries_supports_nested_subpackages(tmp_path: Path, monkeypatch) -> None:
    src_root = tmp_path / "packages" / "evo-demo" / "src"
    nested_dir = src_root / "evo_demo_pkg" / "overrides" / "geostatistics"
    nested_dir.mkdir(parents=True)
    (src_root / "evo_demo_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "evo_demo_pkg" / "overrides" / "__init__.py").write_text("", encoding="utf-8")
    (nested_dir / "__init__.py").write_text("", encoding="utf-8")
    (nested_dir / "_internal.py").write_text('__all__ = ["Hidden"]\nHidden = object\n', encoding="utf-8")
    (nested_dir / "custom_task.py").write_text(
        '__all__ = ["CustomTaskOverride"]\n\nclass CustomTaskOverride:\n    pass\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        discovered = gen_api_docs._discover_typed_entries(
            tmp_path,
            [("evo-demo", "packages/evo-demo/src/evo_demo_pkg/overrides", "evo_demo_pkg.overrides")],
        )
    finally:
        for mod_name in [name for name in sys.modules if name == "evo_demo_pkg" or name.startswith("evo_demo_pkg.")]:
            sys.modules.pop(mod_name, None)

    assert list(discovered.keys()) == ["evo-demo"]
    assert list(discovered["evo-demo"].keys()) == ["custom-task"]
    entries = discovered["evo-demo"]["custom-task"]
    assert len(entries) == 1
    assert entries[0].class_name == "CustomTaskOverride"
    assert entries[0].namespace == "evo_demo_pkg.overrides.geostatistics.custom_task.CustomTaskOverride"
    assert entries[0].github_url == (
        f"{gen_api_docs.GITHUB_BASE_URL}/packages/evo-demo/src/evo_demo_pkg/overrides/geostatistics/custom_task.py"
    )

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_uv_lock_sources.py"

spec = importlib.util.spec_from_file_location("check_uv_lock_sources", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load validator module from {SCRIPT_PATH}")

validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


def test_repo_uv_lock_files_use_only_default_pypi_sources() -> None:
    lockfiles = validator.find_uv_lock_files(REPO_ROOT)

    assert lockfiles == [REPO_ROOT / "code-samples" / "uv.lock", REPO_ROOT / "uv.lock"]

    for lockfile in lockfiles:
        assert validator.validate_lock_file(lockfile) is True


def test_custom_registry_source_is_rejected(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        """
version = 1

[[package]]
name = "demo"
version = "1.0.0"
source = { registry = "https://packages.example.com/simple" }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert validator.validate_lock_file(lockfile) is False


def test_editable_and_virtual_sources_are_allowed(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        """
version = 1

[[package]]
name = "workspace-member"
version = "0.1.0"
source = { editable = "packages/member" }

[[package]]
name = "virtual-member"
version = "0.1.0"
source = { virtual = "." }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert validator.validate_lock_file(lockfile) is True


def test_unsupported_non_registry_source_is_rejected(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        """
version = 1

[[package]]
name = "demo"
version = "1.0.0"
source = { git = "https://example.com/repo.git" }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert validator.validate_lock_file(lockfile) is False

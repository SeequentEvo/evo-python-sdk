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

import json
import os
from pathlib import Path

# Root directories
CODE_SAMPLES_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_SAMPLES_DIR.parent

# Default location for .env file with credentials
DEFAULT_ENV_FILE = REPO_ROOT / ".github" / "scripts" / ".env"

# ---------------------------------------------------------------------------
# Notebook discovery
# ---------------------------------------------------------------------------

# Directories to scan for notebooks (extend this list as new dirs are added).
NOTEBOOK_DIRS: list[Path] = [
    CODE_SAMPLES_DIR,
]

# Notebooks that can be fully executed without authentication or external API
# access.
EXECUTABLE_NOTEBOOKS: list[str] = [
    "code-samples/common-tasks/working-with-parquet.ipynb",
]

# Notebooks that contain CI auth markers but are NOT fully self-contained for
# CI execution (e.g. require interactive widgets or a browser).
AUTH_EXCLUDE_NOTEBOOKS: list[str] = [
    # Requires user to manually copy a block model UUID from a previous cell's output.
    "code-samples/blockmodels/api-examples.ipynb",
    # Cascading failure: file_info set by an earlier interactive cell.
    "code-samples/files/sdk-examples.ipynb",
    # Intermittent 404: file upload may not propagate before the read-back poll.
    "code-samples/files/api-examples.ipynb",
    # Requires a pre-existing pointset object ID to be set manually before running.
    "code-samples/geoscience-objects/download-pointset/download-pointset.ipynb",
]

# Auth notebooks are auto-detected by scanning code cells for CI-compatible auth
# patterns.
_CI_AUTH_MARKERS = ("_create_ci_manager", "get_manual_auth")


# Directory name patterns to ignore during notebook discovery.
_IGNORE_DIRS = {".venv", ".ipynb_checkpoints", "__pycache__", "node_modules", ".tox"}


def discover_notebooks(root: Path | None = None) -> list[Path]:
    """Return every .ipynb file under *root* (or all configured dirs).

    Skips virtual environments, checkpoint dirs, and other non-source trees.
    """
    roots = [root] if root is not None else NOTEBOOK_DIRS
    notebooks: list[Path] = []
    for directory in roots:
        for nb in sorted(directory.rglob("*.ipynb")):
            if not any(part in _IGNORE_DIRS for part in nb.parts):
                notebooks.append(nb)
    return notebooks


def _relative_posix(notebook_path: Path) -> str | None:
    """Return the POSIX path relative to REPO_ROOT, or None if outside."""
    try:
        return notebook_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def notebook_id(notebook_path: Path) -> str:
    """Generate a readable pytest node-id from a notebook path."""
    return _relative_posix(notebook_path) or str(notebook_path)


def is_executable(notebook_path: Path) -> bool:
    """Return True if the notebook is in the executable allow-list."""
    rel = _relative_posix(notebook_path)
    return rel is not None and rel in EXECUTABLE_NOTEBOOKS


def is_auth_notebook(notebook_path: Path) -> bool:
    """Return True if the notebook requires authentication credentials."""
    if is_executable(notebook_path):
        return False

    rel = _relative_posix(notebook_path)
    if rel is None or rel in AUTH_EXCLUDE_NOTEBOOKS:
        return False

    try:
        nb = json.loads(notebook_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    return any(
        any(marker in "".join(cell.get("source", [])) for marker in _CI_AUTH_MARKERS)
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def load_env_file(env_file: Path | None = None) -> dict[str, str]:
    """Load environment variables from a .env file.

    Returns a dict of the variables found. Does not modify ``os.environ``.
    """
    env_path = env_file or DEFAULT_ENV_FILE
    if not env_path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip()
    return env_vars


_AUTH_ENV_KEYS = ("EVO_CLIENT_ID", "EVO_CLIENT_SECRET", "EVO_ORG_ID", "EVO_HUB_URL")
_AUTH_FIELD_NAMES = ("client_id", "client_secret", "org_id", "hub_url")


def get_auth_credentials(env_file: Path | None = None) -> dict[str, str | None]:
    """Get authentication credentials from environment or .env file.

    Checks ``os.environ`` first (for CI secrets), then falls back to .env file.

    Returns a dict with keys: client_id, client_secret, org_id, hub_url.
    Values are ``None`` if not found.
    """
    file_env = load_env_file(env_file)
    return {
        field: os.environ.get(env_key) or file_env.get(env_key)
        for field, env_key in zip(_AUTH_FIELD_NAMES, _AUTH_ENV_KEYS)
    }

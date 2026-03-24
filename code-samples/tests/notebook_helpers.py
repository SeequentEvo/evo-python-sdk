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
    "code-samples/auth-and-evo-discovery/native-app-token.ipynb",
    "code-samples/workspaces/bonus/move-objects.ipynb",
    # Requires user to manually copy a block model UUID from a previous cell's output.
    "code-samples/blockmodels/api-examples.ipynb",
    # SDK bug: unsupported data type 'large_string' in evo-blockmodels.
    "code-samples/blockmodels/reports.ipynb",
    # Cascading failure: file_info set by an earlier interactive cell.
    "code-samples/files/sdk-examples.ipynb",
    # Tries to upload/download from evo-demo.static.evo.seequent.com which is unreachable.
    "code-samples/geoscience-objects/drilling-campaign/download-a-drilling-campaign/sdk-examples.ipynb",
    "code-samples/geoscience-objects/publish-triangular-mesh/publish-triangular-mesh.ipynb",
    # Requires a previously published object_id that doesn't exist in the CI workspace.
    "code-samples/geoscience-objects/download-pointset/download-pointset.ipynb",
    # Data type mismatch: 'Hole ID' treated as scalar instead of category.
    "code-samples/geoscience-objects/publish-pointset/publish-pointset.ipynb",
    # geosoft is Windows-only; excluded on all platforms since CI matrix covers Windows separately.
    "code-samples/geoscience-objects/publish-regular-2d-grid/publish-regular-2d-grid.ipynb",
]

# Auth notebooks are auto-detected by scanning code cells for CI-compatible auth
# patterns.
_CI_AUTH_MARKERS: tuple[str, ...] = ("_create_ci_manager", "get_manual_auth")


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


def notebook_id(notebook_path: Path) -> str:
    """Generate a readable pytest node-id from a notebook path."""
    try:
        return str(notebook_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(notebook_path)


def is_executable(notebook_path: Path) -> bool:
    """Return True if the notebook is in the executable allow-list."""
    try:
        rel = str(notebook_path.relative_to(REPO_ROOT))
    except ValueError:
        return False
    return rel in EXECUTABLE_NOTEBOOKS


def is_auth_notebook(notebook_path: Path) -> bool:
    """Return True if the notebook requires authentication credentials."""
    if is_executable(notebook_path):
        return False
    try:
        rel = str(notebook_path.relative_to(REPO_ROOT))
    except ValueError:
        return False
    if rel in AUTH_EXCLUDE_NOTEBOOKS:
        return False
    try:
        with open(notebook_path) as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if any(marker in source for marker in _CI_AUTH_MARKERS):
                return True
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return False


def load_env_file(env_file: Path | None = None) -> dict[str, str]:
    """Load environment variables from a .env file.

    Returns a dict of the variables found. Does not modify os.environ.
    """
    env_path = env_file or DEFAULT_ENV_FILE
    env_vars: dict[str, str] = {}

    if not env_path.exists():
        return env_vars

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

    return env_vars


def get_auth_credentials(env_file: Path | None = None) -> dict[str, str | None]:
    """Get authentication credentials from environment or .env file.

    Checks os.environ first (for CI secrets), then falls back to .env file.

    Returns a dict with keys: client_id, client_secret, org_id, hub_url, workspace_id.
    Values are None if not found.
    """
    # Load from .env file as fallback
    file_env = load_env_file(env_file)

    def get_var(name: str) -> str | None:
        return os.environ.get(name) or file_env.get(name)

    return {
        "client_id": get_var("EVO_CLIENT_ID"),
        "client_secret": get_var("EVO_CLIENT_SECRET"),
        "org_id": get_var("EVO_ORG_ID"),
        "hub_url": get_var("EVO_HUB_URL"),
        "workspace_id": get_var("EVO_WORKSPACE_ID"),
    }

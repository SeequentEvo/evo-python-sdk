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

# Notebooks that require authentication credentials (service app with client_id
# and client_secret). These will only run if EVO_CLIENT_ID and EVO_CLIENT_SECRET
# environment variables are set.
# Note: Only include notebooks that are self-contained and can run end-to-end
# without additional manual setup (like workspace selection).
AUTH_NOTEBOOKS: list[str] = [
    "code-samples/workspaces/api-examples.ipynb",
    "code-samples/workspaces/sdk-examples.ipynb",
]


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
    try:
        rel = str(notebook_path.relative_to(REPO_ROOT))
    except ValueError:
        return False
    return rel in AUTH_NOTEBOOKS


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


def has_auth_credentials(env_file: Path | None = None) -> bool:
    """Return True if service app credentials (client_id + client_secret) are available."""
    creds = get_auth_credentials(env_file)
    return bool(creds["client_id"] and creds["client_secret"])

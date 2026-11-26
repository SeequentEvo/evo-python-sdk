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

import ast
import asyncio
import importlib
import sys
from pathlib import Path

import nbformat
import pytest
from notebook_auth_patch import patch_notebook_auth_for_ci
from notebook_helpers import (
    CI_AUTH_NOTEBOOKS,
    CODE_SAMPLES_DIR,
    discover_notebooks,
    get_auth_credentials,
    is_auth_notebook,
    is_executable,
    notebook_id,
    sort_by_dependencies,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUTH_HELPER_DIR = CODE_SAMPLES_DIR

# ---------------------------------------------------------------------------
# Collect all notebooks once so parametrize can use them
# ---------------------------------------------------------------------------
ALL_NOTEBOOKS: list[Path] = discover_notebooks()
EXEC_NOTEBOOKS: list[Path] = [nb for nb in ALL_NOTEBOOKS if is_executable(nb)]
AUTH_NOTEBOOKS: list[Path] = sort_by_dependencies(
    [nb for nb in ALL_NOTEBOOKS if notebook_id(nb) in CI_AUTH_NOTEBOOKS and is_auth_notebook(nb)]
)

# Standard-library top-level module names (Python 3.10+).  We skip these
# during the import check because they are always available and never need
# to be installed.
_STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names)

# Extra names to skip – these are provided by the Jupyter/IPython runtime
# or are otherwise unavailable outside a live kernel.
_EXTRA_SKIP: frozenset[str] = frozenset(
    {
        "IPython",
        "ipywidgets",
        "google",  # google.colab – optional, not always present
        "auth_helper",  # Local module in code-samples root
    }
)

_SKIP_IMPORT_CHECK: frozenset[str] = _STDLIB_MODULES | _EXTRA_SKIP

# Platform-conditional imports: mapping from module name to a predicate
# that returns True when the module is expected to be available.
_PLATFORM_SPECIFIC_IMPORTS: dict[str, tuple[set[str], tuple[int, int] | None]] = {
    # (allowed platforms, max python version exclusive)
    "geosoft": ({"win32"}, (3, 14)),
}


def _get_platform_skip_imports() -> frozenset[str]:
    """Return imports to skip on the current platform/version."""
    current_platform = sys.platform
    current_version = sys.version_info[:2]
    skip = set()
    for mod, (platforms, max_version) in _PLATFORM_SPECIFIC_IMPORTS.items():
        if current_platform not in platforms or (max_version is not None and current_version >= max_version):
            skip.add(mod)
    return frozenset(skip)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_notebook(path: Path) -> nbformat.NotebookNode:
    """Read and return a notebook, raising on malformed JSON."""
    return nbformat.read(str(path), as_version=4)


def _extract_imports(source: str) -> set[str]:
    """Return the set of top-level package names imported by *source*.

    Handles ``import foo`` and ``from foo.bar import baz`` by extracting the
    root package name (``foo``).  Lines that fail to parse (e.g. because of
    top-level ``await``) are silently skipped – we only care about imports.
    """
    modules: set[str] = set()
    for line in source.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _can_import(module_name: str) -> bool:
    """Return True if *module_name* can be imported in the current environment."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS, ids=[notebook_id(nb) for nb in ALL_NOTEBOOKS])
class TestNotebookValidation:
    """Validate every discovered notebook without executing it."""

    def test_valid_syntax(self, notebook_path: Path) -> None:
        """Notebook must be well-formed JSON that conforms to nbformat v4."""
        nb = _read_notebook(notebook_path)
        nbformat.validate(nb)

    def test_imports_resolvable(self, notebook_path: Path) -> None:
        """All imported packages must be installed in the current environment."""
        nb = _read_notebook(notebook_path)

        all_imports: set[str] = set()
        for cell in nb.cells:
            if cell.cell_type == "code":
                all_imports |= _extract_imports(cell.source)

        # Skip standard-library, runtime-only, platform-conditional, and local sibling modules
        local_modules = {p.stem for p in notebook_path.parent.rglob("*.py")}
        to_check = all_imports - _SKIP_IMPORT_CHECK - _get_platform_skip_imports() - local_modules

        missing = sorted(mod for mod in to_check if not _can_import(mod))

        assert not missing, (
            f"The following imports could not be resolved: {', '.join(missing)}. Are all dependencies installed?"
        )


@pytest.mark.parametrize("notebook_path", EXEC_NOTEBOOKS, ids=[notebook_id(nb) for nb in EXEC_NOTEBOOKS])
class TestNotebookExecution:
    """Execute notebooks that do not require authentication."""

    def test_executes_without_error(self, notebook_path: Path) -> None:
        """Notebook must run to completion without raising."""
        from nbclient import NotebookClient

        nb = _read_notebook(notebook_path)
        client = NotebookClient(
            nb,
            timeout=300,
            kernel_name="python3",
        )
        # Run in the notebook's own directory so relative paths resolve.
        client.execute(cwd=str(notebook_path.parent))


@pytest.mark.parametrize("notebook_path", AUTH_NOTEBOOKS, ids=[notebook_id(nb) for nb in AUTH_NOTEBOOKS])
class TestAuthNotebookExecution:
    """Execute notebooks that require authentication credentials.

    These tests only run when EVO_CLIENT_ID and EVO_CLIENT_SECRET are set
    (either in environment or in .github/scripts/.env).
    """

    def test_executes_with_auth(self, notebook_path: Path, ci_workspace: str) -> None:
        """Notebook must run to completion with valid credentials."""
        import os

        from nbclient import NotebookClient

        creds = get_auth_credentials()

        # Map credential fields back to environment variable names
        _FIELD_TO_ENV = {
            "client_id": "EVO_CLIENT_ID",
            "client_secret": "EVO_CLIENT_SECRET",
            "org_id": "EVO_ORG_ID",
            "hub_url": "EVO_HUB_URL",
        }
        env = {
            **os.environ,
            "CI": "true",
            "EVO_WORKSPACE_ID": ci_workspace,
            **{_FIELD_TO_ENV[k]: v for k, v in creds.items() if v},
        }

        nb = _read_notebook(notebook_path)
        nb = patch_notebook_auth_for_ci(
            nb,
            auth_helper_dir=AUTH_HELPER_DIR,
        )
        nb.cells.insert(
            0,
            nbformat.v4.new_code_cell(
                f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
                "from notebook_helpers import reuse_ci_objects\n"
                f"reuse_ci_objects({notebook_path.stem!r})"
            ),
        )

        client = NotebookClient(
            nb,
            timeout=600,
            kernel_name="python3",
        )

        client.execute(cwd=str(notebook_path.parent), env=env)


# ---------------------------------------------------------------------------
# Shared workspace fixture
# ---------------------------------------------------------------------------


def _build_workspace_client(creds: dict):
    """Return (ws_client, org_uuid) built from *creds*, or (None, None) if creds are missing."""
    import os
    from uuid import UUID

    from evo.aio import AioTransport
    from evo.common import APIConnector
    from evo.oauth import ClientCredentialsAuthorizer, EvoScopes, OAuthConnector
    from evo.workspaces import WorkspaceAPIClient

    hub_url = creds.get("hub_url") or os.environ.get("EVO_HUB_URL")
    org_id = creds.get("org_id") or os.environ.get("EVO_ORG_ID")
    client_id = creds.get("client_id") or os.environ.get("EVO_CLIENT_ID")
    client_secret = creds.get("client_secret") or os.environ.get("EVO_CLIENT_SECRET")

    if not all([hub_url, org_id, client_id, client_secret]):
        return None, None

    transport = AioTransport(user_agent="Evo SDK CI/1.0")
    authorizer = ClientCredentialsAuthorizer(
        oauth_connector=OAuthConnector(transport=transport, client_id=client_id, client_secret=client_secret),
        scopes=EvoScopes.all_evo,
    )
    connector = APIConnector(base_url=hub_url, transport=transport, authorizer=authorizer)
    return WorkspaceAPIClient(connector=connector, org_id=UUID(org_id)), UUID(org_id)


CI_WORKSPACE_NAME = "evo-sdk-notebook-tests-ci"


async def _get_or_create_ci_workspace(creds: dict) -> str | None:
    """Return the ID of the shared CI workspace, creating it only if none exists.

    The workspace is reused across runs and never deleted, so CI does not leave a
    soft-deleted workspace behind on every run.
    """
    ws_client, _ = _build_workspace_client(creds)
    if ws_client is None:
        return None

    existing = [
        ws
        for ws in await ws_client.list_all_workspaces(name=CI_WORKSPACE_NAME, deleted=False)
        if ws.display_name == CI_WORKSPACE_NAME
    ]
    if existing:
        # If concurrent runs ever created duplicates, always pick the same one.
        workspace = min(existing, key=lambda ws: (ws.created_at, str(ws.id)))
        print(f"\n[workspace] Reusing workspace: {workspace.id}")
    else:
        workspace = await ws_client.create_workspace(name=CI_WORKSPACE_NAME)
        print(f"\n[workspace] Created workspace: {workspace.id}")
    return str(workspace.id)


@pytest.fixture(scope="session")
def ci_workspace():
    """Provide the shared CI workspace to notebooks that need authentication."""
    import os

    run_auth = os.environ.get("EVO_RUN_AUTH_NOTEBOOKS")
    if run_auth == "false":
        pytest.skip("Authenticated notebooks run in the designated CI job only.")

    workspace_id = asyncio.run(_get_or_create_ci_workspace(get_auth_credentials()))
    if workspace_id is None:
        if run_auth == "true":
            pytest.fail("Missing Evo credentials for the authenticated notebook CI job.")
        pytest.skip("Missing Evo credentials for authenticated notebooks.")

    return workspace_id

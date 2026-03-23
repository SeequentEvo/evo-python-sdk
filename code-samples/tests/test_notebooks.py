"""Automated tests for sample Jupyter notebooks.

Three levels of validation are applied:

1. **Syntax** – every notebook is valid JSON conforming to the nbformat schema.
2. **Imports** – every ``import`` / ``from … import`` found in code cells
   resolves to an installed package (catches stale or misspelled deps).
3. **Execution** – notebooks that need no authentication are executed
   end-to-end via *nbclient* and must complete without errors.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import nbformat
import pytest
from notebook_helpers import (
    discover_notebooks,
    is_executable,
    notebook_id,
)

# ---------------------------------------------------------------------------
# Collect all notebooks once so parametrize can use them
# ---------------------------------------------------------------------------
ALL_NOTEBOOKS: list[Path] = discover_notebooks()
EXEC_NOTEBOOKS: list[Path] = [nb for nb in ALL_NOTEBOOKS if is_executable(nb)]

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
    }
)

_SKIP_IMPORT_CHECK: frozenset[str] = _STDLIB_MODULES | _EXTRA_SKIP

# Platform-conditional imports: mapping from module name to the platform(s)
# where it IS available. On other platforms, the import check is skipped.
_PLATFORM_SPECIFIC_IMPORTS: dict[str, set[str]] = {
    "geosoft": {"win32"},
}


def _get_platform_skip_imports() -> frozenset[str]:
    """Return imports to skip on the current platform."""
    current_platform = sys.platform
    return frozenset(mod for mod, platforms in _PLATFORM_SPECIFIC_IMPORTS.items() if current_platform not in platforms)


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
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module.split(".")[0])
    return modules


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

        # Remove standard-library and runtime-only modules
        to_check = all_imports - _SKIP_IMPORT_CHECK

        # Skip local sibling .py modules that live next to the notebook
        local_modules = {p.stem for p in notebook_path.parent.rglob("*.py")}
        to_check -= local_modules

        # Skip platform-conditional packages that aren't available on this OS
        to_check -= _get_platform_skip_imports()

        missing: list[str] = []
        for mod in sorted(to_check):
            try:
                importlib.import_module(mod)
            except ImportError:
                missing.append(mod)

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

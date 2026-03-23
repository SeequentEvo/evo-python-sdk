"""Notebook discovery and classification helpers."""

from __future__ import annotations

from pathlib import Path

# Root directories
CODE_SAMPLES_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_SAMPLES_DIR.parent

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

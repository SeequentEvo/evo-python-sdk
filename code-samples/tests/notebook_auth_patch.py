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

import copy
import re
from pathlib import Path

import nbformat


class NotebookAuthPatchError(RuntimeError):
    """Raised when a notebook auth cell cannot be patched safely."""


_AUTH_CALL_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]*)"
    r"(?P<target>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*=\s*await\s+ServiceManagerWidget\.with_auth_code\s*\("
)


def _ci_auth_source(*, auth_helper_dir: Path, target: str) -> str:
    """Return CI auth source code that creates the same manager variable."""
    auth_helper_dir = auth_helper_dir.resolve()

    return f"""import sys

sys.path.insert(0, {str(auth_helper_dir)!r})
from auth_helper import _create_ci_manager

{target} = await _create_ci_manager()
"""


def _find_matching_paren(source: str, open_paren_index: int) -> int:
    """Find the matching closing parenthesis for the opening parenthesis at open_paren_index."""
    depth = 0
    quote: str | None = None
    triple_quote = False
    escape = False
    i = open_paren_index

    while i < len(source):
        char = source[i]

        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif triple_quote and source.startswith(quote * 3, i):
                quote = None
                triple_quote = False
                i += 2
            elif not triple_quote and char == quote:
                quote = None

        else:
            if char in {"'", '"'}:
                if source.startswith(char * 3, i):
                    quote = char
                    triple_quote = True
                    i += 2
                else:
                    quote = char
                    triple_quote = False

            elif char == "#":
                newline_index = source.find("\n", i)
                if newline_index == -1:
                    return -1
                i = newline_index

            elif char == "(":
                depth += 1

            elif char == ")":
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    return -1


def _consume_login_call(source: str, start_index: int) -> int:
    """
    Consume the trailing .login(...) call after ServiceManagerWidget.with_auth_code(...).

    start_index should be the index immediately after the closing parenthesis of
    ServiceManagerWidget.with_auth_code(...).

    Returns the index immediately after the .login(...) call.
    """
    i = start_index

    while i < len(source) and source[i] in " \t\r\n":
        i += 1

    if not source.startswith(".login", i):
        raise NotebookAuthPatchError(
            "Found ServiceManagerWidget.with_auth_code(...) but it was not followed by .login()."
        )

    i += len(".login")

    while i < len(source) and source[i] in " \t":
        i += 1

    if i >= len(source) or source[i] != "(":
        raise NotebookAuthPatchError("Found .login but not .login(...).")

    close_paren_index = _find_matching_paren(source, i)

    if close_paren_index == -1:
        raise NotebookAuthPatchError("Could not find the closing parenthesis for .login(...).")

    end_index = close_paren_index + 1

    while end_index < len(source) and source[end_index] in " \t":
        end_index += 1

    if end_index < len(source) and source[end_index] == "\n":
        end_index += 1

    return end_index


def _replace_interactive_auth_call(source: str, *, auth_helper_dir: Path) -> tuple[str, bool]:
    """
    Replace the usual interactive notebook auth call with CI auth.

    Replaces:

        manager = await ServiceManagerWidget.with_auth_code(
            redirect_url=redirect_url,
            client_id=client_id,
        ).login()

    with:

        import sys

        sys.path.insert(0, "/abs/path/to/code-samples")
        from auth_helper import _create_ci_manager

        manager = await _create_ci_manager()

    The assignment target is preserved. For example, if the notebook used
    `service_manager = ...`, the replacement will use `service_manager = ...`.
    """
    match = _AUTH_CALL_PATTERN.search(source)

    if match is None:
        return source, False

    target = match.group("target")
    open_paren_index = match.end() - 1

    close_paren_index = _find_matching_paren(source, open_paren_index)

    if close_paren_index == -1:
        raise NotebookAuthPatchError(
            "Could not find the closing parenthesis for ServiceManagerWidget.with_auth_code(...)."
        )

    end_index = _consume_login_call(source, close_paren_index + 1)

    replacement = _ci_auth_source(auth_helper_dir=auth_helper_dir, target=target)

    patched_source = source[: match.start()] + replacement + source[end_index:]

    return patched_source, True


def _remove_service_manager_widget_import(source: str) -> str:
    """
    Remove ServiceManagerWidget from simple evo.notebooks import lines.

    Examples:
        from evo.notebooks import ServiceManagerWidget

    becomes removed entirely.

        from evo.notebooks import ServiceManagerWidget, SomethingElse

    becomes:
        from evo.notebooks import SomethingElse
    """
    lines = source.splitlines(keepends=True)
    output: list[str] = []

    for line in lines:
        match = re.match(r"^(\s*)from\s+evo\.notebooks\s+import\s+(.+?)(\s*(#.*)?\n?)$", line)

        if not match:
            output.append(line)
            continue

        indent = match.group(1)
        imports = match.group(2)
        suffix = match.group(3)

        names = [name.strip() for name in imports.split(",")]
        names = [name for name in names if name and name != "ServiceManagerWidget"]

        if names:
            output.append(f"{indent}from evo.notebooks import {', '.join(names)}{suffix}")

    return "".join(output)


def _remove_simple_assignment(source: str, variable_name: str) -> str:
    """
    Remove simple one-line assignments to an auth variable.

    Examples:
        client_id = "<your-client-id>"
        redirect_url = "<your-redirect-url>"
        client_id = os.environ.get("EVO_CLIENT_ID", "<your-client-id>")
    """
    pattern = re.compile(rf"(?m)^[ \t]*{re.escape(variable_name)}\s*=\s*.+?(?:\n|$)")
    return pattern.sub("", source)


def _remove_obvious_auth_comments(source: str) -> str:
    """Remove simple comments that only describe interactive auth credentials."""
    patterns = [
        r"(?m)^[ \t]*#\s*Evo app credentials\s*\n",
        r"(?m)^[ \t]*#\s*Enter your client ID and callback URL\.?\s*\n",
    ]

    for pattern in patterns:
        source = re.sub(pattern, "", source)

    return source


def _collapse_excess_blank_lines(source: str) -> str:
    """Collapse excessive blank lines after patching."""
    source = re.sub(r"\n{3,}", "\n\n", source)
    return source.strip() + "\n" if source.strip() else ""


def _clean_same_cell_interactive_auth_artifacts(source: str) -> str:
    """
    Clean up same-cell interactive auth artifacts after replacing the auth call.

    This deliberately only cleans the cell where the auth call was replaced.

    It preserves unrelated setup in the same cell, for example:
        from evo.blockmodels import BlockModelAPIClient
        input_location = "sample-data"

    It removes now-unused interactive auth setup in that same cell, for example:
        from evo.notebooks import ServiceManagerWidget
        client_id = "<your-client-id>"
        redirect_url = "<your-redirect-url>"
    """
    source = _remove_service_manager_widget_import(source)
    source = _remove_simple_assignment(source, "client_id")
    source = _remove_simple_assignment(source, "redirect_url")
    source = _remove_obvious_auth_comments(source)
    source = _collapse_excess_blank_lines(source)

    return source


def patch_notebook_auth_for_ci(
    notebook: nbformat.NotebookNode,
    *,
    auth_helper_dir: Path,
) -> nbformat.NotebookNode:
    """
    Return a patched copy of a notebook where the interactive auth call is replaced
    with CI auth.

    This does not mutate the notebook file on disk.
    """
    patched = copy.deepcopy(notebook)
    replacement_count = 0

    for cell in patched.cells:
        if cell.cell_type != "code":
            continue

        original_source = cell.source or ""

        replaced_source, replaced = _replace_interactive_auth_call(
            original_source,
            auth_helper_dir=auth_helper_dir,
        )

        if not replaced:
            continue

        replacement_count += 1
        cell.source = _clean_same_cell_interactive_auth_artifacts(replaced_source)

    if replacement_count == 0:
        raise NotebookAuthPatchError("No interactive ServiceManagerWidget auth call found to replace.")

    if replacement_count > 1:
        raise NotebookAuthPatchError(f"Expected to replace exactly one auth call, but replaced {replacement_count}.")

    return patched

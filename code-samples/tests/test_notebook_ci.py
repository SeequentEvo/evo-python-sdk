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

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import nbclient
import nbformat
import pytest
import test_notebooks
from notebook_auth_patch import patch_notebook_auth_for_ci
from notebook_helpers import CI_AUTH_NOTEBOOKS, CODE_SAMPLES_DIR, notebook_id, reuse_ci_objects

from evo.common.exceptions import ForbiddenException, NotFoundException
from evo.objects import ObjectAPIClient


def test_selected_auth_notebooks_are_discovered_and_patchable():
    notebooks = test_notebooks.AUTH_NOTEBOOKS
    ids = [notebook_id(notebook) for notebook in notebooks]
    assert set(ids) == CI_AUTH_NOTEBOOKS
    assert "publish-pointset" in ids[0]
    assert "download-pointset" in ids[1]

    for path in notebooks:
        patched = patch_notebook_auth_for_ci(nbformat.read(path, as_version=4), auth_helper_dir=CODE_SAMPLES_DIR)
        code = "\n".join(cell.source for cell in patched.cells if cell.cell_type == "code")
        assert "from auth_helper import _create_ci_manager" in code
        assert repr(str(CODE_SAMPLES_DIR)) in code
        assert "ServiceManagerWidget.with_auth_code" not in code


def _workspace(name, ident, day):
    return SimpleNamespace(
        display_name=name,
        id=UUID(int=ident),
        created_at=datetime(2026, 1, day, tzinfo=timezone.utc),
    )


def test_reuses_oldest_exact_match(monkeypatch):
    name = test_notebooks.CI_WORKSPACE_NAME
    oldest = _workspace(name, 1, 1)
    newer = _workspace(name, 2, 2)
    other = _workspace(name + "-other", 3, 1)
    client = SimpleNamespace(
        list_all_workspaces=AsyncMock(return_value=[newer, other, oldest]),
        create_workspace=AsyncMock(),
    )
    monkeypatch.setattr(test_notebooks, "_build_workspace_client", lambda creds: (client, None))

    assert asyncio.run(test_notebooks._get_or_create_ci_workspace({})) == str(oldest.id)
    client.list_all_workspaces.assert_awaited_once_with(name=name, deleted=False)
    client.create_workspace.assert_not_awaited()


@pytest.mark.parametrize("existing", [[], [_workspace(test_notebooks.CI_WORKSPACE_NAME + "-other", 2, 1)]])
def test_creates_workspace_when_no_exact_match(monkeypatch, existing):
    name = test_notebooks.CI_WORKSPACE_NAME
    created = _workspace(name, 1, 1)
    client = SimpleNamespace(
        list_all_workspaces=AsyncMock(return_value=existing),
        create_workspace=AsyncMock(return_value=created),
    )
    monkeypatch.setattr(test_notebooks, "_build_workspace_client", lambda creds: (client, None))

    assert asyncio.run(test_notebooks._get_or_create_ci_workspace({})) == str(created.id)
    client.create_workspace.assert_awaited_once_with(name=name)


def test_selected_ci_job_fails_when_credentials_are_missing(monkeypatch):
    monkeypatch.setenv("EVO_RUN_AUTH_NOTEBOOKS", "true")
    monkeypatch.setattr(test_notebooks, "get_auth_credentials", lambda: {})
    monkeypatch.setattr(test_notebooks, "_build_workspace_client", lambda creds: (None, None))

    with pytest.raises(pytest.fail.Exception, match="Missing Evo credentials"):
        next(test_notebooks.ci_workspace.__wrapped__())


def test_other_matrix_jobs_skip_without_accessing_workspace(monkeypatch):
    monkeypatch.setenv("EVO_RUN_AUTH_NOTEBOOKS", "false")
    get_credentials = AsyncMock()
    monkeypatch.setattr(test_notebooks, "get_auth_credentials", get_credentials)

    with pytest.raises(pytest.skip.Exception):
        next(test_notebooks.ci_workspace.__wrapped__())
    get_credentials.assert_not_called()


def test_auth_execution_passes_workspace_to_notebook(monkeypatch):
    client = Mock()
    monkeypatch.setattr(nbclient, "NotebookClient", Mock(return_value=client))
    monkeypatch.setattr(
        test_notebooks,
        "get_auth_credentials",
        lambda: {"client_id": "id", "client_secret": "secret", "org_id": "org", "hub_url": "hub"},
    )
    path = test_notebooks.AUTH_NOTEBOOKS[0]

    test_notebooks.TestAuthNotebookExecution().test_executes_with_auth(path, "shared")

    assert client.execute.call_args.kwargs["env"]["EVO_WORKSPACE_ID"] == "shared"
    setup = nbclient.NotebookClient.call_args.args[0].cells[0].source
    assert f"reuse_ci_objects({path.stem!r})" in setup


def _publish_in_new_kernel(monkeypatch, create, client):
    monkeypatch.setattr(ObjectAPIClient, "create_geoscience_object", create)
    reuse_ci_objects("sample")
    return asyncio.run(ObjectAPIClient.create_geoscience_object(client, "a/Name_1.json", {"uuid": None}))


def test_reuse_ci_objects_creates_then_updates_same_path(monkeypatch):
    metadata = SimpleNamespace(id=UUID(int=1))
    updated = SimpleNamespace(metadata=SimpleNamespace(id=metadata.id))
    existing = SimpleNamespace(metadata=metadata, update=AsyncMock(return_value=updated))
    create = AsyncMock(return_value=metadata)
    client = SimpleNamespace(
        download_object_by_path=AsyncMock(side_effect=[NotFoundException(404, "Not Found", None, None), existing])
    )

    assert _publish_in_new_kernel(monkeypatch, create, client) is metadata
    assert _publish_in_new_kernel(monkeypatch, create, client) is updated.metadata

    assert [call.args[0] for call in client.download_object_by_path.await_args_list] == [
        "notebook-ci/sample/1.json",
        "notebook-ci/sample/1.json",
    ]
    create.assert_awaited_once_with(client, "notebook-ci/sample/1.json", {"uuid": None})
    existing.update.assert_awaited_once_with({"uuid": metadata.id})


def test_reuse_ci_objects_propagates_lookup_errors(monkeypatch):
    create = AsyncMock()
    client = SimpleNamespace(
        download_object_by_path=AsyncMock(side_effect=ForbiddenException(403, "Forbidden", None, None))
    )

    with pytest.raises(ForbiddenException):
        _publish_in_new_kernel(monkeypatch, create, client)
    create.assert_not_awaited()

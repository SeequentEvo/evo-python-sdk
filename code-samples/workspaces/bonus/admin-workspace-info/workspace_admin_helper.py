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

"""Admin helper for aggregating workspace information across an Evo instance.

This module provides the :class:`WorkspaceAdminHelper` class, which uses the
Workspaces **admin** endpoints together with the File and Geoscience Object API
clients to produce comprehensive workspace reports.  All workspace and user
operations go through the admin API paths (``/workspace/admin/orgs/{org_id}/…``)
so that instance admins can access *any* workspace regardless of their own
workspace-level role.

.. note::
    The authenticated user **must** be an instance admin.  Call
    :meth:`WorkspaceAdminHelper.check_admin_rights` immediately after
    construction to verify this before running any other methods.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

import ipywidgets as widgets
from IPython.display import display

from evo.common import APIConnector
from evo.common.exceptions import ForbiddenException
from evo.files import FileAPIClient
from evo.files.data import FileMetadata
from evo.notebooks._consts import DEFAULT_DISCOVERY_URL
from evo.notebooks._helpers import build_img_widget
from evo.objects import ObjectAPIClient
from evo.objects.data import ObjectMetadata
from evo.workspaces import User, Workspace, WorkspaceAPIClient
from evo.workspaces import parse as _parse
from evo.workspaces.endpoints.api import AdminApi, LicenseAccessApi


class WorkspaceAdminHelper:
    """Aggregates administrative information across workspaces in an Evo instance.

    All workspace and user listing is performed through the Workspaces **admin**
    API endpoints, which require the authenticated user to be an instance admin.

    This helper orchestrates calls to:

    * :class:`~evo.workspaces.endpoints.api.AdminApi` — admin workspace and user listings.
    * :class:`~evo.objects.ObjectAPIClient` — geoscience objects within each workspace.
    * :class:`~evo.files.FileAPIClient` — files within each workspace.

    Parameters
    ----------
    workspace_client:
        An authenticated :class:`~evo.workspaces.WorkspaceAPIClient`.
    connector:
        The :class:`~evo.common.APIConnector` used to create per-workspace
        File and Object API clients and to call the admin endpoints.
    org_id:
        The UUID of the Evo instance.
    hub_code:
        The short hub code (e.g. ``"aus1"``), passed to the License Access
        endpoint to scope the admin-rights check to the correct hub.
    """

    def __init__(
        self,
        workspace_client: WorkspaceAPIClient,
        connector: APIConnector,
        org_id: UUID,
        hub_code: str,
    ) -> None:
        self._workspace_client = workspace_client
        self._connector = connector
        self._org_id = org_id
        self._hub_code = hub_code
        self._admin_api = AdminApi(connector)
        self._license_api = LicenseAccessApi(connector)

    async def check_admin_rights(self) -> bool:
        """Return ``True`` if the authenticated user is an instance admin.

        Calls the License Access endpoint and inspects the ``entitlements.roles``
        field of the response.  A role name that contains ``"admin"`` (case-
        insensitive) is treated as confirmation of instance-admin rights.
        A ``403 Forbidden`` response means the user has no access at all.

        :returns: ``True`` if the user has instance-admin rights, ``False`` otherwise.
        """
        try:
            response = await self._license_api.v2_license_access_evo_identity_v2_license_access_get(
                org_id=str(self._org_id),
                hub=self._hub_code,
                service=["evo"],
                required_scope=["evo.workspace"],
            )
            return any("admin" in role.lower() for role in response.entitlements.roles)
        except ForbiddenException:
            return False

    async def list_workspaces(self) -> list[Workspace]:
        """Return all workspaces in the instance using the admin endpoint.

        Unlike the standard ``list_all_workspaces`` method, this call uses the
        admin API path which returns *every* workspace in the instance, not just
        those the current user has a direct role in.

        :returns: A sorted list of :class:`~evo.workspaces.Workspace` objects.
        :raises RuntimeError: If the admin API call fails.
        """
        workspaces: list[Workspace] = []
        offset = 0
        limit = 50

        while True:
            response = await self._admin_api.list_workspaces_admin(
                org_id=str(self._org_id),
                limit=limit,
                offset=offset,
            )
            for item in response.results:
                workspaces.append(_parse.workspace_model(item, self._org_id, self._connector.base_url))
            offset += limit
            if offset >= response.links.total:
                break

        return sorted(workspaces, key=lambda w: w.display_name)

    async def get_users_for_workspace(self, workspace_id: UUID) -> list[User]:
        """Return all users with a role in *workspace_id* using the admin endpoint.

        The admin endpoint allows retrieving users from any workspace regardless
        of the caller's own role in that workspace.

        :param workspace_id: The UUID of the target workspace.
        :returns: A list of :class:`~evo.workspaces.User` objects.
        :raises RuntimeError: Wraps any API error with a descriptive message.
        """
        try:
            response = await self._admin_api.list_user_roles_admin(
                org_id=str(self._org_id),
                workspace_id=str(workspace_id),
            )
            return [_parse.user_model(item) for item in response.results]
        except Exception as exc:
            raise RuntimeError(f"Failed to list users for workspace {workspace_id}: {exc}") from exc

    async def get_objects_for_workspace(self, workspace: Workspace) -> list[ObjectMetadata]:
        """Return all geoscience objects in *workspace* (auto-paginated).

        :param workspace: The target :class:`~evo.workspaces.Workspace`.
        :returns: A list of :class:`~evo.objects.data.ObjectMetadata` objects.
        :raises RuntimeError: Wraps any API error with a descriptive message.
        """
        environment = workspace.get_environment()
        object_client = ObjectAPIClient(environment=environment, connector=self._connector)

        all_objects: list[ObjectMetadata] = []
        offset = 0
        limit = 1000

        try:
            while True:
                page = await object_client.list_objects(offset=offset, limit=limit)
                all_objects.extend(page.items())
                offset += limit
                if offset >= page.total:
                    break
        except Exception as exc:
            raise RuntimeError(
                f"Failed to list objects for workspace '{workspace.display_name}'. You are not part of this workspace: {exc}"
            ) from exc

        return all_objects

    async def get_files_for_workspace(self, workspace: Workspace) -> list[FileMetadata]:
        """Return all files in *workspace* (auto-paginated).

        :param workspace: The target :class:`~evo.workspaces.Workspace`.
        :returns: A list of :class:`~evo.files.data.FileMetadata` objects.
        :raises RuntimeError: Wraps any API error with a descriptive message.
        """
        environment = workspace.get_environment()
        file_client = FileAPIClient(environment=environment, connector=self._connector)

        all_files: list[FileMetadata] = []
        offset = 0
        limit = 5000

        try:
            while True:
                page = await file_client.list_files(offset=offset, limit=limit)
                all_files.extend(page.items())
                offset += limit
                if offset >= page.total:
                    break
        except Exception as exc:
            raise RuntimeError(f"Failed to list files for workspace '{workspace.display_name}': {exc}") from exc

        return all_files

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    async def get_workspace_report(
        self,
        workspace: Workspace,
        include_objects: bool = True,
        include_files: bool = True,
    ) -> WorkspaceReport:
        """Build a :class:`WorkspaceReport` for a single workspace.

        Errors when fetching users, objects, or files are captured in the
        corresponding ``*_error`` field of the report rather than propagated,
        so that a single failing workspace does not abort an entire run.

        :param workspace: The workspace to report on.
        :param include_objects: Whether to fetch geoscience objects.
        :param include_files: Whether to fetch files.
        :returns: A populated :class:`WorkspaceReport`.
        """
        report = WorkspaceReport(workspace=workspace)

        try:
            report.users = await self.get_users_for_workspace(workspace.id)
        except Exception as exc:
            report.users_error = str(exc)

        if include_objects:
            try:
                report.objects = await self.get_objects_for_workspace(workspace)
            except Exception as exc:
                report.objects_error = str(exc)

        if include_files:
            try:
                report.files = await self.get_files_for_workspace(workspace)
            except Exception as exc:
                report.files_error = str(exc)

        return report

    async def get_all_workspace_reports(
        self,
        include_objects: bool = True,
        include_files: bool = True,
        workspace_ids: list[UUID] | None = None,
    ) -> list[WorkspaceReport]:
        """Build :class:`WorkspaceReport` objects for all (or a filtered set of) workspaces.

        :param include_objects: Whether to fetch geoscience objects for each workspace.
        :param include_files: Whether to fetch files for each workspace.
        :param workspace_ids: An optional allow-list of workspace UUIDs. When
            provided only those workspaces are reported on.
        :returns: A list of :class:`WorkspaceReport` objects, one per workspace.
        """
        workspaces = await self.list_workspaces()

        if workspace_ids is not None:
            id_set = set(workspace_ids)
            workspaces = [w for w in workspaces if w.id in id_set]

        reports: list[WorkspaceReport] = []
        for workspace in workspaces:
            report = await self.get_workspace_report(
                workspace,
                include_objects=include_objects,
                include_files=include_files,
            )
            reports.append(report)

        return reports


class InstanceSelectorWidget:
    """Interactive widget for selecting an Evo instance after login.

    Uses the Evo Discovery API to enumerate all instances available to the
    authenticated user.  Each instance is expected to have exactly one hub;
    that hub URL is captured automatically when an instance is selected.

    The selections are exposed via :meth:`get_org_id` and :meth:`get_hub_url`
    so that downstream cells can build their API clients without hard-coding any
    configuration.

    Parameters
    ----------
    transport:
        The :class:`~evo.aio.AioTransport` instance used for HTTP requests.
    authorizer:
        The authenticated :class:`~evo.oauth.AuthorizationCodeAuthorizer`.
    """

    _DISCOVERY_URL = DEFAULT_DISCOVERY_URL

    def __init__(self, transport, authorizer) -> None:
        self._transport = transport
        self._authorizer = authorizer
        self._organizations: list = []
        self._hub_url: str | None = None
        self._hub_code: str | None = None

        self.refresh_btn = widgets.Button(
            description="Refresh Instances",
            button_style="info",
            layout=widgets.Layout(margin="2px 5px 2px 0px", align_self="center"),
        )
        self.refresh_btn.style.button_color = "#265C7F"
        self.refresh_btn.on_click(self._on_refresh_click)

        self._loading_widget = build_img_widget("loading.gif")
        self._loading_widget.layout.display = "none"

        self.instance_selector = widgets.Dropdown(
            options=[("Select an instance...", None)],
            value=None,
            description="Instance",
            disabled=True,
            style={"description_width": "80px"},
            layout=widgets.Layout(margin="2px 5px 2px 0px", align_self="flex-start"),
        )
        self.instance_selector.observe(self._on_org_changed, names="value")

        self.info_display = widgets.HTML(value="")

        self.widget = widgets.VBox(
            [
                widgets.HBox(
                    [build_img_widget("EvoBadgeCharcoal_FV.png"), self.refresh_btn, self._loading_widget],
                    layout=widgets.Layout(align_items="center"),
                ),
                widgets.HBox([self.instance_selector]),
                self.info_display,
            ],
            layout=widgets.Layout(align_items="flex-start"),
        )

    def _on_refresh_click(self, btn) -> None:
        import asyncio

        asyncio.create_task(self.refresh())

    def _on_org_changed(self, change) -> None:
        org_id = change["new"]
        self._hub_url = None
        self._hub_code = None
        self.info_display.value = ""

        if org_id is None:
            return

        selected_org = next((o for o in self._organizations if o.id == org_id), None)
        if selected_org is None or not selected_org.hubs:
            self.info_display.value = (
                "<div style='margin-top:10px; padding:10px; background-color:#fff3cd; border-radius:5px;'>"
                "⚠ No hub found for this instance."
                "</div>"
            )
            return

        # Take the first (and typically only) hub for the instance.
        hub = selected_org.hubs[0]
        self._hub_url = hub.url
        self._hub_code = hub.code

        self.info_display.value = (
            "<div style='margin-top:10px; padding:10px; background-color:#f0f8e8; border-radius:5px;'>"
            "<b>Selected configuration</b><br/>"
            f"<b>Instance:</b> {selected_org.display_name}<br/>"
            f"<b>Instance ID:</b> <code>{selected_org.id}</code><br/>"
            f"<b>Hub URL:</b> <code>{hub.url}</code>"
            "</div>"
        )

    async def refresh(self) -> None:
        """Fetch instances from the Discovery API and populate the dropdown."""
        self._loading_widget.layout.display = "flex"
        self.refresh_btn.disabled = True
        self.instance_selector.disabled = True
        self.info_display.value = ""

        try:
            from evo.discovery import DiscoveryAPIClient

            async with APIConnector(self._DISCOVERY_URL, self._transport, self._authorizer) as idp_connector:
                discovery_client = DiscoveryAPIClient(idp_connector)
                self._organizations = await discovery_client.list_organizations()

            if self._organizations:
                org_options = [("Select an instance...", None)]
                org_options.extend([(org.display_name, org.id) for org in self._organizations])
                self.instance_selector.options = org_options
                self.instance_selector.value = None
                # Auto-select when there is only one instance.
                if len(self._organizations) == 1:
                    self.instance_selector.value = self._organizations[0].id
            else:
                self.instance_selector.options = [("No instances found", None)]
                self.info_display.value = (
                    "<div style='margin-top:10px; padding:10px; background-color:#fff3cd; border-radius:5px;'>"
                    "⚠ No instances found."
                    "</div>"
                )

        except Exception as exc:
            self.info_display.value = (
                "<div style='margin-top:10px; padding:10px; background-color:#f8d7da; border-radius:5px;'>"
                f"⚠ Error loading instances: {exc}"
                "</div>"
            )

        finally:
            self._loading_widget.layout.display = "none"
            self.refresh_btn.disabled = False
            self.instance_selector.disabled = False

    async def display(self) -> None:
        """Render the widget and auto-load instances from the Discovery API."""
        display(self.widget)
        await self.refresh()

    def get_org_id(self) -> UUID | None:
        """Return the UUID of the selected instance, or ``None`` if not yet selected."""
        return self.instance_selector.value

    def get_org_name(self) -> str | None:
        """Return the display name of the selected instance, or ``None`` if not yet selected."""
        org_id = self.instance_selector.value
        if org_id is None:
            return None
        org = next((o for o in self._organizations if o.id == org_id), None)
        return org.display_name if org else None

    def get_hub_url(self) -> str | None:
        """Return the hub URL for the selected instance, or ``None`` if not yet selected."""
        return self._hub_url

    def get_hub_code(self) -> str | None:
        """Return the hub code for the selected instance, or ``None`` if not yet selected."""
        return self._hub_code


class UserBrowserWidget:
    """Widget for browsing users per workspace with a dropdown selector.

    Parameters
    ----------
    df_users:
        A DataFrame with columns: Workspace, Full Name, Email, User ID, Role.
    """

    def __init__(self, df_users) -> None:
        workspace_names = sorted(df_users["Workspace"].unique())

        self._df = df_users
        self._label = widgets.Label(value="Workspace:")
        self._dropdown = widgets.Dropdown(
            options=workspace_names,
            description="",
            style={"description_width": "0px"},
            layout=widgets.Layout(width="400px"),
        )
        self._title = widgets.HTML()
        self._output = widgets.Output()

        self._dropdown.observe(self._render, names="value")
        self._render()

        self.widget = widgets.VBox(
            [
                widgets.HBox([self._label, self._dropdown]),
                self._title,
                self._output,
            ],
            layout=widgets.Layout(align_items="flex-start"),
        )

    def _render(self, _=None) -> None:
        ws_name = self._dropdown.value
        group = (
            self._df[self._df["Workspace"] == ws_name][["Full Name", "Email", "User ID", "Role"]]
            .sort_values("Full Name")
            .reset_index(drop=True)
        )
        total = len(group)
        self._title.value = f"<b>{ws_name}</b> — {total} user{'s' if total != 1 else ''}"
        with self._output:
            self._output.clear_output(wait=True)
            display(group)

    def show(self) -> None:
        """Display the widget."""
        display(self.widget)


@dataclass
class WorkspaceReport:
    """A comprehensive admin report for a single workspace.

    Attributes
    ----------
    workspace:
        The workspace this report belongs to.
    users:
        All users that have a role in the workspace.
    objects:
        All geoscience objects found in the workspace.
    files:
        All files found in the workspace.
    users_error:
        Error message if fetching users failed, ``None`` otherwise.
    objects_error:
        Error message if fetching objects failed, ``None`` otherwise.
    files_error:
        Error message if fetching files failed, ``None`` otherwise.
    """

    workspace: Workspace
    users: list[User] = field(default_factory=list)
    objects: list[ObjectMetadata] = field(default_factory=list)
    files: list[FileMetadata] = field(default_factory=list)
    users_error: str | None = None
    objects_error: str | None = None
    files_error: str | None = None

    @property
    def users_by_role(self) -> dict[str, list[User]]:
        """Return users grouped by their workspace role name (``"owner"``, ``"editor"``, ``"viewer"``)."""
        grouped: dict[str, list[User]] = defaultdict(list)
        for user in self.users:
            role_name = user.role.name if user.role else "unknown"
            grouped[role_name].append(user)
        return dict(grouped)

    @property
    def objects_by_schema(self) -> dict[str, list[ObjectMetadata]]:
        """Return geoscience objects grouped by their schema ID string."""
        grouped: dict[str, list[ObjectMetadata]] = defaultdict(list)
        for obj in self.objects:
            grouped[str(obj.schema_id)].append(obj)
        return dict(grouped)

    @property
    def total_file_size_bytes(self) -> int:
        """Total size of all files in the workspace, in bytes."""
        return sum(f.size for f in self.files)

    @property
    def total_file_size_mb(self) -> float:
        """Total size of all files in the workspace, in megabytes (rounded to 2 dp)."""
        return round(self.total_file_size_bytes / (1024 * 1024), 2)

    def summary_dict(self) -> dict:
        """Return a flat dictionary suitable for display in a pandas DataFrame row."""
        return {
            "Workspace Name": self.workspace.display_name,
            "Workspace ID": str(self.workspace.id),
            "User Role": self.workspace.user_role.name if self.workspace.user_role else "N/A",
            "Total Users": len(self.users),
            "Owners": len(self.users_by_role.get("owner", [])),
            "Editors": len(self.users_by_role.get("editor", [])),
            "Viewers": len(self.users_by_role.get("viewer", [])),
            "Total Objects": len(self.objects),
            "Object Schema Types": len(self.objects_by_schema),
            "Total Files": len(self.files),
            "Total File Size (MB)": self.total_file_size_mb,
            "Labels": ", ".join(self.workspace.labels) if self.workspace.labels else "",
            "Created At": str(self.workspace.created_at),
            "Updated At": str(self.workspace.updated_at),
        }


def format_size(size_bytes: int) -> str:
    """Return a human-readable file size string (B / KB / MB / GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"

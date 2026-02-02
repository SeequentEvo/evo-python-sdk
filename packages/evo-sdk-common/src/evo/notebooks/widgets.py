#  Copyright © 2025 Bentley Systems, Incorporated
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

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, cast
from uuid import UUID

import ipywidgets as widgets
from aiohttp.typedefs import StrOrURL
from IPython.display import display

from evo import logging
from evo.aio import AioTransport
from evo.common import APIConnector, BaseAPIClient, Environment
from evo.common.exceptions import UnauthorizedException
from evo.common.interfaces import IAuthorizer, ICache, IContext, IFeedback, ITransport
from evo.discovery import Hub, Organization
from evo.oauth import AnyScopes, EvoScopes, OAuthConnector
from evo.service_manager import ServiceManager
from evo.workspaces import Workspace

from ._consts import (
    DEFAULT_BASE_URI,
    DEFAULT_CACHE_LOCATION,
    DEFAULT_DISCOVERY_URL,
    DEFAULT_REDIRECT_URL,
)
from ._helpers import FileName, build_button_widget, build_img_widget, init_cache
from .authorizer import AuthorizationCodeAuthorizer
from .env import DotEnv

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "FeedbackWidget",
    "HubSelectorWidget",
    "ObjectSearchWidget",
    "OrgSelectorWidget",
    "ServiceManagerWidget",
    "WorkspaceSelectorWidget",
]


class DropdownSelectorWidget(widgets.HBox, Generic[T]):
    UNSELECTED: tuple[str, T]

    def __init__(self, label: str, env: DotEnv) -> None:
        self._env = env
        self.dropdown_widget = widgets.Dropdown(
            options=[self.UNSELECTED],
            description=label,
            value=self.UNSELECTED[1],
            layout=widgets.Layout(margin="5px 5px 5px 5px", align_self="flex-start"),
        )
        self.dropdown_widget.disabled = True
        self.dropdown_widget.observe(self._update_selected, names="value")

        self._loading_widget = build_img_widget("loading.gif")
        self._loading_widget.layout.display = "none"

        super().__init__([self.dropdown_widget, self._loading_widget])

    def _get_options(self) -> list[tuple[str, T]]:
        raise NotImplementedError("Subclasses must implement this method.")

    def _on_selected(self, value: T | None) -> None: ...

    @contextlib.contextmanager
    def _loading(self) -> Iterator[None]:
        self.dropdown_widget.disabled = True
        self._loading_widget.layout.display = "flex"
        try:
            yield
        finally:
            self._loading_widget.layout.display = "none"
            self.dropdown_widget.disabled = False

    def _update_selected(self, _: dict) -> None:
        self.selected = new_value = self.dropdown_widget.value
        self._on_selected(new_value if new_value != self.UNSELECTED[1] else None)

    def refresh(self) -> None:
        logger.debug(f"Refreshing {self.__class__.__name__} options...")
        self.dropdown_widget.disabled = True
        selected = self.selected
        self.dropdown_widget.options = options = [self.UNSELECTED] + self._get_options()
        if len(options) == 2 and selected == self.UNSELECTED[1]:
            # Automatically select the only option if there is only one and no missing option was previously selected.
            self.selected = new_value = options[1][1]
        else:
            # Otherwise, ensure the selected option is still valid.
            for _, value in options:
                if value == selected:
                    self.selected = new_value = selected
                    break
            else:
                # If the selected option is no longer valid, reset to the unselected value.
                self.selected = new_value = self.UNSELECTED[1]

        # Make sure the new value is passed to the _on_selected method.
        self._on_selected(new_value if new_value != self.UNSELECTED[1] else None)

        # Disable the widget if there are no options to select.
        self.dropdown_widget.disabled = len(options) <= 1

    @classmethod
    def _serialize(cls, value: T) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def _deserialize(cls, value: str) -> T:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def selected(self) -> T:
        value = self._env.get(f"{self.__class__.__name__}.selected", self._serialize(self.UNSELECTED[1]))
        return self._deserialize(value)

    @selected.setter
    def selected(self, value: T) -> None:
        self._env.set(f"{self.__class__.__name__}.selected", self._serialize(value))
        self.dropdown_widget.value = value

    @property
    def disabled(self) -> bool:
        return self.dropdown_widget.disabled

    @disabled.setter
    def disabled(self, value: bool) -> None:
        self.dropdown_widget.disabled = value


_NULL_UUID = UUID(int=0)


class _UUIDSelectorWidget(DropdownSelectorWidget[UUID]):
    @classmethod
    def _serialize(cls, value: UUID) -> str:
        return str(value)

    @classmethod
    def _deserialize(cls, value: str) -> UUID:
        return UUID(value)


class OrgSelectorWidget(_UUIDSelectorWidget):
    UNSELECTED = ("Select Organisation", _NULL_UUID)

    def __init__(self, env: DotEnv, manager: ServiceManager) -> None:
        self._manager = manager
        super().__init__("Organisation", env)

    def _get_options(self) -> list[tuple[str, UUID]]:
        return [(org.display_name, org.id) for org in self._manager.list_organizations()]

    def _on_selected(self, value: UUID | None) -> None:
        self._manager.set_current_organization(value)


class HubSelectorWidget(DropdownSelectorWidget[str]):
    UNSELECTED = ("Select Hub", "")

    def __init__(self, env: DotEnv, manager: ServiceManager, org_selector: OrgSelectorWidget) -> None:
        self._manager = manager
        super().__init__("Hub", env)
        org_selector.dropdown_widget.observe(self._on_org_selected, names="value")

    def _on_org_selected(self, _: dict) -> None:
        self.refresh()

    def _get_options(self) -> list[tuple[str, str]]:
        return [(hub.display_name, hub.code) for hub in self._manager.list_hubs()]

    def _on_selected(self, value: str | None) -> None:
        self._manager.set_current_hub(value)

    @classmethod
    def _serialize(cls, value: str) -> str:
        return value

    @classmethod
    def _deserialize(cls, value: str) -> str:
        return value


class WorkspaceSelectorWidget(_UUIDSelectorWidget):
    UNSELECTED = ("Select Workspace", _NULL_UUID)

    def __init__(self, env: DotEnv, manager: ServiceManager, hub_selector: HubSelectorWidget) -> None:
        self._manager = manager
        super().__init__("Workspace", env)
        hub_selector.dropdown_widget.observe(self._on_hub_selected, names="value")

    async def refresh_workspaces(self) -> None:
        with self._loading():
            await self._manager.refresh_workspaces()
            self.refresh()

    def _on_hub_selected(self, _: dict) -> asyncio.Future:
        self.disabled = True
        return asyncio.ensure_future(self.refresh_workspaces())

    def _on_selected(self, value: UUID | None) -> None:
        self._manager.set_current_workspace(value)

    def _get_options(self) -> list[tuple[str, UUID]]:
        return [(ws.display_name, ws.id) for ws in self._manager.list_workspaces()]


# Generic type variable for the client factory method.
T_client = TypeVar("T_client", bound=BaseAPIClient)


class _ServiceManagerWidgetMeta(type(widgets.HBox), type(IContext)):
    """Metaclass that combines ipywidgets and pure interfaces metaclasses."""

    pass


class ServiceManagerWidget(widgets.HBox, IContext, metaclass=_ServiceManagerWidgetMeta):
    def __init__(self, transport: ITransport, authorizer: IAuthorizer, discovery_url: str, cache: ICache) -> None:
        """
        :param transport: The transport to use for API requests.
        :param authorizer: The authorizer to use for API requests.
        :param discovery_url: The URL of the Evo Discovery service.
        :param cache: The cache to use for storing tokens and other data.
        """
        self._authorizer = authorizer
        self._cache = cache
        self._service_manager = ServiceManager(
            transport=transport,
            authorizer=authorizer,
            discovery_url=discovery_url,
            cache=cache,
        )
        env = DotEnv(cache)

        self._btn = build_button_widget("Sign In")
        self._btn.on_click(self._on_click)
        self._org_selector = OrgSelectorWidget(env, self._service_manager)
        self._hub_selector = HubSelectorWidget(env, self._service_manager, self._org_selector)
        self._workspace_selector = WorkspaceSelectorWidget(env, self._service_manager, self._hub_selector)

        self._loading_widget = build_img_widget("loading.gif")
        self._loading_widget.layout.display = "none"

        self._prompt_area = widgets.Output()
        self._prompt_area.layout.display = "none"

        col_1 = widgets.VBox(
            [
                widgets.HBox([build_img_widget("EvoBadgeCharcoal_FV.png"), self._btn, self._loading_widget]),
                widgets.HBox([self._org_selector]),
                widgets.HBox([self._hub_selector]),
                widgets.HBox([self._workspace_selector]),
            ]
        )
        col_2 = widgets.VBox([self._prompt_area])

        super().__init__(
            [col_1, col_2],
            layout={
                "display": "flex",
                "flex_flow": "row",
                "justify_content": "space-between",
                "align_items": "center",
            },
        )
        display(self)

    @classmethod
    def with_auth_code(
        cls,
        client_id: str,
        base_uri: str = DEFAULT_BASE_URI,
        discovery_url: str = DEFAULT_DISCOVERY_URL,
        redirect_url: str = DEFAULT_REDIRECT_URL,
        client_secret: str | None = None,
        cache_location: FileName = DEFAULT_CACHE_LOCATION,
        oauth_scopes: AnyScopes = EvoScopes.all_evo | EvoScopes.offline_access,
        proxy: StrOrURL | None = None,
    ) -> ServiceManagerWidget:
        """Create a ServiceManagerWidget with an authorization code authorizer.

        To use it, you will need an OAuth client ID. See the documentation for information on how to obtain this:
        https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens

        Chain this method with the login method to authenticate the user and obtain an access token:

        ```python
        manager = await ServiceManagerWidget.with_auth_code(client_id="your-client-id").login()
        ```

        :param client_id: The client ID to use for authentication.
        :param base_uri: The OAuth server base URI.
        :param discovery_url: The URL of the Evo Discovery service.
        :param redirect_url: The local URL to redirect the user back to after authorisation.
        :param client_secret: The client secret to use for authentication.
        :param cache_location: The location of the cache file.
        :param oauth_scopes: The OAuth scopes to request.
        :param proxy: The proxy URL to use for API requests.

        :returns: The new ServiceManagerWidget.
        """
        cache = init_cache(cache_location)
        transport = AioTransport(user_agent=client_id, proxy=proxy)
        authorizer = AuthorizationCodeAuthorizer(
            oauth_connector=OAuthConnector(
                transport=transport,
                base_uri=base_uri,
                client_id=client_id,
                client_secret=client_secret,
            ),
            redirect_url=redirect_url,
            scopes=oauth_scopes,
            env=DotEnv(cache),
        )
        return cls(transport, authorizer, discovery_url, cache)

    async def _login_with_auth_code(self, timeout_seconds: int) -> None:
        """Login using an authorization code authorizer.

        This method will attempt to reuse an existing token from the environment file. If no token is found, the user will
        be prompted to log in.

        :param timeout_seconds: The number of seconds to wait for the user to log in.
        """
        authorizer = cast(AuthorizationCodeAuthorizer, self._authorizer)
        if not await authorizer.reuse_token():
            await authorizer.login(timeout_seconds=timeout_seconds)

    async def login(self, timeout_seconds: int = 180) -> ServiceManagerWidget:
        """Authenticate the user and obtain an access token.

        Only the notebook authorizer implementations are supported by this method.

        This method returns the current instance of the ServiceManagerWidget to allow for method chaining.

        ```python
        manager = await ServiceManagerWidget.with_auth_code(client_id="your-client-id").login()
        ```

        :param timeout_seconds: The maximum time (in seconds) to wait for the authorisation process to complete.

        :returns: The current instance of the ServiceManagerWidget.
        """
        # Open the transport without closing it to avoid the overhead of opening it multiple times.
        await self._service_manager._transport.open()
        with self._loading():
            match self._authorizer:
                case AuthorizationCodeAuthorizer():
                    await self._login_with_auth_code(timeout_seconds)
                case unknown:
                    raise NotImplementedError(f"ServiceManagerWidget cannot login using {type(unknown).__name__}.")

            # Refresh the services after logging in.
            await self.refresh_services()
        return self

    @property
    def cache(self) -> ICache:
        return self._cache

    def _update_btn(self, signed_in: bool) -> None:
        if signed_in:
            self._btn.description = "Refresh Evo Services"
        else:
            self._btn.description = "Sign In"

    def _on_click(self, _: widgets.Button) -> asyncio.Future:
        return asyncio.ensure_future(self.refresh_services())

    @contextlib.contextmanager
    def _loading(self) -> Iterator[None]:
        self._btn.disabled = True
        self._loading_widget.layout.display = "flex"
        try:
            yield
        finally:
            self._loading_widget.layout.display = "none"
            self._btn.disabled = False

    @contextlib.contextmanager
    def _loading_services(self) -> Iterator[None]:
        self._org_selector.disabled = True
        self._hub_selector.disabled = True
        self._workspace_selector.disabled = True
        try:
            yield
        finally:
            self._org_selector.refresh()
            self._hub_selector.refresh()

    @contextlib.contextmanager
    def _prompt(self) -> Iterator[widgets.Output]:
        self._prompt_area.layout.display = "flex"
        try:
            yield self._prompt_area
        finally:
            self._prompt_area.layout.display = "none"
            self._prompt_area.clear_output()

    async def refresh_services(self) -> None:
        with self._loading():
            with self._loading_services():
                try:
                    await self._service_manager.refresh_organizations()
                except UnauthorizedException:  # Expired token or user not logged in.
                    # Attempt to log in again.
                    await self.login()

                    # Try refresh the services again after logging in.
                    await self._service_manager.refresh_organizations()
            await self._workspace_selector.refresh_workspaces()
            self._update_btn(True)

    @property
    def organizations(self) -> list[Organization]:
        return self._service_manager.list_organizations()

    @property
    def hubs(self) -> list[Hub]:
        return self._service_manager.list_hubs()

    @property
    def workspaces(self) -> list[Workspace]:
        return self._service_manager.list_workspaces()

    def get_connector(self) -> APIConnector:
        """Get an API connector for the currently selected hub.

        :returns: The API connector.

        :raises SelectionError: If no organization or hub is currently selected.
        """
        return self._service_manager.get_connector()

    def get_environment(self) -> Environment:
        """Get an environment with the currently selected organization, hub, and workspace.

        :returns: The environment.

        :raises SelectionError: If no organization, hub, or workspace is currently selected.
        """
        return self._service_manager.get_environment()

    def get_org_id(self) -> UUID:
        """Gets the ID of the currently selected organization.

        :return: The organization ID.
        :raises SelectionError: If no organization is currently selected.
        """
        return self._service_manager.get_org_id()

    def get_cache(self) -> ICache:
        """
        Gets the cache for this context.

        :returns: The cache.
        """
        return self._cache

    def create_client(self, client_class: type[T_client], *args: Any, **kwargs: Any) -> T_client:
        """Create a client for the currently selected workspace.

        :param client_class: The class of the client to create.

        :returns: The new client.

        :raises SelectionError: If no organization, hub, or workspace is currently selected.
        """
        return self._service_manager.create_client(client_class, *args, **kwargs)


class FeedbackWidget(IFeedback):
    """Simple feedback widget for displaying progress and messages to the user."""

    def __init__(self, label: str) -> None:
        """
        :param label: The label for the feedback widget.
        """
        label = widgets.Label(label)
        self._progress = widgets.FloatProgress(value=0, min=0, max=1, style={"bar_color": "#265C7F"})
        self._progress.layout.width = "400px"
        self._msg = widgets.Label("", style={"font_style": "italic"})
        self._widget = widgets.HBox([label, self._progress, self._msg])
        self._last_message = ""
        display(self._widget)

    def progress(self, progress: float, message: str | None = None) -> None:
        """Progress the feedback and update the text to message.

        This can raise an exception to cancel the current operation.

        :param progress: A float between 0 and 1 representing the progress of the operation as a percentage.
        :param message: An optional message to display to the user.
        """
        self._progress.value = progress
        self._progress.description = f"{progress * 100:5.1f}%"
        if message is not None:
            self._msg.value = message


class ObjectSearchWidget(widgets.VBox):
    """A widget for searching and selecting geoscience objects by name.

    This widget provides a user-friendly interface for discovering objects in an Evo workspace
    without requiring knowledge of UUIDs or the low-level ObjectAPIClient.

    Features:
    - Text search with debounced input (300ms delay)
    - Optional filtering by object type (e.g., "pointset", "block-model")
    - Displays matching objects in a dropdown
    - Shows metadata, versions, and attributes for the selected object
    - Caches object list per workspace to minimize API calls

    Example usage:
        ```python
        from evo.notebooks import ServiceManagerWidget, ObjectSearchWidget
        from evo.objects.typed import PointSet

        manager = await ServiceManagerWidget.with_auth_code(client_id="...").login()

        # Search for pointsets containing "Ag" in the name
        picker = ObjectSearchWidget(manager, search="Ag", object_type="pointset")

        # Once user selects an object, load it as a typed PointSet
        pointset = await PointSet.from_reference(manager, picker.selected_reference)
        df = await pointset.locations.as_dataframe()
        ```
    """

    # Mapping of sub_classification to user-friendly names
    _TYPE_DISPLAY_NAMES: dict[str, str] = {
        "pointset": "Pointset",
        "block-model": "Block Model",
        "regular-3d-grid": "Regular 3D Grid",
        "regular-masked-3d-grid": "Regular Masked 3D Grid",
        "tensor-3d-grid": "Tensor 3D Grid",
        "drilling-campaign": "Drilling Campaign",
        "downhole-collection": "Downhole Collection",
        "triangulated-surface-mesh": "Triangulated Surface Mesh",
        "variogram": "Variogram",
    }

    def __init__(
        self,
        context: IContext,
        search: str = "",
        object_type: str | None = None,
        auto_display: bool = True,
    ) -> None:
        """Create a new ObjectSearchWidget.

        :param context: The context (e.g., ServiceManagerWidget) providing environment and connector.
        :param search: Initial search text to filter objects by name.
        :param object_type: Optional object type filter (e.g., "pointset", "block-model").
        :param auto_display: Whether to automatically display the widget (default True).
        """
        self._context = context
        self._object_type = object_type
        self._cached_objects: list[Any] = []  # List of ObjectMetadata
        self._cache_workspace_id: UUID | None = None
        self._debounce_task: asyncio.Task | None = None
        self._selected_metadata: Any | None = None  # ObjectMetadata

        # Build UI components
        self._search_input = widgets.Text(
            value=search,
            placeholder="Type to search objects by name...",
            description="Search:",
            layout=widgets.Layout(width="400px"),
        )
        self._search_input.observe(self._on_search_change, names="value")

        type_options = [("All types", None)] + [
            (display_name, type_key) for type_key, display_name in self._TYPE_DISPLAY_NAMES.items()
        ]
        self._type_filter = widgets.Dropdown(
            options=type_options,
            value=object_type,
            description="Type:",
            layout=widgets.Layout(width="250px"),
        )
        self._type_filter.observe(self._on_type_change, names="value")

        self._results_dropdown = widgets.Dropdown(
            options=[("No results", None)],
            value=None,
            description="Object:",
            layout=widgets.Layout(width="500px"),
            disabled=True,
        )
        self._results_dropdown.observe(self._on_selection_change, names="value")

        self._loading_indicator = widgets.Label("")
        self._status_label = widgets.Label("")

        # Metadata display area
        self._metadata_output = widgets.Output(layout=widgets.Layout(width="100%", border="1px solid #ddd"))

        # Layout
        search_row = widgets.HBox([self._search_input, self._type_filter, self._loading_indicator])
        results_row = widgets.HBox([self._results_dropdown, self._status_label])

        super().__init__([search_row, results_row, self._metadata_output])

        # Trigger initial search if search text provided
        if search:
            asyncio.ensure_future(self._perform_search())

        if auto_display:
            display(self)

    @property
    def selected_reference(self) -> Any | None:
        """Get the ObjectReference for the selected object.

        Returns None if no object is selected.
        Use this with typed object classes like PointSet.from_reference().
        """
        if self._selected_metadata is None:
            return None
        return self._selected_metadata.url

    @property
    def selected_name(self) -> str | None:
        """Get the name of the selected object, or None if nothing is selected."""
        if self._selected_metadata is None:
            return None
        return self._selected_metadata.name

    @property
    def selected_metadata(self) -> Any | None:
        """Get the full ObjectMetadata for the selected object, or None if nothing is selected."""
        return self._selected_metadata

    def _on_search_change(self, change: dict) -> None:
        """Handle search input changes with debouncing."""
        if self._debounce_task is not None:
            self._debounce_task.cancel()
        self._debounce_task = asyncio.ensure_future(self._debounced_search())

    async def _debounced_search(self) -> None:
        """Wait for debounce delay then perform search."""
        await asyncio.sleep(0.3)  # 300ms debounce
        await self._perform_search()

    def _on_type_change(self, change: dict) -> None:
        """Handle type filter changes."""
        self._object_type = change["new"]
        # Clear cache to force re-fetch with new filter
        self._cached_objects = []
        self._cache_workspace_id = None
        asyncio.ensure_future(self._perform_search())

    def _on_selection_change(self, change: dict) -> None:
        """Handle object selection changes."""
        selected_id = change["new"]
        if selected_id is None:
            self._selected_metadata = None
            self._clear_metadata_display()
            return

        # Find the selected object in cached objects
        for obj in self._cached_objects:
            if obj.id == selected_id:
                self._selected_metadata = obj
                asyncio.ensure_future(self._display_metadata(obj))
                break

    async def _perform_search(self) -> None:
        """Perform the object search and update results."""
        self._loading_indicator.value = "Loading..."
        self._results_dropdown.disabled = True

        try:
            # Fetch objects if cache is invalid
            current_workspace_id = self._context.get_environment().workspace_id
            if self._cache_workspace_id != current_workspace_id or not self._cached_objects:
                await self._fetch_objects()
                self._cache_workspace_id = current_workspace_id

            # Filter by search text (case-insensitive partial match)
            search_text = self._search_input.value.lower().strip()
            filtered = self._cached_objects

            if search_text:
                filtered = [obj for obj in filtered if search_text in obj.name.lower()]

            # Update dropdown
            if filtered:
                options = [
                    (f"{obj.name} ({self._get_type_display(obj.schema_id.sub_classification)})", obj.id)
                    for obj in filtered
                ]
                self._results_dropdown.options = options
                self._results_dropdown.value = options[0][1]  # Select first result
                self._results_dropdown.disabled = False
                self._status_label.value = f"Found {len(filtered)} object(s)"
            else:
                self._results_dropdown.options = [("No matching objects", None)]
                self._results_dropdown.value = None
                self._results_dropdown.disabled = True
                self._status_label.value = "No results"
                self._clear_metadata_display()

        except Exception as e:
            self._status_label.value = f"Error: {e}"
            logger.exception("Error searching objects")
        finally:
            self._loading_indicator.value = ""

    async def _fetch_objects(self) -> None:
        """Fetch all objects from the workspace (with pagination)."""
        # Lazy import to avoid circular dependencies
        from evo.objects import ObjectAPIClient

        client = ObjectAPIClient.from_context(self._context)

        # Fetch all objects with pagination
        all_objects = await client.list_all_objects(limit_per_request=500)

        # Filter by object type client-side (API requires full schema path with version)
        if self._object_type:
            self._cached_objects = [
                obj for obj in all_objects
                if obj.schema_id.sub_classification == self._object_type
            ]
        else:
            self._cached_objects = all_objects

    def _get_type_display(self, sub_classification: str) -> str:
        """Get user-friendly display name for an object type."""
        return self._TYPE_DISPLAY_NAMES.get(sub_classification, sub_classification)

    def _clear_metadata_display(self) -> None:
        """Clear the metadata display area."""
        self._metadata_output.clear_output()

    async def _display_metadata(self, obj: Any) -> None:
        """Display metadata, versions, and attributes for the selected object."""
        self._metadata_output.clear_output()

        with self._metadata_output:
            try:
                # Basic metadata
                print("=" * 60)
                print(f"📦 {obj.name}")
                print("=" * 60)
                print(f"Type:        {self._get_type_display(obj.schema_id.sub_classification)}")
                print(f"Path:        {obj.path}")
                print(f"Object ID:   {obj.id}")
                print(f"Schema:      {obj.schema_id}")


                print()
                print(f"Created:     {obj.created_at.strftime('%Y-%m-%d %H:%M:%S')}", end="")
                if obj.created_by and obj.created_by.name:
                    print(f" by {obj.created_by.name}")
                else:
                    print()

                print(f"Modified:    {obj.modified_at.strftime('%Y-%m-%d %H:%M:%S')}", end="")
                if obj.modified_by and obj.modified_by.name:
                    print(f" by {obj.modified_by.name}")
                else:
                    print()

                if obj.stage:
                    print(f"Stage:       {obj.stage.name}")

                # Fetch and display versions
                print()
                print("-" * 60)
                print("📋 Versions")
                print("-" * 60)
                await self._display_versions(obj)

                # Fetch and display spatial info and attributes from downloaded object
                print()
                await self._display_attributes(obj)

            except Exception as e:
                print(f"Error loading details: {e}")
                logger.exception("Error displaying object metadata")

    async def _display_versions(self, obj: Any) -> None:
        """Fetch and display versions for the object."""
        try:
            from evo.objects import ObjectAPIClient

            client = ObjectAPIClient.from_context(self._context)
            versions = await client.list_versions_by_id(obj.id)

            if not versions:
                print("  No versions found")
                return

            for i, version in enumerate(versions):
                version_label = "latest" if i == 0 else f"v{len(versions) - i}"
                created_str = version.created_at.strftime("%Y-%m-%d %H:%M")

                # Handle created_by as either ServiceUser object or string
                if version.created_by:
                    if hasattr(version.created_by, 'name') and version.created_by.name:
                        created_by = version.created_by.name
                    else:
                        created_by = str(version.created_by)
                else:
                    created_by = "unknown"

                stage_str = f" [{version.stage.name}]" if version.stage else ""
                print(f"  {version_label}: {created_str} by {created_by}{stage_str}")

        except Exception as e:
            print(f"  Error loading versions: {e}")

    async def _display_attributes(self, obj: Any) -> None:
        """Fetch and display attributes for the latest version of the object."""
        try:
            from evo.objects import ObjectAPIClient

            client = ObjectAPIClient.from_context(self._context)
            downloaded = await client.download_object_by_id(obj.id)
            obj_dict = downloaded.as_dict()

            # Display spatial info from downloaded object
            crs = obj_dict.get("coordinate_reference_system")
            if crs:
                if isinstance(crs, dict):
                    epsg = crs.get("epsg_code")
                    if epsg:
                        print(f"CRS:         EPSG:{epsg}")
                elif isinstance(crs, str):
                    print(f"CRS:         {crs}")

            bbox = obj_dict.get("bounding_box")
            if bbox and isinstance(bbox, dict):
                print(f"Bounding Box:")
                print(f"  X: [{bbox.get('min_x', 0):.2f}, {bbox.get('max_x', 0):.2f}]")
                print(f"  Y: [{bbox.get('min_y', 0):.2f}, {bbox.get('max_y', 0):.2f}]")
                print(f"  Z: [{bbox.get('min_z', 0):.2f}, {bbox.get('max_z', 0):.2f}]")

            # Extract attributes based on object type
            print()
            print("-" * 60)
            print("📊 Attributes")
            print("-" * 60)
            attributes = self._extract_attributes(downloaded, obj.schema_id.sub_classification)

            if not attributes:
                print("  No attributes found")
                return

            for attr_name, attr_type in attributes:
                print(f"  • {attr_name} ({attr_type})")

        except Exception as e:
            print(f"  Error loading attributes: {e}")

    def _extract_attributes(self, downloaded_object: Any, sub_classification: str) -> list[tuple[str, str]]:
        """Extract attribute names and types from a downloaded object based on its type."""
        attributes = []

        def _extract_from_list(attrs: Any, suffix: str = "") -> None:
            """Helper to extract attributes from a list of attribute dicts."""
            if not isinstance(attrs, list):
                return
            for attr in attrs:
                if isinstance(attr, dict):
                    name = attr.get("name", "unknown")
                    attr_type = attr.get("attribute_type", "unknown")
                    display_name = f"{name}{suffix}" if suffix else name
                    attributes.append((display_name, attr_type))

        try:
            obj_dict = downloaded_object.as_dict()

            # Different object types store attributes in different places
            if sub_classification == "pointset":
                # Pointsets have attributes in locations.attributes
                locations = obj_dict.get("locations")
                if isinstance(locations, dict):
                    _extract_from_list(locations.get("attributes", []))

            elif sub_classification in ("regular-3d-grid", "regular-masked-3d-grid", "tensor-3d-grid"):
                # Grids have cell_attributes and vertex_attributes
                _extract_from_list(obj_dict.get("cell_attributes", []), " (cell)")
                _extract_from_list(obj_dict.get("vertex_attributes", []), " (vertex)")

            elif sub_classification == "block-model":
                # Block models reference the Block Model Service
                bm_ref = obj_dict.get("block_model_reference")
                if isinstance(bm_ref, dict):
                    _extract_from_list(bm_ref.get("attributes", []))

            else:
                # Generic fallback - look for common attribute patterns
                for key in ("attributes", "cell_attributes", "vertex_attributes"):
                    if key in obj_dict:
                        _extract_from_list(obj_dict[key])

        except Exception as e:
            logger.debug(f"Error extracting attributes: {e}")

        return attributes


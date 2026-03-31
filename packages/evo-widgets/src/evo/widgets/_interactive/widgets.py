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

"""Modern anywidget-based Jupyter widgets for Seequent Evo SDK."""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
from collections.abc import Iterator
from typing import Any, Generic, TypeVar
from uuid import UUID

import anywidget
import traitlets
from aiohttp.typedefs import StrOrURL

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
from ._helpers import FileName, init_cache
from .authorizer import InteractiveAuthorizer
from .env import DotEnv
from ..urls import get_portal_url_from_reference, get_viewer_url_from_reference, serialize_object_reference

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "display_object_links",
    "DropdownSelectorWidget",
    "FeedbackWidget",
    "HubSelectorWidget",
    "OrgSelectorWidget",
    "ServiceManagerWidget",
    "WorkspaceSelectorWidget",
]

# Path to static files
_STATIC_DIR = pathlib.Path(__file__).parent.parent / "static"


# Resolve metaclass conflict between anywidget (MetaHasTraits) and pure_interface (InterfaceType)
class _FeedbackMeta(type(anywidget.AnyWidget), type(IFeedback)):
    pass


class FeedbackWidget(anywidget.AnyWidget, IFeedback, metaclass=_FeedbackMeta):
    """Simple feedback widget for displaying progress and messages to the user.

    This is a modern anywidget-based implementation that works across different
    Jupyter environments.
    """

    _esm = _STATIC_DIR / "feedback.js"
    _css = _STATIC_DIR / "feedback.css"

    # Synced traits
    label = traitlets.Unicode("").tag(sync=True)
    progress_value = traitlets.Float(0.0).tag(sync=True)
    message = traitlets.Unicode("").tag(sync=True)

    def __init__(self, label: str) -> None:
        """
        :param label: The label for the feedback widget.
        """
        super().__init__(label=label)
        self._last_message = ""

    def progress(self, progress: float, message: str | None = None) -> None:
        """Progress the feedback and update the text to message.

        This can raise an exception to cancel the current operation.

        :param progress: A float between 0 and 1 representing the progress of the operation as a percentage.
        :param message: An optional message to display to the user.
        """
        self.progress_value = progress
        if message is not None:
            self.message = message


class DropdownSelectorWidget(anywidget.AnyWidget, Generic[T]):
    """Base dropdown selector widget with loading state.

    This is a modern anywidget-based implementation.
    """

    _esm = _STATIC_DIR / "dropdown.js"
    _css = _STATIC_DIR / "dropdown.css"

    UNSELECTED: tuple[str, T]

    # Synced traits
    label = traitlets.Unicode("").tag(sync=True)
    options = traitlets.List(traitlets.Tuple()).tag(sync=True)
    value = traitlets.Any(allow_none=True).tag(sync=True)
    disabled = traitlets.Bool(True).tag(sync=True)
    loading = traitlets.Bool(False).tag(sync=True)

    def __init__(self, label: str, env: DotEnv) -> None:
        self._env = env
        super().__init__(
            label=label,
            options=[self.UNSELECTED],
            value=self.UNSELECTED[1],
            disabled=True,
        )
        self.observe(self._on_value_change, names="value")

    def _get_options(self) -> list[tuple[str, T]]:
        raise NotImplementedError("Subclasses must implement this method.")

    def _on_selected(self, value: T | None) -> None:
        ...

    def _on_value_change(self, change: dict) -> None:
        new_value = change["new"]
        self._persist_selection(new_value)
        self._on_selected(new_value if new_value != self.UNSELECTED[1] else None)

    def _persist_selection(self, value: T) -> None:
        self._env.set(f"{self.__class__.__name__}.selected", self._serialize(value))

    def refresh(self) -> None:
        logger.debug(f"Refreshing {self.__class__.__name__} options...")
        self.disabled = True
        self.loading = True

        try:
            selected = self._get_persisted_selection()
            new_options = [self.UNSELECTED] + self._get_options()
            self.options = new_options

            if len(new_options) == 2 and selected == self.UNSELECTED[1]:
                # Auto-select if only one option
                new_value = new_options[1][1]
            else:
                # Check if selected is still valid
                new_value = self.UNSELECTED[1]
                for _, val in new_options:
                    if val == selected:
                        new_value = selected
                        break

            self.value = new_value
            self._on_selected(new_value if new_value != self.UNSELECTED[1] else None)
            self.disabled = len(new_options) == 1
        finally:
            self.loading = False

    def _get_persisted_selection(self) -> T:
        value = self._env.get(f"{self.__class__.__name__}.selected", self._serialize(self.UNSELECTED[1]))
        return self._deserialize(value)

    @classmethod
    def _serialize(cls, value: T) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def _deserialize(cls, value: str) -> T:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def selected(self) -> T:
        return self.value

    @selected.setter
    def selected(self, value: T) -> None:
        self.value = value


_NULL_UUID = UUID(int=0)


class _UUIDSelectorWidget(DropdownSelectorWidget[UUID]):
    @classmethod
    def _serialize(cls, value: UUID) -> str:
        return str(value)

    @classmethod
    def _deserialize(cls, value: str) -> UUID:
        return UUID(value)


class OrgSelectorWidget(_UUIDSelectorWidget):
    """Organisation selection dropdown widget."""

    UNSELECTED = ("Select Organisation", _NULL_UUID)

    def __init__(self, env: DotEnv, manager: ServiceManager) -> None:
        self._manager = manager
        super().__init__("Organisation", env)

    def _get_options(self) -> list[tuple[str, UUID]]:
        return [(org.display_name, org.id) for org in self._manager.list_organizations()]

    def _on_selected(self, value: UUID | None) -> None:
        self._manager.set_current_organization(value)


class HubSelectorWidget(DropdownSelectorWidget[str]):
    """Hub selection dropdown widget."""

    UNSELECTED = ("Select Hub", "")

    def __init__(self, env: DotEnv, manager: ServiceManager, org_selector: OrgSelectorWidget) -> None:
        self._manager = manager
        super().__init__("Hub", env)
        org_selector.observe(self._on_org_selected, names="value")

    def _on_org_selected(self, change: dict) -> None:
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
    """Workspace selection dropdown widget."""

    UNSELECTED = ("Select Workspace", _NULL_UUID)

    def __init__(self, env: DotEnv, manager: ServiceManager, hub_selector: HubSelectorWidget) -> None:
        self._manager = manager
        super().__init__("Workspace", env)
        hub_selector.observe(self._on_hub_selected, names="value")

    async def refresh_workspaces(self) -> None:
        self.loading = True
        try:
            await self._manager.refresh_workspaces()
            self.refresh()
        finally:
            self.loading = False

    def _on_hub_selected(self, change: dict) -> asyncio.Future:
        self.disabled = True
        return asyncio.ensure_future(self.refresh_workspaces())

    def _on_selected(self, value: UUID | None) -> None:
        self._manager.set_current_workspace(value)

    def _get_options(self) -> list[tuple[str, UUID]]:
        return [(ws.display_name, ws.id) for ws in self._manager.list_workspaces()]


# Generic type variable for the client factory method.
T_client = TypeVar("T_client", bound=BaseAPIClient)


class _ServiceManagerWidgetMeta(type(anywidget.AnyWidget), type(IContext)):
    """Metaclass that combines anywidget and pure interfaces metaclasses."""

    pass


class ServiceManagerWidget(anywidget.AnyWidget, IContext, metaclass=_ServiceManagerWidgetMeta):
    """Main authentication and service discovery widget.

    This is a modern anywidget-based implementation that provides authentication,
    organization/hub/workspace selection, and API client creation.
    """

    _esm = _STATIC_DIR / "service_manager.js"
    _css = _STATIC_DIR / "service_manager.css"

    # Button state
    button_text = traitlets.Unicode("Sign In").tag(sync=True)
    button_disabled = traitlets.Bool(False).tag(sync=True)
    button_clicked = traitlets.Bool(False).tag(sync=True)
    main_loading = traitlets.Bool(False).tag(sync=True)

    # Prompt area
    prompt_text = traitlets.Unicode("").tag(sync=True)
    show_prompt = traitlets.Bool(False).tag(sync=True)

    # Organization dropdown
    org_options = traitlets.List(traitlets.Tuple()).tag(sync=True)
    org_value = traitlets.Any(allow_none=True).tag(sync=True)
    org_loading = traitlets.Bool(False).tag(sync=True)

    # Hub dropdown
    hub_options = traitlets.List(traitlets.Tuple()).tag(sync=True)
    hub_value = traitlets.Any(allow_none=True).tag(sync=True)
    hub_loading = traitlets.Bool(False).tag(sync=True)

    # Workspace dropdown
    ws_options = traitlets.List(traitlets.Tuple()).tag(sync=True)
    ws_value = traitlets.Any(allow_none=True).tag(sync=True)
    ws_loading = traitlets.Bool(False).tag(sync=True)

    def __init__(self, transport: ITransport, authorizer: IAuthorizer, discovery_url: str, cache: ICache) -> None:
        """
        :param transport: The transport to use for API requests.
        :param authorizer: The authorizer to use for API requests.
        :param discovery_url: The URL of the Evo Discovery service.
        :param cache: The cache to use for storing tokens and other data.
        """
        super().__init__()

        self._authorizer = authorizer
        self._cache = cache
        self._env = DotEnv(cache)
        self._service_manager = ServiceManager(
            transport=transport,
            authorizer=authorizer,
            discovery_url=discovery_url,
            cache=cache,
        )

        # Initialize dropdown options
        self.org_options = [("Select Organisation", str(_NULL_UUID))]
        self.org_value = str(_NULL_UUID)
        self.hub_options = [("Select Hub", "")]
        self.hub_value = ""
        self.ws_options = [("Select Workspace", str(_NULL_UUID))]
        self.ws_value = str(_NULL_UUID)

        # Observe changes
        self.observe(self._on_button_click, names="button_clicked")
        self.observe(self._on_org_change, names="org_value")
        self.observe(self._on_hub_change, names="hub_value")
        self.observe(self._on_ws_change, names="ws_value")

        # Auto-display the widget (for backward compatibility with evo.notebooks)
        from IPython.display import display

        display(self)

    def _on_button_click(self, change: dict) -> None:
        if change["new"]:
            self.button_clicked = False
            asyncio.ensure_future(self.refresh_services())

    def _on_org_change(self, change: dict) -> None:
        value = change["new"]
        if value and value != str(_NULL_UUID):
            self._service_manager.set_current_organization(UUID(value))
            self._refresh_hubs()
        else:
            self._service_manager.set_current_organization(None)

    def _on_hub_change(self, change: dict) -> None:
        value = change["new"]
        if value:
            self._service_manager.set_current_hub(value)
            asyncio.ensure_future(self._refresh_workspaces())
        else:
            self._service_manager.set_current_hub(None)

    def _on_ws_change(self, change: dict) -> None:
        value = change["new"]
        if value and value != str(_NULL_UUID):
            self._service_manager.set_current_workspace(UUID(value))
        else:
            self._service_manager.set_current_workspace(None)

    @contextlib.contextmanager
    def _loading(self) -> Iterator[None]:
        """Context manager that sets loading and disabled state during operations."""
        self.main_loading = True
        self.button_disabled = True
        try:
            yield
        finally:
            self.main_loading = False
            self.button_disabled = False

    def _refresh_orgs(self) -> None:
        orgs = self._service_manager.list_organizations()
        self.org_options = [("Select Organisation", str(_NULL_UUID))] + [
            (org.display_name, str(org.id)) for org in orgs
        ]

    def _refresh_hubs(self) -> None:
        hubs = self._service_manager.list_hubs()
        self.hub_options = [("Select Hub", "")] + [(hub.display_name, hub.code) for hub in hubs]

    async def _refresh_workspaces(self) -> None:
        self.ws_loading = True
        try:
            await self._service_manager.refresh_workspaces()
            workspaces = self._service_manager.list_workspaces()
            self.ws_options = [("Select Workspace", str(_NULL_UUID))] + [
                (ws.display_name, str(ws.id)) for ws in workspaces
            ]
        finally:
            self.ws_loading = False

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
        authorizer = InteractiveAuthorizer(
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
        """Login using an authorization code authorizer."""
        authorizer = self._authorizer
        if isinstance(authorizer, InteractiveAuthorizer):
            if not await authorizer.reuse_token():
                await authorizer.login(timeout_seconds=timeout_seconds)

    async def login(self, timeout_seconds: int = 180) -> ServiceManagerWidget:
        """Authenticate the user and obtain an access token.

        :param timeout_seconds: The maximum time (in seconds) to wait for the authorisation process to complete.
        :returns: The current instance of the ServiceManagerWidget.
        """
        await self._service_manager._transport.open()

        with self._loading():
            if isinstance(self._authorizer, InteractiveAuthorizer):
                await self._login_with_auth_code(timeout_seconds)
            else:
                raise NotImplementedError(f"ServiceManagerWidget cannot login using {type(self._authorizer).__name__}.")

            await self.refresh_services()

        return self

    async def refresh_services(self) -> None:
        with self._loading():
            try:
                await self._service_manager.refresh_organizations()
            except UnauthorizedException:
                await self.login()
                await self._service_manager.refresh_organizations()

            self._refresh_orgs()
            self._refresh_hubs()
            await self._refresh_workspaces()
            self.button_text = "Refresh Evo Services"

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
        """Get an API connector for the currently selected hub."""
        return self._service_manager.get_connector()

    def get_environment(self) -> Environment:
        """Get an environment with the currently selected organization, hub, and workspace."""
        return self._service_manager.get_environment()

    def get_org_id(self) -> UUID:
        """Gets the ID of the currently selected organization."""
        return self._service_manager.get_org_id()

    def get_cache(self) -> ICache:
        """Gets the cache for this context."""
        return self._cache

    def create_client(self, client_class: type[T_client], *args: Any, **kwargs: Any) -> T_client:
        """Create a client for the currently selected workspace."""
        return self._service_manager.create_client(client_class, *args, **kwargs)


def display_object_links(object_reference: Any, label: str = "Object links") -> None:
    """Display Evo Viewer and Portal links for a geoscience object.

    In a Jupyter environment, this renders styled HTML with two clickable links:
    - "Open in Evo Viewer" - Opens the 3D viewer for the object
    - "Open in Evo Portal" - Opens the object's overview page

    Outside of notebooks, this function does nothing.

    :param object_reference: Object reference - can be:
        - A string URL
        - An ObjectReference
        - A typed object (PointSet, BlockModel, Regular3DGrid, etc.)
        - ObjectMetadata or DownloadedObject
    :param label: Label text to display above the links
    """
    try:
        from IPython.display import HTML, display
    except ImportError:
        return

    try:
        ref_str = serialize_object_reference(object_reference)
        viewer_url = get_viewer_url_from_reference(ref_str)
        portal_url = get_portal_url_from_reference(ref_str)
    except Exception:
        return

    style = (
        "margin:8px 0;padding:8px 12px;border:1px solid #e0e0e0;border-radius:6px;"
        "background:#fafafa;font-family:Inter,Segoe UI,Arial,sans-serif;"
    )
    link_style = "margin-right:16px;color:#0066cc;text-decoration:none;"
    html = f"""
    <div style='{style}'>
      <div style='font-weight:600;color:#333;margin-bottom:4px'>{label}</div>
      <a style='{link_style}' href='{viewer_url}' target='_blank'>🔍 Open in Evo Viewer</a>
      <a style='{link_style}' href='{portal_url}' target='_blank'>📋 Open in Evo Portal</a>
    </div>
    """
    display(HTML(html))

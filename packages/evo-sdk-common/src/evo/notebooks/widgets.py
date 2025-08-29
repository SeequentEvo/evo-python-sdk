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
from evo.common.interfaces import IAuthorizer, ICache, IFeedback, ITransport
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


class ServiceManagerWidget(widgets.HBox):
    def __init__(self, transport: ITransport, authorizer: IAuthorizer, discovery_url: str, cache: ICache) -> None:
        """
        :param transport: The transport to use for API requests.
        :param authorizer: The authorizer to use for API requests.
        :param discovery_url: The URL of the Evo Discovery service.
        :param cache: The cache to use for storing tokens and other data.
        """
        self._authorizer = authorizer
        self._service_manager = ServiceManager(
            transport=transport,
            authorizer=authorizer,
            discovery_url=discovery_url,
        )
        self._cache = cache
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

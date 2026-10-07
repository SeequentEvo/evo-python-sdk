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
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
from uuid import UUID, uuid4

import anywidget
import traitlets
from IPython.display import display

from evo import logging
from evo.common import APIConnector, BaseAPIClient, Environment
from evo.common.exceptions import UnauthorizedException
from evo.common.interfaces import IAuthorizer, ICache, IContext, IFeedback, ITransport
from evo.discovery import Hub, Organization
from evo.oauth import AccessTokenAuthorizer, AnyScopes, EvoScopes, OAuthConnector
from evo.oauth.exceptions import OAuthError
from evo.service_manager import ServiceManager
from evo.workspaces import Workspace

from ._consts import (
    DEFAULT_BASE_URI,
    DEFAULT_CACHE_LOCATION,
    DEFAULT_DISCOVERY_URL,
    DEFAULT_REDIRECT_URL,
)
from ._helpers import FileName, asset_data_uri, init_cache, read_asset_text
from .authorizer import AuthorizationCodeAuthorizer
from .env import DotEnv

if TYPE_CHECKING:
    from aiohttp.typedefs import StrOrURL

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = [
    "FeedbackWidget",
    "OrgSelectorWidget",
    "ServiceManagerWidget",
    "WorkspaceSelectorWidget",
]

_STYLESHEET = read_asset_text("widgets.css")


class DropdownSelectorWidget(anywidget.AnyWidget, Generic[T]):
    _esm = read_asset_text("selector.js")
    _css = _STYLESHEET

    UNSELECTED: tuple[str, T]

    label = traitlets.Unicode("").tag(sync=True)
    # Options are serialized to strings because widget state is synchronised to the frontend as JSON.
    options = traitlets.List(traitlets.List(traitlets.Unicode())).tag(sync=True)
    value = traitlets.Unicode("").tag(sync=True)
    disabled = traitlets.Bool(True).tag(sync=True)
    loading = traitlets.Bool(False).tag(sync=True)
    spinner = traitlets.Unicode("").tag(sync=True)

    def __init__(self, label: str, env: DotEnv) -> None:
        self._env = env
        unselected_label, unselected_value = self.UNSELECTED
        super().__init__(
            label=label,
            options=[[unselected_label, self._serialize(unselected_value)]],
            value=self._serialize(unselected_value),
            disabled=True,
            spinner=asset_data_uri("loading.gif"),
        )
        self.observe(self._update_selected, names="value")

    def _get_options(self) -> list[tuple[str, T]]:
        raise NotImplementedError("Subclasses must implement this method.")

    def _on_selected(self, value: T | None) -> None: ...

    @contextlib.contextmanager
    def _loading(self) -> Iterator[None]:
        self.disabled = True
        self.loading = True
        try:
            yield
        finally:
            self.loading = False
            self.disabled = False

    def _update_selected(self, _: dict) -> None:
        self.selected = new_value = self._deserialize(self.value)
        self._on_selected(new_value if new_value != self.UNSELECTED[1] else None)

    def refresh(self) -> None:
        logger.debug(f"Refreshing {self.__class__.__name__} options...")
        self.disabled = True
        selected = self.selected
        options = [self.UNSELECTED] + self._get_options()
        self.options = [[label, self._serialize(value)] for label, value in options]
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
        self.disabled = len(options) <= 1

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
        self.value = self._serialize(value)


_NULL_UUID = UUID(int=0)


class _UUIDSelectorWidget(DropdownSelectorWidget[UUID]):
    @classmethod
    def _serialize(cls, value: UUID) -> str:
        return str(value)

    @classmethod
    def _deserialize(cls, value: str) -> UUID:
        return UUID(value)


class OrgSelectorWidget(_UUIDSelectorWidget):
    UNSELECTED = ("Select Instance", _NULL_UUID)

    def __init__(self, env: DotEnv, manager: ServiceManager) -> None:
        self._manager = manager
        super().__init__("Instance", env)

    def _get_options(self) -> list[tuple[str, UUID]]:
        return [(org.display_name, org.id) for org in self._manager.list_organizations()]

    def _on_selected(self, value: UUID | None) -> None:
        self._manager.set_current_organization(value)
        # Auto-select the first hub for the selected organization
        if value is not None:
            hubs = self._manager.list_hubs()
            if hubs:
                self._manager.set_current_hub(hubs[0].code)


class WorkspaceSelectorWidget(_UUIDSelectorWidget):
    UNSELECTED = ("Select Workspace", _NULL_UUID)

    def __init__(self, env: DotEnv, manager: ServiceManager, org_selector: OrgSelectorWidget) -> None:
        self._manager = manager
        super().__init__("Workspace", env)
        org_selector.observe(self._on_org_selected, names="value")

    async def refresh_workspaces(self) -> None:
        with self._loading():
            await self._manager.refresh_workspaces()
            self.refresh()

    def _on_org_selected(self, _: dict) -> asyncio.Future:
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
    _esm = read_asset_text("service_manager.js")
    _css = _STYLESHEET

    logo = traitlets.Unicode("").tag(sync=True)
    spinner = traitlets.Unicode("").tag(sync=True)
    button_text = traitlets.Unicode("").tag(sync=True)
    disabled = traitlets.Bool(False).tag(sync=True)
    loading = traitlets.Bool(False).tag(sync=True)
    message = traitlets.Unicode("").tag(sync=True)
    browser_token_required = traitlets.Bool(False).tag(sync=True)
    browser_token_channel = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        transport: ITransport,
        authorizer: IAuthorizer,
        discovery_url: str,
        cache: ICache,
        *,
        browser_token: bool = False,
    ) -> None:
        """
        :param transport: The transport to use for API requests.
        :param authorizer: The authorizer to use for API requests.
        :param discovery_url: The URL of the Evo Discovery service.
        :param cache: The cache to use for storing tokens and other data.
        """
        self._authorizer = authorizer
        self._cache = cache
        self._browser_token_event = asyncio.Event() if browser_token else None
        self._browser_token: str | None = None
        self._browser_token_error: str | None = None
        self._browser_channel = None
        self._browser_channel_listener = None
        browser_token_channel = (
            self._open_browser_token_channel() if browser_token and sys.platform == "emscripten" else ""
        )
        self._service_manager = ServiceManager(
            transport=transport,
            authorizer=authorizer,
            discovery_url=discovery_url,
            cache=cache,
        )
        env = DotEnv(cache)

        super().__init__(
            logo=asset_data_uri("EvoBadgeCharcoal_FV.png"),
            spinner=asset_data_uri("loading.gif"),
            button_text="Refresh Evo Services" if self._is_externally_authorized else "Sign In",
            browser_token_required=browser_token,
            browser_token_channel=browser_token_channel,
        )
        self.on_msg(self._handle_frontend_msg)

        self._org_selector = OrgSelectorWidget(env, self._service_manager)
        self._workspace_selector = WorkspaceSelectorWidget(env, self._service_manager, self._org_selector)

        display(self, self._org_selector, self._workspace_selector)

    def _open_browser_token_channel(self) -> str:
        from js import BroadcastChannel
        from pyodide.ffi import create_proxy

        channel_id = uuid4().hex
        channel = BroadcastChannel.new(f"evo-browser-token-{channel_id}")
        listener = create_proxy(lambda message: self._handle_frontend_msg(self, message.data.to_py(), []))
        channel.addEventListener("message", listener)
        self._browser_channel = channel
        self._browser_channel_listener = listener
        return channel_id

    def _close_browser_token_channel(self) -> None:
        if self._browser_channel is not None:
            self._browser_channel.removeEventListener("message", self._browser_channel_listener)
            self._browser_channel.close()
            self._browser_channel_listener.destroy()
            self._browser_channel = None
            self._browser_channel_listener = None

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
        from evo.aio import AioTransport

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

    @classmethod
    def with_access_token(
        cls,
        access_token: str | None = None,
        discovery_url: str = DEFAULT_DISCOVERY_URL,
        cache_location: FileName = DEFAULT_CACHE_LOCATION,
        transport: ITransport | None = None,
        user_agent: str = "evo-sdk-common",
    ) -> ServiceManagerWidget:
        """Create a ServiceManagerWidget from an access token that was issued elsewhere.

        This is intended for hosted environments, such as JupyterLite, where the page hosting the notebook has already
        signed the user in. The token is not refreshed, so a new one must be supplied when it expires.

        ```python
        manager = await ServiceManagerWidget.with_access_token().login()
        ```

        :param access_token: The access token to authorise requests with. When omitted in Pyodide, the widget frontend
            reads it from the hosting page's local storage during login.
        :param discovery_url: The URL of the Evo Discovery service.
        :param cache_location: The location of the cache file.
        :param transport: The transport to use for API requests. Defaults to the browser `fetch` transport in a Pyodide
            runtime, and to `AioTransport` elsewhere.
        :param user_agent: The value to provide in the `User-Agent` header.

        :returns: The new ServiceManagerWidget.
        """
        in_pyodide = sys.platform == "emscripten"

        if access_token is None:
            if not in_pyodide:
                raise ValueError("An access token must be provided when not running in a Pyodide runtime.")

        if transport is None:
            if in_pyodide:
                from evo.pyodide import JsTransport

                transport = JsTransport(user_agent=user_agent)
            else:
                from evo.aio import AioTransport

                transport = AioTransport(user_agent=user_agent)

        return cls(
            transport,
            AccessTokenAuthorizer(access_token or ""),
            discovery_url,
            init_cache(cache_location),
            browser_token=access_token is None,
        )

    async def _receive_browser_token(self, timeout_seconds: int) -> None:
        event = self._browser_token_event
        if event is None:
            return
        try:
            await asyncio.wait_for(event.wait(), timeout_seconds)
        except TimeoutError as exc:
            raise OAuthError(
                "Timed out waiting for the browser access token. Display the widget and try again."
            ) from exc
        finally:
            self._close_browser_token_channel()
        if self._browser_token_error:
            raise OAuthError(self._browser_token_error)
        if not self._browser_token:
            raise OAuthError("No access token received from the hosting page. Sign in first.")

        self._authorizer = AccessTokenAuthorizer(self._browser_token)
        self._service_manager._authorizer = self._authorizer
        self._browser_token = None
        self._browser_token_event = None

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
            await self._receive_browser_token(timeout_seconds)
            match self._authorizer:
                case AuthorizationCodeAuthorizer():
                    await self._login_with_auth_code(timeout_seconds)
                case AccessTokenAuthorizer():
                    pass  # The token was issued elsewhere, so there is nothing to do.
                case unknown:
                    raise NotImplementedError(f"ServiceManagerWidget cannot login using {type(unknown).__name__}.")

            # Refresh the services after logging in.
            await self.refresh_services()
        return self

    @property
    def cache(self) -> ICache:
        return self._cache

    @property
    def _is_externally_authorized(self) -> bool:
        """Whether credentials come from outside the widget, so it cannot initiate a sign in itself."""
        return isinstance(self._authorizer, AccessTokenAuthorizer)

    def _update_btn(self, signed_in: bool) -> None:
        if signed_in or self._is_externally_authorized:
            self.button_text = "Refresh Evo Services"
        else:
            self.button_text = "Sign In"

    def _handle_frontend_msg(self, _widget: object, content: object, _buffers: list) -> asyncio.Future | None:
        if isinstance(content, dict) and content.get("type") == "click":
            return asyncio.ensure_future(self.refresh_services())
        if isinstance(content, dict) and content.get("type") == "browser_token" and self._browser_token_event:
            self._browser_token = content.get("token") if isinstance(content.get("token"), str) else None
            self._browser_token_error = content.get("error") if isinstance(content.get("error"), str) else None
            self._browser_token_event.set()
        return None

    @contextlib.contextmanager
    def _loading(self) -> Iterator[None]:
        self.disabled = True
        self.loading = True
        try:
            yield
        finally:
            self.loading = False
            self.disabled = False

    @contextlib.contextmanager
    def _loading_services(self) -> Iterator[None]:
        self._org_selector.disabled = True
        self._workspace_selector.disabled = True
        try:
            yield
        finally:
            self._org_selector.refresh()

    async def refresh_services(self) -> None:
        with self._loading():
            with self._loading_services():
                try:
                    await self._service_manager.refresh_organizations()
                except UnauthorizedException as exc:  # Expired token or user not logged in.
                    if isinstance(self._authorizer, AccessTokenAuthorizer):
                        raise OAuthError(
                            "The access token is no longer valid, and cannot be refreshed from here. Sign in again"
                            " and create a new ServiceManagerWidget with the new token."
                        ) from exc

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


class _ProgressWidget(anywidget.AnyWidget):
    _esm = read_asset_text("feedback.js")
    _css = _STYLESHEET

    label = traitlets.Unicode("").tag(sync=True)
    value = traitlets.Float(0.0).tag(sync=True)
    message = traitlets.Unicode("").tag(sync=True)


class FeedbackWidget(IFeedback):
    """Simple feedback widget for displaying progress and messages to the user."""

    def __init__(self, label: str) -> None:
        """
        :param label: The label for the feedback widget.
        """
        self._widget = _ProgressWidget(label=label)
        self._last_message = ""
        display(self._widget)

    def progress(self, progress: float, message: str | None = None) -> None:
        """Progress the feedback and update the text to message.

        This can raise an exception to cancel the current operation.

        :param progress: A float between 0 and 1 representing the progress of the operation as a percentage.
        :param message: An optional message to display to the user.
        """
        self._widget.value = progress
        if message is not None:
            self._widget.message = message

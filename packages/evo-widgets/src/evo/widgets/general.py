"""ServiceManagerWidget implementation using anywidget."""

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

import anywidget
import traitlets

from evo import logging
from evo.aio import AioTransport
from evo.common import APIConnector, BaseAPIClient, Environment
from evo.common.exceptions import UnauthorizedException
from evo.common.interfaces import IAuthorizer, ICache, IFeedback, ITransport
from evo.discovery import Hub, Organization
from evo.oauth import AnyScopes, EvoScopes, OAuthConnector
from evo.service_manager import ServiceManager
from evo.workspaces import Workspace
from ._helpers import FileName, init_cache


from ._consts import (
    DEFAULT_BASE_URI,
    DEFAULT_CACHE_LOCATION,
    DEFAULT_DISCOVERY_URL,
    DEFAULT_REDIRECT_URL,
)
from .authorizer import AuthorizationCodeAuthorizer
from .env import DotEnv




__all__ = ["ServiceManagerWidget"]


class ServiceManagerWidget(anywidget.AnyWidget):
    """Interactive widget for managing Evo services authentication and selection."""
    
    _esm = Path(__file__).parent / "static" / "service_manager.js"
    _css = Path(__file__).parent / "static" / "service_manager.css"
    
    # Authentication state
    signed_in = traitlets.Bool(False).tag(sync=True)
    loading = traitlets.Bool(False).tag(sync=True)
    
    # Dropdown options and selections
    organizations = traitlets.List([]).tag(sync=True)
    selected_org_id = traitlets.Unicode("").tag(sync=True)
    
    hubs = traitlets.List([]).tag(sync=True)
    selected_hub_code = traitlets.Unicode("").tag(sync=True)
    
    workspaces = traitlets.List([]).tag(sync=True)
    selected_workspace_id = traitlets.Unicode("").tag(sync=True)
    
    # Button state
    button_text = traitlets.Unicode("Sign In").tag(sync=True)
    
    # Messages from frontend
    action = traitlets.Unicode("").tag(sync=True)
    
    def __init__(self, transport:ITransport, authorizer:IAuthorizer, discovery_url:str, cache:ICache, **kwargs):
        """Initialize the ServiceManagerWidget.
        
        Args:
            transport: The transport to use for API requests
            authorizer: The authorizer to use for API requests
            discovery_url: The URL of the Evo Discovery service
            cache: The cache to use for storing tokens and other data
        """
        super().__init__(**kwargs)
        
        self._authorizer = authorizer
        self._transport = transport
        self._discovery_url = discovery_url
        self._cache = cache
        self._env = DotEnv(cache)
        self._service_manager = None
        
        # Initialize service manager if dependencies provided
        if transport and authorizer and discovery_url:
            self._service_manager = ServiceManager(
                transport=transport,
                authorizer=authorizer,
                discovery_url=discovery_url,
            )
        
        # Observe action changes from frontend
        self.observe(self._on_action, names=['action'])
        self.observe(self._on_org_change, names=['selected_org_id'])
        self.observe(self._on_hub_change, names=['selected_hub_code'])
        self.observe(self._on_workspace_change, names=['selected_workspace_id'])
    
    @classmethod
    def with_auth_code(
        cls,
        client_id: str,
        base_uri: str = DEFAULT_BASE_URI,
        discovery_url: str = DEFAULT_DISCOVERY_URL,
        redirect_url: str = DEFAULT_REDIRECT_URL,
        client_secret: str | None = None,
        cache_location: str = DEFAULT_CACHE_LOCATION,
        oauth_scopes=None,
        proxy=None,
    ):
        """Create a ServiceManagerWidget with an authorization code authorizer.
        
        Args:
            client_id: The client ID to use for authentication
            base_uri: The OAuth server base URI
            discovery_url: The URL of the Evo Discovery service
            redirect_url: The local URL to redirect the user back to after authorisation
            client_secret: The client secret to use for authentication
            cache_location: The location of the cache file
            oauth_scopes: The OAuth scopes to request
            proxy: The proxy URL to use for API requests
            
        Returns:
            The new ServiceManagerWidget
        """

        
        # Initialize cache
        cache = init_cache(cache_location)
        ignorefile = cache.root / ".gitignore"
        ignorefile.write_text("*\n")
        
        # Set default scopes if not provided
        if oauth_scopes is None:
            oauth_scopes = EvoScopes.all_evo | EvoScopes.offline_access
        
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
    
    async def login(self, timeout_seconds: int = 180):
        """Authenticate the user and obtain an access token.
        
        Args:
            timeout_seconds: The maximum time (in seconds) to wait for authorisation
            
        Returns:
            The current instance of the ServiceManagerWidget
        """
        
        
        # Open transport
        await self._service_manager._transport.open()
        
        self.loading = True
        try:
            # Handle authorization
            if isinstance(self._authorizer, AuthorizationCodeAuthorizer):
                if not await self._authorizer.reuse_token():
                    await self._authorizer.login(timeout_seconds=timeout_seconds)
            
            # Refresh services after login
            await self.refresh_services()
            
        finally:
            self.loading = False
        
        return self
    
    @property
    def cache(self):
        """Get the cache instance used by this widget.
        
        Returns:
            The cache instance
        """
        return self._cache
    
    async def refresh_services(self):
        """Refresh the list of organizations, hubs, and workspaces."""
        
        self.loading = True
        try:
            try:
                await self._service_manager.refresh_organizations()
            except UnauthorizedException:
                # Re-login if token expired
                await self.login()
                await self._service_manager.refresh_organizations()
            
            # Update organizations list
            orgs = self._service_manager.list_organizations()
            self.organizations = [
                {"id": str(org.id), "name": org.display_name}
                for org in orgs
            ]
            
            # Update signed in state and button
            self.signed_in = True
            self.button_text = "Refresh Evo Services"
            
            # Refresh hubs and workspaces if already selected
            if self.selected_org_id:
                self._refresh_hubs()
                if self.selected_hub_code:
                    await self._refresh_workspaces()
                    
        finally:
            self.loading = False
    
    def _refresh_hubs(self):
        """Refresh the list of hubs for the selected organization."""
        if self.selected_org_id:
            try:
                hub_list = self._service_manager.list_hubs()
                self.hubs = [
                    {"code": hub.code, "name": hub.display_name}
                    for hub in hub_list
                ]
            except Exception:
                self.hubs = []
        else:
            self.hubs = []
    
    async def _refresh_workspaces(self):
        """Refresh the list of workspaces for the selected hub."""
        if self.selected_hub_code:
            try:
                await self._service_manager.refresh_workspaces()
                ws_list = self._service_manager.list_workspaces()
                self.workspaces = [
                    {"id": str(ws.id), "name": ws.display_name}
                    for ws in ws_list
                ]
            except Exception:
                self.workspaces = []
        else:
            self.workspaces = []
    
    def _on_action(self, change):
        """Handle action messages from the frontend."""
        action = change['new']
        if action == 'refresh':
            asyncio.create_task(self.refresh_services())
    
    def _on_org_change(self, change):
        """Handle organization selection changes."""
        org_id = change['new']
        if org_id and self._service_manager:
            try:
                uuid_org_id = UUID(org_id) if org_id else None
                self._service_manager.set_current_organization(uuid_org_id)
                self._refresh_hubs()
            except Exception:
                pass
    
    def _on_hub_change(self, change):
        """Handle hub selection changes."""
        hub_code = change['new']
        if self._service_manager:
            self._service_manager.set_current_hub(hub_code if hub_code else None)
            asyncio.create_task(self._refresh_workspaces())
    
    def _on_workspace_change(self, change):
        """Handle workspace selection changes."""
        workspace_id = change['new']
        if workspace_id and self._service_manager:
            try:
                uuid_ws_id = UUID(workspace_id) if workspace_id else None
                self._service_manager.set_current_workspace(uuid_ws_id)
            except Exception:
                pass
    
    def get_connector(self):
        """Get an API connector for the currently selected hub."""
        return self._service_manager.get_connector()
    
    def get_environment(self):
        """Get an environment with the currently selected organization, hub, and workspace."""
        return self._service_manager.get_environment()
    
    def create_client(self, client_class, *args, **kwargs):
        """Create a client for the currently selected workspace."""
        return self._service_manager.create_client(client_class, *args, **kwargs)



class FeedbackWidget(anywidget.AnyWidget):
    """Simple feedback widget for displaying progress and messages to the user."""
    
    _esm = Path(__file__).parent / "static" / "feedback.js"
    _css = Path(__file__).parent / "static" / "feedback.css"
    
    # Widget state
    label = traitlets.Unicode("").tag(sync=True)
    progress_value = traitlets.Float(0.0).tag(sync=True)
    progress_percent = traitlets.Unicode("0.0%").tag(sync=True)
    message = traitlets.Unicode("").tag(sync=True)
    
    def __init__(self, label: str, **kwargs):
        """Initialize the FeedbackWidget.
        
        Args:
            label: The label for the feedback widget
        """
        super().__init__(**kwargs)
        self.label = label
        self._last_message = ""
    
    def progress(self, progress: float, message: str | None = None) -> None:
        """Update the progress and optional message.
        
        This can raise an exception to cancel the current operation.
        
        Args:
            progress: A float between 0 and 1 representing the progress (0-100%)
            message: An optional message to display to the user
        """
        # Clamp progress between 0 and 1
        progress = max(0.0, min(1.0, progress))
        
        self.progress_value = progress
        self.progress_percent = f"{progress * 100:5.1f}%"
        
        if message is not None:
            self.message = message
            self._last_message = message

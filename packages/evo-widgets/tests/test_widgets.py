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

"""Tests for the interactive widget state logic in evo.widgets._interactive.widgets."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest


class TestLoadingContextManager(unittest.TestCase):
    """Tests for the _loading() context manager."""

    def _create_widget(self) -> MagicMock:
        """Create a mock widget with the _loading context manager behavior."""
        from evo.widgets._interactive.widgets import ServiceManagerWidget

        # We need to instantiate enough to test _loading() behavior
        # Patch out the dependencies
        with (
            patch("evo.widgets._interactive.widgets.ServiceManager"),
            patch("evo.widgets._interactive.widgets.DotEnv"),
            patch("IPython.display.display"),
        ):
            mock_transport = MagicMock()
            mock_authorizer = MagicMock()
            mock_cache = MagicMock()
            widget = ServiceManagerWidget(
                transport=mock_transport,
                authorizer=mock_authorizer,
                discovery_url="https://example.com",
                cache=mock_cache,
            )
        return widget

    def test_loading_sets_state_on_entry(self) -> None:
        """_loading() should set main_loading and button_disabled to True on entry."""
        widget = self._create_widget()

        self.assertFalse(widget.main_loading)
        self.assertFalse(widget.button_disabled)

        with widget._loading():
            self.assertTrue(widget.main_loading)
            self.assertTrue(widget.button_disabled)

    def test_loading_clears_state_on_exit(self) -> None:
        """_loading() should clear main_loading and button_disabled on exit."""
        widget = self._create_widget()

        with widget._loading():
            pass

        self.assertFalse(widget.main_loading)
        self.assertFalse(widget.button_disabled)

    def test_loading_clears_on_exception(self) -> None:
        """_loading() should clear state even when exception is raised."""
        widget = self._create_widget()

        with self.assertRaises(ValueError):
            with widget._loading():
                raise ValueError("test error")

        self.assertFalse(widget.main_loading)
        self.assertFalse(widget.button_disabled)


class TestOrgChangeHandler(unittest.TestCase):
    """Tests for _on_org_change() handler."""

    def _create_widget(self) -> MagicMock:
        """Create a mock widget with mocked service manager."""
        from evo.widgets._interactive.widgets import ServiceManagerWidget

        with (
            patch("evo.widgets._interactive.widgets.ServiceManager") as MockSM,
            patch("evo.widgets._interactive.widgets.DotEnv"),
            patch("IPython.display.display"),
        ):
            mock_sm_instance = MagicMock()
            MockSM.return_value = mock_sm_instance

            mock_transport = MagicMock()
            mock_authorizer = MagicMock()
            mock_cache = MagicMock()
            widget = ServiceManagerWidget(
                transport=mock_transport,
                authorizer=mock_authorizer,
                discovery_url="https://example.com",
                cache=mock_cache,
            )
            widget._mock_service_manager = mock_sm_instance
        return widget

    def test_org_change_sets_organization(self) -> None:
        """_on_org_change() should call set_current_organization with the UUID."""
        widget = self._create_widget()
        test_uuid = "12345678-1234-5678-1234-567812345678"
        widget._mock_service_manager.refresh_workspaces = AsyncMock()
        widget._mock_service_manager.list_workspaces.return_value = []

        async def run() -> None:
            widget._on_org_change({"new": test_uuid})
            await asyncio.sleep(0)

        asyncio.run(run())
        widget._mock_service_manager.set_current_organization.assert_called_once_with(UUID(test_uuid))

    def test_org_change_auto_selects_hub(self) -> None:
        """_on_org_change() should auto-select the first hub."""
        widget = self._create_widget()
        test_uuid = "12345678-1234-5678-1234-567812345678"

        mock_hub = MagicMock()
        mock_hub.code = "hub-1"
        widget._mock_service_manager.list_hubs.return_value = [mock_hub]
        widget._mock_service_manager.refresh_workspaces = AsyncMock()
        widget._mock_service_manager.list_workspaces.return_value = []

        async def run() -> None:
            widget._on_org_change({"new": test_uuid})
            await asyncio.sleep(0)

        asyncio.run(run())
        widget._mock_service_manager.set_current_hub.assert_called_once_with("hub-1")

    def test_org_change_no_hubs_does_not_crash(self) -> None:
        """_on_org_change() should handle case when no hubs are available."""
        widget = self._create_widget()
        test_uuid = "12345678-1234-5678-1234-567812345678"

        widget._mock_service_manager.list_hubs.return_value = []
        widget._mock_service_manager.refresh_workspaces = AsyncMock()
        widget._mock_service_manager.list_workspaces.return_value = []

        async def run() -> None:
            widget._on_org_change({"new": test_uuid})
            await asyncio.sleep(0)

        asyncio.run(run())
        widget._mock_service_manager.set_current_hub.assert_not_called()

    def test_org_change_triggers_workspace_refresh(self) -> None:
        """_on_org_change() should call refresh_workspaces."""
        widget = self._create_widget()
        test_uuid = "12345678-1234-5678-1234-567812345678"
        widget._mock_service_manager.list_hubs.return_value = []
        widget._mock_service_manager.refresh_workspaces = AsyncMock()
        widget._mock_service_manager.list_workspaces.return_value = []

        async def run() -> None:
            widget._on_org_change({"new": test_uuid})
            await asyncio.sleep(0)

        asyncio.run(run())
        widget._mock_service_manager.refresh_workspaces.assert_called_once()

    def test_org_change_skips_null(self) -> None:
        """_on_org_change() should do nothing for null UUID."""
        from evo.widgets._interactive.widgets import _NULL_UUID

        widget = self._create_widget()

        widget._on_org_change({"new": str(_NULL_UUID)})

        widget._mock_service_manager.set_current_organization.assert_not_called()


class TestAsyncErrorHandling(unittest.TestCase):
    """Tests for _run_async() and _handle_async_error()."""

    def _create_widget(self) -> MagicMock:
        """Create a mock widget."""
        from evo.widgets._interactive.widgets import ServiceManagerWidget

        with (
            patch("evo.widgets._interactive.widgets.ServiceManager"),
            patch("evo.widgets._interactive.widgets.DotEnv"),
            patch("IPython.display.display"),
        ):
            mock_transport = MagicMock()
            mock_authorizer = MagicMock()
            mock_cache = MagicMock()
            widget = ServiceManagerWidget(
                transport=mock_transport,
                authorizer=mock_authorizer,
                discovery_url="https://example.com",
                cache=mock_cache,
            )
        return widget

    def test_handle_async_error_logs_exception(self) -> None:
        """_handle_async_error() should log exceptions from failed futures."""
        widget = self._create_widget()

        mock_future = MagicMock()
        mock_future.cancelled.return_value = False
        mock_future.exception.return_value = ValueError("test error")

        with patch("evo.widgets._interactive.widgets.logger") as mock_logger:
            widget._handle_async_error(mock_future)

        mock_logger.error.assert_called_once()
        self.assertIn("test error", str(mock_logger.error.call_args))

    def test_handle_async_error_ignores_cancelled(self) -> None:
        """_handle_async_error() should not log cancelled futures."""
        widget = self._create_widget()

        mock_future = MagicMock()
        mock_future.cancelled.return_value = True

        with patch("evo.widgets._interactive.widgets.logger") as mock_logger:
            widget._handle_async_error(mock_future)

        mock_logger.error.assert_not_called()

    def test_handle_async_error_ignores_success(self) -> None:
        """_handle_async_error() should not log successful futures."""
        widget = self._create_widget()

        mock_future = MagicMock()
        mock_future.cancelled.return_value = False
        mock_future.exception.return_value = None

        with patch("evo.widgets._interactive.widgets.logger") as mock_logger:
            widget._handle_async_error(mock_future)

        mock_logger.error.assert_not_called()


@pytest.mark.asyncio
class TestRefreshServicesNoRecursion:
    """Tests to verify refresh_services() doesn't cause infinite recursion."""

    async def test_refresh_services_calls_login_helper_not_login(self) -> None:
        """On UnauthorizedException, refresh_services should call _login_with_auth_code, not login()."""
        from evo.common.exceptions import UnauthorizedException
        from evo.widgets._interactive.widgets import ServiceManagerWidget

        # Create a properly initialized UnauthorizedException
        unauth_error = UnauthorizedException(status=401, reason="Unauthorized", content=None, headers=None)

        with (
            patch("evo.widgets._interactive.widgets.ServiceManager") as MockSM,
            patch("evo.widgets._interactive.widgets.DotEnv"),
            patch("IPython.display.display"),
        ):
            mock_sm_instance = MagicMock()
            mock_sm_instance.refresh_organizations = AsyncMock(side_effect=[unauth_error, None])
            mock_sm_instance.list_organizations.return_value = []
            mock_sm_instance.refresh_workspaces = AsyncMock()
            mock_sm_instance.list_workspaces.return_value = []
            MockSM.return_value = mock_sm_instance

            mock_transport = MagicMock()
            mock_authorizer = MagicMock()
            mock_cache = MagicMock()
            widget = ServiceManagerWidget(
                transport=mock_transport,
                authorizer=mock_authorizer,
                discovery_url="https://example.com",
                cache=mock_cache,
            )

            widget._login_with_auth_code = AsyncMock()
            widget.login = AsyncMock()

            await widget.refresh_services()

            widget._login_with_auth_code.assert_called_once_with(timeout_seconds=180)
            widget.login.assert_not_called()


if __name__ == "__main__":
    unittest.main()

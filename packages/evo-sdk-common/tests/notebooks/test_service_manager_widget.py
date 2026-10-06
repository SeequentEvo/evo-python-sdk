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

import json
import tempfile
import unittest
from unittest import mock
from uuid import UUID

from evo.aio import AioTransport
from evo.common import RequestMethod
from evo.common.exceptions import SelectionError
from evo.common.interfaces import IContext
from evo.common.test_tools import MockResponse, TestTransport
from evo.notebooks import ServiceManagerWidget
from evo.oauth import AccessTokenAuthorizer
from evo.oauth.exceptions import OAuthError

from ..data import load_test_data

DISCOVERY_URL = "https://discover.test"
EMPTY_WORKSPACE_LIST = json.dumps(
    {
        "results": [],
        "links": {
            "first": "http://first",
            "last": "http://last",
            "next": None,
            "previous": None,
            "count": 0,
            "total": 0,
        },
    }
)


class _ServiceManagerWidgetTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_location = self._tmp.name

    def build_widget(self, transport: TestTransport | None = None) -> ServiceManagerWidget:
        return ServiceManagerWidget.with_access_token(
            "test-token",
            discovery_url=DISCOVERY_URL,
            cache_location=self.cache_location,
            transport=transport or TestTransport(base_url=DISCOVERY_URL),
        )


class TestWithAccessToken(_ServiceManagerWidgetTestCase):
    def test_uses_access_token_authorizer(self) -> None:
        widget = self.build_widget()
        assert isinstance(widget._authorizer, AccessTokenAuthorizer)

    async def test_authorizes_requests_with_the_token(self) -> None:
        widget = self.build_widget()
        headers = await widget._authorizer.get_default_headers()
        assert headers == {"Authorization": "Bearer test-token"}

    def test_is_a_context(self) -> None:
        """The playground widget was only duck-typed, so this is the main benefit of unifying the two."""
        assert isinstance(self.build_widget(), IContext)

    def test_honours_an_explicit_transport(self) -> None:
        transport = TestTransport(base_url=DISCOVERY_URL)
        widget = self.build_widget(transport)
        assert widget._service_manager._transport is transport

    def test_defaults_to_aiohttp_outside_pyodide(self) -> None:
        widget = ServiceManagerWidget.with_access_token("test-token", cache_location=self.cache_location)
        assert isinstance(widget._service_manager._transport, AioTransport)

    def test_requires_a_token_outside_pyodide(self) -> None:
        with self.assertRaisesRegex(ValueError, "access token must be provided"):
            ServiceManagerWidget.with_access_token(cache_location=self.cache_location)

    def test_reads_the_token_from_the_browser_in_pyodide(self) -> None:
        with (
            mock.patch("evo.notebooks.widgets.sys.platform", "emscripten"),
            mock.patch.dict(
                "sys.modules", {"evo.pyodide": mock.MagicMock(get_browser_access_token=lambda: "browser-token")}
            ),
        ):
            widget = self.build_widget()
        assert isinstance(widget._authorizer, AccessTokenAuthorizer)

    def test_button_invites_a_refresh_rather_than_a_sign_in(self) -> None:
        assert self.build_widget()._btn.description == "Refresh Evo Services"

    def test_button_invites_a_sign_in_for_the_auth_code_flow(self) -> None:
        widget = ServiceManagerWidget.with_auth_code(client_id="test", cache_location=self.cache_location)
        assert widget._btn.description == "Sign In"

    def test_does_not_persist_the_token(self) -> None:
        """Unlike the auth code flow, an externally issued token must not be written to the cache."""
        self.build_widget()
        written = [path.read_text() for path in self._tmp_files()]
        assert not any("test-token" in content for content in written)

    def _tmp_files(self) -> list:
        from pathlib import Path

        return [path for path in Path(self.cache_location).rglob("*") if path.is_file()]


class TestLoginWithAccessToken(_ServiceManagerWidgetTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.transport = TestTransport(base_url=DISCOVERY_URL)
        self.discovery_data = json.dumps(load_test_data("successful_service_discovery.json"))
        self.widget = self.build_widget(self.transport)

    def _respond(self, discovery_status: int = 200) -> None:
        async def handler(*, method: RequestMethod, url: str, **kwargs: object) -> MockResponse:
            if "discovery" in url:
                return MockResponse(
                    status_code=discovery_status,
                    content=self.discovery_data if discovery_status == 200 else "",
                    headers={"Content-Type": "application/json"},
                )
            return MockResponse(
                status_code=200, content=EMPTY_WORKSPACE_LIST, headers={"Content-Type": "application/json"}
            )

        self.transport.request.side_effect = handler

    async def test_login_does_not_require_an_interactive_flow(self) -> None:
        self._respond()
        assert await self.widget.login() is self.widget

    async def test_login_refreshes_the_service_list(self) -> None:
        self._respond()
        await self.widget.login()
        assert self.widget.organizations

    def _select_first_real_org(self):
        # The first org in the test data uses the nil UUID, which is also the selector's "unselected" sentinel.
        org = next(org for org in self.widget.organizations if org.id != UUID(int=0))
        self.widget._org_selector.dropdown_widget.value = org.id
        return org

    async def test_login_selects_an_organization_and_hub(self) -> None:
        self._respond()
        await self.widget.login()

        org = self._select_first_real_org()

        assert self.widget.get_org_id() == org.id
        assert isinstance(self.widget.get_org_id(), UUID)
        assert self.widget.hubs

    async def test_get_environment_requires_a_workspace(self) -> None:
        self._respond()
        await self.widget.login()
        self._select_first_real_org()

        with self.assertRaises(SelectionError):
            self.widget.get_environment()

    async def test_connector_targets_the_selected_hub(self) -> None:
        self._respond()
        await self.widget.login()
        self._select_first_real_org()

        connector = self.widget.get_connector()
        assert connector.base_url.startswith(self.widget.hubs[0].url)

    async def test_expired_token_reports_that_it_cannot_be_refreshed(self) -> None:
        """A token issued elsewhere cannot be renewed here, so this must not recurse into login()."""
        self._respond(discovery_status=401)
        with self.assertRaisesRegex(OAuthError, "cannot be refreshed from here"):
            await self.widget.login()

    async def test_unsupported_authorizer_is_rejected(self) -> None:
        self.widget._authorizer = mock.Mock()
        with self.assertRaises(NotImplementedError):
            await self.widget.login()

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

import tempfile
import unittest
from unittest import mock

from anywidget import AnyWidget

from evo.common.interfaces import IContext
from evo.notebooks import ServiceManagerWidget


class TestServiceManagerWidget(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.widget = ServiceManagerWidget.with_auth_code(client_id="test", cache_location=self._tmp.name)
        self.addCleanup(self.widget.close)

    def test_is_anywidget_and_context(self) -> None:
        assert isinstance(self.widget, AnyWidget)
        assert isinstance(self.widget, IContext)

    def test_sign_in_button_updates_for_auth_state(self) -> None:
        assert self.widget.button_text == "Sign In"

        self.widget._update_btn(True)
        assert self.widget.button_text == "Refresh Evo Services"

        self.widget._update_btn(False)
        assert self.widget.button_text == "Sign In"

    async def test_frontend_click_refreshes_services(self) -> None:
        with mock.patch.object(self.widget, "refresh_services", new_callable=mock.AsyncMock) as refresh:
            task = self.widget._handle_frontend_msg(self.widget, {"type": "click"}, [])
            assert task is not None
            await task

        refresh.assert_awaited_once()

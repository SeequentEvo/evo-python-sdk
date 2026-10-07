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
import json
import unittest
from unittest import mock

from evo.common import HTTPHeaderDict, RequestMethod
from evo.common.exceptions import ClientValueError, TransportError
from evo.common.interfaces import ITransport

from ._stubs import FakeBlob, FakeFetch, FakeFormData, FakeResponse, install

install()

from evo.pyodide import JsTransport  # noqa: E402
from evo.pyodide import transport as transport_module  # noqa: E402


class TestJsTransport(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.transport = JsTransport(user_agent="test-agent")

    def patch_fetch(self, *responses: FakeResponse, error: Exception | None = None) -> FakeFetch:
        fetch = FakeFetch(*responses, error=error)
        patcher = mock.patch.object(transport_module, "pyfetch", fetch)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fetch

    def test_implements_transport_interface(self) -> None:
        assert isinstance(self.transport, ITransport)

    def test_default_user_agent(self) -> None:
        assert JsTransport().user_agent == "evo-sdk-common"

    async def test_open_and_close_are_noops(self) -> None:
        await self.transport.open()
        await self.transport.close()

    async def test_async_context_manager(self) -> None:
        async with self.transport as entered:
            assert entered is self.transport

    async def test_get_request(self) -> None:
        fetch = self.patch_fetch(FakeResponse(status=200, body=b"hello", headers={"Content-Type": "text/plain"}))

        response = await self.transport.request(RequestMethod.GET, "https://example.test/thing")

        assert response.status == 200
        assert response.data == b"hello"
        assert response.headers["Content-Type"] == "text/plain"
        assert fetch.calls[0][0] == "https://example.test/thing"
        assert fetch.last_call["method"] == "GET"

    async def test_adds_default_user_agent(self) -> None:
        fetch = self.patch_fetch(FakeResponse())
        await self.transport.request(RequestMethod.GET, "https://example.test/")
        assert fetch.last_call["headers"]["User-Agent"] == "test-agent"

    async def test_does_not_override_user_agent(self) -> None:
        fetch = self.patch_fetch(FakeResponse())
        await self.transport.request(
            RequestMethod.GET, "https://example.test/", headers=HTTPHeaderDict({"User-Agent": "caller"})
        )
        assert fetch.last_call["headers"]["User-Agent"] == "caller"

    async def test_body_and_post_params_are_mutually_exclusive(self) -> None:
        self.patch_fetch(FakeResponse())
        with self.assertRaises(ClientValueError):
            await self.transport.request(
                RequestMethod.POST, "https://example.test/", post_params=[("a", "b")], body={"a": "b"}
            )

    async def test_serialises_object_body_as_json(self) -> None:
        fetch = self.patch_fetch(FakeResponse())

        await self.transport.request(RequestMethod.POST, "https://example.test/", body={"name": "value"})

        assert json.loads(fetch.last_call["body"]) == {"name": "value"}
        assert fetch.last_call["headers"]["Content-Type"] == "application/json"

    async def test_preserves_explicit_content_type_for_object_body(self) -> None:
        fetch = self.patch_fetch(FakeResponse())

        await self.transport.request(
            RequestMethod.POST,
            "https://example.test/",
            headers=HTTPHeaderDict({"Content-Type": "application/vnd.custom+json"}),
            body={"name": "value"},
        )

        assert fetch.last_call["headers"]["Content-Type"] == "application/vnd.custom+json"

    async def test_passes_string_body_through_unchanged(self) -> None:
        fetch = self.patch_fetch(FakeResponse())
        await self.transport.request(RequestMethod.POST, "https://example.test/", body="raw")
        assert fetch.last_call["body"] == "raw"

    async def test_url_encodes_form_post_params(self) -> None:
        fetch = self.patch_fetch(FakeResponse())

        await self.transport.request(
            RequestMethod.POST,
            "https://example.test/",
            headers=HTTPHeaderDict({"Content-Type": "application/x-www-form-urlencoded"}),
            post_params=[("one", "1"), ("two", "2")],
        )

        assert fetch.last_call["body"] == "one=1&two=2"

    async def test_builds_form_data_for_multipart_post_params(self) -> None:
        fetch = self.patch_fetch(FakeResponse())

        await self.transport.request(
            RequestMethod.POST,
            "https://example.test/",
            headers=HTTPHeaderDict({"Content-Type": "multipart/form-data"}),
            post_params=[("file", b"bytes"), ("name", "value")],
        )

        body = fetch.last_call["body"]
        assert isinstance(body, FakeFormData)
        assert body.fields[0][0] == "file"
        assert isinstance(body.fields[0][1], FakeBlob)
        assert body.fields[0][1].parts == [b"bytes"]
        assert body.fields[1] == ("name", "value")
        assert "Content-Type" not in fetch.last_call["headers"]

    async def test_head_request_uses_server_headers_without_reading_body(self) -> None:
        response = FakeResponse(status=200, headers={"Content-Length": "10", "Accept-Ranges": "bytes"})
        fetch = self.patch_fetch(response)

        result = await self.transport.request(RequestMethod.HEAD, "https://example.test/file")

        assert fetch.last_call["method"] == "HEAD"
        assert result.data == b""
        assert result.headers["Content-Length"] == "10"
        assert result.headers["Accept-Ranges"] == "bytes"
        assert not response.body_read

    async def test_head_probes_a_byte_if_cors_hides_required_headers(self) -> None:
        fetch = self.patch_fetch(FakeResponse(), FakeResponse(status=206, headers={"Content-Range": "bytes 0-0/987"}))

        result = await self.transport.request(RequestMethod.HEAD, "https://example.test/file")

        assert [call[1]["method"] for call in fetch.calls] == ["HEAD", "GET"]
        assert fetch.last_call["headers"]["Range"] == "bytes=0-0"
        assert result.headers["Content-Length"] == "987"
        assert result.headers["Accept-Ranges"] == "bytes"

    async def test_head_does_not_claim_range_support_without_evidence(self) -> None:
        probe = FakeResponse(status=200, body=b"large file")
        self.patch_fetch(FakeResponse(), probe)

        result = await self.transport.request(RequestMethod.HEAD, "https://example.test/file")

        assert "Accept-Ranges" not in result.headers
        assert "Content-Length" not in result.headers
        assert not probe.body_read

    async def test_head_request_keeps_headers_reported_by_the_server(self) -> None:
        response = FakeResponse(status=200, headers={"Content-Length": "99", "Accept-Ranges": "none"})
        self.patch_fetch(response)

        result = await self.transport.request(RequestMethod.HEAD, "https://example.test/file")

        assert result.headers["Content-Length"] == "99"
        assert result.headers["Accept-Ranges"] == "none"

    async def test_redirect_response_has_no_body(self) -> None:
        response = FakeResponse(status=302, headers={"Location": "https://example.test/elsewhere"})
        self.patch_fetch(response)

        result = await self.transport.request(RequestMethod.GET, "https://example.test/")

        assert result.status == 302
        assert result.data == b""
        assert result.headers["Location"] == "https://example.test/elsewhere"
        assert not response.body_read

    async def test_followed_task_submission_redirect_is_restored_as_303(self) -> None:
        response = FakeResponse(
            status=202, url="https://example.test/compute/orgs/org/topic/task/job/status", redirected=True
        )
        self.patch_fetch(response)

        result = await self.transport.request(RequestMethod.POST, "https://example.test/compute/orgs/org/topic/task")

        assert result.status == 303
        assert result.reason == "See Other"
        assert result.headers["Location"] == "https://example.test/compute/orgs/org/topic/task/job/status"

    async def test_unrelated_followed_redirect_fails_instead_of_returning_final_response(self) -> None:
        response = FakeResponse(status=200, url="https://example.test/file", redirected=True)
        self.patch_fetch(response)

        with self.assertRaisesRegex(TransportError, "original response is unavailable"):
            await self.transport.request(RequestMethod.PUT, "https://example.test/files/1")

    async def test_unredirected_202_is_left_alone(self) -> None:
        self.patch_fetch(FakeResponse(status=202, body=b"accepted"))

        result = await self.transport.request(RequestMethod.POST, "https://example.test/tasks")

        assert result.status == 202
        assert result.data == b"accepted"

    async def test_synthesises_content_range_when_hidden_by_cors(self) -> None:
        self.patch_fetch(FakeResponse(status=206, body=b"0123"))

        result = await self.transport.request(
            RequestMethod.GET, "https://example.test/file", headers=HTTPHeaderDict({"Range": "bytes=0-3"})
        )

        assert result.headers["Content-Range"] == "bytes 0-3/*"

    async def test_keeps_content_range_reported_by_the_server(self) -> None:
        self.patch_fetch(FakeResponse(status=206, body=b"0123", headers={"Content-Range": "bytes 0-3/10"}))

        result = await self.transport.request(
            RequestMethod.GET, "https://example.test/file", headers=HTTPHeaderDict({"Range": "bytes=0-3"})
        )

        assert result.headers["Content-Range"] == "bytes 0-3/10"

    async def test_does_not_invent_content_range_if_server_ignores_range(self) -> None:
        self.patch_fetch(FakeResponse(status=200, body=b"full file"))

        result = await self.transport.request(
            RequestMethod.GET, "https://example.test/file", headers=HTTPHeaderDict({"Range": "bytes=0-3"})
        )

        assert "Content-Range" not in result.headers

    async def test_does_not_invent_content_range_if_length_mismatches(self) -> None:
        self.patch_fetch(FakeResponse(status=206, body=b"012345"))

        result = await self.transport.request(
            RequestMethod.GET, "https://example.test/file", headers=HTTPHeaderDict({"Range": "bytes=0-3"})
        )

        assert "Content-Range" not in result.headers

    async def test_ignores_malformed_range_header(self) -> None:
        self.patch_fetch(FakeResponse(status=206, body=b"0123"))

        result = await self.transport.request(
            RequestMethod.GET, "https://example.test/file", headers=HTTPHeaderDict({"Range": "nonsense"})
        )

        assert "Content-Range" not in result.headers

    async def test_wraps_fetch_failures_in_transport_error(self) -> None:
        self.patch_fetch(error=RuntimeError("network down"))

        with self.assertRaises(TransportError) as ctx:
            await self.transport.request(RequestMethod.GET, "https://example.test/")

        assert isinstance(ctx.exception.caused_by, RuntimeError)

    async def test_connection_timeout_aborts_fetch(self) -> None:
        signals = []

        async def stalled_fetch(url: str, **options: object) -> FakeResponse:
            signals.append(options["signal"])
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        with mock.patch.object(transport_module, "pyfetch", stalled_fetch):
            with self.assertRaises(TransportError) as ctx:
                await self.transport.request(RequestMethod.GET, "https://example.test/", request_timeout=(0.01, 1))

        assert isinstance(ctx.exception.caused_by, TimeoutError)
        assert signals[0].aborted

    async def test_total_timeout_aborts_body_read(self) -> None:
        class StalledResponse(FakeResponse):
            async def bytes(self) -> bytes:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        fetch = self.patch_fetch(StalledResponse())

        with self.assertRaises(TransportError) as ctx:
            await self.transport.request(RequestMethod.GET, "https://example.test/", request_timeout=0.01)

        assert isinstance(ctx.exception.caused_by, TimeoutError)
        assert fetch.last_call["signal"].aborted

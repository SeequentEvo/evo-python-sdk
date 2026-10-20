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

from __future__ import annotations

try:
    from js import AbortController, Blob, FormData

    from pyodide.ffi import to_js
    from pyodide.http import pyfetch
except ImportError:
    raise ImportError("JsTransport cannot be used because it is not running in a Pyodide runtime.")

import asyncio
import json
import re
from time import monotonic
from types import TracebackType
from urllib.parse import urlencode, urlsplit

from evo.common import HTTPHeaderDict, HTTPResponse, RequestMethod
from evo.common.exceptions import ClientValueError, TransportError
from evo.common.interfaces import ITransport
from evo.logging import getLogger

__all__ = ["JsTransport"]

logger = getLogger("pyodide.transport")

_FORM_CONTENT_TYPES = ("application/x-www-form-urlencoded", "multipart/form-data")


class JsTransport(ITransport):
    """An `ITransport` implementation that issues requests through the browser `fetch` API.

    See `evo.common.interfaces.ITransport` for more detail.

    Unlike other transports, this one cannot avoid following redirects: the browser exposes manually handled redirects
    as opaque responses with status 0. Redirects are therefore followed, and the responses the SDK relies on are
    reconstructed from the final response.
    """

    def __init__(self, user_agent: str | None = None) -> None:
        """
        :param user_agent: The value to provide in the `User-Agent` header.
        """
        self.user_agent = user_agent or "evo-sdk-common"

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> ITransport:
        return self

    async def __aexit__(
        self,
        exc_type: type[Exception] | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def _build_body(
        self,
        headers: dict[str, str],
        post_params: list[tuple[str, str | bytes]],
        body: object | str | bytes | None,
    ) -> object:
        content_type = headers.get("Content-Type", "")
        if post_params and any(form_type in content_type for form_type in _FORM_CONTENT_TYPES):
            if content_type == "application/x-www-form-urlencoded":
                return urlencode(post_params)
            headers.pop("Content-Type")
            form_data = FormData.new()
            for key, value in post_params:
                form_data.append(key, Blob.new(to_js([value])) if isinstance(value, bytes) else value)
            return form_data

        if body is not None and not isinstance(body, (str, bytes)):
            headers.setdefault("Content-Type", "application/json")
            return json.dumps(body)

        return body

    async def request(
        self,
        method: RequestMethod,
        url: str,
        headers: HTTPHeaderDict | None = None,
        post_params: list[tuple[str, str | bytes]] | None = None,
        body: object | str | bytes | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> HTTPResponse:
        if post_params is not None and body is not None:
            raise ClientValueError(msg="HTTP body and post parameters cannot be used at the same time.")

        controller = AbortController.new()
        deadline = monotonic() + request_timeout if isinstance(request_timeout, (int, float)) else None
        connect_timeout, read_timeout = request_timeout if isinstance(request_timeout, tuple) else (None, None)

        def timeout_for(stage: str) -> float | None:
            if deadline is not None:
                return max(0, deadline - monotonic())
            return connect_timeout if stage == "connect" else read_timeout

        async def fetch(fetch_url: str, **options: object) -> object:
            return await asyncio.wait_for(
                pyfetch(fetch_url, signal=controller.signal, **options), timeout=timeout_for("connect")
            )

        try:
            request_headers = dict(headers or {})
            request_headers.setdefault("User-Agent", self.user_agent)
            request_body = self._build_body(request_headers, post_params or [], body)

            is_head = method == RequestMethod.HEAD
            resp = await fetch(
                url,
                method=str(method),
                headers=request_headers,
                body=request_body,
                redirect="follow",
            )
            resp_headers = HTTPHeaderDict(resp.headers)

            if (
                is_head
                and resp.status == 200
                and ("Content-Length" not in resp_headers or "Accept-Ranges" not in resp_headers)
            ):
                probe_headers = {**request_headers, "Range": "bytes=0-0"}
                probe = await fetch(url, method="GET", headers=probe_headers, redirect="follow")
                content_range = HTTPHeaderDict(probe.headers).get("Content-Range", "")
                match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
                if probe.status == 206 and match:
                    resp_headers["Content-Length"] = match.group(1)
                    resp_headers["Accept-Ranges"] = "bytes"

            if resp.redirected:
                request_path = urlsplit(url).path.rstrip("/")
                status_path = urlsplit(str(resp.url)).path
                if (
                    method == RequestMethod.POST
                    and resp.status == 202
                    and re.fullmatch(r".*/compute/orgs/[^/]+/[^/]+/[^/]+", request_path)
                    and re.fullmatch(re.escape(request_path) + r"/[^/]+/status", status_path)
                ):
                    resp_headers["Location"] = str(resp.url)
                    return HTTPResponse(status=303, data=b"", reason="See Other", headers=resp_headers)
                raise TransportError(msg="Browser fetch followed a redirect whose original response is unavailable.")

            # Redirect responses carry their information in the Location header and have no body to read.
            if 300 <= resp.status < 400:
                return HTTPResponse(status=resp.status, data=b"", reason=resp.status_text, headers=resp_headers)

            if is_head:
                return HTTPResponse(status=resp.status, data=b"", reason=resp.status_text, headers=resp_headers)

            data = await asyncio.wait_for(resp.bytes(), timeout=timeout_for("read"))
            range_header = request_headers.get("Range") or request_headers.get("range")
            if range_header and resp.status == 206 and "Content-Range" not in resp_headers:
                if match := re.fullmatch(r"bytes=(\d+)-(\d+)", range_header):
                    start, end = map(int, match.groups())
                    if end >= start and len(data) == end - start + 1:
                        resp_headers["Content-Range"] = f"bytes {start}-{end}/*"

            return HTTPResponse(status=resp.status, data=data, reason=resp.status_text, headers=resp_headers)
        except TransportError:
            raise
        except Exception as e:
            raise TransportError(msg="Could not complete HTTP request", caused_by=e)
        finally:
            controller.abort()

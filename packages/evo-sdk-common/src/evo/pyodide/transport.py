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
    from js import FormData

    from pyodide.http import pyfetch
except ImportError:
    raise ImportError("JsTransport cannot be used because it is not running in a Pyodide runtime.")

import contextlib
import json
from types import TracebackType
from urllib.parse import urlencode

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
            form_data = FormData.new()
            for key, value in post_params:
                form_data.append(key, value)
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

        try:
            request_headers = dict(headers or {})
            request_headers.setdefault("User-Agent", self.user_agent)
            request_body = self._build_body(request_headers, post_params or [], body)

            # Browsers often strip headers from HEAD responses due to CORS, but the SDK needs Content-Length and
            # Accept-Ranges, so issue a GET and discard the body instead.
            is_head = method == RequestMethod.HEAD
            actual_method = "GET" if is_head else str(method)

            resp = await pyfetch(
                url,
                method=actual_method,
                headers=request_headers,
                body=request_body,
                redirect="follow",
            )
            resp_headers = HTTPHeaderDict(resp.headers)

            # Fetch hides the original 303 when following a compute task submission redirect. Recreate it from the
            # final status URL so the SDK can construct and poll the submitted job.
            if method == RequestMethod.POST and resp.redirected and resp.status == 202:
                resp_headers["Location"] = str(resp.url)
                return HTTPResponse(status=303, data=b"", reason="See Other", headers=resp_headers)

            # Redirect responses carry their information in the Location header and have no body to read.
            if 300 <= resp.status < 400:
                return HTTPResponse(status=resp.status, data=b"", reason=resp.status_text, headers=resp_headers)

            data = await resp.bytes()

            if is_head:
                resp_headers.setdefault("Accept-Ranges", "bytes")
                resp_headers.setdefault("Content-Length", str(len(data)))
                return HTTPResponse(status=resp.status, data=b"", reason=resp.status_text, headers=resp_headers)

            range_header = request_headers.get("Range") or request_headers.get("range")
            if range_header and "Content-Range" not in resp_headers:
                with contextlib.suppress(IndexError, ValueError):
                    start, end = range_header.split("=")[1].split("-")
                    resp_headers["Content-Range"] = f"bytes {start}-{end}/*"

            return HTTPResponse(status=resp.status, data=data, reason=resp.status_text, headers=resp_headers)
        except Exception as e:
            raise TransportError(msg="Could not complete HTTP request", caused_by=e)

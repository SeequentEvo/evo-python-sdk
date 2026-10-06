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

"""Stand-ins for the `js` and `pyodide` modules that only exist inside a Pyodide runtime.

`evo.pyodide` cannot be imported without them, so `install()` must be called before importing anything from it.
"""

from __future__ import annotations

import base64
import importlib.machinery
import json
import sys
import time
import types
from typing import Any


class FakeFormData:
    """Stand-in for the browser `FormData` object."""

    def __init__(self) -> None:
        self.fields: list[tuple[str, str | bytes]] = []

    @classmethod
    def new(cls) -> FakeFormData:
        return cls()

    def append(self, key: str, value: str | bytes) -> None:
        self.fields.append((key, value))


class FakeLocalStorage:
    """Stand-in for the browser `localStorage` object."""

    def __init__(self, items: dict[str, str] | None = None) -> None:
        self._items = dict(items or {})

    def getItem(self, key: str) -> str | None:  # noqa: N802 - mirrors the browser API.
        return self._items.get(key)

    def setItem(self, key: str, value: str) -> None:  # noqa: N802 - mirrors the browser API.
        self._items[key] = value


class FakeResponse:
    """Stand-in for the response object returned by `pyodide.http.pyfetch`."""

    def __init__(
        self,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        url: str = "https://example.test/",
        redirected: bool = False,
        status_text: str = "OK",
    ) -> None:
        self.status = status
        self.headers = dict(headers or {})
        self.url = url
        self.redirected = redirected
        self.status_text = status_text
        self.body_read = False
        self._body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    async def bytes(self) -> bytes:
        self.body_read = True
        return self._body

    async def json(self) -> Any:
        self.body_read = True
        return json.loads(self._body)


class FakeFetch:
    """Stand-in for `pyodide.http.pyfetch` that records calls and returns canned responses."""

    def __init__(self, *responses: FakeResponse, error: Exception | None = None) -> None:
        self._responses = list(responses)
        self._error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1][1]

    async def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self._error is not None:
            raise self._error
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def jwt(expires_at: float) -> str:
    """Build an unsigned JWT carrying only an `exp` claim."""
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expires_at}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def valid_jwt() -> str:
    return jwt(time.time() + 600)


def expired_jwt() -> str:
    return jwt(time.time() - 10)


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    # find_spec() and importlib machinery reject modules without a spec.
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


def install() -> None:
    """Register the stand-in runtime modules so that `evo.pyodide` can be imported."""
    if "js" not in sys.modules:
        js = _new_module("js")
        js.FormData = FakeFormData
        js.localStorage = FakeLocalStorage()
        sys.modules["js"] = js

    if "pyodide" not in sys.modules:
        pyodide = _new_module("pyodide")
        pyodide.__path__ = []
        http = _new_module("pyodide.http")

        async def pyfetch(*args: Any, **kwargs: Any) -> FakeResponse:
            raise AssertionError("pyfetch must be patched by the test.")

        http.pyfetch = pyfetch
        pyodide.http = http
        sys.modules["pyodide"] = pyodide
        sys.modules["pyodide.http"] = http

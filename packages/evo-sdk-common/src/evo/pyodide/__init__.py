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

"""Support for running the Evo SDK inside a Pyodide runtime, such as JupyterLite.

This package may only be imported from a Pyodide runtime. It provides a transport that routes requests through the
browser `fetch` API, and helpers for picking up credentials issued by the hosting page.
"""

from ._browser import get_browser_access_token, get_browser_config
from .transport import JsTransport

__all__ = [
    "JsTransport",
    "get_browser_access_token",
    "get_browser_config",
]

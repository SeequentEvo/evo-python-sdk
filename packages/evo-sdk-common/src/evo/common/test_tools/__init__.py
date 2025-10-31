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

"""Evo Client Common Test Tools

The contents of this module are utilities that make it easy to test service client libraries. These utilities should
only be used for unit tests.
"""

import sys
import warnings

from .consts import ACCESS_TOKEN, BASE_URL, HUB, ORG, WORKSPACE_ID
from .http import (
    AbstractTestRequestHandler,
    MockResponse,
    TestAuthorizer,
    TestHTTPHeaderDict,
    TestTransport,
    TestWithConnector,
)
from .io import (
    DownloadRequestHandler,
    MultiDownloadRequestHandler,
    StorageDestinationRequestHandler,
    TestWithDownloadHandler,
    TestWithUploadHandler,
    UrlGenerator,
)
from .storage import TestWithStorage
from .utils import long_test, utc_datetime, utc_time

if "pytest" not in sys.modules:
    # Issue a warning whenever this module is imported.
    warnings.warn(__doc__)

__all__ = [
    "ACCESS_TOKEN",
    "BASE_URL",
    "HUB",
    "ORG",
    "WORKSPACE_ID",
    "AbstractTestRequestHandler",
    "DownloadRequestHandler",
    "MockResponse",
    "MultiDownloadRequestHandler",
    "StorageDestinationRequestHandler",
    "TestAuthorizer",
    "TestHTTPHeaderDict",
    "TestTransport",
    "TestWithConnector",
    "TestWithDownloadHandler",
    "TestWithStorage",
    "TestWithUploadHandler",
    "UrlGenerator",
    "long_test",
    "utc_datetime",
    "utc_time",
]

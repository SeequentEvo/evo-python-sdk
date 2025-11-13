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
"""
File API
=============

The File API provides the ability to manage files of any type or size, associated with
your Evo workspace. Enable your product with Evo connected workflows by integrating with the Seequent Evo
File API. Most file formats and sizes are accepted.

Files can be referenced by their UUID, or by a user-defined file path. Files are versioned, so updating or
replacing them will create a new version of the file. The latest version of the file is always returned
unless a specific version is requested.

For more information on using the File API, see [Overview](https://developer.seequent.com/docs/guides/file/), or the API references here.


This code is generated from the OpenAPI specification for File API.
API version: 2.10.0
"""

# Import endpoint apis.
from .api import FileV2Api

__all__ = [
    "FileV2Api",
]

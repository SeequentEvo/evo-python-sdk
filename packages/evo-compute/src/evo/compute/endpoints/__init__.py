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
Compute Task API
=============

Normally this code would be generated from an OpenAPI specification. However, the API specification for
the Compute Task API is new and not yet finalized. Therefore, this code is written against the intended API,
with some customisation as needed to make it work.

The client implementation should still abstract these details away from the user, so that the user can interact
with the API in a more user-friendly way, and so that this implementation can be changed without affecting the user.

This code is based on the OpenAPI specification for Compute Task API.
API version: 0.0.1
"""

# Import endpoint apis.
from .api import JobApi, TaskApi

__all__ = [
    "JobApi",
    "TaskApi",
]

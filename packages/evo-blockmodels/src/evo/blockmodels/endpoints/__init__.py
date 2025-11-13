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
Block Model API
=============


    The Block Model API provides the ability to manage and report on block models in your Evo workspaces. Enable your
    product with Evo connected workflows by integrating with the Seequent Block Model API.

    The Block Model API supports a range of sub-blocking options, and both full and partial updates to block models,
    including updates of specified columns and/or sub-volumes. Block models are versioned, and the service supports
    reporting on the material content of current or previous versions, and comparing the content between versions of
    a block model.

    For more information on the Block Model API, see [Overview](/docs/guides/blockmodel/), or the API references here.


This code is generated from the OpenAPI specification for Block Model API.
API version: 1.41.3
"""

# Import endpoint apis.
from .api import ColumnOperationsApi, JobsApi, MetadataApi, OperationsApi, ReportsApi, UnitsApi, VersionsApi

__all__ = [
    "ColumnOperationsApi",
    "JobsApi",
    "MetadataApi",
    "OperationsApi",
    "ReportsApi",
    "UnitsApi",
    "VersionsApi",
]

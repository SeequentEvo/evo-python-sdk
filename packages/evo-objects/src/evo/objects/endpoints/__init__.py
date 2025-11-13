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
Geoscience Object API
=============


The Geoscience Object API enables technological integrations at the object level. It enables users to access and utilise their data across all products through a common and accessible data structure.

A Geoscience Object is a data structure that represents a concrete geological, geotechnical, or geophysical concept. Geoscience Objects can be referenced by their UUID or by a user-defined object path.

For more information on using the Geoscience Object API, see the [Geoscience Object API overview](/docs/guides/objects), or the API references here.


This code is generated from the OpenAPI specification for Geoscience Object API.
API version: 1.21.0
"""

# Import endpoint apis.
from .api import DataApi, MetadataApi, ObjectsApi, StagesApi

__all__ = [
    "DataApi",
    "MetadataApi",
    "ObjectsApi",
    "StagesApi",
]

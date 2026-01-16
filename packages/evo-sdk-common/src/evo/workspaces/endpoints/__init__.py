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
Workspaces API
=============

The Workspaces API enables users to organize, maintain, and store project data, but does not use or process the data. The workspace APIs allow you to manage:
- Workspaces
- User roles within workspaces
- Workspace thumbnails

There are three pre-defined roles within workspaces:

- Owner: can perform all actions in the workspace
- Editor: can perform all actions excluding deleting of a workspace
- Viewer: can view the workspace

These user roles can be assigned to users in a workspace. Once a role has been assigned it can be replaced or removed.
Users can also retrieve user roles, the role of a particular user, and their own role if applicable.
For more information on using the Workspaces API, see the [Workspaces API overview](https://developer.seequent.com/docs/guides/workspaces/), or the API references here.


This code is generated from the OpenAPI specification for Workspaces API.
API version: 1.0
"""

# Import endpoint apis.
from .api import (
    AdminApi,
    DiscoveryApi,
    FoldersApi,
    GeneralApi,
    InstanceUsersApi,
    LicenseAccessApi,
    ThumbnailsApi,
    TokenApi,
    WorkspacesApi,
)

__all__ = [
    "AdminApi",
    "DiscoveryApi",
    "FoldersApi",
    "GeneralApi",
    "InstanceUsersApi",
    "LicenseAccessApi",
    "ThumbnailsApi",
    "TokenApi",
    "WorkspacesApi",
]

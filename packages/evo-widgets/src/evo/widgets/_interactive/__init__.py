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

"""Interactive anywidget-based widgets for Jupyter notebooks.

This subpackage provides modern, interactive widgets built with anywidget
for richer notebook experiences including authentication, service discovery,
and object search.

Example usage::

    from evo.widgets import ServiceManagerWidget

    # Create and authenticate
    manager = await ServiceManagerWidget.with_auth_code(
        client_id="your-client-id"
    ).login()
    manager  # Display the widget

    # Use the manager to create clients
    client = manager.create_client(ObjectAPIClient)
"""

from .authorizer import AuthorizationCodeAuthorizer
from .env import DotEnv
from .widgets import (
    DropdownSelectorWidget,
    FeedbackWidget,
    HubSelectorWidget,
    OrgSelectorWidget,
    ServiceManagerWidget,
    WorkspaceSelectorWidget,
    display_object_links,
)

__all__ = [
    "AuthorizationCodeAuthorizer",
    "display_object_links",
    "DotEnv",
    "DropdownSelectorWidget",
    "FeedbackWidget",
    "HubSelectorWidget",
    "OrgSelectorWidget",
    "ServiceManagerWidget",
    "WorkspaceSelectorWidget",
]

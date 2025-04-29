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

from __future__ import annotations

from .authorizer import (
    AccessTokenAuthorizer,
    AuthorizationCodeAuthorizer,
    ClientCredentialsAuthorizer,
    DeviceFlowAuthorizer,
)
from .data import AccessToken, DeviceFlowResponse, OAuthScopes, UserAccessToken
from .exceptions import OAuthError, OIDCError
from .oauth_redirect_handler import OAuthRedirectHandler
from .oidc import OIDCConnector

__all__ = [
    "AccessToken",
    "AccessTokenAuthorizer",
    "AuthorizationCodeAuthorizer",
    "ClientCredentialsAuthorizer",
    "DeviceFlowAuthorizer",
    "DeviceFlowResponse",
    "OAuthError",
    "OAuthRedirectHandler",
    "OAuthScopes",
    "OIDCConnector",
    "OIDCError",
    "UserAccessToken",
]

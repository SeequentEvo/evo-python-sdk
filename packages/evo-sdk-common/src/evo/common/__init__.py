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

from .connector import APIConnector, NoAuth
from .context import StaticContext
from .data import (
    DependencyStatus,
    EmptyResponse,
    Environment,
    HealthCheckType,
    HTTPHeaderDict,
    HTTPResponse,
    Page,
    RequestMethod,
    ResourceMetadata,
    ServiceHealth,
    ServiceStatus,
    ServiceUser,
)
from .interfaces import IAuthorizer, ICache, IContext, IFeedback, ITransport
from .service import BaseAPIClient
from .urls import (
    get_evo_base_url,
    get_hub_code,
    get_portal_url,
    get_portal_url_from_environment,
    get_portal_url_from_reference,
    get_viewer_url,
    get_viewer_url_from_environment,
    get_viewer_url_from_reference,
    parse_object_reference_url,
    serialize_object_reference,
)

__all__ = [
    "APIConnector",
    "BaseAPIClient",
    "DependencyStatus",
    "EmptyResponse",
    "Environment",
    "EvoContext",
    "HTTPHeaderDict",
    "HTTPResponse",
    "HealthCheckType",
    "IAuthorizer",
    "ICache",
    "IContext",
    "IFeedback",
    "ITransport",
    "NoAuth",
    "Page",
    "RequestMethod",
    "ResourceMetadata",
    "ServiceHealth",
    "ServiceStatus",
    "ServiceUser",
    "StaticContext",
    "get_evo_base_url",
    "get_hub_code",
    "get_portal_url",
    "get_portal_url_from_environment",
    "get_portal_url_from_reference",
    "get_viewer_url",
    "get_viewer_url_from_environment",
    "get_viewer_url_from_reference",
    "parse_object_reference_url",
    "serialize_object_reference",
]

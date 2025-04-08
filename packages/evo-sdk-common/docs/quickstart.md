# Getting Started

## Basic usage

Just want to get going? This is the simplest setup to interact with an Evo service.

``` python 
from evo.aio import AioTransport
from evo.common import APIConnector, BaseAPIClient
from evo.common.utils import BackoffIncremental
from evo.discovery import DiscoveryAPIClient
from evo.oauth import AuthorizationCodeAuthorizer, OIDCConnector
from evo.workspaces import WorkspaceAPIClient

# Configure the transport.
transport = AioTransport(
    user_agent="evo-sdk-common-poc",
    max_attempts=3,
    backoff_method=BackoffIncremental(2),
    num_pools=4,
    verify_ssl=True,
)

# Login to the Evo platform.
# User Login
authorizer = AuthorizationCodeAuthorizer(
    redirect_url="<redirect_url>",
    oidc_connector=OIDCConnector(
        transport=transport,
        oidc_issuer="<issuer_url>",
        client_id="<client_id>",
    ),
)
await authorizer.login()

# Select an Organization.
async with APIConnector("https://discover.api.seequent.com", transport, authorizer) as api_connector:
    discovery_client = DiscoveryAPIClient(api_connector)
    organizations = await discovery_client.list_organizations()

selected_org = organizations[0]

# Select a hub and create a connector.
hub_connector = APIConnector(selected_org.hubs[0].url, transport, authorizer)

# Select a Workspace.
async with hub_connector:
    workspace_client = WorkspaceAPIClient(hub_connector, selected_org.id)
    workspaces = await workspace_client.list_workspaces()

workspace = workspaces[0]
workspace_env = workspace.get_environment()

# Interact with a service.
async with hub_connector:
    service_client = BaseAPIClient(workspace_env, hub_connector)
    ...

```

## What's going on?

### Configure your Transport Layer

The `ITransport` interface is used to make HTTP requests to Evo APIs. The `AioTransport` class is an implementation
based on the `aiohttp` library, which is an optional dependency. Different HTTP client libraries can be substituted by
implementing a facade that implements the `ITransport` interface.

Transport objects must be re-entrant so that they can be used by multiple coroutines at the same time. `AioTransport`
uses an internal counter to track the number of places where the transport is being used. When the counter reaches zero,
the underlying HTTP client session is closed, and any related resources are released. The next time the transport is
opened, a new session will be created.

``` python
from evo.aio import AioTransport
from evo.common.utils import BackoffIncremental

# Configure the transport.
transport = AioTransport(
    user_agent="evo-sdk-common-poc",
    max_attempts=3,
    backoff_method=BackoffIncremental(2),
    num_pools=4,
    verify_ssl=True,
)

# We can open the transport outside a context manager so that the underlying session is left open. This can save
# time if we are going to make multiple batches of requests in the same area of code. Ideally, the transport should
# be closed when it is no longer needed.
await transport.open()
```

### Log in to Evo

The Evo service requires users to log in to access their data. This is done using the OAuth 2.0 protocol. The
`evo.oauth.AuthorizationCodeAuthorizer` class is used to authenticate users and obtain an access token. The access
token is used to make requests to the Evo service.

``` python
authorizer = AuthorizationCodeAuthorizer(
    redirect_url="<redirect_url>",
    oidc_connector=OIDCConnector(
        transport=transport,
        oidc_issuer="<issuer_url>",
        client_id="<client_id>",
    ),
)
await authorizer.login()
```

Alternatively, a client of `client credientials` grant type can use the `ClientCredentialsAuthorizer` for authorization into Evo. This allows for service to service requests, instead of user login and redirects. 

``` python
from evo.oauth import OIDCConnector, ClientCredentialsAuthorizer, OAuthScopes

ISSUER_URI = "https://ims.bentley.com"
CLIENT_NAME = "<client_name>"
CLIENT_SECRET = "<client_secret>"
CLIENT_ID = CLIENT_NAME.lower()

authorizer = ClientCredentialsAuthorizer(
    oidc_connector=OIDCConnector(
        transport=transport,
        oidc_issuer=ISSUER_URI,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    ),
    scopes=OAuthScopes.all_evo,
)

# Authorize the client.
await authorizer.authorize()
```

### Using the Discovery API

The discovery API is used to find the organizations and hubs that a user has access to. This is done using the
`evo.discovery.DiscoveryAPIClient` class. The `list_organizations()` method is used to get a
list of organizations. Each listed organization includes a list of hubs that are used by that organization.

``` python
async with APIConnector("https://discover.api.seequent.com", transport, authorizer) as api_connector:
    discovery_client = DiscoveryAPIClient(api_connector)
    organizations = await discovery_client.list_organizations()
```

### Using the Workspace API

The Workspace API is used to find the workspaces that a user has access to. This is done using the
`evo.workspaces.WorkspaceAPIClient` class. The `list_workspaces()` method is used to get
a list of workspaces belonging to the specified organization on the specified hub.

``` python
async with hub_connector:
    workspace_client = WorkspaceAPIClient(hub_connector, selected_org.id)
    workspaces = await workspace_client.list_workspaces()
```

### Interacting with services

Service clients are used to interact with individual services. Workspace objects have a `get_environment()` method that
returns an environment object. This object can be used by one or more service clients to interact with different
services using the same organization and workspace. The hub connector that was used for workspace discovery can be
reused for interacting with services.

``` python
async with hub_connector:
    service_client = BaseAPIClient(workspace_env, hub_connector)
    ...
```

## Skip Organization and Workspace Discovery

If you already know the hub URL, organization ID, and workspace ID, you can skip the discovery steps and directly create
connector and environment objects.

``` python
from uuid import UUID

from evo.aio import AioTransport
from evo.common import APIConnector, BaseAPIClient, Environment
from evo.common.utils import BackoffIncremental
from evo.oauth import AuthorizationCodeAuthorizer, OIDCConnector

transport = AioTransport(
    user_agent="evo-sdk-common-poc",
    max_attempts=3,
    backoff_method=BackoffIncremental(2),
    num_pools=4,
    verify_ssl=True,
)
authorizer = AuthorizationCodeAuthorizer(
    redirect_url="<redirect_url>",
    oidc_connector=OIDCConnector(
        transport=transport,
        oidc_issuer="<issuer_url>",
        client_id="<client_id>",
    ),
)
await authorizer.login()

environment = Environment(
    hub_url="<hub-url>",
    org_id=UUID("<organization-id>"),
    workspace_id=UUID("<workspace-id>"),
)
hub_connector = APIConnector(environment.hub_url, transport, authorizer)
async with hub_connector:
    service_client = BaseAPIClient(environment, hub_connector)
    ...
```

## File I/O

Some utility classes for robust file I/O operations have been implemented in the `evo.common.io` module. These classes
are designed to handle large files and provide a simple interface for reading and writing data in chunks.

The most common use case for these classes is to transfer large files between local storage and remote storage. `evo.common.io.HTTPSource` and `evo.common.io.StorageDestination` each have static methods that make it easy to upload and download files.

Downloading large files is very standardized across service providers, but uploading large files is a different story. Evo APIs have a unique way to support chunked file upload, so an Evo-specific implementation is provided in `evo.common.io.StorageDestination`.

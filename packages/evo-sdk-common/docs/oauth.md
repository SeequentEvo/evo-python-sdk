# OAuth Examples

The OAuth implementation provided requires additional dependencies to be installed. To install the required dependencies, make sure you depend on `evo-sdk-common[all]`.

## OpenID Connect

The provided OAuth implementation depends on OpenID Connect discovery to retrieve the OAuth endpoints. This approach means we can support multiple OAuth providers without needing to hardcode the endpoints for each provider. For an OAuth provider to be supported, it must support the OpenID Connect discovery, in addition to the `code` response type, and the `authorization_code` grant type.

The `OIDCConnector` is the central component for all of our OAuth workflows.

``` python
import logging
from evo.aio import AioTransport
from evo.oauth import OIDCConnector

logging.basicConfig(level=logging.DEBUG)

ISSUER_URI = "https://ims.bentley.com"
REDIRECT_URL = "http://localhost:3000/signin-oidc"
USER_AGENT = "EvoPythonSDK"
CLIENT_ID = "<client-id>"

connector = OIDCConnector(
    transport=AioTransport(
        user_agent=USER_AGENT,
    ),
    oidc_issuer=ISSUER_URI,
    client_id=CLIENT_ID,
)
```

## Manage Access Tokens

The `OAuth` library provides authorizer classes to handle different OAuth flows. The `AuthorizationCodeAuthorizer` can be used for user access tokens, and the `ClientCredientialsAuthorizer` can be used for service to service authentication.
### Simplest way to manage user access tokens

`AuthorizationCodeAuthorizer` is the simplest way to manage user access tokens. Logging in will open a browser window to the authorisation URL and wait for the user to authenticate and authorise the application. The `AuthorizationCodeAuthorizer` object allows the user access token to be used in API requests.

``` python
from evo.oauth import AuthorizationCodeAuthorizer, OAuthScopes

authorizer = AuthorizationCodeAuthorizer(
    oidc_connector=connector,
    redirect_url=REDIRECT_URL,
    scopes=OAuthScopes.all_evo | OAuthScopes.offline_access,
)
await authorizer.login()
print(await authorizer.get_default_headers())
```

## Refreshing the access token

`AuthorizationCodeAuthorizer.refresh_token()` can be used to refresh the access token when it expires. If the authorization server did not return a refresh token, the function will raise `OAuthError`. If you try refreshing the token within 5 minutes of the last refresh, the token will not be refreshed and the method will return `False`. Similarly, if there is any error during the refresh, the method will return `False`.

This is how `APIConnector` automatically refreshes the access token when it expires.

> Note: You MUST request the `offline_access` scope at login to get a refresh token. Offline access is not included by default in any of the predefined scope groups.

``` python
refreshed = await authorizer.refresh_token()
print(f"The token was {'' if refreshed else 'not '}refreshed.")
```

## Using the OAuthRedirectHandler

The `OAuthRedirectHandler` wraps the `OIDCConnector` and implements a localhost HTTP server to handle the OAuth redirect. This is useful for applications that cannot open a browser window, such as a command-line application. The `OAuthRedirectHandler` is an asynchronous context manager that manages the lifecycle of the HTTP server.

``` python
from evo.oauth import OAuthRedirectHandler, OAuthScopes

async with OAuthRedirectHandler(connector, REDIRECT_URL) as handler:
    result = await handler.login(OAuthScopes.offline_access)

print(f"Access token: {result.access_token}")
```

## Client Credientials Authentication

Using `ClientCredientialsAuthorizer` we can handle service to service authentication.

``` python
import logging
from evo.aio import AioTransport
from evo.oauth import ClientCredentialsAuthorizer, OAuthScopes, OIDCConnector

logging.basicConfig(level=logging.DEBUG)

ISSUER_URI = "https://ims.bentley.com"
USER_AGENT = "EvoPythonSDK"
CLIENT_ID = "<client-id>"
CLIENT_SECRET = "<client-secret>"

authorizer = ClientCredentialsAuthorizer(
    oidc_connector=OIDCConnector(
        transport=AioTransport(
            user_agent=USER_AGENT,
        ),
        oidc_issuer=ISSUER_URI,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    ),
    scopes=OAuthScopes.openid | OAuthScopes.profile | OAuthScopes.evo_discovery | OAuthScopes.evo_workspace,
)

print(await authorizer.get_default_headers())
```

## Device Flow Authentication

Using `DeviceFlowAuthorizer` we can handle device flow authentication. This is useful for devices that do not have a browser or a keyboard.

``` python
import logging
from evo.aio import AioTransport
from evo.oauth import DeviceFlowAuthorizer, OAuthScopes, OIDCConnector

logging.basicConfig(level=logging.INFO)

ISSUER_URI = "https://ims.bentley.com"
USER_AGENT = "Evo-SDK-Device-Flow"
CLIENT_ID = "<client-id>"

authorizer = DeviceFlowAuthorizer(
    oidc_connector=OIDCConnector(
        transport=AioTransport(
            user_agent=USER_AGENT,
        ),
        oidc_issuer=ISSUER_URI,
        client_id=CLIENT_ID,
    ),
    scopes=OAuthScopes.all_evo,
)

async with authorizer.login() as flow:
    print("+------------------------------------------------------------------------+")
    print("|                                                                        |")
    print("| Using a browser on another device, visit:                              |")
    print(f"| {flow.verification_uri:<70} |")
    print("|                                                                        |")
    print("| And enter the code:                                                    |")
    print(f"| {flow.user_code:<70} |")
    if flow.verification_uri_complete is not None:
        print("|                                                                        |")
        print("| Or visit this URL:                                                     |")
        print(f"| {flow.verification_uri_complete:<70} |")
    print("|                                                                        |")
    print("+------------------------------------------------------------------------+")

print(await authorizer.get_default_headers())
```

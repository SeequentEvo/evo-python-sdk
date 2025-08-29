# OAuth examples

The OAuth implementation provided requires additional dependencies to be installed. To install the required dependencies,
make sure you depend on `evo-sdk-common[notebooks]`. If you installed the `evo-sdk` package, this optional dependency
will already be installed for you.

```shell
pip install evo-sdk-common[notebooks]
```

**Note:** If you'd like to run these examples interactively, you can use the [OAuth Jupyter notebook](examples/oauth.ipynb).

## Configure the connector

The `OAuthConnector` is the central component for all of our OAuth workflows.

```python
from evo.aio import AioTransport
from evo.oauth import OAuthConnector

# OAuth client app credentials
# See: https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens
REDIRECT_URL = "http://localhost:3000/signin-callback"
CLIENT_NAME = "Your Client Name"
CLIENT_ID = "<client-id>"

connector = OAuthConnector(
    transport=AioTransport(user_agent=CLIENT_NAME),
    client_id=CLIENT_ID,
)
```

## Manage access tokens

The `OAuth` library provides authorizer classes to handle different OAuth flows. The `AuthorizationCodeAuthorizer` can be used for user access tokens, and the `ClientCredientialsAuthorizer` can be used for service to service authentication.

### Simplest way to manage user access tokens

`AuthorizationCodeAuthorizer` is the simplest way to manage user access tokens. Logging in will open a browser window to the authorisation URL and wait for the user to authenticate and authorise the application. The `AuthorizationCodeAuthorizer` object allows the user access token to be used in API requests.

```python
from evo.oauth import AuthorizationCodeAuthorizer, EvoScopes

authorizer = AuthorizationCodeAuthorizer(
    oauth_connector=connector,
    redirect_url=REDIRECT_URL,
    scopes=EvoScopes.all_evo | EvoScopes.offline_access,
)
await authorizer.login()
print(await authorizer.get_default_headers())
```

## Refreshing the access token

`AuthorizationCodeAuthorizer.refresh_token()` can be used to refresh the access token when it expires. If the authorization server did not return a refresh token, the function will raise `OAuthError`. If you try refreshing the token within 5 minutes of the last refresh, the token will not be refreshed and the method will return `False`. Similarly, if there is any error during the refresh, the method will return `False`.

This is how `APIConnector` automatically refreshes the access token when it expires.

**Note:** You MUST request the `offline_access` scope at login to get a refresh token. Offline access is not included by default in any of the predefined scope groups.

```python
refreshed = await authorizer.refresh_token()
print(f"The token was {'' if refreshed else 'not '}refreshed.")
```

## Using the OAuthRedirectHandler

The `OAuthRedirectHandler` wraps the `OAuthConnector` and implements a localhost HTTP server to handle the OAuth redirect. This is useful for applications that cannot open a browser window, such as a command-line application. The `OAuthRedirectHandler` is an asynchronous context manager that manages the lifecycle of the HTTP server.

```python
from evo.oauth import OAuthRedirectHandler, EvoScopes

async with OAuthRedirectHandler(connector, REDIRECT_URL) as handler:
    result = await handler.login(EvoScopes.offline_access)

print(f"Access token: {result.access_token}")
```

## Client credentials authentication

The `ClientCredientialsAuthorizer` allows you to handle service-to-service authentication.

```python
from evo.aio import AioTransport
from evo.oauth import ClientCredentialsAuthorizer, EvoScopes, OAuthConnector

# OAuth client app credentials
# See: https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens
CLIENT_NAME = "Your Client Name"
CLIENT_ID = "<client-id>"
CLIENT_SECRET = "<client-secret>"

authorizer = ClientCredentialsAuthorizer(
    oauth_connector=OAuthConnector(
        transport=AioTransport(user_agent=USER_AGENT),
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    ),
    scopes=EvoScopes.evo_discovery | EvoScopes.evo_workspace,
)

print(await authorizer.get_default_headers())
```

## Specifying a token with AccessTokenAuthorizer

If you already have an access token, and you're happy to manage the validity of it and refresh it yourself, you can
use the `AccessTokenAuthorizer` to skip all OAuth processes and simply use the token to authorise requests.

```python
from evo.oauth import AccessTokenAuthorizer

authorizer = AccessTokenAuthorizer(access_token="your-access-token-here")

print(authorizer.get_default_headers())
```

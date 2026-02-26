# evo-python-sdk

[GitHub repository](https://github.com/SeequentEvo/evo-python-sdk)

`evo-python-sdk` is designed for developers, data scientists, geologists, and geostatisticians who want to work with Seequent Evo APIs and geoscience data.

## Quick start for notebooks

Once you have an Evo app registered and the SDK installed, you can load and work with geoscience objects in just a few lines of code:

```python
# Authenticate with Evo
from evo.notebooks import ServiceManagerWidget

manager = await ServiceManagerWidget.with_auth_code(
    client_id="<your-client-id>",
    cache_location="./notebook-data",
).login()
```

```python
# Enable rich HTML display for Evo objects in Jupyter
%load_ext evo.widgets

# Load an object by file path or UUID
from evo.objects.typed import object_from_uuid, object_from_path

obj = await object_from_path(manager, "<your-object-path>")

# OR

obj = await object_from_uuid(manager, "<your-object-uuid>")
obj  # Displays object info with links to Evo Portal and Viewer
```

```python
# Get data as a pandas DataFrame
df = await obj.to_dataframe()
df.head()
```

Typed objects like `PointSet`, `BlockModel`, and `Variogram` provide pretty-printed output in Jupyter with clickable links to view your data in Evo. As support for more geoscience objects is added, geologists and geostatisticians can interact with points, variograms, block models, grids, and more — all through intuitive Python classes.

For a hands-on introduction, see the [simplified object interactions](https://github.com/SeequentEvo/evo-python-sdk/tree/main/code-samples/geoscience-objects/simplified-object-interactions/) notebook. For a complete geostatistical workflow including variogram modelling and kriging estimation, see the [running kriging compute](https://github.com/SeequentEvo/evo-python-sdk/tree/main/code-samples/geoscience-objects/running-kriging-compute/) notebook.

## Getting started with Evo code samples

For detailed information about creating Evo apps, the authentication setup, available code samples, and step-by-step guides for working with the Jupyter notebooks, please refer to the [**Quick start guide**](https://developer.seequent.com/docs/guides/getting-started/quick-start-guide), or [**code-samples**](https://github.com/SeequentEvo/evo-python-sdk/tree/main/code-samples) section of the repository. 

This comprehensive guide will walk you through everything required to get started with Evo APIs. 

## Getting started with the Evo SDK

The evo-python-sdk contains a number of sub-packages. You may choose to install the `evo-sdk` package, which includes all
sub-packages and optional dependencies (e.g. Jupyter notebook support), or choose a specific package to install:

| Package | Version | Description |
| --- | --- | --- |
| evo-sdk | <a href="https://pypi.org/project/evo-sdk/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk" /></a> | A metapackage that installs all available Seequent Evo SDKs, including Jupyter notebook examples. |
| evo-sdk-common ([discovery](evo-sdk-common/discovery/DiscoveryAPIClient.md) and [workspaces](evo-sdk-common/workspaces/WorkspaceAPIClient.md)) | <a href="https://pypi.org/project/evo-sdk-common/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk-common" /></a> | A shared library that provides common functionality for integrating with Seequent's client SDKs. |
| evo-files ([api](evo-files/FileAPIClient.md)) | <a href="https://pypi.org/project/evo-files/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-files" /></a> | A service client for interacting with the Evo File API. |
| evo-objects ([introduction](evo-objects/Introduction.md), [typed objects](evo-objects/TypedObjects.md), [api](evo-objects/ObjectAPIClient.md)) | <a href="https://pypi.org/project/evo-objects/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-objects" /></a> | Typed Python classes and an API client for geoscience objects — points, grids, variograms, and more. |
| evo-colormaps ([api](evo-colormaps/ColormapAPIClient.md)) | <a href="https://pypi.org/project/evo-colormaps/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-colormaps" /></a> | A service client to create colour mappings and associate them to geoscience data with the Colormap API.|
| evo-blockmodels ([introduction](evo-blockmodels/Introduction.md), [typed objects](evo-blockmodels/TypedObjects.md), [api](evo-blockmodels/BlockModelAPIClient.md)) | <a href="https://pypi.org/project/evo-blockmodels/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-blockmodels" /></a> | Typed block model interactions, reports, and an API client for managing block models in Evo. |
| evo-widgets ([introduction](evo-widgets/Introduction.md)) | <a href="https://pypi.org/project/evo-widgets/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-widgets" /></a> | Widgets and presentation layer — rich HTML rendering of typed geoscience objects in Jupyter notebooks. |
| evo-compute ([introduction](evo-compute/Introduction.md), [typed objects](evo-compute/TypedObjects.md), [api](evo-compute/JobClient.md)) | <a href="https://pypi.org/project/evo-compute/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-compute" /></a> | Run compute tasks (e.g. kriging estimation) via the Compute Tasks API.|

### Getting started with SDK development

Now that you have installed the Evo SDK, you can get started by configuring your API connector, and performing a
basic API call to list the organizations that you have access to:

```python
from evo.aio import AioTransport
from evo.oauth import OAuthConnector, AuthorizationCodeAuthorizer
from evo.discovery import DiscoveryAPIClient
from evo.common import APIConnector
import asyncio

transport = AioTransport(user_agent="Your Application Name")
connector = OAuthConnector(transport=transport, client_id="\<YOUR_CLIENT_ID\>")
authorizer = AuthorizationCodeAuthorizer(oauth_connector=connector, redirect_url="http://localhost:3000/signin-callback")

async def main():
    await authorizer.login()
    await discovery()

async def discovery():
    async with APIConnector("https://discover.api.seequent.com", transport, authorizer) as api_connector:
        discovery_client = DiscoveryAPIClient(api_connector)
        organizations = await discovery_client.list_organizations()
        print("Organizations:", organizations)

asyncio.run(main())
```

For next steps, start with the packages most relevant to your workflow:

**Getting started — typed objects & visualisation:**

* [`evo-objects`](evo-objects/Introduction.md): load and work with points, grids, variograms, and other geoscience objects as typed Python classes
* [`evo-blockmodels`](evo-blockmodels/Introduction.md): create, query, and report on block models with typed interactions
* [`evo-compute`](evo-compute/Introduction.md): run compute tasks such as kriging estimation
* [`evo-widgets`](evo-widgets/Introduction.md): rich HTML rendering of typed geoscience objects in Jupyter notebooks

**API clients [For developers]:**

* `evo-sdk-common` ([`discovery`](evo-sdk-common/discovery/DiscoveryAPIClient.md) and [`workspaces`](evo-sdk-common/workspaces/WorkspaceAPIClient.md)): foundation for all Evo SDKs, including arbitrary API requests
* [`evo-files`](evo-files/FileAPIClient.md): low-level File API client
* [`evo-objects` API](evo-objects/ObjectAPIClient.md): low-level Geoscience Object API client
* [`evo-colormaps`](evo-colormaps/ColormapAPIClient.md): Colormap API client
* [`evo-blockmodels` API](evo-blockmodels/BlockModelAPIClient.md): low-level Block Model API client
* [`evo-compute` API](evo-compute/JobClient.md): low-level Compute Tasks API client
* [Seequent Developer Portal](https://developer.seequent.com/docs/guides/getting-started/quick-start-guide): guides, tutorials, and API references

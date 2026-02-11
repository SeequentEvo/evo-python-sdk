# evo-python-sdk

[GitHub repository](https://github.com/SeequentEvo/evo-python-sdk)

## Getting started with Evo code samples

For detailed information about creating Evo apps, the authentication setup, available code samples, and step-by-step guides for working with the Jupyter notebooks, please refer to the [**code-samples**](https://github.com/SeequentEvo/evo-python-sdk/tree/main/code-samples) section of the repository. 

This comprehensive guide will walk you through everything required to get started with Evo APIs. 

## Getting started with Evo SDK development

The evo-python-sdk contains a number of sub-packages. You may choose to install the `evo-sdk` package, which includes all
sub-packages and optional dependencies (e.g. Jupyter notebook support), or choose a specific package to install:

| Package | Version | Description |
| --- | --- | --- |
| evo-sdk | <a href="https://pypi.org/project/evo-sdk/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk" /></a> | A metapackage that installs all available Seequent Evo SDKs, including Jupyter notebook examples. |
| evo-sdk-common ([discovery](evo-python-sdk/evo-sdk-common/discovery) and [workspaces](evo-python-sdk/evo-sdk-common/workspaces)) | <a href="https://pypi.org/project/evo-sdk-common/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk-common" /></a> | A shared library that provides common functionality for integrating with Seequent's client SDKs. |
| [evo-files](evo-python-sdk/evo-files) | <a href="https://pypi.org/project/evo-files/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-files" /></a> | A service client for interacting with the Evo File API. |
| [evo-objects](evo-python-sdk/evo-objects) | <a href="https://pypi.org/project/evo-objects/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-objects" /></a> | A geoscience object service client library designed to help get up and running with the Geoscience Object API. |
| [evo-colormaps](evo-python-sdk/evo-colormaps)  | <a href="https://pypi.org/project/evo-colormaps/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-colormaps" /></a> | A service client to create colour mappings and associate them to geoscience data with the Colormap API.|
| [evo-blockmodels](evo-python-sdk/evo-blockmodels) | <a href="https://pypi.org/project/evo-blockmodels/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-blockmodels" /></a> | The Block Model API provides the ability to manage and report on block models in your Evo workspaces. |
| [evo-compute](evo-python-sdk/evo-compute)  | <a href="https://pypi.org/project/evo-compute/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-compute" /></a> | A service client to send jobs to the Compute Tasks API.|

### Getting started

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

For next steps and more information about using Evo, see:

* `evo-sdk-common` ([`discovery`](evo-python-sdk/evo-sdk-common/discovery) and [`workspaces`](evo-python-sdk/evo-sdk-common/workspaces)): providing the foundation for all Evo SDKs, as well as tools
  for performing arbitrary Seequent Evo API requests
* [`evo-files`](evo-python-sdk/evo-files): for interacting with the File API
* [`evo-objects`](evo-python-sdk/evo-objects): for interacting with the Geoscience Object API
* [`evo-colormaps`](evo-python-sdk/evo-colormaps): for interacting with the Colormap API
* [`evo-blockmodels`](evo-python-sdk/evo-blockmodels): for interacting with the Block Model API
* [`evo-compute`](evo-python-sdk/evo-compute): for interacting with the Compute Tasks API

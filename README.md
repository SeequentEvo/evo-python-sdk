<p align="center"><a href="https://seequent.com" target="_blank"><picture><source media="(prefers-color-scheme: dark)" srcset="https://developer.seequent.com/img/seequent-logo-dark.svg" alt="Seequent logo" width="400" /><img src="https://developer.seequent.com/img/seequent-logo.svg" alt="Seequent logo" width="400" /></picture></a></p>
<p align="center">
    <a href="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml"><img src="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml/badge.svg" alt="" /></a>
</p>
<p align="center">
    <a href="https://developer.seequent.com/" target="_blank">Seequent Developer Portal</a>
    &bull; <a href="https://community.seequent.com/group/19-evo" target="_blank">Seequent Community</a>
    &bull; <a href="https://seequent.com" target="_blank">Seequent website</a>
</p>

## Evo

Evo is a unified platform for geoscience teams. It enables access, connection, computation, and management of subsurface data. This empowers better decision-making, simplified collaboration, and accelerated innovation. Evo is built on open APIs, allowing developers to build custom integrations and applications. Our open schemas, code examples, and SDK are available for the community to use and extend.

Evo is powered by Seequent, a Bentley organisation.

## Prerequisites

Before you get started, make sure you have:

* **A registered Evo app**

    *Evo apps* provide the credentials necessary to generate Evo access tokens, which in turn provide access to your Evo data. An app can be created by you or by a member of your team.
    
    Register an Evo app in the [Bentley Developer Portal](https://developer.bentley.com/my-apps). For in-depth instructions, follow this [guide](https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens) on the Seequent Developer Portal.

    NOTE: You must have a **Bentley developer account** in order to create apps. If you try to register an app using the link above but find that you don't have permission, contact your account administrator to get access.

* **A local copy of this repository**

    Clone the repository using Git or download it as a ZIP file from the green **Code** button at the top of the page.

* **A Python code editor, eg. VS Code, PyCharm**
    
    For running and editing the sample notebooks and other source code files.

## About this repository

`evo-python-sdk` is designed for developers, data scientists, and technical users who want to work with Seequent Evo APIs and geoscience data. 

* To quickly learn how to use Evo APIs, start with the [Getting started with Evo code samples](#getting-started-with-evo-code-samples) section, which contains practical, end-to-end Jupyter notebook examples for common workflows. Most new users should begin with this section.

* If you are interested in the underlying SDKs or need to understand the implementation details, explore the [Getting started with Evo SDK development](#getting-started-with-evo-sdk-development) section, which contains the source code for each Evo SDK. 

* To learn about contributing to this repository, take a look at the [Contributing](#contributing) section.

## Getting started with Evo code samples

For detailed information about creating Evo apps, the authentication setup, available code samples, and step-by-step guides for working with the Jupyter notebooks, please refer to the [**code-samples/README.md**](code-samples/README.md) file. 

This comprehensive guide will walk you through everything required to get started with Evo APIs. 

## Getting started with Evo SDK development

This repository contains a number of sub-packages. You may choose to install the `evo-sdk` package, which includes all
sub-packages and optional dependencies (e.g. Jupyter notebook support), or choose a specific package to install:

| Package | Version | Description |
| --- | --- | --- |
| [evo-sdk](README.md) | <a href="https://pypi.org/project/evo-sdk/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk" /></a> | A metapackage that installs all available Seequent Evo SDKs, including Jupyter notebook examples. |
| [evo-sdk-common](packages/evo-sdk-common/README.md) | <a href="https://pypi.org/project/evo-sdk-common/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk-common" /></a> | A shared library that provides common functionality for integrating with Seequent's client SDKs. |
| [evo-files](packages/evo-files/README.md) | <a href="https://pypi.org/project/evo-files/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-files" /></a> | A service client for interacting with the Evo File API. |
| [evo-objects](packages/evo-objects/README.md) | <a href="https://pypi.org/project/evo-objects/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-objects" /></a> | A geoscience object service client library designed to help get up and running with the Geoscience Object API. |
| [evo-colormaps](packages/evo-colormaps/README.md)  | <a href="https://pypi.org/project/evo-colormaps/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-colormaps" /></a> | A service client to create colour mappings and associate them to geoscience data with the Colormap API.|
| [evo-blockmodels](packages/evo-blockmodels/README.md) | <a href="https://pypi.org/project/evo-blockmodels/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-blockmodels" /></a> | The Block Model API provides the ability to manage and report on block models in your Evo workspaces. |
| [evo-compute](packages/evo-compute/README.md)  | <a href="https://pypi.org/project/evo-compute/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-compute" /></a> | A service client to send jobs to the Compute Tasks API.|

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
connector = OAuthConnector(transport=transport, client_id="<YOUR_CLIENT_ID>")
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
* [`evo-sdk-common`](packages/evo-sdk-common/README.md): providing the foundation for all Evo SDKs, as well as tools
  for performing arbitrary Seequent Evo API requests
* [`evo-files`](packages/evo-files/README.md): for interacting with the File API
* [`evo-objects`](packages/evo-objects/README.md): for interacting with the Geoscience Object API
* [`evo-colormaps`](packages/evo-colormaps/README.md): for interacting with the Colormap API
* [`evo-blockmodels`](packages/evo-blockmodels/README.md): for interacting with the Block Model API
* [`evo-compute`](packages/evo-compute/README.md): for interacting with the Compute Tasks API
* [Seequent Developer Portal](https://developer.seequent.com/docs/guides/getting-started/quick-start-guide): for guides,
  tutorials, and API references

## Contributing

Thank you for your interest in contributing to Seequent software. Please have a look over our [contribution guide](./CONTRIBUTING.md).

### Getting started

All Python SDKs in this monorepo are managed with [uv](https://docs.astral.sh/uv/).
We use [workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) in order to manage the different SDKs
published out of this repository.

With workspaces, `uv lock` operates on the entire workspace at once. `uv run` and `uv sync` operate on the workspace root by default, though both accept a `--package` argument allowing you to run a command in a particular workspace member from any workspace directory.

### Install UV

To install UV on your machine, run one of the following convenience scripts from the root of the repo. These scripts ensure everyone is using the same version.

#### Windows
```shell
./scripts/install-uv.ps1
```

#### Linux / macOS
```shell
./scripts/install-uv.sh
```
You can run the same script again whenever the version in the `UV_VERSION` file changes. It will replace your existing installation of UV.

### Install pre-commit hooks

Once you've installed UV, install pre-commit hooks. These are used to standardise development workflows for all contributors:

```shell
uv run pre-commit install
```

### Setting up and running Jupyter notebooks

Notebooks can be run in your tool of choice (e.g. VS Code). To use Jupyter (the default):

```shell
uv sync --all-packages --all-extras
```

Then, in the directory of the notebook(s) you want to run:

```shell
uv run jupyter notebook
```

A browser should launch where you can open the notebooks for the current directory.

## Code of conduct

We rely on an open, friendly, inclusive environment. To help us ensure this remains possible, please familiarise yourself with our [code of conduct](./CODE_OF_CONDUCT.md).

## License

The Python SDK for Evo is open source and licensed under the [Apache 2.0 license](./LICENSE.md).

Copyright © 2025 Bentley Systems, Incorporated.

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Test line

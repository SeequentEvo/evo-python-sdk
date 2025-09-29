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

## SDKs

This repository contains a number of sub-packages. You may choose to install the `evo-sdk` package, which includes all
sub-packages and optional dependencies (e.g. Jupyter notebook support), or choose a specific package to install:

| Package | Version | Description |
| --- | --- | --- |
| [evo-sdk](README.md) | <a href="https://pypi.org/project/evo-sdk/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk" /></a> | A metapackage that installs all available Seequent Evo SDKs, including Jupyter notebook examples. |
| [evo-sdk-common](packages/evo-sdk-common/README.md) | <a href="https://pypi.org/project/evo-sdk-common/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk-common" /></a> | A shared library that provides common functionality for integrating with Seequent's client SDKs. |
| [evo-files](packages/evo-files/README.md) | <a href="https://pypi.org/project/evo-files/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-files" /></a> | A service client for interacting with the Evo File API. |
| [evo-objects](packages/evo-objects/README.md) | <a href="https://pypi.org/project/evo-objects/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-objects" /></a> | A geoscience object service client library designed to help get up and running with the Geoscience Object API. |

## Pre-requisites

* Python ^3.10
* An [application registered in Bentley](https://developer.bentley.com/register/?product=seequent-evo)

## Installation

To install the `evo-sdk` package, including all sub-packages, run the following command:

```shell
pip install evo-sdk
```

Seequent Evo APIs use OAuth for authentication. In order to support it in this example, we'll be using the
[asyncio library](https://pypi.org/project/asyncio/) to power the OAuth callback process.

```shell
pip install asyncio
```

## Getting started

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

Windows:
```shell
./scripts/install-uv.ps1
```

UNIX-like:
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

Then in the directory of the notebook(s) you want to run, type:

```shell
uv run jupyter notebook
```

It should open a browser where you can open the notebooks for the current directory.

## Code of conduct

We rely on an open, friendly, inclusive environment. To help us ensure this remains possible, please familiarise yourself with our [code of conduct](./CODE_OF_CONDUCT.md).

## License

The Python SDK for Evo is open source and licensed under the [Apache 2.0 license.](./LICENSE.md).

Copyright © 2025 Bentley Systems, Incorporated.

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

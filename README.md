<p align="center"><a href="https://seequent.com" target="_blank"><picture><source media="(prefers-color-scheme: dark)" srcset="https://developer.seequent.com/img/seequent-logo-dark.svg" alt="Seequent logo" width="400" /><img src="https://developer.seequent.com/img/seequent-logo.svg" alt="Seequent logo" width="400" /></picture></a></p>
<p align="center">
    <a href="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/on-push.yaml"><img src="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/on-push.yaml/badge.svg" alt="" /></a>
</p>
<p align="center">
    <a href="https://developer.seequent.com/" target="_blank">Seequent Developer Portal</a>
    &bull; <a href="https://community.seequent.com/" target="_blank">Seequent Community</a>
    &bull; <a href="https://seequent.com" target="_blank">Seequent website</a>
</p>

## Evo

Evo is a unified platform for geoscience teams. It enables access, connection, computation, and management of subsurface data. This empowers better decision-making, simplified collaboration, and accelerated innovation. Evo is built on open APIs, allowing developers to build custom integrations and applications. Our open schemas, code examples, and SDK are available for the community to use and extend. 

Evo is powered by Seequent, a Bentley organisation.

## Getting started

All Python SDKs in this monorepo are managed with [uv](https://docs.astral.sh/uv/). 
We use [workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) in order to manage the different SDKs
published out of this repository. 

With workspaces, `uv lock` operates on the entire workspace at once. `uv run` and `uv sync` operate on the workspace root by default, though both accept a `--package` argument allowing you to run a command in a particular workspace member from any workspace directory.

## Install UV
To install uv on your machine, run one of the following convenience scripts from the root of the repo. These scripts ensure everyone is using the same version.

Windows:
```
./scripts/install-uv.ps1
```

UNIX-like:
```
./scripts/install-uv.sh
```
You can run the same script again whenever the version in the UV_VERSION file changes. It will replace your existing installation of uv.

### Install pre-commit hooks

Once you've installed UV, install pre-commit hooks. These are used to standardise development workflows for all contributors:

```
uv run pre-commit install
```

## SDKs
- [evo-client-common](packages/evo-client-common/README.md)
  - A shared library that provides common functionality for integrating with Seequent's client SDKs. 
- [evo-files](packages/evo-files/README.md)
  - A service client for interacting with the Evo File API.
- [evo-objects](packages/evo-objects/README.md)
  - A geoscience object service client library designed to help get up and running with geoscience objects. 

## Contributing

Thank you for your interest in contributing to Seequent software. Please have a look over our [contribution guide](./CONTRIBUTING.md).

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

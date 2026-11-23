<p align="center"><a href="https://seequent.com" target="_blank"><picture><source media="(prefers-color-scheme: dark)" srcset="https://developer.seequent.com/img/seequent-logo-dark.svg" alt="Seequent logo" width="400" /><img src="https://developer.seequent.com/img/seequent-logo.svg" alt="Seequent logo" width="400" /></picture></a></p>
<p align="center">
    <a href="https://pypi.org/project/evo-sdk-common/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk-common" /></a>
    <a href="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml"><img src="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml/badge.svg" alt="" /></a>
</p>
<p align="center">
    <a href="https://developer.seequent.com/" target="_blank">Seequent Developer Portal</a>
    &bull; <a href="https://community.seequent.com/group/19-evo" target="_blank">Seequent Community</a>
    &bull; <a href="https://seequent.com" target="_blank">Seequent website</a>
</p>

# Evo SDK Common

Evo SDK Common is a Python package that establishes a common framework for use by client libraries that interact
with Seequent Evo APIs. It can also be used to power arbitrary API requests to Seequent Evo APIs, where an existing
client SDK may not exist yet.

## Pre-requisites

* Python 3.11.5 or higher
* An [application registered in Bentley](https://developer.bentley.com/register/?product=seequent-evo)

## Installation

```shell
pip install evo-sdk-common
```

To use the SDK within a Jupyter notebook, include the `notebooks` optional dependency:

```shell
pip install evo-sdk-common[notebooks]
```

## Usage

See the [Getting Started](docs/quickstart.md) guide to learn how to use this library. There is more detailed information about authentication options in the [OAuth examples](docs/oauth.md), and a selection of interactive Jupyter notebooks in the [examples folder](docs/examples).

Here's a small example from the [Getting Started](docs/quickstart.md) guide showing to use `evo-sdk-common` to list the
organizations you have access to:

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

### Connection and file-transfer limits

`AioTransport(max_connections=...)` controls **total simultaneous HTTP connections** across all hosts.
Its default is now **100** (previously 4), matching aiohttp. Passing `max_connections=4` preserves the old
connection limit; `0` disables it. The legacy `num_pools` argument is a deprecated alias that emits a
`DeprecationWarning`; passing both raises `ValueError`.

File uploads and downloads share two independent, session-owned limits:

| Setting | Default | Scope |
|---|---:|---|
| `max_concurrent_transfers` | 16 | Active files, acquired before requesting their URL or opening the local file; held through commit/cleanup |
| `max_concurrent_chunks` | 16 | Active chunks, acquired before reading data and held through writing it |
| File operation's `max_workers` | 4 | Worker tasks per file, rather than one task per chunk |

With the default 4 MiB chunks, at most 64 MiB of active chunk payloads are admitted across these transfers.
This is not a bound on total process memory, preparation work, queued files or cache size. File/chunk limits must
be positive integers and are separate from HTTP connections; keeping them below the HTTP limit leaves capacity
for API traffic but does not reserve connections. Increasing `max_connections` alone does not increase buffered chunks.

Limits are shared by all clients using one open `AioTransport`, recreated on reopening it, and independent across
transports. There is no process-global semaphore or registry. Custom `ITransport` implementations can expose a
`transfer_limits` property returning one `evo.common.io.TransferLimits(max_files=..., max_chunks=...)` per session.
Without that optional property, existing custom transports retain per-file limiting only.

Storage operations retry transient HTTP statuses 408, 429, 500, 502, 503 and 504, raw connection failures and
timeouts with the existing `Retry` backoff policy. Failed chunk attempts cancel and drain workers, then retry only
unfinished chunks using the same tracker and staged block list. Completed chunks are not deliberately resent;
a lost acknowledgement can still cause an idempotent block PUT to be repeated. URL acquisition and final commits
have separate attempt budgets. Exhausted transport retries are not wrapped in another retry loop. Existing
401/403 URL renewal remains supported; permanent failures and exhausted budgets are reported to the caller.

## Contributing

For instructions on contributing to the development of this library, please refer to the [evo-python-sdk documentation](https://github.com/seequentevo/evo-python-sdk).

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

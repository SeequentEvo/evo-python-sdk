<p align="center"><a href="https://seequent.com" target="_blank"><picture><source media="(prefers-color-scheme: dark)" srcset="https://developer.seequent.com/img/seequent-logo-dark.svg" alt="Seequent logo" width="400" /><img src="https://developer.seequent.com/img/seequent-logo.svg" alt="Seequent logo" width="400" /></picture></a></p>
<p align="center">
    <a href="https://pypi.org/project/evo-objects/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-objects" /></a>
    <a href="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml"><img src="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml/badge.svg" alt="" /></a>
</p>
<p align="center">
    <a href="https://developer.seequent.com/" target="_blank">Seequent Developer Portal</a>
    &bull; <a href="https://community.seequent.com/group/19-evo" target="_blank">Seequent Community</a>
    &bull; <a href="https://seequent.com" target="_blank">Seequent website</a>
</p>

# Evo Object API Client

The Geoscience Object API is Seequent's next-generation cloud offering for geoscience
data, empowering our users to build responsive modern workflows. 

## Pre-requisites

* Python ^3.10
* An [application registered in Bentley](https://developer.bentley.com/register/?product=seequent-evo)

## Installation

```shell
pip install evo-objects
```

## Usage

See [the evo-sdk-common documentation](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-sdk-common/README.md)
for information on how to authenticate, then select the organisation, hub and workspace that you would like to use.

### Interacting with the Geoscience Object API

To get up and running quickly with the Evo Objects SDK, start by configuring your
[environment and API connector](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-sdk-common/docs/quickstart.md).

```python
from evo.objects import ObjectAPIClient

service_client = ObjectAPIClient(environment, connector)
service_health = await service_client.get_service_health()
service_health.raise_for_status()

# The data client is an optional utility that provides helpers for uploading and downloading
# parquet data via pyarrow.Table objects.
data_client = service_client.get_data_client(manager.cache)
...
```

Listing objects is simple, just call the `ObjectAPIClient.list_objects()` method.

```python
offset = 0
while True:
    page = await service_client.list_objects(offset=offset, limit=10)
    if offset == 0:
        print(f"Found {page.total} object{'' if page.total == 1 else 's'}")
    for object in page:
        print(f"{object.path}: <{object.schema_id}> ({object.id})")

    if page.is_last:
        break
    else:
        offset = page.next_offset
```

You can also get a list of all objects. Internally, this recursively calls the `list_objects()` method until all objects are fetched.

Check out the other methods on the `ObjectAPIClient` for more details on how to upload and download objects, or get object versions.

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

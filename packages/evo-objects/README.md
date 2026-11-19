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

* Python 3.11.5 or higher
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

### Typed downhole collections

`DownholeCollection` provides a DataFrame-based API for creating and reading downhole objects. A collection can contain
distance tables, interval tables, or both. `hole_index` is an integer key in the object's `hole_id` lookup table, not a
row position or a pandas categorical code. The creation helper uses dense zero-based keys in sorted hole-ID order unless
an explicit `hole_indices` mapping is supplied. Persisted objects may use arbitrary integer keys. Collection holes may
contain a subset or repeated chunks.

```python
import pandas as pd

from evo.objects.typed import DownholeCollection, DownholeCollectionData
from evo.objects.typed.downhole_collection import IntervalCollection
from evo.objects.utils.downhole import hole_chunks_from_ids

intervals = pd.DataFrame({"from": [0.0], "to": [1.5], "lithology": ["sandstone"]})
collections = [
    IntervalCollection(
        name="geology",
        holes=hole_chunks_from_ids(pd.Series(["DH-01"])),
        table=intervals,
        unit="m",  # Collection coordinate units are set explicitly here.
    )
]

# Build DownholeCollectionData with matching path, location-hole chunks, and collar properties,
# then create it with: await DownholeCollection.create(context, data)
```

Use `await dhc.location.to_dataframe()` and `await dhc.location.path_to_dataframe()` to read collars and paths.
Distance and interval tables provide `to_dataframe()` and `to_dataframe_by_hole()`. Before reading a large object, call
`await dhc.prefetch_collections("geology")` to warm only the requested collection data (and location data by default).
Attribute descriptions for ordinary attributes round-trip through `DataFrame.attrs["attribute_descriptions"]`.
Collection coordinate units come only from `DistanceCollection.unit` or `IntervalCollection.unit`; unit metadata on the
distance or `from` column is ignored.

### Concurrent persistence

Typed creation and replacement prepare and upload independent attributes, submodels and collections
concurrently. DataFrame conversion, Parquet compression and hashing run in the default thread pool, not on the
event loop. Completed files can upload while other files are still being prepared; preparation never holds an
upload slot. The default DataFrame path passes cached-file metadata to uploading rather than retaining intermediate
Arrow tables. Existing `upload_table`/`upload_category_table` overrides still intercept DataFrame uploads, but may
retain their Arrow inputs until the override returns.

Identical tables share an immutable, hash-named Parquet cache file. Completed files are published atomically
without replacing an existing file, including across threads and processes. On POSIX, the cache file system
must support hard links; publication errors propagate rather than falling back to an overwrite.
Publication and temporary-file cleanup use bounded retries for `PermissionError` on all platforms.

Results retain input order. Parallel table/attribute replacement stages results and checks for attribute conflicts
before applying changes. Failure cancels and drains sibling tasks; cancellation waits for running cache writers
and chunk tasks before returning. Uploaded blobs are not rolled back on failure. Subclasses that customize
`DataTable.from_dataframe` must put commit-time derived-metadata updates in `_apply_uploaded_table` for those
updates to run during combined table/attribute replacement; perform fallible checks before modifying live state.

File admission and buffered chunks are controlled by the shared transport, not a global typed-upload semaphore:

```python
from evo.aio import AioTransport

transport = AioTransport(
    user_agent="Your Application Name",
    max_connections=100,            # Total simultaneous HTTP connections (replaces the deprecated num_pools).
    max_concurrent_transfers=16,    # Files, before requesting a SAS URL or opening the cache file.
    max_concurrent_chunks=16,       # Buffered chunks across file uploads and downloads.
)
# Supply this transport to the APIConnector used by your context.
```

The defaults allow up to 64 MiB of active chunk payloads (16 x 4 MiB), excluding input frames, preparation,
HTTP overhead and cached disk data. They are starting defaults, not a total-memory guarantee. Custom transports
can opt into shared limits as described in `evo-sdk-common`; otherwise per-file limits still apply.
Do not modify input DataFrames or clear/evict their cache while persistence is running. Pending preparation
jobs and completed cached files are not bounded by the transfer limits.

Transient storage HTTP errors (408, 429, 500, 502, 503, 504), raw connection failures and timeouts use bounded
backoff in the common I/O layer. Chunk retries retain completed blocks; final commits and URL acquisition have
their own retry budgets. Permanent errors and exhausted transport retries propagate, and cancellation is never
treated as success. The default is three consecutive attempts with exponential backoff; successful chunks reset
the chunk-transfer attempt counter. This does not retry the whole object or discard successfully prepared files.

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

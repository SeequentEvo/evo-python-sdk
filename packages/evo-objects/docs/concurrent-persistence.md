# Concurrent persistence

Typed object creation and replacement prepare and upload independent attributes and collections concurrently.
This happens automatically and preserves input order.

## Adjusting transfer limits

Use the transport supplied to your `APIConnector` to tune concurrency:

```python
from evo.aio import AioTransport

transport = AioTransport(
    user_agent="Your Application Name",
    max_connections=100,
    max_concurrent_transfers=16,
    max_concurrent_chunks=16,
)
```

These are the defaults. `max_connections` limits simultaneous HTTP connections.
`max_concurrent_transfers` limits active file transfers, while `max_concurrent_chunks` limits buffered chunks
across uploads and downloads. Clients using the same open transport share these limits.

Lower the transfer and chunk limits to reduce resource use. With the default 4 MiB chunks, 16 buffered chunks
use up to 64 MiB for their payloads. This is **not a total-memory limit**: input DataFrames, data preparation
and HTTP processing need additional memory. Queued work and disk cache size are not bounded by these settings.

See [connection and file-transfer limits](../../evo-sdk-common/README.md#connection-and-file-transfer-limits)
for custom transports and migration from `num_pools`.

## Choosing and managing a file cache

Object data is prepared as Parquet files in a local cache before upload. The cache also allows downloaded data
to be reused. Identical prepared tables share a cache file.

You can choose the cache directory when constructing the data client:

```python
from pathlib import Path

from evo.common.utils import Cache

cache = Cache(Path.home() / ".evo-cache", mkdir=True)
data_client = service_client.get_data_client(cache)
context = data_client.get_static_context()
```

Use this `context` with typed creation methods. In notebooks, you can instead use `manager.cache`; see the
[notebook utilities example](../../evo-sdk-common/docs/examples/notebook-utilities.ipynb).

On POSIX systems, including Linux and macOS, the cache filesystem must support hard links. Some removable
drives and network or virtual filesystems do not. If cache writes fail because hard links are unsupported,
choose a cache directory on a supported local filesystem. The SDK reports the failure rather than risk
overwriting a file another operation is using. Windows does not require hard links.

Allow enough disk space for prepared and downloaded data. Do not modify input DataFrames or clear the cache
while persistence is running. Once all operations using it have finished, `data_client.clear_cache()` removes
the cached object data for that workspace. This does not delete data stored in Evo.

## Failures and cancellation

Temporary network and storage errors are retried automatically with bounded backoff. Permanent errors and
exhausted retries are raised to the caller; an entire object operation is not automatically retried.

Combined table and attribute replacement applies local changes only after its uploads succeed. If an upload
fails, the other uploads in that operation are cancelled. Cancellation waits for in-progress file work to
finish, so it may not return immediately. Data already uploaded to Evo is not rolled back.

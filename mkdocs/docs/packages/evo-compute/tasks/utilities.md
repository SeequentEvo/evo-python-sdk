[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/utilities)

# Utilities Tasks

Discoverable compute tasks in the `utilities` topic. Call any task below through `ComputeClient` as `await client.utilities.<task>.run(...)`, or by name via `await client.arun("utilities", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `FileInput` | `str | FileHandle` | A File API URL or an SDK file/upload handle exposing `.url` or `.metadata.url`. |

## Tasks

### compress

`client.utilities.compress` · Task: `utilities/compress` · Version: `1.1.0` · Preview (`preview=True` required)

Downloads a set of files referenced by File API URLs, bundles them into a single archive (ZIP or TAR with gzip/bzip2/xz compression), uploads the archive back to the File API, and returns a downloadable reference to it.

```python
await client.utilities.compress.run(
    *,
    files: list[FileInput],
    destination: FileInput,
    output_name: str = ...,
    archive_format: Literal["zip", "zip-stored", "tar", "tar.gz", "tar.bz2", "tar.xz"] = ...,
    compression_level: int | None = ...,
    preview: bool = False,
) -> UtilitiesCompressResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `files` | `list[FileInput]` | File API references (URLs) for the files to include in the archive. | _required_ |
| `destination` | `FileInput` | File API reference (path- or id-based URL) specifying where the resulting archive should be uploaded. | _required_ |
| `output_name` | `str` | Friendly name of the resulting archive returned in the task results. | `'download.zip'` |
| `archive_format` | `Literal["zip", "zip-stored", "tar", "tar.gz", "tar.bz2", "tar.xz"]` | Archive/compression format: 'zip' (default), 'zip-stored', 'tar', 'tar.gz', 'tar.bz2', or 'tar.xz'. | `'zip'` |
| `compression_level` | `int | None` | Compression level (0-9), higher means smaller output but slower. Applies to 'zip', 'tar.gz', 'tar.bz2', and 'tar.xz'. Not supported for 'zip-stored' and 'tar'. Defaults to the format's default level. Bounds: `>= 0`, `<= 9`. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`UtilitiesCompressResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `output` | `UtilitiesCompressResultArchive` | — | Reference to the uploaded archive, which can be downloaded via the File API. |

#### Nested Shapes

**`UtilitiesCompressResultArchive`** (`ResultNode`)

Reference to the uploaded archive, which can be downloaded via the File API.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a file. |
| `name` | `str` | — | The name of the file, typically the terminal segment of the path. |

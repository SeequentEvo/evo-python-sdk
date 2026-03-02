# evo-files

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-files/src/evo/files/)

File SDK provides the ability to manage files of any type or size, associated with your Evo workspace. Enable your product with Evo connected workflows by integrating with the Seequent Evo File API. Most file formats and sizes are accepted.

## Usage

```python
from evo.files import FileAPIClient

file_client = FileAPIClient(environment, connector)
files = await file_client.list_files(workspace_id)
```

See the [FileAPIClient](FileAPIClient.md) page for the full API reference.


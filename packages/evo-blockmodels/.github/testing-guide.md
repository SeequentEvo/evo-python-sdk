# Testing Guide for evo-blockmodels

## Running Tests

### With uv (preferred)
```bash
cd packages/evo-blockmodels
uv sync --group test
uv run pytest tests/ -v
```

### With local source (if uv version mismatch)
```powershell
cd packages/evo-blockmodels
$env:PYTHONPATH="src;..\evo-sdk-common\src"
python -m pytest tests/ -v
```

### Run specific test file
```bash
python -m pytest tests/test_typed_regular_block_model.py -v
```

### Run specific test
```bash
python -m pytest tests/test_typed_regular_block_model.py::TestRegularBlockModelCreate::test_create_with_data -v
```

## Important: Data Type Requirements

### Unit IDs
Always use valid unit IDs from the `Units` class:
```python
from evo.blockmodels import Units

# Valid
units = {"grade": Units.GRAMS_PER_TONNE}  # "g/t"

# Invalid - will cause 422 error
units = {"grade": "g/m3"}  # Not a valid unit ID
```

### Geometry Column Types
The Block Model Service requires `i`, `j`, `k` columns to be `uint32`:
```python
# The dataframe_to_pyarrow() utility handles this automatically
from evo.blockmodels.typed._utils import dataframe_to_pyarrow

df = pd.DataFrame({"i": [0, 1], "j": [0, 1], "k": [0, 1], "grade": [1.0, 2.0]})
table = dataframe_to_pyarrow(df)  # Casts i,j,k to uint32
```

## Test Structure

### Base Classes

```python
from evo.common.test_tools import TestWithConnector, TestWithStorage

class TestMyFeature(TestWithConnector, TestWithStorage):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)  # Provides: connector, transport, environment
        TestWithStorage.setUp(self)    # Provides: cache
        self.setup_universal_headers(get_header_metadata("evo.blockmodels.client"))
```

### Creating Test Context

```python
from evo.common import StaticContext

self._context = StaticContext.from_environment(
    environment=self.environment,
    connector=self.connector,
    cache=self.cache,
)
```

## Mocking Strategies

### 1. Request Handlers (for full integration)

Best for testing complete workflows with HTTP requests:

```python
class MyRequestHandler(JobPollingRequestHandler):
    def __init__(self, result, job_response, pending_request=0):
        super().__init__(job_response, pending_request)
        self._result = result

    async def request(self, method, url, headers=None, body=None, **kwargs) -> MockResponse:
        match method:
            case RequestMethod.POST if url.endswith("/block-models"):
                return MockResponse(status_code=201, content=self._result.model_dump_json())
            case RequestMethod.POST if url.endswith("/uploaded"):
                job_url, _ = url.rsplit("/", 1)
                return MockResponse(status_code=201, content=json.dumps({"job_url": job_url}))
            case RequestMethod.PATCH:
                return MockResponse(status_code=202, content=self._update_result.model_dump_json())
            case RequestMethod.GET:
                return self.job_poll()  # From JobPollingRequestHandler
            case _:
                return self.not_found()
```

### 2. HTTP Response (for simple cases)

```python
with self.transport.set_http_response(
    200,
    mock_response.model_dump_json(),
    headers={"Content-Type": "application/json"},
):
    result = await client.list_block_models()
```

### 3. Mock Client Methods (for typed layer)

Best for testing typed classes without HTTP complexity:

```python
from unittest import mock

with (
    mock.patch.object(BlockModelAPIClient, "get_block_model") as mock_get,
    mock.patch.object(BlockModelAPIClient, "query_block_model_as_table") as mock_query,
    mock.patch.object(BlockModelAPIClient, "list_versions") as mock_versions,
):
    mock_get.return_value = mock_metadata
    mock_query.return_value = pyarrow_table
    mock_versions.return_value = [mock_version]
    
    result = await RegularBlockModel.get(context, bm_id)
```

### 4. Mock Upload (for data operations)

```python
with mock.patch("evo.common.io.upload.StorageDestination") as mock_dest:
    mock_dest.upload_file = mock.AsyncMock()
    result = await RegularBlockModel.create(context, data)
    mock_dest.upload_file.assert_called_once()
```

## Creating Mock Data

### Block Model Metadata

```python
def _mock_block_model(environment) -> models.BlockModel:
    return models.BlockModel(
        bbox=models.BBoxXYZ(
            x_minmax=models.FloatRange(min=0, max=10),
            y_minmax=models.FloatRange(min=0, max=10),
            z_minmax=models.FloatRange(min=0, max=10),
        ),
        block_rotation=[models.Rotation(axis=RotationAxis.x, angle=20)],
        bm_uuid=BM_UUID,
        name="Test BM",
        description="Test Block Model",
        coordinate_reference_system="EPSG:4326",
        size_unit_id="m",
        workspace_id=environment.workspace_id,
        org_uuid=environment.org_id,
        model_origin=models.Location(x=0, y=0, z=0),
        normalized_rotation=[0, 20, 0],
        size_options=models.SizeOptionsRegular(
            model_type="regular",
            n_blocks=models.Size3D(nx=10, ny=10, nz=10),
            block_size=models.BlockSize(x=1, y=1, z=1),
        ),
        geoscience_object_id=GOOSE_UUID,
        created_at=DATE,
        created_by=MODEL_USER,
        last_updated_at=DATE,
        last_updated_by=MODEL_USER,
    )
```

### Version

```python
def _mock_version(version_id, version_uuid, goose_version_id, bbox=None, columns=()):
    return models.Version(
        base_version_id=None if version_id == 1 else version_id - 1,
        bbox=bbox,
        bm_uuid=BM_UUID,
        comment="",
        created_at=DATE,
        created_by=MODEL_USER,
        geoscience_version_id=goose_version_id,
        mapping=models.Mapping(columns=list(columns)),
        parent_version_id=version_id - 1,
        version_id=version_id,
        version_uuid=version_uuid,
    )
```

### Data Classes (for typed layer)

```python
from evo.blockmodels.data import BlockModel as BlockModelData, RegularGridDefinition, Version

mock_metadata = BlockModelData(
    environment=self.environment,
    id=BM_UUID,
    name="Test BM",
    description="Test Block Model",
    created_at=DATE,
    created_by=USER,
    grid_definition=RegularGridDefinition(
        model_origin=[0, 0, 0],
        rotations=[(RotationAxis.x, 20)],
        n_blocks=[10, 10, 10],
        block_size=[1.0, 1.0, 1.0],
    ),
    coordinate_reference_system="EPSG:4326",
    size_unit_id="m",
    bbox=BM_BBOX,
    last_updated_at=DATE,
    last_updated_by=USER,
    geoscience_object_id=GOOSE_UUID,
)
```

## Common Test Patterns

### Testing Create Operations

```python
async def test_create_with_data(self):
    self.transport.set_request_handler(
        CreateRequestHandler(
            create_result=_mock_create_result(self.environment),
            job_response=JobResponse(job_status=JobStatus.COMPLETE, payload=FIRST_VERSION),
            update_result=UPDATE_RESULT,
            update_job_response=JobResponse(job_status=JobStatus.COMPLETE, payload=SECOND_VERSION),
        )
    )
    
    with mock.patch("evo.common.io.upload.StorageDestination") as mock_dest:
        mock_dest.upload_file = mock.AsyncMock()
        result = await RegularBlockModel.create(self.context, data)
        mock_dest.upload_file.assert_called_once()
    
    self.assertEqual(result.id, BM_UUID)
```

### Testing Get Operations

```python
async def test_get_block_model(self):
    with (
        mock.patch.object(BlockModelAPIClient, "get_block_model") as mock_get,
        mock.patch.object(BlockModelAPIClient, "query_block_model_as_table") as mock_query,
        mock.patch.object(BlockModelAPIClient, "list_versions") as mock_versions,
    ):
        mock_get.return_value = mock_metadata
        mock_query.return_value = test_table
        mock_versions.return_value = [mock_version]
        
        result = await RegularBlockModel.get(self.context, BM_UUID)
    
    self.assertEqual(result.id, BM_UUID)
    self.assertEqual(len(result.cell_data), 3)
```

### Testing Update Operations

```python
async def test_update_attributes(self):
    self.transport.set_request_handler(
        UpdateRequestHandler(
            update_result=UPDATE_RESULT,
            job_response=JobResponse(job_status=JobStatus.COMPLETE, payload=UPDATED_VERSION),
        )
    )
    
    with mock.patch("evo.common.io.upload.StorageDestination") as mock_dest:
        mock_dest.upload_file = mock.AsyncMock()
        new_version = await block_model.update_attributes(new_data, new_columns=["col1"])
    
    self.assertEqual(new_version.version_id, 2)
```

## Debugging Tips

1. **Print HTTP requests**: Add logging to request handlers
2. **Check job payloads**: Ensure `job_response.payload` matches expected model
3. **Validate Pydantic models**: Use `.model_dump()` to inspect model state
4. **Check PYTHONPATH**: Ensure local source is in path when testing

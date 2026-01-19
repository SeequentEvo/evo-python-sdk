# Copilot Instructions for evo-blockmodels

## Package Overview

The `evo-blockmodels` package provides a Python SDK for the Seequent Evo Geoscience Block Model API. It supports creating, querying, and updating block models with versioning and typed access patterns.

## Related: BlockModelRef in evo-objects

For workflows that need block models as Geoscience Objects (e.g., compute tasks), use `BlockModelRef` from `evo.objects.typed`:

```python
from evo.objects.typed import BlockModelRef
from evo.objects import ObjectReference

# Load an existing block model reference using its geoscience_object_id
object_ref = ObjectReference.new(
    environment=manager.get_environment(),
    object_id=block_model.metadata.geoscience_object_id,
)
bm_ref = await BlockModelRef.from_reference(manager, object_ref)

# Access data through the Block Model Service
df = await bm_ref.get_data(columns=["*"])

# Add attributes
await bm_ref.add_attribute(data_df, "new_attribute", unit=Units.KG_PER_CUBIC_METRE)
```

## Architecture

### Core Components

1. **`client.py`** - `BlockModelAPIClient`: Main API client for block model operations
   - Uses `OperationsApi`, `VersionsApi`, `ColumnOperationsApi`, `JobsApi`, `MetadataApi`, `UnitsApi`
   - Requires `IContext` with environment, connector, and cache
   - Job-based async operations with polling

2. **`typed/`** - Typed access layer (similar to `evo-objects/typed/`)
   - `RegularBlockModel` - High-level typed wrapper for regular block models
   - `RegularBlockModelData` - Dataclass for creating block models
   - `types.py` - Local type definitions (Point3, Size3i, Size3d, BoundingBox)
   - `units.py` - Unit constants and utilities
   - `_utils.py` - DataFrame/PyArrow conversion utilities

3. **`endpoints/`** - Auto-generated API models and endpoints
   - `models.py` - Pydantic models generated from OpenAPI spec
   - `api/` - API endpoint classes (auto-generated, do not edit)

4. **`data.py`** - Data classes for block model metadata
   - `BlockModel`, `Version`, grid definition classes

### Key Patterns

#### Using Units

Always use the `Units` class for unit IDs to avoid invalid unit errors:

```python
from evo.blockmodels import Units

# Creating block model with units
bm_data = RegularBlockModelData(
    name="My Model",
    size_unit_id=Units.METRES,
    units={
        "grade": Units.GRAMS_PER_TONNE,
        "density": Units.TONNES_PER_CUBIC_METRE,
    },
    ...
)

# Adding attribute with unit
await bm_ref.add_attribute(df, "metal_content", unit=Units.KG_PER_CUBIC_METRE)

# Get all available units from the service
from evo.blockmodels import get_available_units
units = await get_available_units(context)
```

Common unit constants:
- Length: `Units.METRES`, `Units.FEET`, `Units.CENTIMETRES`
- Mass: `Units.KILOGRAMS`, `Units.TONNES`, `Units.GRAMS`
- Density: `Units.KG_PER_CUBIC_METRE`, `Units.TONNES_PER_CUBIC_METRE`, `Units.GRAMS_PER_CUBIC_CENTIMETRE`
- Grade: `Units.GRAMS_PER_TONNE`, `Units.PERCENT`, `Units.PPM`

#### Creating Block Models

```python
from evo.blockmodels import RegularBlockModel, RegularBlockModelData, Point3, Size3i, Size3d, Units

data = RegularBlockModelData(
    name="My Model",
    origin=Point3(0, 0, 0),
    n_blocks=Size3i(10, 10, 10),
    block_size=Size3d(1.0, 1.0, 1.0),
    cell_data=my_dataframe,  # pd.DataFrame with i,j,k + attributes
    size_unit_id=Units.METRES,
    units={"grade": Units.GRAMS_PER_TONNE},
)
block_model = await RegularBlockModel.create(context, data)

# The block model has an associated Geoscience Object
print(f"GOOSE ID: {block_model.metadata.geoscience_object_id}")
```

#### DataFrame Requirements

DataFrames must include geometry columns with correct types:
- **Columns**: `i`, `j`, `k` (block indices) - will be cast to `uint32`
- **OR**: `x`, `y`, `z` (coordinates)

The `dataframe_to_pyarrow()` utility automatically casts i,j,k to `uint32` as required by the Block Model Service.

#### Querying Block Models

```python
# Typed access (returns pd.DataFrame with column names, not UUIDs)
block_model = await RegularBlockModel.get(context, bm_id)
df = block_model.cell_data

# Via BlockModelRef
df = await bm_ref.get_data(columns=["*"])  # Returns user-friendly column names
df = await bm_ref.get_data(columns=["grade", "density"])
```

#### Updating Attributes

```python
# Via BlockModelRef
await bm_ref.add_attribute(df, "new_col", unit=Units.GRAMS_PER_TONNE)

await bm_ref.update_attributes(
    df,
    new_columns=["col1", "col2"],
    update_columns={"existing_col"},
    delete_columns={"old_col"},
    units={"col1": Units.PERCENT},
)
```

### Async Job Pattern

Block model operations are async with job polling:
1. Call API endpoint → returns `job_url`
2. Poll `get_job_status` until `COMPLETE` or `FAILED`
3. Extract payload from completed job response

### Upload Workflow

For data uploads (create with data, add columns, update columns):
1. Call update endpoint → returns `upload_url` and `job_uuid`
2. Write data to cache as parquet file
3. Upload to `upload_url` using `BlockModelUpload`
4. Call `notify_upload_complete`
5. Poll job until complete

## Testing Patterns

### Test Base Classes
- `TestWithConnector` - Provides mock `connector`, `transport`, `environment`
- `TestWithStorage` - Provides mock `cache`
- Use `StaticContext.from_environment()` to create test context

### Mocking Client Methods
```python
with mock.patch.object(BlockModelAPIClient, "get_block_model") as mock_get:
    mock_get.return_value = mock_metadata
    result = await RegularBlockModel.get(context, bm_id)
```

## Dependencies

- `evo-sdk-common` - Common SDK utilities, IContext, ICache, IFeedback
- `pydantic>=2` - Data validation
- `pyarrow>=19` - Parquet file handling (optional)
- `pandas>=2` - DataFrame support (optional, with pyarrow)


## Code Style

- Line length: 120 characters
- Use type hints for all function signatures
- Use dataclasses with `frozen=True, kw_only=True` for immutable data
- Follow existing patterns in `evo-objects/typed/` for typed access


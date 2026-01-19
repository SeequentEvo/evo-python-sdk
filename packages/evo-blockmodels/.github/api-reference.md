# Block Model API Reference

## Endpoints Overview

The Block Model API is organized into these endpoint groups:

### OperationsApi
- `create_block_model` - Create a new block model with grid definition
- `delete_block_model` - Delete a block model
- `restore_block_model` - Restore a deleted block model

### VersionsApi
- `list_block_model_versions` - List all versions of a block model
- `retrieve_block_model_version` - Get specific version metadata
- `query_block_model_latest_as_post` - Query block data (returns job URL)
- `get_deltas_for_block_model` - Check for changes between versions

### ColumnOperationsApi
- `update_block_model_from_latest_version` - Start update workflow (add/update/delete columns)
- `notify_upload_complete` - Signal that data upload is complete

### JobsApi
- `get_job_status` - Poll job status and get result payload

### MetadataApi
- `list_block_models` - List block models in workspace
- `retrieve_block_model` - Get block model metadata by ID

### UnitsApi
- `get_units` - Get list of available units for column values

## Units

The Block Model Service requires specific unit IDs. Use the `Units` class for valid unit IDs:

```python
from evo.blockmodels import Units, get_available_units

# Common unit constants
Units.METRES                  # "m"
Units.GRAMS_PER_TONNE         # "g/t"
Units.TONNES_PER_CUBIC_METRE  # "t/m3"
Units.KG_PER_CUBIC_METRE      # "kg/m3"
Units.PERCENT                 # "%"
Units.PPM                     # "ppm"

# Query all available units from the service
units = await get_available_units(context)
```

### Unit Types
- `LENGTH` - m, cm, mm, km, ft, in, yd, mi
- `MASS` - g, kg, t, lb, oz, oz_tr
- `VOLUME` - m3, cm3, L, ft3, gal_us
- `MASS_PER_VOLUME` - kg/m3, g/cm3, t/m3, lb/ft3
- `MASS_PER_MASS` - g/t, kg/t, %, ppm, ppb, oz/t, lb/t
- `VOLUME_PER_VOLUME` - %_vol
- `VALUE_PER_MASS` - $/t, $/oz

## Data Flow

### Create Block Model

```
1. POST /block-models
   Request: CreateData (name, grid definition, origin, rotations)
   Response: BlockModelAndJobURL (bm_uuid, job_url)

2. GET /jobs/{job_id} (poll until COMPLETE)
   Response: JobResponse (job_status, payload: Version)
```

### Update with Data

```
1. PATCH /block-models/{bm_id}/blocks
   Request: UpdateDataLiteInput (columns: new/update/delete/rename)
   Response: UpdateWithUrl (job_uuid, upload_url)

2. PUT {upload_url}
   Upload parquet file with data

3. POST /block-models/{bm_id}/jobs/{job_id}/uploaded
   Response: UpdatedResponse (job_url)

4. GET /jobs/{job_id} (poll until COMPLETE)
   Response: JobResponse (job_status, payload: Version)
```

### Query Block Model

```
1. POST /block-models/{bm_id}/blocks
   Request: QueryCriteria (columns, bbox, version_uuid, output_options)
   Response: QueryResult (job_url)

2. GET /jobs/{job_id} (poll until COMPLETE)
   Response: JobResponse (job_status, payload: QueryDownload with download_url)

3. GET {download_url}
   Download parquet file with queried data
```

## Key Models

### Grid Definitions

```python
# Regular grid
SizeOptionsRegular(
    model_type="regular",
    n_blocks=Size3D(nx=10, ny=10, nz=10),
    block_size=BlockSize(x=1.0, y=1.0, z=1.0),
)

# Fully sub-blocked
SizeOptionsFullySubBlocked(
    model_type="fully_sub_blocked",
    n_parent_blocks=Size3D(...),
    n_subblocks_per_parent=Size3D(...),
    parent_block_size=BlockSize(...),
)

# Octree
SizeOptionsOctree(
    model_type="octree",
    n_parent_blocks=Size3D(...),
    n_subblocks_per_parent=Size3D(...),
    parent_block_size=BlockSize(...),
)

# Flexible
SizeOptionsFlexible(
    model_type="flexible",
    n_parent_blocks=Size3D(...),
    n_subblocks_per_parent=Size3D(...),
    parent_block_size=BlockSize(...),
)
```

### Column Operations

```python
UpdateColumnsLiteInput(
    new=[ColumnLite(title="col1", data_type=DataType.Float64, unit_id="g/t")],
    update=["existing_col"],  # List of column titles to update
    delete=["old_col"],       # List of column titles to delete
    rename=[RenameLite(title="old", new_title="new")],
    update_metadata=[UpdateMetadataLite(title="col", values=UpdateMetadataValues(unit_id="kg"))],
)
```

### Query Criteria

```python
QueryCriteria(
    columns=["*"],  # or specific column titles/IDs
    bbox=BBox(      # Optional bounding box filter
        i_minmax=IntRange(min=0, max=10),
        j_minmax=IntRange(min=0, max=10),
        k_minmax=IntRange(min=0, max=10),
    ),
    version_uuid=uuid,  # Optional, defaults to latest
    geometry_columns=GeometryColumns.coordinates,  # or "indices"
    output_options=OutputOptionsParquet(
        file_format="parquet",
        column_headers=ColumnHeaderType.name,  # or "id"
        exclude_null_rows=True,
    ),
)
```

### Job Status

```python
class JobStatus(Enum):
    PENDING_UPLOAD = "PENDING_UPLOAD"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
```

## Data Types

Supported column data types (from PyArrow):
- `Boolean`, `Int8`, `Int16`, `Int32`, `Int64`
- `UInt8`, `UInt16`, `UInt32`, `UInt64`
- `Float16`, `Float32`, `Float64`
- `Utf8` (strings)
- `Date32`, `Timestamp`

## System Columns

Reserved columns (cannot be modified):
- `i`, `j`, `k` - Block indices
- `x`, `y`, `z` - Centroid coordinates
- `dx`, `dy`, `dz` - Sub-block sizes (sub-blocked models)
- `sidx` - Sub-block index (fully sub-blocked/octree)
- `start_si`, `start_sj`, `start_sk`, `end_si`, `end_sj`, `end_sk` - Flexible model indices
- `version_id` - Version identifier

## Error Handling

Common error types:
- `BadRequestException` (400) - Invalid request data
- `UnauthorizedException` (401) - Authentication required
- `ForbiddenException` (403) - Insufficient permissions
- `NotFoundException` (404) - Resource not found
- `JobFailedException` - Job completed with failure status


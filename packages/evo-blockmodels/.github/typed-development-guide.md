# Typed Block Model Development Guide

## Overview

The `typed/` module provides high-level, pandas DataFrame-based access to block models, similar to `evo-objects/typed/`. This guide covers extending and maintaining the typed access layer.

## Module Structure

```
typed/
├── __init__.py              # Public exports
├── types.py                 # Local type definitions (Point3, Size3i, etc.)
├── units.py                 # Unit constants and utilities
├── _utils.py                # Internal conversion utilities
└── regular_block_model.py   # RegularBlockModel and RegularBlockModelData
```

## Using Units

Always use the `Units` class for unit IDs to avoid validation errors:

```python
from evo.blockmodels import Units

# Common unit constants
Units.METRES            # "m"
Units.GRAMS_PER_TONNE   # "g/t"
Units.TONNES_PER_CUBIC_METRE  # "t/m3"
Units.KG_PER_CUBIC_METRE      # "kg/m3"
Units.PERCENT           # "%"
Units.PPM               # "ppm"

# Query available units from the service
from evo.blockmodels import get_available_units
units = await get_available_units(context)
for unit in units:
    print(f"{unit.unit_id}: {unit.description}")
```

## DataFrame Requirements

### Geometry Columns

DataFrames must include geometry columns:
- **Regular grids**: `i`, `j`, `k` (indices) OR `x`, `y`, `z` (coordinates)
- **Sub-blocked grids**: Additional `sidx` or `dx`, `dy`, `dz` columns

### Data Type Requirements

The Block Model Service requires specific data types:
- `i`, `j`, `k` columns must be `uint32` (not `int64`)
- The `dataframe_to_pyarrow()` utility handles this automatically

```python
# In _utils.py - automatically casts i,j,k to uint32
def dataframe_to_pyarrow(df: pd.DataFrame) -> pa.Table:
    table = pa.Table.from_pandas(df)
    
    # Cast i, j, k columns to uint32 as required by Block Model Service
    for col_name in ["i", "j", "k"]:
        if col_name in table.column_names:
            col = table.column(col_name)
            if col.type != pa.uint32():
                table = table.set_column(
                    table.schema.get_field_index(col_name),
                    col_name,
                    col.cast(pa.uint32())
                )
    return table
```

## Adding New Block Model Types

To add support for a new block model type (e.g., SubBlockedBlockModel):

### 1. Create Data Class

```python
# In typed/subblocked_block_model.py
@dataclass(frozen=True, kw_only=True)
class SubBlockedBlockModelData:
    name: str
    origin: Point3
    n_parent_blocks: Size3i
    n_subblocks_per_parent: Size3i
    parent_block_size: Size3d
    rotations: list[tuple[RotationAxis, float]] = field(default_factory=list)
    cell_data: pd.DataFrame | None = None
    description: str | None = None
    crs: str | None = None
    size_unit_id: str | None = None
    units: dict[str, str] = field(default_factory=dict)
```

### 2. Create Typed Class

```python
class SubBlockedBlockModel:
    def __init__(
        self,
        client: BlockModelAPIClient,
        metadata: BlockModel,
        version: Version,
        cell_data: pd.DataFrame,
    ) -> None:
        self._client = client
        self._metadata = metadata
        self._version = version
        self._cell_data = cell_data

    @classmethod
    async def create(cls, context: IContext, data: SubBlockedBlockModelData, ...) -> SubBlockedBlockModel:
        ...

    @classmethod
    async def get(cls, context: IContext, bm_id: UUID, ...) -> SubBlockedBlockModel:
        ...
```

### 3. Export from Module

```python
# In typed/__init__.py
from .subblocked_block_model import SubBlockedBlockModel, SubBlockedBlockModelData

__all__ = [
    ...,
    "SubBlockedBlockModel",
    "SubBlockedBlockModelData",
]
```

## BlockModelRef in evo-objects

The `BlockModelRef` class in `evo-objects` acts as a proxy to the Block Model Service:

```python
from evo.objects.typed import BlockModelRef
from evo.objects import ObjectReference

# When a block model is created, it has an associated Geoscience Object
# Use the geoscience_object_id to load the reference
object_ref = ObjectReference.new(
    environment=context.get_environment(),
    object_id=block_model.metadata.geoscience_object_id,
)
bm_ref = await BlockModelRef.from_reference(context, object_ref)

# Access data (returns DataFrame with column names, not UUIDs)
df = await bm_ref.get_data(columns=["*"])

# Add attributes (uses dataframe_to_pyarrow for proper uint32 casting)
await bm_ref.add_attribute(df, "new_col", unit=Units.GRAMS_PER_TONNE)
```

## Testing New Typed Classes

### Run Tests

```bash
cd packages/evo-blockmodels
$env:PYTHONPATH="src;..\evo-sdk-common\src"
python -m pytest tests/test_typed_regular_block_model.py -v
```

### Mock Client Methods

```python
from unittest import mock

with (
    mock.patch.object(BlockModelAPIClient, "get_block_model") as mock_get,
    mock.patch.object(BlockModelAPIClient, "query_block_model_as_table") as mock_query,
):
    mock_get.return_value = mock_metadata
    mock_query.return_value = pyarrow_table
    
    result = await RegularBlockModel.get(context, bm_id)
```

## Common Pitfalls

1. **Invalid unit IDs**: Always use `Units` class constants, not arbitrary strings
2. **Wrong column types**: i,j,k must be `uint32` - use `dataframe_to_pyarrow()`
3. **Forgetting exports**: Always update `__init__.py` files
4. **Grid definition type checking**: Use `isinstance()` checks for grid types
5. **Async methods**: All API operations must be `async def`
6. **Cache requirement**: Upload operations require cache to be configured
7. **Column headers**: Use `ColumnHeaderType.name` to get user-friendly column names instead of UUIDs


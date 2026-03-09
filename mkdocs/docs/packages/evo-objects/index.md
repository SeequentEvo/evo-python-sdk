# evo-objects

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-objects/src/evo/objects/)

The `evo-objects` package provides both a low-level API client and typed Python classes for working with geoscience objects in Evo.

## Typed Objects

The typed objects module provides intuitive Python classes for working with Evo geoscience objects. Instead of dealing with raw API responses, you work with `PointSet`, `Regular3DGrid`, `Variogram`, and other domain-specific types that provide:

- Simple property access (e.g., `pointset.num_points`, `grid.bounding_box`)
- `to_dataframe()` for getting data as pandas DataFrames
- Rich HTML display in Jupyter notebooks (via `%load_ext evo.widgets`)
- Clickable links to Evo Portal and Viewer

### Loading objects

Three convenience functions let you load any typed object by reference, path, or UUID:

```python
from evo.objects.typed import object_from_path, object_from_uuid, object_from_reference

# By file path in the workspace
obj = await object_from_path(manager, "my-folder/assay-data")

# By UUID
obj = await object_from_uuid(manager, "b208a6c9-6881-4b97-b02d-acb5d81299bb")

# By full object reference URL
obj = await object_from_reference(manager, reference_url)
```

The correct typed class (`PointSet`, `Regular3DGrid`, etc.) is selected automatically based on the object's schema.

### Class hierarchy

All typed geoscience objects inherit from [`BaseObject`](evo-objects/typed-objects/base/BaseObject), which provides common properties (name, description, tags) and operations (update, refresh, validate). Objects with spatial data extend [`BaseSpatialObject`](evo-objects/typed-objects/spatial/BaseSpatialObject), adding bounding box and coordinate reference system support.

**Objects** — each has a corresponding `Data` class used for creation:

| Object                                                                                     | Data class                                                                                         | Description                                    |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [`PointSet`](evo-objects/typed-objects/pointset)                                           | [`PointSetData`](evo-objects/typed-objects/pointset/PointSetData)                                  | Point cloud with XYZ locations and attributes  |
| [`Regular3DGrid`](evo-objects/typed-objects/regular-grid/Regular3DGrid)                    | [`Regular3DGridData`](evo-objects/typed-objects/regular-grid/Regular3DGridData)                    | Uniform-cell 3D grid with cell and vertex data |
| [`RegularMasked3DGrid`](evo-objects/typed-objects/regular-masked-grid/RegularMasked3DGrid) | [`RegularMasked3DGridData`](evo-objects/typed-objects/regular-masked-grid/RegularMasked3DGridData) | Regular grid with masked (inactive) cells      |
| [`Tensor3DGrid`](evo-objects/typed-objects/tensor-grid/Tensor3DGrid)                       | [`Tensor3DGridData`](evo-objects/typed-objects/tensor-grid/Tensor3DGridData)                       | Grid with variable cell sizes per axis         |
| [`Variogram`](evo-objects/typed-objects/variogram)                                         | [`VariogramData`](evo-objects/typed-objects/variogram/VariogramData)                               | Variogram model for geostatistical analysis    |
| [`BlockModel`](evo-objects/typed-objects/block-model-ref/BlockModel)                       | —                                                                                                  | Proxy to Block Model Service (see below)       |

**Supporting types** — see [common types](evo-objects/typed-objects/types/BoundingBox) (`BoundingBox`, `Rotation`, `Point3`, `Size3d`, etc.) and [attributes](evo-objects/typed-objects/attributes) (`Attributes`, `Attribute`).

### BlockModel (via evo-blockmodels)

The `BlockModel` type is a geoscience object that acts as a proxy to the Block Model Service. When `evo-blockmodels` is installed (`pip install evo-objects[blockmodels]`), the full range of block model operations is available directly on the `BlockModel` object — no need to use the low-level `BlockModelAPIClient`.

```python
from evo.objects.typed import BlockModel, RegularBlockModelData, Point3, Size3d, Size3i

data = RegularBlockModelData(
    name="My Block Model",
    origin=Point3(x=0, y=0, z=0),
    n_blocks=Size3i(nx=10, ny=10, nz=5),
    block_size=Size3d(dx=2.5, dy=5.0, dz=5.0),
    cell_data=my_dataframe,
)
bm = await BlockModel.create_regular(manager, data)
```

```python
# Load an existing block model
bm = await object_from_path(manager, "my-folder/block-model")

# Get data as a DataFrame
df = await bm.to_dataframe()

# Add a new attribute
await bm.add_attribute(data_df, "new_attribute", unit="g/t")

# Create and run a report
report = await bm.create_report(spec)
result = await report.run(manager)
df = result.to_dataframe()
```

After a compute task (e.g., kriging) adds attributes on the server, call `refresh()` to update the local object:

```python
bm = await bm.refresh()
bm.attributes  # Now shows newly added attributes
```

For the full `evo-blockmodels` typed API (RegularBlockModel, Reports, Units), see the [evo-blockmodels documentation](./evo-blockmodels).

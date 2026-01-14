# Typed Objects Reference

This document provides API reference for typed object classes in `evo.objects.typed`. These are the recommended classes for working with Evo geoscience objects in notebooks.

## Common Imports

```python
from evo.objects.typed import (
    # Object classes
    PointSet, PointSetData,
    BlockModel, RegularBlockModelData,
    Regular3DGrid, Regular3DGridData,
    RegularMasked3DGrid, RegularMasked3DGridData,
    Tensor3DGrid, Tensor3DGridData,
    
    # Helper types
    Point3, Size3i, Size3d, Rotation, EpsgCode, BoundingBox,
)

from evo.blockmodels import Units  # For block model units
```

---

## PointSet

A collection of points in 3D space with associated attributes.

### Loading an Existing Pointset

```python
from evo.objects.typed import PointSet

# From ObjectSearchWidget selection
pointset = await PointSet.from_reference(manager, picker.selected_reference)

# From direct URL
pointset = await PointSet.from_reference(manager, "https://evo.../objects/...")
```

### Creating a New Pointset

```python
import pandas as pd
from evo.objects.typed import PointSet, PointSetData, EpsgCode

# DataFrame must have x, y, z columns, plus any attribute columns
df = pd.DataFrame({
    "x": [100, 200, 300],
    "y": [400, 500, 600],
    "z": [10, 20, 30],
    "grade": [1.5, 2.3, 0.8],  # attribute
})

data = PointSetData(
    name="My Pointset",
    coordinate_reference_system=EpsgCode(32632),
    locations=df,
)

pointset = await PointSet.create(manager, data)
```

### Accessing Data

```python
# Get all locations and attributes as DataFrame
df = await pointset.locations.as_dataframe()

# Access a specific attribute
attr_df = await pointset.locations.attributes["grade"].as_dataframe()

# List available attributes
for attr in pointset.locations.attributes:
    print(f"{attr.name}: {attr.attribute_type}")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Object name |
| `metadata` | `ObjectMetadata` | Full metadata including URL, created_at, etc. |
| `bounding_box` | `BoundingBox` | Spatial extent of the points |
| `coordinate_reference_system` | `EpsgCode` | CRS of the pointset |
| `locations` | `Locations` | Access to point coordinates and attributes |

---

## BlockModel

A block model stored in the Block Model Service with a Geoscience Object reference.

### Loading an Existing Block Model

```python
from evo.objects.typed import BlockModel

block_model = await BlockModel.from_reference(manager, picker.selected_reference)
```

### Creating a Regular Block Model

```python
import pandas as pd
from evo.objects.typed import BlockModel, RegularBlockModelData, Point3, Size3i, Size3d
from evo.blockmodels import Units

# Create block data with x, y, z coordinates or i, j, k indices
df = pd.DataFrame({
    "x": [...],  # centroid x coordinates
    "y": [...],  # centroid y coordinates
    "z": [...],  # centroid z coordinates
    "grade": [...],
    "density": [...],
})

data = RegularBlockModelData(
    name="My Block Model",
    origin=Point3(x=0, y=0, z=0),
    n_blocks=Size3i(nx=10, ny=10, nz=5),
    block_size=Size3d(dx=25.0, dy=25.0, dz=10.0),
    cell_data=df,
    crs="EPSG:32632",
    size_unit_id=Units.METRES,
    units={"grade": Units.GRAMS_PER_TONNE, "density": Units.TONNES_PER_CUBIC_METRE},
)

block_model = await BlockModel.create_regular(manager, data)
```

### Accessing Data

```python
# Get all block data
df = await block_model.get_data(columns=["*"])

# Get specific columns
df = await block_model.get_data(columns=["grade", "density"])
```

### Adding Attributes

```python
# Add a new attribute column
df_new = pd.DataFrame({
    "x": df["x"],
    "y": df["y"],
    "z": df["z"],
    "metal_content": df["grade"] * df["density"],
})

version = await block_model.add_attribute(
    df_new,
    attribute_name="metal_content",
    unit=Units.KG_PER_CUBIC_METRE,
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Object name |
| `block_model_uuid` | `UUID` | UUID in Block Model Service |
| `geometry` | `BlockModelGeometry` | Origin, n_blocks, block_size, rotation |
| `attributes` | `list[BlockModelAttribute]` | Available attributes with units |
| `bounding_box` | `BoundingBox` | Spatial extent |
| `metadata` | `ObjectMetadata` | Geoscience Object metadata |

---

## Regular3DGrid

A regular 3D grid with uniform cell sizes.

### Loading an Existing Grid

```python
from evo.objects.typed import Regular3DGrid

grid = await Regular3DGrid.from_reference(manager, picker.selected_reference)
```

### Creating a New Grid

```python
import numpy as np
import pandas as pd
from evo.objects.typed import Regular3DGrid, Regular3DGridData, Point3, Size3i, Size3d, Rotation

data = Regular3DGridData(
    name="My Grid",
    origin=Point3(0, 0, 0),
    size=Size3i(10, 10, 5),  # number of cells
    cell_size=Size3d(25.0, 25.0, 10.0),
    rotation=Rotation(0, 0, 0),
    cell_data=pd.DataFrame({
        "temperature": np.random.rand(10 * 10 * 5),
    }),
)

grid = await Regular3DGrid.create(manager, data)
```

### Accessing Data

```python
# Get cell data
cell_df = await grid.cells.as_dataframe()

# Get vertex data (if available)
vertex_df = await grid.vertices.as_dataframe()

# Access specific attribute
attr_df = await grid.cells.attributes["temperature"].as_dataframe()
```

### Updating Data

```python
# Update cell data
await grid.cells.set_dataframe(pd.DataFrame({
    "temperature": new_values,
}))

# Update metadata
grid.name = "Updated Grid Name"
await grid.update()
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Object name |
| `origin` | `Point3` | Grid origin point |
| `size` | `Size3i` | Number of cells (nx, ny, nz) |
| `cell_size` | `Size3d` | Cell dimensions (dx, dy, dz) |
| `rotation` | `Rotation` | Grid rotation |
| `bounding_box` | `BoundingBox` | Spatial extent |
| `cells` | `Dataset` | Cell data and attributes |
| `vertices` | `Dataset` | Vertex data and attributes |

---

## RegularMasked3DGrid

A regular 3D grid with a boolean mask indicating active cells.

### Creating with Mask

```python
import numpy as np
from evo.objects.typed import RegularMasked3DGrid, RegularMasked3DGridData, Point3, Size3i, Size3d

nx, ny, nz = 20, 20, 10
total_cells = nx * ny * nz

# Create mask (True = active cell)
mask = np.ones(total_cells, dtype=bool)
mask[:100] = False  # Mask out first 100 cells

data = RegularMasked3DGridData(
    name="Masked Grid",
    origin=Point3(0, 0, 0),
    size=Size3i(nx, ny, nz),
    cell_size=Size3d(50.0, 50.0, 10.0),
    mask=mask,
    cell_data=None,  # Add later
)

grid = await RegularMasked3DGrid.create(manager, data)
```

---

## Tensor3DGrid

A 3D grid with variable cell sizes along each axis.

```python
from evo.objects.typed import Tensor3DGrid, Tensor3DGridData

# Cell sizes can vary along each axis
data = Tensor3DGridData(
    name="Tensor Grid",
    origin=Point3(0, 0, 0),
    u_cell_sizes=[10, 20, 30, 40],  # 4 cells along x with varying sizes
    v_cell_sizes=[15, 15, 15],      # 3 cells along y
    w_cell_sizes=[5, 10, 15, 20],   # 4 cells along z
)

grid = await Tensor3DGrid.create(manager, data)
```

---

## Helper Types

### Point3

A 3D point with x, y, z coordinates.

```python
from evo.objects.typed import Point3

origin = Point3(x=100.0, y=200.0, z=50.0)

# Access components
print(origin.x, origin.y, origin.z)
```

### Size3i

Integer size in 3D (for number of cells/blocks).

```python
from evo.objects.typed import Size3i

n_blocks = Size3i(nx=10, ny=20, nz=5)
```

### Size3d

Float size in 3D (for dimensions).

```python
from evo.objects.typed import Size3d

block_size = Size3d(dx=25.0, dy=25.0, dz=10.0)
```

### Rotation

Rotation angles (dip_azimuth, dip, pitch).

```python
from evo.objects.typed import Rotation

rotation = Rotation(dip_azimuth=90, dip=0, pitch=0)
```

### EpsgCode

Coordinate reference system.

```python
from evo.objects.typed import EpsgCode

crs = EpsgCode(32632)  # WGS 84 / UTM zone 32N
```

### BoundingBox

Spatial bounds of an object.

```python
# Access from any typed object
bb = pointset.bounding_box
print(f"X range: {bb.min_x} to {bb.max_x}")
print(f"Y range: {bb.min_y} to {bb.max_y}")
print(f"Z range: {bb.min_z} to {bb.max_z}")
```

---

## Units Class

Common unit IDs for block model attributes.

```python
from evo.blockmodels import Units

# Length
Units.METRES       # "m"
Units.FEET         # "ft"

# Mass
Units.GRAMS        # "g"
Units.KILOGRAMS    # "kg"
Units.TONNES       # "t"

# Grade (mass per mass)
Units.GRAMS_PER_TONNE   # "g/t"
Units.PERCENT           # "%"
Units.PPM               # "ppm"

# Density (mass per volume)
Units.KG_PER_CUBIC_METRE      # "kg/m3"
Units.TONNES_PER_CUBIC_METRE  # "t/m3"
```


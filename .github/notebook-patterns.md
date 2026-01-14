# Common Notebook Patterns

Copy-paste code patterns for common Evo notebook workflows. These patterns use typed objects and widgets — the recommended approach for geologist-friendly notebooks.

---

## Notebook Format

Notebooks in this repository use the **Python percent format**:

```python
#%% md
# Title

Markdown content here.
#%% md
## Section
#%%
# Code cell
import pandas as pd
```

- `#%% md` — Markdown cell
- `#%%` — Code cell  
- Save with `.ipynb` extension

---

## Authentication

### Standard Authentication

```python
from evo.notebooks import ServiceManagerWidget

manager = await ServiceManagerWidget.with_auth_code(
    client_id="<your-client-id>",
    cache_location="./notebook-data",
).login()
```

### With Custom Discovery URL (Integration Environment)

```python
from evo.notebooks import ServiceManagerWidget

manager = await ServiceManagerWidget.with_auth_code(
    client_id="<your-client-id>",
    discovery_url="https://int-discover.test.api.seequent.com",
    cache_location="./notebook-data",
).login()
```

---

## Display Object Links

After loading or creating an object, display convenient links to view it in Evo:

```python
from evo.notebooks import display_object_links

# Works with any typed object (PointSet, BlockModel, Regular3DGrid, etc.)
display_object_links(pointset, label="My Pointset")

# Also works with ObjectReference, ObjectMetadata, or URL strings
display_object_links(picker.selected_reference, label="Selected Object")
display_object_links("https://...", label="Object URL")
```

This renders styled HTML with two clickable links:
- **Open in Evo Viewer** — 3D visualization of the object
- **Open in Evo Portal** — Object overview page with metadata

---

## Object Discovery

### Search for Object by Partial Name

```python
from evo.notebooks import ObjectSearchWidget

# Search all object types
picker = ObjectSearchWidget(manager, search="sample")

# Search specific type
picker = ObjectSearchWidget(manager, search="Ag_LMS", object_type="pointset")
```

### Available Object Types for Filtering

- `"pointset"` — Point clouds and sample data
- `"block-model"` — Block models
- `"regular-3d-grid"` — Regular 3D grids
- `"regular-masked-3d-grid"` — Masked regular 3D grids
- `"tensor-3d-grid"` — Variable cell size grids
- `"drilling-campaign"` — Drilling campaigns
- `"downhole-collection"` — Downhole data
- `"triangulated-surface-mesh"` — Surface meshes
- `"variogram"` — Variogram models

### Get Selected Object Reference

```python
# After user selects from picker widget
reference = picker.selected_reference  # ObjectReference
name = picker.selected_name            # str
metadata = picker.selected_metadata    # ObjectMetadata
```

---

## Loading Existing Objects

### Load from ObjectSearchWidget

```python
from evo.objects.typed import PointSet

pointset = await PointSet.from_reference(manager, picker.selected_reference)
```

### Load from Direct URL

```python
from evo.objects.typed import PointSet

url = "https://evo.integration.seequent.com/org-id/workspaces/.../objects/uuid"
pointset = await PointSet.from_reference(manager, url)
```

### Load Different Object Types

```python
from evo.objects.typed import (
    PointSet,
    BlockModel,
    Regular3DGrid,
    RegularMasked3DGrid,
    Tensor3DGrid,
)

# Each type uses the same pattern
pointset = await PointSet.from_reference(manager, reference)
block_model = await BlockModel.from_reference(manager, reference)
grid = await Regular3DGrid.from_reference(manager, reference)
```

---

## Creating Objects from Local Data

### Pointset from CSV

```python
import pandas as pd
from evo.objects.typed import PointSet, PointSetData, EpsgCode

# Load CSV (must have x, y, z columns)
df = pd.read_csv("my_points.csv")

# Create pointset
data = PointSetData(
    name="My Pointset",
    coordinate_reference_system=EpsgCode(32632),  # UTM zone 32N
    locations=df,
)

pointset = await PointSet.create(manager, data)
print(f"Created: {pointset.metadata.url}")
```

### Block Model from DataFrame

```python
import pandas as pd
from evo.objects.typed import BlockModel, RegularBlockModelData, Point3, Size3i, Size3d
from evo.blockmodels import Units

# Prepare data with x, y, z coordinates
df = pd.DataFrame({
    "x": [...],
    "y": [...],
    "z": [...],
    "grade": [...],
})

data = RegularBlockModelData(
    name="My Block Model",
    origin=Point3(x=0, y=0, z=0),
    n_blocks=Size3i(nx=10, ny=10, nz=5),
    block_size=Size3d(dx=25.0, dy=25.0, dz=10.0),
    cell_data=df,
    crs="EPSG:32632",
    size_unit_id=Units.METRES,
    units={"grade": Units.GRAMS_PER_TONNE},
)

block_model = await BlockModel.create_regular(manager, data)
```

### Regular 3D Grid from NumPy

```python
import numpy as np
import pandas as pd
from evo.objects.typed import Regular3DGrid, Regular3DGridData, Point3, Size3i, Size3d

nx, ny, nz = 10, 10, 5
values = np.random.rand(nx * ny * nz)

data = Regular3DGridData(
    name="My Grid",
    origin=Point3(0, 0, 0),
    size=Size3i(nx, ny, nz),
    cell_size=Size3d(25.0, 25.0, 10.0),
    cell_data=pd.DataFrame({"value": values}),
)

grid = await Regular3DGrid.create(manager, data)
```

---

## Accessing Object Data

### Pointset Data

```python
# Get all locations and attributes as DataFrame
df = await pointset.locations.as_dataframe()

# List attribute names
for attr in pointset.locations.attributes:
    print(f"{attr.name}: {attr.attribute_type}")

# Get single attribute
attr_df = await pointset.locations.attributes["grade"].as_dataframe()
```

### Block Model Data

```python
# Get all columns
df = await block_model.get_data(columns=["*"])

# Get specific columns
df = await block_model.get_data(columns=["grade", "density"])

# List attributes
for attr in block_model.attributes:
    print(f"{attr.name}: {attr.attribute_type} ({attr.unit})")
```

### Grid Data

```python
# Get cell data
cell_df = await grid.cells.as_dataframe()

# Get vertex data
vertex_df = await grid.vertices.as_dataframe()

# Get specific attribute
attr_df = await grid.cells.attributes["temperature"].as_dataframe()
```

---

## Modifying Objects

### Add Attribute to Block Model

```python
from evo.blockmodels import Units

# Prepare data with coordinates
df_new = pd.DataFrame({
    "x": df["x"],
    "y": df["y"],
    "z": df["z"],
    "new_attr": calculated_values,
})

version = await block_model.add_attribute(
    df_new,
    attribute_name="new_attr",
    unit=Units.PERCENT,
)
```

### Update Grid Cell Data

```python
await grid.cells.set_dataframe(pd.DataFrame({
    "value": new_values,
}))
await grid.update()
```

### Update Object Metadata

```python
grid.name = "Updated Name"
grid.description = "New description"
await grid.update()
```

---

## Working with Coordinates

### Parse Workspace URL

Extract workspace context from an Evo workspace URL:

```python
# Example URL: https://evo.integration.seequent.com/829e6621-0ab6-4d7d-96bb-2bb5b407a5fe/workspaces/350mt/783b6eef-01b9-42a7-aaf4-35e153e6fcbe/overview
# The widget selector in manager handles workspace selection automatically
```

### Set CRS on Objects

```python
from evo.objects.typed import EpsgCode

# Using EpsgCode
crs = EpsgCode(32632)  # WGS 84 / UTM zone 32N

# Or as string in block models
crs = "EPSG:32632"
```

---

## Error Handling (Optional)

For more robust notebooks, wrap operations in try/except:

```python
try:
    pointset = await PointSet.from_reference(manager, picker.selected_reference)
    df = await pointset.locations.as_dataframe()
    print(f"Loaded {len(df)} points")
except Exception as e:
    print(f"Error loading pointset: {e}")
```

---

## Display Progress

Use `FeedbackWidget` for long operations:

```python
from evo.notebooks import FeedbackWidget

fb = FeedbackWidget("Downloading data")
df = await pointset.locations.as_dataframe(fb=fb)
```


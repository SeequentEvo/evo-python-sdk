# Notebook Generation Guide for Evo Python SDK

## Overview

This guide provides instructions for AI agents generating Jupyter notebooks that interact with Seequent Evo. The target audience is **geologists with limited Python experience** who need simple, readable notebooks.

## Notebook Format

**IMPORTANT:** Notebooks in this repository use the **Python percent format**, not standard JSON `.ipynb` format.

- Use `#%% md` for markdown cells
- Use `#%%` for code cells
- Save files with `.ipynb` extension

Example structure:
```python
#%% md
# My Notebook Title

Some markdown description here.
#%% md
## Section Header
#%%
# Python code here
from evo.notebooks import ServiceManagerWidget

manager = await ServiceManagerWidget.with_auth_code(...).login()
#%% md
## Next Section
#%%
# More code
```

## Core Principles

1. **Use typed objects by default** — Always use `PointSet`, `BlockModel`, `Regular3DGrid`, etc. from `evo.objects.typed`
2. **Use widgets for discovery** — Use `ObjectSearchWidget` when users need to find objects by name
3. **Keep notebooks minimal** — Clear markdown headers, few cells, no unnecessary complexity
4. **Never expose low-level SDK** — Do not use `ObjectAPIClient` or `BlockModelAPIClient` unless explicitly requested by an advanced developer
5. **Show Evo links** — Use `display_object_links()` after loading/creating objects for easy navigation to Evo Portal and Viewer

## When to Use Each Pattern

| User Request | Pattern to Use |
|-------------|----------------|
| "Find an object called X" | `ObjectSearchWidget` + `TypedClass.from_reference()` |
| "Download object from URL" | `TypedClass.from_reference(manager, url)` |
| "Create object from CSV" | `pd.read_csv()` + `TypedClassData(...)` + `TypedClass.create()` |
| "I need low-level API access" | Only then use `ObjectAPIClient` (advanced users only) |

## Standard Notebook Structure

### 1. Authentication Cell (Required)

Always start with authentication using `ServiceManagerWidget`:

```python
from evo.notebooks import ServiceManagerWidget

# Evo app credentials
client_id = "<your-client-id>"  # Replace with your client ID

manager = await ServiceManagerWidget.with_auth_code(
    client_id=client_id,
    cache_location="./notebook-data",
).login()
```

### 2. Object Discovery Cell (When user knows object name but not UUID)

Use `ObjectSearchWidget` to let users search and select objects:

```python
from evo.notebooks import ObjectSearchWidget

# Search for objects by partial name
picker = ObjectSearchWidget(manager, search="Ag_LMS", object_type="pointset")
```

The widget provides:
- **Search input** with debounced search (300ms) — type partial name to filter
- **Type filter dropdown** — filter by object type (pointset, block-model, etc.)
- **Results dropdown** — select from matching objects
- **Metadata display** showing for the selected object:
  - Basic info: name, type, path, object ID, schema version
  - Timestamps: created/modified dates and users
  - Stage (if applicable)
  - CRS and bounding box (fetched from object data)
  - Version history with dates and creators
  - Available attributes with their types

Properties to access selection:
- `picker.selected_reference` — `ObjectReference` to pass to `from_reference()`
- `picker.selected_name` — display name of selected object
- `picker.selected_metadata` — full `ObjectMetadata` object

### 3. Load Object Cell

Load the selected object using typed classes:

```python
from evo.objects.typed import PointSet

# Load the selected pointset
pointset = await PointSet.from_reference(manager, picker.selected_reference)
```

### 4. Access Data Cell

Access data as pandas DataFrame:

```python
# Get pointset locations and attributes as DataFrame
df = await pointset.locations.as_dataframe()
df.head()
```

### 5. Display Links to Evo Portal/Viewer (Optional but Recommended)

After loading an object, display convenient links to view it in Evo:

```python
from evo.notebooks import display_object_links

# Show clickable links to open in Evo Viewer and Portal
display_object_links(pointset, label="My Pointset")
```

## Typed Object Classes

Use these imports from `evo.objects.typed`:

| Object Type | Class | Data Class |
|-------------|-------|------------|
| Pointset | `PointSet` | `PointSetData` |
| Block Model | `BlockModel` | `RegularBlockModelData` |
| Regular 3D Grid | `Regular3DGrid` | `Regular3DGridData` |
| Regular Masked 3D Grid | `RegularMasked3DGrid` | `RegularMasked3DGridData` |
| Tensor 3D Grid | `Tensor3DGrid` | `Tensor3DGridData` |

## Common Helper Types

```python
from evo.objects.typed import (
    Point3,      # 3D point (x, y, z)
    Size3i,      # Integer size (nx, ny, nz)
    Size3d,      # Float size (dx, dy, dz)
    Rotation,    # Rotation angles
    EpsgCode,    # Coordinate reference system
    BoundingBox, # Spatial bounds
)
```

## Units (for Block Models)

```python
from evo.blockmodels import Units

# Common unit IDs
Units.METRES              # "m"
Units.GRAMS_PER_TONNE     # "g/t"
Units.TONNES_PER_CUBIC_METRE  # "t/m3"
Units.PERCENT             # "%"
```

## Example Workflows

### Download a Pointset by Name

```python
from evo.notebooks import ServiceManagerWidget, ObjectSearchWidget, display_object_links
from evo.objects.typed import PointSet

# 1. Authenticate
manager = await ServiceManagerWidget.with_auth_code(
    client_id="your-client-id",
    cache_location="./notebook-data",
).login()

# 2. Search for object
picker = ObjectSearchWidget(manager, search="Ag_LMS1", object_type="pointset")

# 3. Load as typed object
pointset = await PointSet.from_reference(manager, picker.selected_reference)

# 4. Display links to Evo Portal and Viewer
display_object_links(pointset, label="Pointset")

# 5. Get data as DataFrame
df = await pointset.locations.as_dataframe()
df.head()
```

### Create a Pointset from CSV

```python
import pandas as pd
from evo.notebooks import ServiceManagerWidget, display_object_links
from evo.objects.typed import PointSet, PointSetData, EpsgCode

# 1. Authenticate
manager = await ServiceManagerWidget.with_auth_code(
    client_id="your-client-id",
    cache_location="./notebook-data",
).login()

# 2. Load data from CSV
df = pd.read_csv("my_points.csv")  # Must have x, y, z columns

# 3. Create pointset
pointset_data = PointSetData(
    name="My Pointset",
    coordinate_reference_system=EpsgCode(32632),
    locations=df,
)
pointset = await PointSet.create(manager, pointset_data)

# 4. Display links to view in Evo
display_object_links(pointset, label="Created Pointset")
print(f"Created: {pointset.metadata.url}")
```

### Load Object by Direct URL

When user provides a full URL:

```python
from evo.notebooks import ServiceManagerWidget
from evo.objects.typed import PointSet

manager = await ServiceManagerWidget.with_auth_code(
    client_id="your-client-id",
    cache_location="./notebook-data",
).login()

# Load directly from URL
url = "https://evo.integration.seequent.com/.../objects/..."
pointset = await PointSet.from_reference(manager, url)
df = await pointset.locations.as_dataframe()
```

## What NOT to Do

❌ **Do not expose ObjectAPIClient to geologists:**
```python
# BAD - too low-level for geologists
from evo.objects import ObjectAPIClient
client = ObjectAPIClient.from_context(manager)
objects = await client.list_all_objects()
```

❌ **Do not require users to find UUIDs manually:**
```python
# BAD - requires user to know the UUID
object_id = "9100d7dc-44e9-4e61-b427-159635dea22f"
```

❌ **Do not use raw dictionaries for object schemas:**
```python
# BAD - too complex for geologists
sample_pointset = {
    "name": "Sample pointset",
    "uuid": None,
    "bounding_box": {...},
    ...
}
```

## Advanced Users Only

If a user explicitly requests low-level API access (e.g., "I need to access the raw API" or "I'm a developer who needs more control"), then you may use:

- `ObjectAPIClient` from `evo.objects`
- `BlockModelAPIClient` from `evo.blockmodels`
- `DownloadedObject` and direct schema manipulation

But always clarify that these are advanced patterns not recommended for typical workflows.

## Related Documentation

- [Typed Objects Reference](typed-objects-reference.md) — Detailed API for each typed class
- [Common Patterns](notebook-patterns.md) — Copy-paste code snippets


# Copilot Instructions for evo-python-sdk

## Repository Overview

The `evo-python-sdk` is a collection of Python packages providing SDK access to Seequent Evo APIs. Each package is independently versioned and published.

## Package Structure

```
packages/
├── evo-sdk-common/      # Common utilities, interfaces, HTTP handling
├── evo-blockmodels/     # Block Model API client (typed access, Units)
├── evo-objects/         # Geoscience Object API client (typed access, BlockModelRef)
├── evo-files/           # File API client
├── evo-compute/         # Compute API client
├── evo-colormaps/       # Colormap utilities
└── evo-workspaces/      # Workspace management (part of common)
```

## Common Patterns

### Package Dependencies

All packages depend on `evo-sdk-common` which provides:
- `IContext`, `ICache`, `IFeedback` interfaces
- `APIConnector`, `BaseAPIClient` for HTTP operations
- `Environment` for workspace/org context
- Test utilities (`TestWithConnector`, `TestWithStorage`)

### Creating API Clients

```python
from evo.common import IContext

class MyAPIClient(BaseAPIClient):
    @classmethod
    def from_context(cls, context: IContext) -> MyAPIClient:
        return cls(
            environment=context.get_environment(),
            connector=context.get_connector(),
            cache=context.get_cache(),
        )
```

### Typed Access Pattern

For high-level typed access (see `evo-objects/typed/` and `evo-blockmodels/typed/`):
- Data classes with `@dataclass(frozen=True, kw_only=True)`
- Typed wrapper classes with `create()`, `get()` class methods
- Properties for metadata access
- `pd.DataFrame` for tabular data
- Use `Units` class for block model unit IDs

### Block Models

```python
from evo.blockmodels import RegularBlockModel, RegularBlockModelData, Units

# Use Units class for valid unit IDs
data = RegularBlockModelData(
    name="My Model",
    units={"grade": Units.GRAMS_PER_TONNE},
    ...
)
block_model = await RegularBlockModel.create(context, data)

# Access via BlockModelRef in evo-objects
from evo.objects.typed import BlockModelRef
bm_ref = await BlockModelRef.from_reference(context, object_ref)
df = await bm_ref.get_data(columns=["*"])
```

### Testing

```python
from evo.common.test_tools import TestWithConnector, TestWithStorage

class TestMyClient(TestWithConnector, TestWithStorage):
    def setUp(self):
        TestWithConnector.setUp(self)
        TestWithStorage.setUp(self)
```

## Development Commands

```bash
# Sync dependencies (from package directory)
uv sync --group test

# Run tests
python -m pytest tests/ -v

# With local source (if uv version mismatch)
$env:PYTHONPATH="src;..\evo-sdk-common\src"
python -m pytest tests/ -v

# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/
```

## Code Style

- Python 3.10+ required
- Line length: 120 characters
- Use type hints everywhere
- Async/await for all API operations
- Pydantic v2 for data models
- PyArrow for parquet handling
- **Prefer enums and helper classes over string literals** for parameters with known options (improves discoverability via autocomplete)

## Notebook Generation

When generating Jupyter notebooks for users, follow these rules:

### Target Audience
Notebooks are primarily for **geologists with limited Python experience**. Keep code minimal and readable.

### Notebook Format (CRITICAL)
Jupyter notebooks (`.ipynb` files) must use **standard JSON format** with proper cell structure.

**CRITICAL: The `source` field MUST be an array of strings (one per line), NOT a single string.**

**CORRECT format - use this:**
```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# My Notebook Title\n",
    "\n",
    "This is markdown content."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv(\"data.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2",
   "metadata": {},
   "source": [
    "## Next Section"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

**WRONG - source as single string (causes cell rendering issues):**
```json
{
  "cell_type": "markdown",
  "source": "# Title\n\nThis is wrong format"
}
```

**WRONG format - percent format is NOT valid for .ipynb files:**
```
#%% md
# My Notebook Title
#%%
import pandas as pd
```

The percent format (`#%% md`, `#%%`) is only a text representation used by some IDEs internally. Files saved in this format will appear as a single unexecutable cell when opened in Jupyter.

### Notebook Output Guidelines (IMPORTANT)
- **Always clear outputs** from notebook cells before committing - outputs can be very large and cause merge conflicts
- Code cells should have `"execution_count": null` and `"outputs": []`
- This keeps notebooks small and easy to review in version control

### Default Approach (IMPORTANT)
- **Always use typed objects** (`PointSet`, `BlockModel`, `Regular3DGrid`, etc.) from `evo.objects.typed`
- **Use `object_from_uuid()` or `object_from_path()`** to load existing objects - these are the preferred methods
- **Use pretty printing** (just output the object) to display objects in Jupyter - this shows Portal/Viewer links automatically
- **Use `visualise_objects()`** to view multiple objects together in the Evo Viewer
- **Use `ObjectSearchWidget`** for object discovery **only when** users know an object name but not its UUID
- **Never expose `ObjectAPIClient`** or `BlockModelAPIClient` to users unless explicitly requested by an advanced developer

### Viewing Multiple Objects Together
Use `visualise_objects()` to create a link that opens multiple objects in the same Evo Viewer scene:

```python
from evo.objects.typed import visualise_objects, object_from_uuid

# Load multiple objects
pointset = await object_from_uuid(manager, "uuid-1")
block_model = await object_from_uuid(manager, "uuid-2")

# Display a link to view both together
visualise_objects([pointset, block_model])

# With custom label
visualise_objects([pointset, block_model], label="View drilling data with block model")
```

### Loading Objects (IMPORTANT)
The preferred methods for loading objects are `object_from_uuid()` and `object_from_path()`:

```python
from evo.objects.typed import object_from_uuid, object_from_path

# Load by UUID (preferred when UUID is known)
pointset = await object_from_uuid(manager, "b208a6c9-6881-4b97-b02d-acb5d81299bb")

# Load by path (alternative)
pointset = await object_from_path(manager, "my-folder/pointset.json")

# Pretty-print the object (shows Portal/Viewer links, metadata, attributes)
pointset  # Just output the object - it has _repr_html_ for Jupyter

# View attributes (also has pretty printing)
pointset.attributes
```

**Do NOT use** `TypedClass.from_reference()` directly - use `object_from_uuid()` or `object_from_path()` instead.

### When to Use Each Pattern
| User Request | Pattern |
|-------------|---------|
| "Find object named X" (no URL given) | `ObjectSearchWidget` + `object_from_uuid()` |
| **User provides Evo Portal URL with `?id=`** | **Extract UUID from URL, use `object_from_uuid()`** |
| "Download object with UUID" | `object_from_uuid(manager, uuid)` |
| "Download object by path" | `object_from_path(manager, path)` |
| "Create from CSV" | `pd.read_csv()` + `TypedClassData()` + `TypedClass.create()` |
| "I need low-level API" | Only then use `ObjectAPIClient` (advanced users) |

**IMPORTANT:** When the user provides an Evo Portal URL containing `?id={object_id}`, extract the UUID and use `object_from_uuid()`.

### Standard Notebook Structure
1. Authentication with `ServiceManagerWidget`
2. Object discovery with `ObjectSearchWidget` (if needed)
3. Load object with `object_from_uuid()` or `object_from_path()`
4. Display object by outputting it (pretty printing shows Portal/Viewer links)
5. Access data with `.as_dataframe()`, `.to_dataframe()`, or `.get_data()`

### Evo Portal URL Format

When users provide an Evo Portal URL, extract the relevant IDs and determine the authentication environment.

**URL Pattern (New Format):**
```
https://{evo_host}/{org_id}/data/{workspace_id}/objects/{object_id}
```

**URL Pattern (Legacy Format - still supported for parsing):**
```
https://{evo_host}/{org_id}/workspaces/{hub_code}/{workspace_id}/{view}?id={object_id}
```

**Example (New Format):**
```
https://evo.integration.seequent.com/829e6621-0ab6-4d7d-96bb-2bb5b407a5fe/data/783b6eef-01b9-42a7-aaf4-35e153e6fcbe/objects/9100d7dc-44e9-4e61-b427-159635dea22f
```

**Example (Legacy Format):**
```
https://evo.integration.seequent.com/829e6621-0ab6-4d7d-96bb-2bb5b407a5fe/workspaces/350mt/783b6eef-01b9-42a7-aaf4-35e153e6fcbe/overview?id=9100d7dc-44e9-4e61-b427-159635dea22f
```

Extracted values:
- `org_id`: `829e6621-0ab6-4d7d-96bb-2bb5b407a5fe`
- `workspace_id`: `783b6eef-01b9-42a7-aaf4-35e153e6fcbe`
- `object_id`: `9100d7dc-44e9-4e61-b427-159635dea22f`

**Authentication Environment by Host:**

| Evo Host | Environment | `base_uri` | `discovery_url` |
|----------|-------------|-----------|-----------------|
| `evo.seequent.com` | Production | (default) | `https://discover.api.seequent.com` |
| `evo.integration.seequent.com` | Integration/QA | `https://qa-ims.bentley.com` | `https://int-discover.test.api.seequent.com` |

**Building ObjectReference from URL (legacy approach):**

Note: The preferred approach is to extract the UUID from the URL and use `object_from_uuid()`.

```python
from evo.objects.typed import object_from_uuid

# Extract UUID from the portal URL (works with both new and legacy formats)
# New URL: https://evo.integration.seequent.com/.../objects/9100d7dc-44e9-4e61-b427-159635dea22f
# Legacy URL: https://evo.integration.seequent.com/.../overview?id=9100d7dc-44e9-4e61-b427-159635dea22f
object_id = "9100d7dc-44e9-4e61-b427-159635dea22f"

# Load the object directly by UUID
pointset = await object_from_uuid(manager, object_id)

# Pretty-print shows Portal/Viewer links
pointset
```

**Complete Example for Integration Environment:**

```python
from evo.notebooks import ServiceManagerWidget

# Integration environment (evo.integration.seequent.com)
manager = await ServiceManagerWidget.with_auth_code(
    client_id="your-client-id",
    base_uri="https://qa-ims.bentley.com",
    discovery_url="https://int-discover.test.api.seequent.com",
    cache_location="./notebook-data",
).login()
```

**Complete Notebook Example (when user provides a URL):**

When a user provides a URL like:
`https://evo.integration.seequent.com/829e6621-0ab6-4d7d-96bb-2bb5b407a5fe/workspaces/350mt/783b6eef-01b9-42a7-aaf4-35e153e6fcbe/overview?id=9100d7dc-44e9-4e61-b427-159635dea22f`

Generate a notebook in standard JSON format (saved as `.ipynb`):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0",
   "metadata": {},
   "source": [
    "# Download PointSet Data from Seequent Evo\n",
    "\n",
    "This notebook downloads the pointset and displays it as a pandas DataFrame."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from evo.notebooks import ServiceManagerWidget\n",
    "\n",
    "client_id = \"<your-client-id>\"  # Replace with your client ID\n",
    "\n",
    "# Integration environment (detected from evo.integration.seequent.com)\n",
    "manager = await ServiceManagerWidget.with_auth_code(\n",
    "    client_id=client_id,\n",
    "    base_uri=\"https://qa-ims.bentley.com\",\n",
    "    discovery_url=\"https://int-discover.test.api.seequent.com\",\n",
    "    cache_location=\"./notebook-data\",\n",
    ").login()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2",
   "metadata": {},
   "source": [
    "## Load the PointSet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from evo.objects.typed import object_from_uuid\n",
    "\n",
    "# Load by UUID extracted from URL\n",
    "# Source: https://evo.integration.seequent.com/.../overview?id=9100d7dc-44e9-4e61-b427-159635dea22f\n",
    "pointset = await object_from_uuid(manager, \"9100d7dc-44e9-4e61-b427-159635dea22f\")\n",
    "\n",
    "# Pretty-print shows object info and Portal/Viewer links\n",
    "pointset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# View the attributes (also has pretty printing)\n",
    "pointset.attributes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5",
   "metadata": {},
   "source": [
    "## View Data as DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get all locations and attributes as a DataFrame\n",
    "df = await pointset.to_dataframe()\n",
    "df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

### Detailed Guides
- [Notebook Generation Guide](notebook-generation-guide.md) — Full instructions
- [Typed Objects Reference](typed-objects-reference.md) — API reference
- [Common Patterns](notebook-patterns.md) — Copy-paste snippets
- [Compute Task Notebook Guide](../packages/evo-compute/.github/compute-notebook-guide.md) — Kriging and compute task patterns

**Note:** These notebook generation rules apply only to Jupyter notebook context, not to SDK development or testing.

## Compute Tasks (Kriging)

When users ask to run compute tasks (especially kriging), follow the patterns in the [Compute Task Notebook Guide](../packages/evo-compute/.github/compute-notebook-guide.md).

**Key patterns for kriging notebooks:**

1. **Authenticate** with `ServiceManagerWidget`
2. **Load source objects** using `object_from_uuid()` and review with pretty printing
3. **Load or create a variogram** using `object_from_uuid()` or `Variogram.create()`
4. **Create a target BlockModel** (preferred over grids) or use existing one
5. **Define kriging parameters** - use `block_model.attributes["name"]` for targets (works for both new and existing attributes)
6. **Run kriging** - use `run()` with list of parameters for multiple scenarios, it shows progress feedback by default ("Running x/y...")
7. **Refresh and review** - reload block model to see new attributes, view with pretty printing
8. **Basic analysis** - query data with `to_dataframe()`, show statistics

**Targeting Block Model attributes:**
```python
# Use block_model.attributes[] for target - creates if doesn't exist, updates if it does
params = KrigingParameters(
    source=source_pointset.attributes["grade"],
    target=block_model.attributes["kriged_grade"],  # Preferred pattern
    variogram=variogram,
    search=SearchNeighbourhood(ellipsoid=search_ellipsoid, max_samples=20),
)

# After kriging, ALWAYS refresh to get the updated block model:
block_model = await block_model.refresh()
```

**Example kriging imports:**
```python
from evo.compute.tasks import run, Source, Target, SearchNeighborhood, Ellipsoid, EllipsoidRanges
from evo.compute.tasks.kriging import KrigingParameters  # Kriging-specific
from evo.objects.typed import (
    object_from_uuid, BlockModel, RegularBlockModelData, Point3, Size3i, Size3d,
    Variogram, VariogramData, SphericalStructure, Anisotropy,
    EllipsoidRanges as VariogramEllipsoidRanges,  # For variogram structure definition
)
from evo.objects.typed.ellipsoid import Rotation as EllipsoidRotation
from evo.blockmodels import Units
from evo.notebooks import ServiceManagerWidget
```

**Ellipsoid visualization pattern:**
```python
# Get ellipsoid from variogram structure with largest range (default)
var_ell = variogram.get_ellipsoid()

# Or explicitly select a structure by index
var_ell = variogram.get_ellipsoid(structure_index=0)

# Scale for search neighborhood (typically 2x variogram range)
search_ell = var_ell.scaled(2.0)

# Generate mesh points for 3D visualization with Plotly
import plotly.graph_objects as go
pts = await source_pointset.to_dataframe()
center = (pts["x"].mean(), pts["y"].mean(), pts["z"].mean())

x, y, z = var_ell.surface_points(center=center)
mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.3, color="blue", name="Variogram")

# Or wireframe for lighter visualization
x, y, z = var_ell.wireframe_points(center=center)
line = go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Variogram")
```

**Variogram curve visualization:**
```python
# Get variogram curves for 2D plotting (principal directions)
major, semi_maj, minor = variogram.get_principal_directions()

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=minor.distance, y=minor.semivariance, name="Minor"))
fig.add_trace(go.Scatter(x=semi_maj.distance, y=semi_maj.semivariance, name="Semi-major"))
fig.add_trace(go.Scatter(x=major.distance, y=major.semivariance, name="Major"))
fig.update_layout(xaxis_title="Distance", yaxis_title="Semivariance")
fig.show()

# Get variogram curve in arbitrary direction
distance, semivariance = variogram.get_direction(azimuth=45, dip=30)
fig.add_trace(go.Scatter(x=distance, y=semivariance, name="Az=45°, Dip=30°"))
```

**Running kriging (single task):**
```python
from evo.compute.tasks import run, Target, SearchNeighborhood
from evo.compute.tasks.kriging import KrigingParameters

params = KrigingParameters(
    source=pointset.attributes["grade"],
    target=block_model.attributes["kriged_grade"],
    variogram=variogram,
    search=SearchNeighborhood(
        ellipsoid=var_ell.scaled(2.0),
        max_samples=20,
    ),
)

# Run with default progress feedback
result = await run(manager, params)
result  # Pretty-print shows Portal link
```

**Running multiple kriging tasks:**
```python
# Create parameter sets for different scenarios
param_sets = [
    KrigingParameters(..., search=SearchNeighborhood(..., max_samples=10)),
    KrigingParameters(..., search=SearchNeighborhood(..., max_samples=20)),
    KrigingParameters(..., search=SearchNeighborhood(..., max_samples=30)),
]

# Run all concurrently - progress shows "Running x/y..."
results = await run(manager, param_sets)
results  # Pretty-print shows all results in a table
results[0]  # Access individual result
```

**Task Registry (for SDK developers):**

The `run()` function uses a task registry to dispatch to the appropriate runner based on parameter types. This enables running different task types together in the future:

```python
# Task runners register themselves when imported
from evo.compute.tasks.common.runner import register_task_runner

# Example: how kriging registers itself (done automatically on import)
register_task_runner(KrigingParameters, _run_kriging_for_registry)

# Future: mixed task types can run together
results = await run(manager, [
    KrigingParameters(...),
    SimulationParameters(...),  # future task type
])
```

## Block Model Reports

Reports provide resource estimation summaries for block models (tonnages, grades, metal content by category).

**Requirements for reports:**
1. **Units on columns** - Report columns must have units defined (use `set_attribute_units()`)
2. **At least one category column** - For grouping results (e.g., domain, rock type)
3. **Density information** - Either a density column OR fixed density value (not both)

**Density configuration rules:**
- **Using density column**: Set `density_column_name="density"` only. Do NOT set `density_unit_id` - the unit comes from the column.
- **Using fixed density**: Set both `density_value=2.7` AND `density_unit_id="t/m3"`. Do NOT set `density_column_name`.

**Example report workflow:**
```python
from evo.blockmodels import Units
from evo.blockmodels.typed import ReportSpecificationData, ReportColumnSpec, ReportCategorySpec

# Step 1: Add a category column (e.g., domain) to the block model
df = await block_model.to_dataframe()
# Create domains by slicing on z-coordinate (example - in practice use geological interpretation)
z_min, z_max = df["z"].min(), df["z"].max()
z_range = z_max - z_min
df["domain"] = df["z"].apply(lambda z: "LMS1" if z < z_min + z_range/3 else ("LMS2" if z < z_min + 2*z_range/3 else "LMS3"))
await block_model.add_attribute(df[["x", "y", "z", "domain"]], "domain")

# Step 2: Set units on report columns
block_model = await block_model.set_attribute_units({
    "Au": Units.GRAMS_PER_TONNE,
})

# Step 3: Create and run the report (using fixed density)
report = await block_model.create_report(ReportSpecificationData(
    name="Gold Resource Report",
    columns=[
        ReportColumnSpec(column_name="Au", aggregation=Aggregation.MASS_AVERAGE, label="Au Grade", output_unit_id=Units.GRAMS_PER_TONNE),
    ],
    categories=[
        ReportCategorySpec(column_name="domain", label="Domain", values=["LMS1", "LMS2", "LMS3"]),
    ],
    mass_unit_id="t",
    density_value=2.7,  # Fixed density - requires density_unit_id
    density_unit_id="t/m3",
    # OR use density_column_name="density" (without density_unit_id) if you have a density column
    run_now=True,
))

# Step 4: Pretty-print shows BlockSync link
report  # Displays report info with link to BlockSync

# Step 5: Get results as DataFrame
result = await report.refresh()
df = result.to_dataframe()
```

**BlockSync URL patterns:**

For **Block Models** (via `block_model.blocksync_url`):
- **Integration**: `https://blocksync.integration.seequent.com/{org_id}/redirect?ws={workspace_id}&bm={block_model_id}`
- **Production**: `https://blocksync.seequent.com/{org_id}/redirect?ws={workspace_id}&bm={block_model_id}`

For **Reports** (via `report.blocksync_url`):
- **Integration**: `https://blocksync.integration.seequent.com/{org_id}/{hub_code}/{workspace_id}/blockmodel/{bm_id}/reports/{report_id}?result_id={result_id}`
- **Production**: `https://blocksync.seequent.com/{org_id}/{hub_code}/{workspace_id}/blockmodel/{bm_id}/reports/{report_id}?result_id={result_id}`

**Key classes:**
- `ReportSpecificationData` - Data for creating a report
- `ReportColumnSpec` - Column definition (name, aggregation, unit)
- `Aggregation` - Enum for aggregation type: `MASS_AVERAGE` for grades, `SUM` for metal content
- `ReportCategorySpec` - Category definition (name, label, values)
- `MassUnits` - Helper with common mass unit IDs (`TONNES`, `KILOGRAMS`, `OUNCES`)
- `Report` - Report wrapper with `create_report()`, `run()`, `refresh()`, `blocksync_url`
- `ReportResult` - Result wrapper with `to_dataframe()`

## Package-Specific Guides

- **evo-compute**: See `packages/evo-compute/.github/` for compute task notebook patterns
- **evo-blockmodels**: See `packages/evo-blockmodels/.github/` for detailed guides
- **evo-objects**: Follow patterns in `src/evo/objects/typed/`

## SDK Development

### Creating New Typed Object Wrappers

When adding support for new Geoscience Object types, use the **annotation-based pattern** with `Annotated` and `SchemaLocation`.

**Quick summary of steps:**
1. Understand the JSON schema (sub-classification, required fields, datasets)
2. Create a `*Data` dataclass for object creation
3. Create `SchemaModel` subclasses for complex nested structures (e.g., `DataTableAndAttributes`)
4. Create the wrapper class inheriting from appropriate bases (`BaseSpatialObject`, `ConstructableObject`)
5. Define properties using `Annotated[Type, SchemaLocation("json_path")]` 
6. Override `_data_to_schema` if complex serialization is needed
7. Implement `_repr_html_` for Jupyter pretty printing
8. Export from `__init__.py` and write comprehensive tests

### Annotation-Based Pattern

Use `Annotated[..., SchemaLocation(...)]` for all typed object properties.

**Pattern for simple properties:**
```python
from typing import Annotated
from evo.objects.typed._model import SchemaLocation
from evo.objects.typed.base import BaseObject

class MyObject(BaseObject):
    # Simple properties - use Annotated with SchemaLocation
    name: Annotated[str, SchemaLocation("name")]
    description: Annotated[str | None, SchemaLocation("description")]
    tags: Annotated[dict[str, str], SchemaLocation("tags")] = {}
    
    # Numeric properties
    sill: Annotated[float, SchemaLocation("sill")]
    count: Annotated[int | None, SchemaLocation("count")]
    
    # Complex nested data stored as dict
    _geometry_raw: Annotated[dict, SchemaLocation("geometry")]
```

**Pattern for sub-models (datasets with attributes):**
```python
from typing import Annotated, ClassVar
from evo.objects.typed._model import SchemaLocation, SchemaModel
from evo.objects.typed._data import DataTable, DataTableAndAttributes
from evo.objects.typed.attributes import Attributes
from evo.objects.data import KnownTableFormat, FLOAT_ARRAY_3

# Define the data table format
class LocationTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = ["x", "y", "z"]

# Define the combined table + attributes structure
class Locations(DataTableAndAttributes):
    _table: Annotated[LocationTable, SchemaLocation("coordinates")]
    attributes: Annotated[Attributes, SchemaLocation("attributes")]

# Use in the typed object
class PointSet(BaseSpatialObject, ConstructableObject):
    locations: Annotated[Locations, SchemaLocation("locations")]
```

**Complete example - PointSet:**
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pandas as pd

from evo.objects import SchemaVersion
from evo.objects.typed._data import DataTable, DataTableAndAttributes
from evo.objects.typed._model import SchemaLocation
from evo.objects.typed.attributes import Attributes
from evo.objects.typed.spatial import BaseSpatialObject, BaseSpatialObjectData
from evo.objects.typed.base import ConstructableObject
from evo.objects.data import FLOAT_ARRAY_3, KnownTableFormat


# Data table for coordinates
class LocationTable(DataTable):
    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = ["x", "y", "z"]


# Combined locations + attributes
class Locations(DataTableAndAttributes):
    _table: Annotated[LocationTable, SchemaLocation("coordinates")]
    attributes: Annotated[Attributes, SchemaLocation("attributes")]


# Data class for creation
@dataclass(frozen=True, kw_only=True)
class PointSetData(BaseSpatialObjectData):
    locations: pd.DataFrame  # Must have x, y, z columns, optional attribute columns


# The typed wrapper class
class PointSet(BaseSpatialObject, ConstructableObject):
    _data_class = PointSetData
    sub_classification = "pointset"
    creation_schema_version = SchemaVersion(major=1, minor=0, patch=0)

    # Sub-model using Annotated pattern
    locations: Annotated[Locations, SchemaLocation("locations")]

    @property
    def num_points(self) -> int:
        return self.locations.length

    @property
    def attributes(self) -> Attributes:
        return self.locations.attributes

    async def to_dataframe(self, *keys: str, fb=NoFeedback) -> pd.DataFrame:
        return await self.locations.to_dataframe(*keys, fb=fb)
```

**Key classes:**
- `SchemaLocation(path)` - Marks a property as coming from JSON schema at the given path
- `DataLocation(path)` - Marks a property as requiring data download from the given path
- `SchemaModel` - Base class for nested structures that can be serialized/deserialized
- `DataTable` - Simple table data (coordinates, cell values, etc.)
- `DataTableAndAttributes` - Table data with associated attributes
- `Attributes` - Collection of named attributes with values
- `ModelContext` - Context passed through nested models, includes `schema_path` for tracking location

### Schema Path Propagation (IMPORTANT for Compute Tasks)

When typed objects have nested structures (e.g., `PointSet.locations.attributes`), the system automatically propagates the schema path through `ModelContext.schema_path`. This is critical for compute tasks that need to reference attributes by their full JMESPath.

**How it works:**
1. Root object creates `ModelContext` with `schema_path=""`
2. When rebuilding sub-models, each level appends its `SchemaLocation` path
3. For `PointSet.locations` → context gets `schema_path="locations"`
4. For `Locations.attributes` → context gets `schema_path="locations.attributes"`
5. `Attribute.expression` uses this path: `f"{self._context.schema_path}[?key=='{self.key}']"`

**Result:** Users can access `pointset.attributes["grade"]` and the system automatically generates the correct JMESPath `locations.attributes[?key=='...']` for compute APIs.

**Attribute expression pattern:**
```python
@property
def expression(self) -> str:
    """The JMESPath expression to access this attribute from the object."""
    base_path = self._context.schema_path or "attributes"
    return f"{base_path}[?key=='{self.key}']"
```

**For compute tasks (kriging, etc.):**
```python
# User-friendly access - attributes are exposed at the top level
source = pointset.attributes["grade"]  # Returns Attribute object

# Internally generates correct JMESPath for API
# source.expression → "locations.attributes[?key=='abc123']"

# Attribute provides to_source_dict() for serialization
params = KrigingParameters(
    source=pointset.attributes["grade"],  # Automatically converted to Source
    target=block_model.attributes["kriged"],  # Works for targets too
    ...
)
```

### Base Classes

- `BaseObject` - Base for all geoscience objects (has `name`, `description`, `tags`, `extensions`)
- `BaseSpatialObject` - Adds `bounding_box` and `coordinate_reference_system`
- `DynamicBoundingBoxSpatialObject` - Bounding box computed from object properties
- `ConstructableObject` - Mixin that adds `create()`, `replace()`, `create_or_replace()` methods

### Enums and Helper Classes (IMPORTANT)

**Prefer enums and helper classes over string literals** for any parameter with a known set of valid options. This improves discoverability via IDE autocomplete.

**Pattern - Use Enum classes:**
```python
from enum import Enum

class Aggregation(str, Enum):
    """Aggregation methods for report columns."""
    SUM = "SUM"
    """Sum of values - use for metal content, volume, etc."""
    MASS_AVERAGE = "MASS_AVERAGE"
    """Mass-weighted average - use for grades, densities, etc."""

# Usage: aggregation=Aggregation.MASS_AVERAGE
```

**Pattern - Use helper classes with class attributes:**
```python
class MassUnits:
    """Common mass unit IDs."""
    TONNES = "t"
    KILOGRAMS = "kg"
    OUNCES = "oz"

# Usage: mass_unit_id=MassUnits.TONNES
```

**When to apply:**
- API parameters with enumerated values (e.g., aggregation types, status codes)
- Unit IDs (e.g., `Units.GRAMS_PER_TONNE`, `MassUnits.TONNES`)
- Any string parameter where users would otherwise need to read docs to discover valid values

**Benefits:**
- IDE autocomplete shows all valid options
- Docstrings on enum values explain when to use each
- Type checking catches invalid values
- Users don't need to memorize string values

**Existing typed objects to reference:**
- `PointSet` - **Best example of annotation pattern** - locations dataset with attributes using `DataTableAndAttributes`
  - Exposes `attributes` at top level via property that delegates to `locations.attributes`
  - Schema path propagation ensures `attribute.expression` returns `locations.attributes[?key=='...']`
  - `to_source_dict()` and `to_target_dict()` methods for compute task integration
- `Regular3DGrid` - Grid with cells/vertices datasets, uses `DynamicBoundingBoxSpatialObject` for computed bounding box
- `RegularMasked3DGrid` - Grid with boolean mask, shows mask validation pattern
- `Tensor3DGrid` - Variable cell sizes, shows array property handling
- `Variogram` - Simple properties with `Annotated[..., SchemaLocation(...)]`, no datasets
  - `variogram.get_ellipsoid()` - Get Ellipsoid from structure with largest volume (default)
  - `variogram.get_ellipsoid(structure_index=0)` - Get Ellipsoid from specific structure
  - `variogram.get_principal_directions()` - Get curve data for 2D plotting (major, semi_major, minor)
  - `variogram.get_direction(azimuth, dip)` - Get curve in arbitrary direction
- `Ellipsoid` - 3D ellipsoid for search neighborhoods and visualization
  - `Ellipsoid(ranges=EllipsoidRanges(...), rotation=Rotation(...))` - Create directly
  - `ellipsoid.scaled(factor)` - Create scaled copy (e.g., for search neighborhood)
  - `ellipsoid.surface_points(center)` - Generate mesh points for Plotly Mesh3d
  - `ellipsoid.wireframe_points(center)` - Generate wireframe for Plotly Scatter3d
- `BlockModel` - Integration with Block Model Service, simple properties with `Annotated` pattern
- `Report` - Uses `Aggregation` enum, `Units` class, `MassUnits` class

### Attribute Integration with Compute Tasks

The `Attribute` and `PendingAttribute` classes provide seamless integration with compute tasks:

```python
# Attribute class key methods/properties:
class Attribute(SchemaModel):
    name: Annotated[str, SchemaLocation("name")]
    _key: Annotated[str | None, SchemaLocation("key")]
    
    @property
    def key(self) -> str:
        """Unique identifier within the attributes collection."""
        return self._key or self.name
    
    @property
    def expression(self) -> str:
        """JMESPath expression for compute APIs."""
        base_path = self._context.schema_path or "attributes"
        return f"{base_path}[?key=='{self.key}']"
    
    @property
    def exists(self) -> bool:
        """True for existing attributes."""
        return True
    
    def to_source_dict(self) -> dict[str, Any]:
        """Serialize as source for compute tasks."""
        return {
            "object": str(self._obj.metadata.url),
            "attribute": self.expression,
        }
    
    def to_target_dict(self) -> dict[str, str]:
        """Serialize as target for compute tasks."""
        return {"operation": "update", "reference": self.expression}


# PendingAttribute for attributes that don't exist yet:
class PendingAttribute:
    def to_target_dict(self) -> dict[str, str]:
        """Creates new attribute on compute task completion."""
        return {"operation": "create", "name": self._name}
```

**Usage in kriging:**
```python
# Source from existing attribute
source = pointset.attributes["grade"]  # Returns Attribute

# Target - creates if doesn't exist, updates if it does
target = block_model.attributes["kriged_grade"]  # Returns Attribute or PendingAttribute

params = KrigingParameters(
    source=source,  # Converted via to_source_dict()
    target=target,  # Converted via to_target_dict()
    variogram=variogram,
    search=SearchNeighborhood(...),
)
```


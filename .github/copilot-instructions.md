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

## Notebook Generation

When generating Jupyter notebooks for users, follow these rules:

### Target Audience
Notebooks are primarily for **geologists with limited Python experience**. Keep code minimal and readable.

### Notebook Format (CRITICAL)
Jupyter notebooks (`.ipynb` files) must use **standard JSON format** with proper cell structure.

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

**WRONG format (do not use) - percent format is NOT valid for .ipynb files:**
```
#%% md
# My Notebook Title
#%%
import pandas as pd
```

The percent format (`#%% md`, `#%%`) is only a text representation used by some IDEs internally. Files saved in this format will appear as a single unexecutable cell when opened in Jupyter.

### Default Approach (IMPORTANT)
- **Always use typed objects** (`PointSet`, `BlockModel`, `Regular3DGrid`, etc.) from `evo.objects.typed`
- **Use `object_from_uuid()` or `object_from_path()`** to load existing objects - these are the preferred methods
- **Use pretty printing** (just output the object) to display objects in Jupyter - this shows Portal/Viewer links automatically
- **Do NOT use `display_object_links()`** - typed objects have built-in pretty printing with links
- **Use `ObjectSearchWidget`** for object discovery **only when** users know an object name but not its UUID
- **Never expose `ObjectAPIClient`** or `BlockModelAPIClient` to users unless explicitly requested by an advanced developer

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

**URL Pattern:**
```
https://{evo_host}/{org_id}/workspaces/{hub_code}/{workspace_id}/{view}?id={object_id}
```

**Example:**
```
https://evo.integration.seequent.com/829e6621-0ab6-4d7d-96bb-2bb5b407a5fe/workspaces/350mt/783b6eef-01b9-42a7-aaf4-35e153e6fcbe/overview?id=9100d7dc-44e9-4e61-b427-159635dea22f
```

Extracted values:
- `org_id`: `829e6621-0ab6-4d7d-96bb-2bb5b407a5fe`
- `workspace_id`: `783b6eef-01b9-42a7-aaf4-35e153e6fcbe`
- `object_id`: `9100d7dc-44e9-4e61-b427-159635dea22f` (from `?id=` query parameter)

**Authentication Environment by Host:**

| Evo Host | Environment | `base_uri` | `discovery_url` |
|----------|-------------|-----------|-----------------|
| `evo.seequent.com` | Production | (default) | `https://discover.api.seequent.com` |
| `evo.integration.seequent.com` | Integration/QA | `https://qa-ims.bentley.com` | `https://int-discover.test.api.seequent.com` |

**Building ObjectReference from URL (legacy approach):**

Note: The preferred approach is to extract the UUID from the URL and use `object_from_uuid()`.

```python
from evo.objects.typed import object_from_uuid

# Extract UUID from the portal URL
# URL: https://evo.integration.seequent.com/.../overview?id=9100d7dc-44e9-4e61-b427-159635dea22f
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
5. **Define kriging parameters** - prefer multiple scenarios with `run_kriging_multiple()`
6. **Run kriging** with `FeedbackWidget` for progress
7. **Refresh and review** - reload block model to see new attributes, view with pretty printing
8. **Basic analysis** - query data with `get_data()`, show statistics

**Example kriging imports:**
```python
from evo.compute.tasks import (
    run_kriging, run_kriging_multiple,
    KrigingParameters, Source, Target,
    OrdinaryKriging, KrigingSearch, Ellipsoid, EllipsoidRanges, Rotation,
)
from evo.objects.typed import object_from_uuid, BlockModel, RegularBlockModelData, Point3, Size3i, Size3d
from evo.objects.typed import Variogram, VariogramData, SphericalStructure, Anisotropy
from evo.objects.typed import EllipsoidRanges as VariogramEllipsoidRanges  # If needed to disambiguate
from evo.blockmodels import Units
from evo.notebooks import ServiceManagerWidget, FeedbackWidget
```

## Package-Specific Guides

- **evo-compute**: See `packages/evo-compute/.github/` for compute task notebook patterns
- **evo-blockmodels**: See `packages/evo-blockmodels/.github/` for detailed guides
- **evo-objects**: Follow patterns in `src/evo/objects/typed/`

## SDK Development

### Creating New Typed Object Wrappers

When adding support for new Geoscience Object types, follow the [Typed Object Development Guide](typed-object-development-guide.md).

**Quick summary of steps:**
1. Understand the JSON schema (sub-classification, required fields, datasets)
2. Create a `*Data` dataclass for object creation
3. Create helper classes for complex nested structures (with `to_dict()` methods)
4. Create the wrapper class inheriting from appropriate bases
5. Define `SchemaProperty` for JSON fields and `DatasetProperty` for datasets
6. Override `_data_to_dict` if complex serialization is needed
7. Implement `_repr_html_` for Jupyter pretty printing
8. Export from `__init__.py` and write comprehensive tests

**Existing typed objects to reference:**
- `PointSet` - Simple object with locations dataset and attributes
- `Variogram` - Complex nested structures (structures, anisotropy, rotation)
- `Regular3DGrid` - Grid with cells and vertices datasets
- `RegularMasked3DGrid` - Grid with boolean mask
- `BlockModel` - Integration with Block Model Service


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
- **Use `ObjectSearchWidget`** for object discovery **only when** users know an object name but not its UUID
- **Always use `display_object_links()`** after loading/creating objects to show Evo Portal and Viewer links
- **Never expose `ObjectAPIClient`** or `BlockModelAPIClient` to users unless explicitly requested by an advanced developer

### When to Use Each Pattern
| User Request | Pattern |
|-------------|---------|
| "Find object named X" (no URL given) | `ObjectSearchWidget` + `TypedClass.from_reference()` |
| **User provides Evo Portal URL with `?id=`** | **Extract IDs from URL, build `ObjectReference` directly** |
| "Download from URL" | Build `ObjectReference` from environment + object_id |
| "Create from CSV" | `pd.read_csv()` + `TypedClassData()` + `TypedClass.create()` |
| "I need low-level API" | Only then use `ObjectAPIClient` (advanced users) |

**IMPORTANT:** When the user provides an Evo Portal URL containing `?id={object_id}`, do **NOT** use `ObjectSearchWidget`. The object ID is already known - build the `ObjectReference` directly from the URL components.

### Standard Notebook Structure
1. Authentication with `ServiceManagerWidget`
2. Object discovery with `ObjectSearchWidget` (if needed)
3. Load object with `TypedClass.from_reference()`
4. Display links with `display_object_links()`
5. Access data with `.as_dataframe()` or `.get_data()`

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

**Building ObjectReference from URL:**

The `ObjectReference` requires a specific HTTPS format. Build it from the environment:

```python
from evo.objects import ObjectReference

# Extract from the portal URL
org_id = "829e6621-0ab6-4d7d-96bb-2bb5b407a5fe"
workspace_id = "783b6eef-01b9-42a7-aaf4-35e153e6fcbe"
object_id = "9100d7dc-44e9-4e61-b427-159635dea22f"

# After authentication, build the reference
environment = manager.get_environment()
prefix = f"{environment.hub_url}/geoscience-object/orgs/{environment.org_id}/workspaces/{environment.workspace_id}/objects"
object_reference = ObjectReference(f"{prefix}/{object_id}")
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
    "from evo.objects import ObjectReference\n",
    "from evo.objects.typed import PointSet\n",
    "from evo.notebooks import display_object_links\n",
    "\n",
    "# Build object reference from URL components\n",
    "# Source: https://evo.integration.seequent.com/.../overview?id=9100d7dc-44e9-4e61-b427-159635dea22f\n",
    "object_id = \"9100d7dc-44e9-4e61-b427-159635dea22f\"\n",
    "\n",
    "environment = manager.get_environment()\n",
    "prefix = f\"{environment.hub_url}/geoscience-object/orgs/{environment.org_id}/workspaces/{environment.workspace_id}/objects\"\n",
    "object_reference = ObjectReference(f\"{prefix}/{object_id}\")\n",
    "\n",
    "# Download the pointset\n",
    "pointset = await PointSet.from_reference(manager, object_reference)\n",
    "\n",
    "display_object_links(pointset)\n",
    "print(f\"Downloaded: {pointset.name}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4",
   "metadata": {},
   "source": [
    "## View Data as DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get all locations and attributes as a DataFrame\n",
    "df = await pointset.locations.as_dataframe()\n",
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

### Detailed Guides
- [Notebook Generation Guide](notebook-generation-guide.md) — Full instructions
- [Typed Objects Reference](typed-objects-reference.md) — API reference
- [Common Patterns](notebook-patterns.md) — Copy-paste snippets

**Note:** These notebook generation rules apply only to Jupyter notebook context, not to SDK development or testing.

## Package-Specific Guides

- **evo-blockmodels**: See `packages/evo-blockmodels/.github/` for detailed guides
- **evo-objects**: Follow patterns in `src/evo/objects/typed/`


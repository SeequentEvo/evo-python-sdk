# evo-widgets

Widgets and presentation layer for the Evo Python SDK — interactive widgets, HTML rendering, URL generation, and IPython formatters for Jupyter notebooks.

## Overview

This package provides:
- **Interactive widgets** for authentication and service discovery (`ServiceManagerWidget`, `FeedbackWidget`)
- **Rich HTML representations** for Evo SDK typed objects in Jupyter notebooks
- **URL generation** utilities for Portal and Viewer links

It decouples presentation logic from the data model classes, keeping the core SDK lightweight for production use while providing a batteries-included experience for notebook users.

## Installation

The package is included automatically when you install `evo-sdk`:

```bash
pip install evo-sdk
```

Or install it directly:

```bash
pip install evo-widgets
```

For interactive widgets (authentication, progress feedback), install with the `notebooks` extra:

```bash
pip install evo-widgets[notebooks]
```

## Interactive Widgets

> **Security Note:** The `InteractiveAuthorizer` stores OAuth tokens in a local `.env` file,
> which is not a secure storage mechanism. It is intended for development and exploration in
> Jupyter notebooks only. Do not use it in production or shared environments, and ensure the
> `.env` file is not committed to source control.

### ServiceManagerWidget

The `ServiceManagerWidget` provides authentication and service discovery in one convenient widget:

```python
from evo.widgets import ServiceManagerWidget

# Authenticate using OAuth authorization code flow
manager = await ServiceManagerWidget.with_auth_code(
    client_id="your-client-id",
    redirect_url="http://localhost:3000/signin-callback",
).login()

# Use the manager to create API clients
from evo.objects import ObjectAPIClient
client = manager.create_client(ObjectAPIClient)
```

### FeedbackWidget

Display progress feedback in notebooks:

```python
from evo.widgets import FeedbackWidget

fb = FeedbackWidget("Loading data...")
for i in range(100):
    fb.progress(i / 100, f"{i}%")
fb.progress(1, "Done")
```

## Rich Display

### Automatic Formatter Registration

When you import from `evo.widgets`, HTML formatters are automatically registered for all Evo SDK typed objects. No manual setup required:

```python
from evo.widgets import ServiceManagerWidget
from evo.objects.typed import object_from_path

manager = await ServiceManagerWidget.with_auth_code(
    client_id="your-client-id"
).login()

# Objects automatically render with rich HTML formatting
grid = await object_from_path(manager, "/My Grid")
grid  # Displays formatted HTML with Portal/Viewer links
```

Typed objects render with:
- Formatted metadata tables
- Clickable links to Portal and Viewer
- Bounding box information
- Coordinate reference system
- Attribute summaries

> **Note:** To disable auto-registration, set `EVO_WIDGETS_NO_AUTO_REGISTER=1` before importing.
> Manual registration is still available via `%load_ext evo.widgets`.

## Features

### HTML Formatters

Rich HTML representations for all typed geoscience objects:

- `PointSet`
- `Regular3DGrid`
- `TensorGrid`
- `Attributes`
- And all other typed objects inheriting from `_BaseObject`

### URL Generation

Generate Portal and Viewer URLs for objects:

```python
from evo.widgets import (
    get_portal_url_for_object,
    get_viewer_url_for_object,
    get_viewer_url_for_objects,
)

# Get Portal URL for a single object
portal_url = get_portal_url_for_object(grid)

# Get Viewer URL for a single object
viewer_url = get_viewer_url_for_object(grid)

# View multiple objects together in the Viewer
url = get_viewer_url_for_objects(manager, [grid, pointset, tensor_grid])
```

### Low-Level URL Utilities

For advanced use cases, low-level URL generation functions are also available:

```python
from evo.widgets import (
    get_evo_base_url,
    get_hub_code,
    get_portal_url,
    get_viewer_url,
    get_portal_url_from_reference,
    get_viewer_url_from_reference,
    serialize_object_reference,
)

# Generate URLs from components
portal_url = get_portal_url(
    org_id="org-123",
    workspace_id="ws-456",
    object_id="obj-789",
    hub_url="https://350mt.api.seequent.com",
)

# Generate URLs from object reference strings
ref_url = "https://350mt.api.seequent.com/geoscience-object/orgs/org-123/workspaces/ws-456/objects/obj-789"
portal_url = get_portal_url_from_reference(ref_url)
viewer_url = get_viewer_url_from_reference(ref_url)

```

## API Reference

### Interactive Widgets

| Class | Description |
|-------|-------------|
| `ServiceManagerWidget` | Authentication and service discovery widget |
| `FeedbackWidget` | Progress bar and status message widget |
| `DropdownSelectorWidget` | Generic dropdown widget base class |

### IPython Extension

| Function | Description |
|----------|-------------|
| `load_ipython_extension(ipython)` | Register HTML formatters for Evo SDK types |
| `unload_ipython_extension(ipython)` | Unregister HTML formatters |

### URL Functions (Object-Based)

| Function | Description |
|----------|-------------|
| `get_portal_url_for_object(obj)` | Generate Portal URL from a typed object |
| `get_viewer_url_for_object(obj)` | Generate Viewer URL from a typed object |
| `get_viewer_url_for_objects(context, objects)` | Generate Viewer URL for multiple objects |

### URL Functions (Low-Level)

| Function | Description |
|----------|-------------|
| `get_evo_base_url(hub_url)` | Get the Evo base URL from a hub URL |
| `get_hub_code(hub_url)` | Extract the hub code from a hub URL |
| `get_portal_url(org_id, workspace_id, object_id, hub_url)` | Generate Portal URL from components |
| `get_viewer_url(org_id, workspace_id, object_ids, hub_url)` | Generate Viewer URL from components |
| `get_portal_url_from_reference(object_reference)` | Generate Portal URL from reference string |
| `get_viewer_url_from_reference(object_reference)` | Generate Viewer URL from reference string |
| `serialize_object_reference(value)` | Serialize various object types to URL string |

### Formatters

| Function | Description |
|----------|-------------|
| `format_base_object(obj)` | Format a typed geoscience object as HTML |
| `format_attributes_collection(obj)` | Format an attributes collection as HTML |

## How It Works

When you import from `evo.widgets` in an IPython/Jupyter environment, HTML formatters are automatically registered using `for_type_by_name`. This approach:

1. **Avoids hard dependencies** — The widgets package doesn't import model classes directly
2. **Works with all typed objects** — Formatters are registered for the base class, so all subclasses are covered
3. **Lazy loading** — Formatters only activate when the relevant types are actually used
4. **Automatic widget loading** — html repr widgets for typed objects are loaded automatically as part of the `evo.widgets` initialization.

You can disable auto-registration by setting `EVO_WIDGETS_NO_AUTO_REGISTER=1` before importing.

## CSS Customization

The HTML output uses CSS classes prefixed with `.evo` and respects Jupyter theme variables (e.g., `--jp-layout-color1`, `--jp-ui-font-color1`) for proper light/dark mode support.

## License

Apache License 2.0 — see [LICENSE.md](LICENSE.md) for details.


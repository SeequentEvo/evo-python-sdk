# evo-widgets

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-widgets/src/evo/widgets/__init__.py)

Widgets and presentation layer for the Evo Python SDK — HTML rendering, URL generation, and IPython formatters for Jupyter notebooks.

## Usage

Load the IPython extension in your notebook to enable rich HTML rendering for all Evo SDK typed objects:

```python
%load_ext evo.widgets
```

After loading, typed objects like `PointSet`, `Regular3DGrid`, `TensorGrid`, and `BlockModel` will automatically render with formatted metadata tables, clickable Portal/Viewer links, and bounding box information.

## URL Functions

Generate URLs to view objects in the Evo Portal and Viewer:

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

## Formatters

Rich HTML representations for all typed geoscience objects:

- `PointSet`, `Regular3DGrid`, `TensorGrid`, `BlockModel`
- `Variogram`
- `Attributes` collections
- `Report` and `ReportResult`
- `TaskResult` and `TaskResults` (compute results)

All formatters are registered automatically when you load the extension with `%load_ext evo.widgets`. They support light/dark mode via Jupyter theme CSS variables.

## How It Works

When you run `%load_ext evo.widgets`, the extension registers HTML formatters with IPython using `for_type_by_name`. This approach:

1. **Avoids hard dependencies** — The widgets package doesn't import model classes directly
2. **Works with all typed objects** — Formatters are registered for the base class, so all subclasses are covered
3. **Lazy loading** — Formatters only activate when the relevant types are actually used


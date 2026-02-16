# Simplified Object Interactions

This example demonstrates the **recommended approach** for most users and geologists to interact with Evo geoscience objects using the Evo Python SDK.

## Overview

The `simplified-object-interactions.ipynb` notebook shows how to use the **typed objects API** (`PointSet`, `Regular3DGrid`, etc.) along with the `evo.widgets` extension for rich HTML display in Jupyter notebooks. This approach provides:

- **Simple, intuitive API** - Work directly with Python dataclasses and pandas DataFrames
- **Automatic bounding box calculation** - No need to manually compute spatial extents
- **Rich HTML display** - Objects render with formatted tables and clickable links to Evo Portal/Viewer
- **Type safety** - IDE autocompletion and validation for object properties

## When to use this approach

Use the typed objects API when you want to:

- Quickly upload point data, grids, or other geoscience objects to Evo
- Download and inspect existing objects with minimal boilerplate
- Work interactively in Jupyter notebooks with visual feedback
- Focus on your data rather than low-level API details

## When to use the schema-based approach

For advanced use cases, you may need the lower-level `evo-schemas` approach (as shown in `publish-pointset/`):

- Working with object types not yet supported by typed objects
- Fine-grained control over schema versions and data references
- Batch processing pipelines where performance is critical

## Requirements

- Python 3.10+
- A Seequent account with Evo entitlement
- An Evo application with client ID and redirect URL (see [Apps and tokens guide](https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens))

## Quick Start

1. Open `simplified-object-interactions.ipynb` in Jupyter
2. Update the `client_id` and `redirect_url` with your Evo app credentials
3. Run the cells to authenticate, create a pointset from CSV data, and view it in Evo

## Sample Data

The `sample-data/WP_assay.csv` file contains drill hole assay data with:
- X, Y, Z coordinates
- Hole ID
- Assay values (CU_pct, AU_gpt, DENSITY)


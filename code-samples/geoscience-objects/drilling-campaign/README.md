# Drilling Campaign Examples

This directory contains two Jupyter notebooks demonstrating how to work with **Evo Drilling Campaign** geoscience objects using the high‑level Python SDK. These examples show how to create drilling campaign objects from CSV data and how to download existing drilling campaign objects back to CSV format.

## Notebooks

| Notebook | Purpose | Location |
|----------|---------|----------|
| `create-a-drilling-campaign/sdk-examples.ipynb` | Create drilling campaign objects from CSV data | Create and publish collar data, planned data, and interim data |
| `download-a-drilling-campaign/sdk-examples.ipynb` | Download drilling campaign objects to CSV format | List, retrieve, and export drilling campaign data from Evo |

## What You Can Do

### Creating a Drilling Campaign
1. Authenticate with Evo.
2. Load drilling campaign data from CSV files (collar, survey, attributes).
3. Create hole ID mappings and define object attributes.
4. Build drilling campaign components (collar, planned, interim data).
5. Assemble and publish the complete drilling campaign object to Evo.

### Downloading a Drilling Campaign
1. Authenticate with Evo.
2. List all drilling campaign objects in a workspace.
3. Select and retrieve a specific drilling campaign object.
4. Download associated data (collar, planned, interim).
5. Export data to CSV format for further analysis.

## Prerequisites

- A Seequent account with Evo entitlements.
- An Evo application (client ID + redirect URL).
- The Python package manager `uv` installed.

Take a look at `code-samples/README.md` for more detailed information.

## Quick Start

From the `code-samples` directory:

```bash
# 1. Create a Python environment and install dependencies with uv
uv sync

# 2. Launch Jupyter (choose your notebook UI)
uv run jupyter lab  # or: uv run jupyter notebook

# 3. Open either notebook
open geoscience-objects/drilling-campaign/create-a-drilling-campaign/sdk-examples.ipynb
open geoscience-objects/drilling-campaign/download-a-drilling-campaign/sdk-examples.ipynb
```

If using VS Code, you can simply open the notebooks directly; the Python / ipykernel environment should point at the synced virtual environment.

## Sample Data

The **create** notebook includes sample CSV data in its `sample-data/` directory with example collar, survey, and attribute files. The **download** notebook works with existing drilling campaign objects in your Evo workspace and does not require sample data.

## Additional Resources

- [Seequent Developer Portal](https://developer.seequent.com)
- [Apps and tokens guide](https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens)
- [Drilling Campaign schema documentation](https://developer.seequent.com/docs/data-structures/geoscience-objects/schemas/drilling-campaign)

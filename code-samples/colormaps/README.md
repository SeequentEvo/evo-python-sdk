# Colormap Examples

This directory contains two complementary Jupyter notebooks demonstrating how to work with **Evo Colormaps** either via the high‑level Python SDK or via direct (low‑level) API calls. It now also includes a concise Quick Start and expanded troubleshooting guidance.

## Notebooks

| Notebook | Approach | When to Use |
|----------|----------|-------------|
| `sdk-examples.ipynb` | High‑level SDK (`ColormapAPIClient`) | Fast prototyping, standard workflows, built‑in models & validation |
| `api-examples.ipynb` | Direct API calls using `APIConnector.call_api()` | Learning raw endpoints, debugging, custom request handling |

## What You Can Do
1. Authenticate with Evo.
2. List objects in a workspace.
3. Create a new colormap and associate it with an object attribute.
4. Retrieve colormap associations for a selected object.
5. Fetch colormap metadata (attribute controls, gradient controls, RGB colours).
6. (API version only) Inspect raw JSON payloads for transparency and troubleshooting.

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

# 3. Open either notebook inside code-samples/colormaps
open code-samples/colormaps/sdk-examples.ipynb
open code-samples/colormaps/api-examples.ipynb
```
If using VS Code, you can simply open the notebooks directly; the Python / ipykernel environment should point at the synced virtual environment.

## Choosing an Approach
- **Start with the SDK** for concise, maintainable code and guardrails.
- **Switch to API** when you need to:
  - Inspect raw responses / headers.
  - Prototype new or beta endpoints not yet wrapped by the SDK.
  - Implement custom pagination, retries, or diagnostics.

## Additional Resources

- [Seequent Developer Portal](https://developer.seequent.com)
- [Apps and tokens guide](https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens)
- [Colormaps documentation](https://developer.seequent.com/docs/guides/colormap)

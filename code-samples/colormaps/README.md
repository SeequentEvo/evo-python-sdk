# Colour Map Examples

This directory contains two complementary Jupyter notebooks demonstrating how to work with **Evo Colour Maps** either via the high‑level Python SDK or via direct (low‑level) API calls. It now also includes a concise Quick Start and expanded troubleshooting guidance.

## Notebooks

| Notebook | Approach | When to Use |
|----------|----------|-------------|
| `sdk-examples.ipynb` | High‑level SDK (`ColormapAPIClient`) | Fast prototyping, standard workflows, built‑in models & validation |
| `api-examples.ipynb` | Direct API calls using `APIConnector.call_api()` | Learning raw endpoints, debugging, custom request handling |

## What You Can Do
1. Authenticate with Evo (Authorization Code flow).
2. List objects in a workspace.
3. Retrieve colour map associations for a selected object.
4. Fetch colour map metadata (attribute controls, gradient controls, RGB colours).
5. (API version only) Inspect raw JSON payloads for transparency and troubleshooting.

## Prerequisites
- A Seequent account with Evo entitlements.
- An Evo application (client ID + redirect URL).
- Python 3.10–3.12 and the project dependencies installed (from repository root).

## Quick Start
From the repository root (one level above `samples/`):
```bash
# 1. Install dependencies (uv preferred, falls back to pip)
uv sync || pip install -e .

# 2. Export required environment variables (example)
export EVO_CLIENT_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
export EVO_REDIRECT_URL="http://localhost:8765/callback"
export EVO_WORKSPACE_ID="workspace-guid"

# (Optional) If you already have a token
# export EVO_ACCESS_TOKEN="eyJhbGciOi..."

# 3. Launch Jupyter (choose your notebook UI)
uv run jupyter lab  # or: uv run jupyter notebook

# 4. Open either notebook inside samples/colormaps
open samples/colormaps/sdk-examples.ipynb
open samples/colormaps/api-examples.ipynb
```
If using VS Code, you can simply open the notebooks directly; the Python / ipykernel environment should point at the synced virtual environment.

## Choosing an Approach
- **Start with the SDK** for concise, maintainable code and guardrails.
- **Switch to API** when you need to:
  - Inspect raw responses / headers.
  - Prototype new or beta endpoints not yet wrapped by the SDK.
  - Implement custom pagination, retries, or diagnostics.

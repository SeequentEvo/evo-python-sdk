# Block Model Samples

The Block Model SDK tutorials are the recommended path for creating, querying, and managing Evo block models. They use the high-level `evo.blockmodels` interfaces and PyArrow tables.

## Start Here

1. [01: Create and Query a Regular Block Model](sdk/01-create-and-query-regular-block-model.ipynb)
   - Create a regular block model, add columns and units, and query all data or a bounding box.
2. [02: Block Model Reports](sdk/02-block-model-reports.ipynb)
   - Create a report-ready block model and calculate grouped resource-estimation summaries.

## Future SDK Tutorials

The SDK tutorial sequence will expand here as new workflows are added:

- Schema and metadata management
- Versions and change tracking
- Query export and cache-backed workflows
- [Advanced grid workflows](sdk/advanced/README.md), including subblocked and octree block models

## Direct API Reference

[Advanced: Block Model API Reference Workflows](api/block-model-api-reference-workflows.ipynb) contains direct service calls, job polling, and request/response payloads. Use it for debugging, direct API integrations, and functionality not yet supported by the high-level SDK.

## Shared Data

- `data/sample-data/` contains input files shared by the SDK and API tutorials.
- `data/notebook-data/` stores notebook-specific state.
- `api/downloads/` stores generated query results from the direct API reference.

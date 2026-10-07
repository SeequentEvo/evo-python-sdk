# Geostatistics with Evo Compute

This suite of notebooks use the [Evo Python SDK](https://github.com/SeequentEvo/evo-python-sdk) to generate geoscience objects on the Seequent Evo platform and run cloud geostatistics tasks to demonstrate common workflows. The notebook ["1-populate-workspace"](1-populate-workspace.ipynb) must be run prior to any of the others to ensure necessary input data and target blockmodel is created in your Seequent Evo workspace.

## Notebooks

1. [Populate workspace data](1-populate-workspace.ipynb) loads the composite data from a CSV, inspects summary statistics, and creates a PointSet geoscience object (`WP comps`), a variogram model for Copper (`Cu_pct Variogram Model`), and a regular block model (`WP Regular BlockModel`). 
2. [Grid-based declustering](2-declustering.ipynb) retrieves the pointSet and block model objects, runs the Evo declustering task to perform inverse distance weighted declustering and compares the weighted composite values to the naive distribution.
3. [Estimation](3-estimation.ipynb) retrieves the composites, variogram model, and block model; visualizes the variogram's principal directions; and runs kriging, inverse distance weighting (IDW), and k-nearest neighbors (KNN). Results (`Cu_OK`, `Cu_IDW`, and `Cu_KNN`) are written directly to the blocksync block model.
4. [Conditional simulation](4-simulation.ipynb) is focused on performing the conditional simulation workflow (also leveraged by Leapfrog's simulation implementation). 

## Before You Start

- You need a Seequent account with access to an Evo organization and workspace, plus an application client ID configured for the authentication flow. Replace `<CLIENT-ID>` in each notebook and sign in using `ServiceManagerWidget`; see [Apps and tokens](https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens). Keep credentials out of committed notebooks.
- Confirm the selected environment, organization, and workspace before running. Note that some Geostatistical tasks remain in "tech preview" and therefore require the `preview=True` flag.

## Seequent Evo Tasks

The notebooks parameterize and orchestrate each task locally in the Jupyter notebook environmment then the computation itself runs on Seequent Evo's cloud platform, which supplies on-demand processing capacity rather than relying on your local machine.

Tasks reference geoscience objects in your Evo workspace and write results back to Evo. Keeping the input data, computation, and outputs on the platform enables low-latency data exchange within the workspace and avoids repeatedly transferring full datasets through the notebook. See the [Geostatistics tasks guide](https://developer.seequent.com/docs/guides/geostatistics-tasks) for task execution and object references.
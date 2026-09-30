# Geoscience Object Samples

These tutorials demonstrate how to create, download, and work with Evo geoscience objects. The recommended workflows use the high-level typed interfaces in `evo.objects` and the `evo.widgets` extension for rich notebook output.

## Start Here

1. [Simplified Object Interactions](simplified-object-interactions/simplified-object-interactions.ipynb)
	- Create, upload, download, and inspect typed geoscience objects such as PointSets and Regular3DGrids.
2. [Create a Downhole Collection](simplified-object-interactions/create-downhole-collection.ipynb)
	- Create a downhole collection using the simplified typed-object workflow.
3. [Download a PointSet](download-pointset/download-pointset.ipynb)
	- Download point-set data and inspect it in a notebook.

## Geostatistical Workflows

- [Run Kriging Compute](running-kriging-compute/running-kriging-compute.ipynb) creates pointsets and variogram models, visualizes them with Plotly, and introduces kriging estimation with Evo Compute.
- [Run Conditional Simulation](running-conditional-simulation/running-conditional-simulation.ipynb) demonstrates conditional simulation workflows.

## Drilling Campaigns

- [Create a Drilling Campaign](drilling-campaign/create-a-drilling-campaign/sdk-examples.ipynb) creates a drilling campaign with the SDK.
- [Download a Drilling Campaign](drilling-campaign/download-a-drilling-campaign/sdk-examples.ipynb) retrieves an existing drilling campaign.

## Direct API Samples

The publishing notebooks use the lower-level `evo-schemas` interfaces. Use them for direct API integrations and workflows that need greater control over request data.

- [Publish a PointSet](publish-pointset/publish-pointset.ipynb)
- [Publish Downhole Intervals](publish-downhole-intervals/publish-downhole-intervals.ipynb)
- [Publish a Downhole Collection](publish-downhole-collection/publish-downhole-collection.ipynb)
- [Publish Line Segments](publish-line-segments/publish-line-segments.ipynb)
- [Publish a Regular 2D Grid](publish-regular-2d-grid/publish-regular-2d-grid.ipynb)
- [Publish a Triangular Mesh](publish-triangular-mesh/publish-triangular-mesh.ipynb)

## Requirements

Each notebook directory contains its own dependency instructions. You need a [supported Python version](../README.md#before-you-start), a Seequent account with Evo access, and Evo application credentials to authenticate in the notebooks.


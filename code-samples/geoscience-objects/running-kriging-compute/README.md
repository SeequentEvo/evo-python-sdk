# Running Kriging Compute

This example demonstrates a complete geostatistical workflow using the Evo Python SDK:

1. **Load downhole assay data** as a PointSet
2. **Define a variogram model** for spatial correlation
3. **Visualize** the pointset and variogram together with Plotly
4. **Run kriging estimation** using Evo Compute (WIP)

## Overview

The workflow uses the WP_assay.csv dataset containing copper (CU_pct) and gold (AU_gpt) assay values from 55 downhole. We'll:

- Create a `PointSet` from the CSV data
- Define a nested spherical `Variogram` model for copper grades
- Extract and scale `Ellipsoid` objects for search neighborhoods
- Visualize the data, variogram curves, and anisotropy ellipsoids with Plotly
- Set up kriging estimation parameters (WIP)

## Dataset Characteristics

- **8,332 sample points** from 55 downhole
- **Spatial extent**: ~936m (X) × ~1,416m (Y) × ~855m (Z)
- **Coordinate system**: EPSG:32650 (UTM Zone 50N)
- **Target attribute**: CU_pct (copper percentage)
  - Mean: 0.95%, Variance: 0.84

## Variogram Model

The variogram uses two nested spherical structures aligned with the dominant orientation of the downhole data:
- **Nugget**: 0.08 (~10% nugget effect)
- **Short-range structure**: Contribution 0.25, ranges 80m × 60m × 40m
- **Long-range structure**: Contribution 0.51, ranges 250m × 180m × 100m
- **Anisotropy**: Dip 70°, Azimuth 15° (NNE strike direction)

## Kriging Compute

The notebook includes work-in-progress sections demonstrating:
- Creating a target `BlockModel` for estimation
- Configuring `KrigingParameters` with search neighborhoods
- Running kriging tasks with `evo.compute`
- Running multiple scenarios in parallel for sensitivity analysis

## Requirements

- Python 3.10+
- Seequent account with Evo entitlement
- Evo application credentials (client ID and redirect URL)

## Quick Start

1. Open `running-kriging-compute.ipynb` in Jupyter
2. Update the `client_id` and `redirect_url` with your Evo app credentials
3. Run the cells to create objects and visualize the geostatistical model


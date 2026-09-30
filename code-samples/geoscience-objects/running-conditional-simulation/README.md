# Running Conditional Simulation Compute

This example demonstrates a complete conditional simulation workflow using the Evo Python SDK:

1. **Load downhole assay data** as a PointSet
2. **Define a normal-score variogram model** for spatial correlation in Gaussian space
3. **Create a target Regular3DGrid** to simulate onto
4. **Run conditional turning-band simulation** using Evo Compute
5. **Inspect** summary statistics, quantiles, risk measures and the validation report
6. **Publish a block model** combining all three inputs

## Overview

The workflow uses the same WP_assay.csv dataset containing copper (CU_pct) and
gold (AU_gpt) assay values from 55 downholes. We'll:

- Create a `PointSet` from the CSV data
- Define a nested spherical `Variogram` in normal-score space for copper grades
- Extract the variogram `Ellipsoid` and scale it into the conditioning search neighbourhood
- Create a `Regular3DGrid` target and run the `consim` task with `evo.compute`
- Publish the outcome as a `BlockModel` in the Block Model Service
- Compare point-support and block-support scenarios run in parallel

## Dataset Characteristics

- **8,332 sample points** from 55 downholes
- **Spatial extent**: ~936m (X) × ~1,416m (Y) × ~855m (Z)
- **Coordinate system**: EPSG:32650 (UTM Zone 50N)
- **Target attribute**: CU_pct (copper percentage)
  - Mean: 0.95%, Variance: 0.84

## Variogram Model

Conditional simulation works in Gaussian space, so the variogram must model the
**normal-score transformed** data.

- **Modelling space**: `normalscore`
- **Sill**: 1.0 (variance of a standard Gaussian)
- **Nugget**: 0.10
- **Short-range structure**: Contribution 0.30, ranges 80m × 60m × 40m
- **Long-range structure**: Contribution 0.60, ranges 250m × 180m × 100m
- **Anisotropy**: Dip 70°, Azimuth 15° (NNE strike direction)

## Conditional Simulation Task

`ConSimParameters` drives the whole workflow. The main groups of settings are:

| Setting | Purpose |
| --- | --- |
| `distribution` | Tail extrapolation for the normal-score transformation |
| `block_discretization` | Sub-block discretization for support correction |
| `number_of_lines` | Turning bands used by the simulator |
| `number_of_simulations` | How many realisations to generate |
| `number_of_simulations_to_save` | How many realisations to publish to the grid |
| `location_wise_quantiles` | Quantiles computed per block across realisations |
| `probability_above_cutoff` | P(grade > cutoff) per block |
| `mean_above_cutoff` | E[grade \| grade > cutoff] per block |
| `perform_validation` | Generate a validation report and dashboard link |

The task evaluates onto a `regular-3d-grid` or `regular-masked-3d-grid` only.

## Block Model Output

The final section folds all three inputs into a single block model via
`BlockModel.create_regular`, which creates the block model in the Block Model Service
alongside the geoscience object that references it.

| Source | Contribution |
| --- | --- |
| Target grid | Block geometry and every simulated attribute - mean, variance, min, max, quantiles, cutoff statistics and any saved realisations |
| PointSet | Distance from each block centre to the nearest conditioning assay |
| Variogram | The modelled major range, used to normalise that distance into a confidence classification |

The well-known outputs are given friendly aliases (`sim_mean`, `sim_p10`,
`prob_above_0.5`, ...); anything else keeps the name the service generated.

Cell data is matched to blocks through `i`, `j`, `k` index columns, which follow the same
x-fastest ordering as the grid's cells.

## Requirements

- Python 3.10+
- Seequent account with Evo entitlement
- Evo application credentials (client ID and redirect URL)

## Quick Start

1. Open `running-conditional-simulation.ipynb` in Jupyter
2. Update the `client_id` and `redirect_url` with your Evo app credentials
3. Run the cells to create the objects and run the simulation

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/geostatistics)

# Geostatistics Tasks

Discoverable compute tasks in the `geostatistics` topic. Call any task below through `ComputeClient` as `await client.geostatistics.<task>.run(...)`, or by name via `await client.arun("geostatistics", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |
| `AttributeInput` | `str | AnyTypedAttribute` | A typed attribute, a JMESPath expression, or a bare attribute name. |
| `AnyTypedAttribute` | `Attribute | BlockModelAttribute | PendingAttribute | BlockModelPendingAttribute` | A typed attribute handle carrying both its parent geoscience object and attribute name. |
| `SearchNeighborhood` | `evo.compute.tasks.SearchNeighborhood` | Typed search ellipsoid and sample/octant/drillhole limit model accepted alongside raw dictionaries. |

## Tasks

### break_ties

`client.geostatistics.break_ties` · Task: `geostatistics/break-ties` · Version: `1.0.1` · Preview (`preview=True` required)

For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/break-ties)

```python
await client.geostatistics.break_ties.run(
    *,
    source: GeostatisticsBreakTiesSource | AnyTypedAttribute,
    target: GeostatisticsBreakTiesTarget | AnyTypedAttribute,
    neighborhood: GeostatisticsBreakTiesBreakTiesSearch | SearchNeighborhood,
    seed: int = ...,
    preview: bool = False,
) -> GeostatisticsBreakTiesResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsBreakTiesSource | AnyTypedAttribute` | Data in which ties will be broken. | _required_ |
| `target` | `GeostatisticsBreakTiesTarget | AnyTypedAttribute` | Object and attribute where the results will be written. | _required_ |
| `neighborhood` | `GeostatisticsBreakTiesBreakTiesSearch | SearchNeighborhood` | Search parameters. In this break ties algorithm, the final value of each tied point is influenced by the mean of nearby points. The search parameters determines which nearby points to use. | _required_ |
| `seed` | `int` | Seed for the random number generator. | `38239342` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsBreakTiesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message that says what happened in the task. |
| `target` | `GeostatisticsBreakTiesResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target that was created or updated. |

#### Nested Shapes

**`GeostatisticsBreakTiesSource`** (`TypedDict`)

Data in which ties will be broken.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object that contains the data in which ties will be broken. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The known values, used to estimate values at unknown locations. |

**`GeostatisticsBreakTiesTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsBreakTiesTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsBreakTiesTarget`** (`TypedDict`)

Object and attribute where the results will be written.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object where the results will be written. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `GeostatisticsBreakTiesTargetAttributeCreate | GeostatisticsBreakTiesTargetAttributeUpdate | AttributeInput` | Yes | The attribute to write the results to. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |

**`GeostatisticsBreakTiesNeighborhoodEllipsoidEllipsoidRanges`** (`TypedDict`)

The ranges of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `major` | `float` | Yes | The major axis length of the ellipsoid Bounds: `> 0`. |
| `semi_major` | `float` | Yes | The semi major axis length of the ellipsoid Bounds: `> 0`. |
| `minor` | `float` | Yes | The minor axis length of the ellipsoid Bounds: `> 0`. |

**`GeostatisticsBreakTiesRotation`** (`TypedDict`)

The rotation of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dip_azimuth` | `float` | No | first rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |
| `dip` | `float` | No | second rotation, about the x-axis, in degrees Bounds: `>= 0`, `<= 180`. |
| `pitch` | `float` | No | third rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |

**`GeostatisticsBreakTiesEllipsoid`** (`TypedDict`)

The ellipsoid, to search for points within

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid_ranges` | `GeostatisticsBreakTiesNeighborhoodEllipsoidEllipsoidRanges` | Yes | The ranges of the ellipsoid |
| `rotation` | `GeostatisticsBreakTiesRotation` | Yes | The rotation of the ellipsoid |

**`GeostatisticsBreakTiesBreakTiesSearch`** (`TypedDict`)

Search parameters.

In this break ties algorithm, the final value of each tied point is influenced by the mean of nearby points.
The search parameters determines which nearby points to use.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid` | `GeostatisticsBreakTiesEllipsoid` | Yes | The ellipsoid, to search for points within |
| `max_samples` | `int` | Yes | The maximum number of neighbors to find for each tied point. Bounds: `>= 1`. |
| `min_samples` | `int` | No | The minimum number of samples to use for each evaluation point. Bounds: `>= 1`. |

**`GeostatisticsBreakTiesResultAttribute`** (`ResultNode`)

Attribute containing the break ties result.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsBreakTiesResultTarget`** (`ResultNode`)

The target that was created or updated.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeostatisticsBreakTiesResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Attribute containing the break ties result. |

### conditional_turning_bands

`client.geostatistics.conditional_turning_bands` · Task: `geostatistics/conditional-turning-bands` · Version: `1.0.1`

Runs the turning band simulator using an existing continuous distribution. For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/conditional-turning-bands)

```python
await client.geostatistics.conditional_turning_bands.run(
    *,
    source: ObjectInput,
    source_attribute: AttributeInput,
    target: ObjectInput,
    neighborhood: GeostatisticsConditionalTurningBandsNeighborhood | SearchNeighborhood,
    distribution: ObjectInput,
    variogram_model: ObjectInput,
    block_discretization: GeostatisticsConditionalTurningBandsBlockDiscretization,
    filter: GeostatisticsConditionalTurningBandsFilter | None = ...,
    source_filter: GeostatisticsConditionalTurningBandsSourceFilter | None = ...,
    kriging_method: Literal["simple", "ordinary"] = ...,
    number_of_lines: int = ...,
    realizations: int = ...,
    random_seed: int = ...,
    preview: bool = False,
) -> GeostatisticsConditionalTurningBandsResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | Reference to the Geoscience Object containing the source data used for conditioning the simulation. Supported object types are point sets, downhole intervals, 3D grids (regular, masked, or tensor) and block models. For grids and block models, the cell or block centroids are used as the conditioning locations. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. | _required_ |
| `source_attribute` | `AttributeInput` | Reference to the attribute containing the source values. | _required_ |
| `target` | `ObjectInput` | Reference to the Geoscience Object containing the grid to evaluate the simulation onto. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. | _required_ |
| `neighborhood` | `GeostatisticsConditionalTurningBandsNeighborhood | SearchNeighborhood` | Search neighborhood for conditioning. | _required_ |
| `distribution` | `ObjectInput` | A non-parametric continuous cumulative distribution used for the normal-score transformation. Created by the continuous-distribution task. Supported schemas: `non-parametric-continuous-cumulative-distribution/[>=1.0,<2]`. | _required_ |
| `variogram_model` | `ObjectInput` | Reference to the variogram model used to model the covariance within the domain. Supported schemas: `variogram/[>=1.1,<2]`. | _required_ |
| `block_discretization` | `GeostatisticsConditionalTurningBandsBlockDiscretization` | Sub-block discretization used for support correction within each grid block. | _required_ |
| `filter` | `GeostatisticsConditionalTurningBandsFilter | None` | Optional filter to apply to the target grid. This allows the simulation to only be applied to a subset of grid locations, defined by filtering an attribute on the target grid. | `None` |
| `source_filter` | `GeostatisticsConditionalTurningBandsSourceFilter | None` | Optional filter to apply to the source object. This allows the simulation to use only a subset of the source data, defined by filtering an attribute on the source object. | `None` |
| `kriging_method` | `Literal["simple", "ordinary"]` | The kriging method to use for the conditioning step. | `'simple'` |
| `number_of_lines` | `int` | Number of lines to use for the turning-band simulation. Bounds: `>= 1`, `<= 1000`. | `500` |
| `realizations` | `int` | Number of realizations (simulations) to run. All realizations will be saved. Bounds: `>= 1`, `<= 100`. | `1` |
| `random_seed` | `int` | Random seed for simulation and tie-breaking. | `38239342` |
| `preview` | `bool` | Whether to include the preview opt-in header. | `False` |

Results (`GeostatisticsConditionalTurningBandsResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `target` | `GeostatisticsConditionalTurningBandsResultResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target grid with simulation results. |

#### Nested Shapes

**`GeostatisticsConditionalTurningBandsNeighborhoodEllipsoidEllipsoidRanges`** (`TypedDict`)

The ranges of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `major` | `float` | Yes | The major axis length of the ellipsoid Bounds: `> 0`. |
| `semi_major` | `float` | Yes | The semi major axis length of the ellipsoid Bounds: `> 0`. |
| `minor` | `float` | Yes | The minor axis length of the ellipsoid Bounds: `> 0`. |

**`GeostatisticsConditionalTurningBandsRotation`** (`TypedDict`)

The rotation of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dip_azimuth` | `float` | No | first rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |
| `dip` | `float` | No | second rotation, about the x-axis, in degrees Bounds: `>= 0`, `<= 180`. |
| `pitch` | `float` | No | third rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |

**`GeostatisticsConditionalTurningBandsEllipsoid`** (`TypedDict`)

The ellipsoid, to search for points within
Alias: anisotropy

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid_ranges` | `GeostatisticsConditionalTurningBandsNeighborhoodEllipsoidEllipsoidRanges` | Yes | The ranges of the ellipsoid |
| `rotation` | `GeostatisticsConditionalTurningBandsRotation` | Yes | The rotation of the ellipsoid |

**`GeostatisticsConditionalTurningBandsNeighborhood`** (`TypedDict`)

Search neighborhood for conditioning.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid` | `GeostatisticsConditionalTurningBandsEllipsoid` | Yes | The ellipsoid, to search for points within Alias: anisotropy |
| `max_samples` | `int` | Yes | The maximum number of samples to use for each evaluation point. Alias: max-samples Bounds: `>= 1`. |
| `min_samples` | `int` | No | The minimum number of samples to use for each evaluation point. Alias: min-samples Bounds: `>= 1`. |

**`GeostatisticsConditionalTurningBandsBlockDiscretization`** (`TypedDict`)

Sub-block discretization used for support correction within each grid block.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `nx` | `int` | No | Number of subdivisions in the x direction Bounds: `>= 1`, `<= 9`. |
| `ny` | `int` | No | Number of subdivisions in the y direction Bounds: `>= 1`, `<= 9`. |
| `nz` | `int` | No | Number of subdivisions in the z direction Bounds: `>= 1`, `<= 9`. |

**`GeostatisticsConditionalTurningBandsFilterCondition`** (`TypedDict`)

A leaf filter condition applied to one attribute of the filtered object.

Exactly one operator field must be provided.

- ``in``: Membership test. Provide string category names for
    ``CategoryAttribute`` objects or integers for integer-coded indices.
- ``not_in``: Exclusion test.
- ``equal``, ``not_equal``: Numeric equality/inequality.
- ``greater_than``, ``greater_than_or_equal_to``, ``less_than``,
    ``less_than_or_equal_to``: Numeric comparisons on numeric attributes.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operator` | `Literal["in", "not_in", "equal", "not_equal", "greater_than", "greater_than_or_equal_to", "less_than", "less_than_or_equal_to"]` | Yes | The filter operator. Use 'in' or 'not_in' for category/membership tests (pair with 'values'). Use 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to' for numeric comparisons (pair with 'threshold'). |
| `values` | `list[str] | None` | No | Category values to match. Required when operator is 'in' or 'not_in'. Use strings for string/category-name attributes; use integers for integer-coded attributes. |
| `threshold` | `float | None` | No | Numeric threshold for comparison. Required when operator is 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to'. |
| `type` | `Literal["condition"]` | No | Identifies this as a leaf filter condition. |
| `attribute` | `AttributeInput` | Yes | Attribute used to filter locations. |

**`GeostatisticsConditionalTurningBandsAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsConditionalTurningBandsFilterCondition | GeostatisticsConditionalTurningBandsAllOfFilter | GeostatisticsConditionalTurningBandsAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsConditionalTurningBandsAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsConditionalTurningBandsFilterCondition | GeostatisticsConditionalTurningBandsAllOfFilter | GeostatisticsConditionalTurningBandsAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsConditionalTurningBandsFilter`** (`TypedDict`)

Optional filter to apply to the target grid. This allows the simulation to only be applied to a subset of grid locations, defined by filtering an attribute on the target grid.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsConditionalTurningBandsFilterCondition | GeostatisticsConditionalTurningBandsAllOfFilter | GeostatisticsConditionalTurningBandsAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsConditionalTurningBandsSourceAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsConditionalTurningBandsFilterCondition | GeostatisticsConditionalTurningBandsSourceAllOfFilter | GeostatisticsConditionalTurningBandsSourceAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsConditionalTurningBandsSourceAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsConditionalTurningBandsFilterCondition | GeostatisticsConditionalTurningBandsSourceAllOfFilter | GeostatisticsConditionalTurningBandsSourceAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsConditionalTurningBandsSourceFilter`** (`TypedDict`)

Optional filter to apply to the source object. This allows the simulation to use only a subset of the source data, defined by filtering an attribute on the source object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsConditionalTurningBandsFilterCondition | GeostatisticsConditionalTurningBandsSourceAllOfFilter | GeostatisticsConditionalTurningBandsSourceAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsConditionalTurningBandsResultSimulations`** (`ResultNode`)

Attribute containing the block simulation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConditionalTurningBandsResultResultTarget`** (`ResultNode`)

The target grid with simulation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `simulations` | `GeostatisticsConditionalTurningBandsResultSimulations | None` | `await node.simulations.load()`, `await node.simulations.to_dataframe()` | Attribute containing the block simulation results. |

### consim

`client.geostatistics.consim` · Task: `geostatistics/consim` · Version: `1.0.1`

For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/conditional-simulation-workflow)

```python
await client.geostatistics.consim.run(
    *,
    source_object: ObjectInput,
    source_attribute: AttributeInput,
    target_object: ObjectInput,
    neighborhood: GeostatisticsConsimNeighborhood | SearchNeighborhood,
    variogram_model: ObjectInput,
    block_discretization: GeostatisticsConsimBlockDiscretization,
    filter: GeostatisticsConsimFilter | None = ...,
    source_filter: GeostatisticsConsimSourceFilter | None = ...,
    distribution: GeostatisticsConsimContinuousDistributionParameters | None = ...,
    kriging_method: Literal["simple", "ordinary"] = ...,
    number_of_lines: int = ...,
    number_of_simulations: int = ...,
    random_seed: int = ...,
    number_of_simulations_to_save: int = ...,
    loss_calculation: GeostatisticsConsimLossCalculation | None = ...,
    location_wise_quantiles: list[float] = ...,
    probability_above_cutoff: list[float] | None = ...,
    mean_above_cutoff: list[float] | None = ...,
    perform_validation: bool = ...,
    report_context: GeostatisticsConsimReportContext | None = ...,
    report_mean_thresholds: GeostatisticsConsimReportMeanThresholds | None = ...,
    preview: bool = False,
) -> GeostatisticsConsimResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source_object` | `ObjectInput` | Reference to the Geoscience Object containing the source data used for conditioning the simulation. Supported object types are point sets, downhole intervals, 3D grids (regular, masked, or tensor) and block models. For grids and block models, the cell or block centroids are used as the conditioning locations. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. | _required_ |
| `source_attribute` | `AttributeInput` | Reference to the attribute containing the source values. The values are transformed to normal-score space using the provided distribution. | _required_ |
| `target_object` | `ObjectInput` | Reference to the Geoscience Object containing the grid to evaluate the simulation onto. A new attribute will be created on this object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. | _required_ |
| `neighborhood` | `GeostatisticsConsimNeighborhood | SearchNeighborhood` | Search neighborhood for conditioning. | _required_ |
| `variogram_model` | `ObjectInput` | Reference to the variogram model used to model the covariance within the domain. Supported schemas: `variogram/[>=1.1,<2]`. | _required_ |
| `block_discretization` | `GeostatisticsConsimBlockDiscretization` | Sub-block discretization used for support correction within each grid block. | _required_ |
| `filter` | `GeostatisticsConsimFilter | None` | Optional filter to apply to the target grid. This allows the simulation to only be applied to a subset of grid locations, defined by filtering an attribute on the target grid. | `None` |
| `source_filter` | `GeostatisticsConsimSourceFilter | None` | Optional filter to apply to the source object. This allows the simulation to use only a subset of the source data, defined by filtering an attribute on the source object. | `None` |
| `distribution` | `GeostatisticsConsimContinuousDistributionParameters | None` | Parameters for a continuous distribution used for normal-score transformation. | `None` |
| `kriging_method` | `Literal["simple", "ordinary"]` | The kriging method to use for the conditioning step. | `'simple'` |
| `number_of_lines` | `int` | Number of lines to use for the turning-band simulation. Bounds: `>= 1`, `<= 1000`. | `500` |
| `number_of_simulations` | `int` | Number of simulations to run. Bounds: `>= 1`, `<= 100`. | `1` |
| `random_seed` | `int` | Random seed for simulation and tie-breaking. | `38239342` |
| `number_of_simulations_to_save` | `int` | Number of simulations to publish to the output object. If perform_validation is true, the minimum value must be at least 1. Bounds: `>= 0`, `<= 100`. | `5` |
| `loss_calculation` | `GeostatisticsConsimLossCalculation | None` | Settings for loss calculations. | `None` |
| `location_wise_quantiles` | `list[float]` | List of quantiles to compute for each location. | `None` |
| `probability_above_cutoff` | `list[float] | None` | List of cutoff values to compute probability above for each location. If None, this calculation is not performed. | `None` |
| `mean_above_cutoff` | `list[float] | None` | List of cutoff values to compute mean above for each location. If None, this calculation is not performed. | `None` |
| `perform_validation` | `bool` | Whether to run validation and generate a validation report. | `False` |
| `report_context` | `GeostatisticsConsimReportContext | None` | Context for the validation report. | `None` |
| `report_mean_thresholds` | `GeostatisticsConsimReportMeanThresholds | None` | Thresholds for the mean comparison in the validation report. | `None` |
| `preview` | `bool` | Whether to include the preview opt-in header. | `False` |

Results (`GeostatisticsConsimResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `target` | `GeostatisticsConsimResultResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target grid with simulation results. |
| `validation_summary` | `GeostatisticsConsimResultValidationSummary | None` | — | Summary of the validation results. |
| `validation_report` | `GeostatisticsConsimResultResultValidationReport | None` | — | Link to the validation report file. |
| `links` | `GeostatisticsConsimResultLinks` | — | Links related to the simulation task. |

#### Nested Shapes

**`GeostatisticsConsimNeighborhoodEllipsoidEllipsoidRanges`** (`TypedDict`)

The ranges of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `major` | `float` | Yes | The major axis length of the ellipsoid Bounds: `> 0`. |
| `semi_major` | `float` | Yes | The semi major axis length of the ellipsoid Bounds: `> 0`. |
| `minor` | `float` | Yes | The minor axis length of the ellipsoid Bounds: `> 0`. |

**`GeostatisticsConsimRotation`** (`TypedDict`)

The rotation of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dip_azimuth` | `float` | No | first rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |
| `dip` | `float` | No | second rotation, about the x-axis, in degrees Bounds: `>= 0`, `<= 180`. |
| `pitch` | `float` | No | third rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |

**`GeostatisticsConsimEllipsoid`** (`TypedDict`)

The ellipsoid, to search for points within
Alias: anisotropy

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid_ranges` | `GeostatisticsConsimNeighborhoodEllipsoidEllipsoidRanges` | Yes | The ranges of the ellipsoid |
| `rotation` | `GeostatisticsConsimRotation` | Yes | The rotation of the ellipsoid |

**`GeostatisticsConsimNeighborhood`** (`TypedDict`)

Search neighborhood for conditioning.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid` | `GeostatisticsConsimEllipsoid` | Yes | The ellipsoid, to search for points within Alias: anisotropy |
| `max_samples` | `int` | Yes | The maximum number of samples to use for each evaluation point. Alias: max-samples Bounds: `>= 1`. |
| `min_samples` | `int` | No | The minimum number of samples to use for each evaluation point. Alias: min-samples Bounds: `>= 1`. |

**`GeostatisticsConsimBlockDiscretization`** (`TypedDict`)

Sub-block discretization used for support correction within each grid block.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `nx` | `int` | No | Number of subdivisions in the x direction Bounds: `>= 1`, `<= 9`. |
| `ny` | `int` | No | Number of subdivisions in the y direction Bounds: `>= 1`, `<= 9`. |
| `nz` | `int` | No | Number of subdivisions in the z direction Bounds: `>= 1`, `<= 9`. |

**`GeostatisticsConsimFilterCondition`** (`TypedDict`)

A leaf filter condition applied to one attribute of the filtered object.

Exactly one operator field must be provided.

- ``in``: Membership test. Provide string category names for
    ``CategoryAttribute`` objects or integers for integer-coded indices.
- ``not_in``: Exclusion test.
- ``equal``, ``not_equal``: Numeric equality/inequality.
- ``greater_than``, ``greater_than_or_equal_to``, ``less_than``,
    ``less_than_or_equal_to``: Numeric comparisons on numeric attributes.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operator` | `Literal["in", "not_in", "equal", "not_equal", "greater_than", "greater_than_or_equal_to", "less_than", "less_than_or_equal_to"]` | Yes | The filter operator. Use 'in' or 'not_in' for category/membership tests (pair with 'values'). Use 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to' for numeric comparisons (pair with 'threshold'). |
| `values` | `list[str] | None` | No | Category values to match. Required when operator is 'in' or 'not_in'. Use strings for string/category-name attributes; use integers for integer-coded attributes. |
| `threshold` | `float | None` | No | Numeric threshold for comparison. Required when operator is 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to'. |
| `type` | `Literal["condition"]` | No | Identifies this as a leaf filter condition. |
| `attribute` | `AttributeInput` | Yes | Attribute used to filter locations. |

**`GeostatisticsConsimAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsConsimFilterCondition | GeostatisticsConsimAllOfFilter | GeostatisticsConsimAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsConsimAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsConsimFilterCondition | GeostatisticsConsimAllOfFilter | GeostatisticsConsimAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsConsimFilter`** (`TypedDict`)

Optional filter to apply to the target grid. This allows the simulation to only be applied to a subset of grid locations, defined by filtering an attribute on the target grid.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsConsimFilterCondition | GeostatisticsConsimAllOfFilter | GeostatisticsConsimAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsConsimSourceAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsConsimFilterCondition | GeostatisticsConsimSourceAllOfFilter | GeostatisticsConsimSourceAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsConsimSourceAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsConsimFilterCondition | GeostatisticsConsimSourceAllOfFilter | GeostatisticsConsimSourceAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsConsimSourceFilter`** (`TypedDict`)

Optional filter to apply to the source object. This allows the simulation to use only a subset of the source data, defined by filtering an attribute on the source object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsConsimFilterCondition | GeostatisticsConsimSourceAllOfFilter | GeostatisticsConsimSourceAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsConsimDistributionTailExtrapolationUpper`** (`TypedDict`)

Power model for extrapolating the upper tail of the distribution.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `power` | `float` | Yes | Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive). Bounds: `> 0.0`, `<= 1.0`. |
| `max` | `float` | Yes | Maximum extent of tail, must be greater than the maximum value in the data. |

**`GeostatisticsConsimDistributionTailExtrapolationLower`** (`TypedDict`)

Power model for extrapolating the lower tail of the distribution. Optional.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `power` | `float` | Yes | Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive). Bounds: `> 0.0`, `<= 1.0`. |
| `min` | `float` | Yes | Minimum extent of tail, must be less than the minimum value in the data. |

**`GeostatisticsConsimTailExtrapolation`** (`TypedDict`)

Tail extrapolation

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `upper` | `GeostatisticsConsimDistributionTailExtrapolationUpper` | Yes | Power model for extrapolating the upper tail of the distribution. |
| `lower` | `GeostatisticsConsimDistributionTailExtrapolationLower | None` | No | Power model for extrapolating the lower tail of the distribution. Optional. |

**`GeostatisticsConsimContinuousDistributionParameters`** (`TypedDict`)

Parameters for a continuous distribution used for normal-score transformation.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `tail_extrapolation` | `GeostatisticsConsimTailExtrapolation | None` | No | Tail extrapolation |
| `weights` | `AttributeInput | None` | No | Optional weights attribute for weighted distribution |

**`GeostatisticsConsimConSimMaterialCategory`** (`TypedDict`)

Material category for loss calculation.

If both cutoff_grade and metal_price are zero, this category is treated as waste.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `cutoff_grade` | `float` | Yes | Cutoff grade for the category Bounds: `>= 0.0`. |
| `metal_price` | `float` | Yes | Price of the metal per unit Bounds: `>= 0.0`. |
| `label` | `str` | Yes | Label for the material category |

**`GeostatisticsConsimLossCalculationTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsConsimLossCalculationTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsConsimLossCalculation`** (`TypedDict`)

Settings for loss calculations.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `material_categories` | `list[GeostatisticsConsimConSimMaterialCategory]` | Yes | Ordered list of material categories, from lowest to highest cutoff grade |
| `processing_cost` | `float` | Yes | Processing cost per unit of material Bounds: `>= 0.0`. |
| `mining_waste_cost` | `float` | Yes | Mining cost per unit of waste material Bounds: `>= 0.0`. |
| `mining_ore_cost` | `float` | Yes | Mining cost per unit of ore material Bounds: `>= 0.0`. |
| `metal_recovery_fraction` | `float` | Yes | Fraction of metal expected to be recovered during processing Bounds: `>= 0.0`, `<= 1.0`. |
| `target_attribute` | `GeostatisticsConsimLossCalculationTargetAttributeCreate | GeostatisticsConsimLossCalculationTargetAttributeUpdate | AttributeInput` | Yes | The target attribute that will be created or updated with loss calculation results. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |

**`GeostatisticsConsimValidationReportContextItem`** (`TypedDict`)

A single item of contextual information to be included in the validation report.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `label` | `str` | Yes | Label of the context item, which will appear on the report. |
| `value` | `str` | Yes | Value of the context item. |

**`GeostatisticsConsimReportContext`** (`TypedDict`)

Context for the validation report.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `title` | `str` | Yes | Title that will appear on the report. |
| `details` | `list[GeostatisticsConsimValidationReportContextItem]` | Yes | Key-value pairs of details to appear on the report. |

**`GeostatisticsConsimReportMeanThresholds`** (`TypedDict`)

Thresholds for the mean comparison in the validation report.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `acceptable` | `float` | Yes | Threshold for the percentage difference between the reference and simulated means that is considered acceptable. Bounds: `>= 0.0`. |
| `marginal` | `float` | Yes | Threshold for the percentage difference between the reference and simulated means that is considered marginal. Bounds: `>= 0.0`. |

**`GeostatisticsConsimResultMean`** (`ResultNode`)

Attribute containing the mean of the simulations.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultVariance`** (`ResultNode`)

Attribute containing the variance of the simulations.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultMinimum`** (`ResultNode`)

Attribute containing the minimum of the simulations.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultMaximum`** (`ResultNode`)

Attribute containing the maximum of the simulations.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultResultTargetSummaryAttributes`** (`ResultNode`)

Attribute containing summary statistics of the simulation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `mean` | `GeostatisticsConsimResultMean` | `await node.mean.load()`, `await node.mean.to_dataframe()` | Attribute containing the mean of the simulations. |
| `variance` | `GeostatisticsConsimResultVariance` | `await node.variance.load()`, `await node.variance.to_dataframe()` | Attribute containing the variance of the simulations. |
| `min` | `GeostatisticsConsimResultMinimum` | `await node.min.load()`, `await node.min.to_dataframe()` | Attribute containing the minimum of the simulations. |
| `max` | `GeostatisticsConsimResultMaximum` | `await node.max.load()`, `await node.max.to_dataframe()` | Attribute containing the maximum of the simulations. |

**`GeostatisticsConsimResultQuantile`** (`ResultNode`)

Attribute containing a quantile of the simulation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |
| `quantile` | `float` | — | Quantile of the simulations, that this attribute represents. Bounds: `>= 0.0`, `<= 1.0`. |

**`GeostatisticsConsimResultSimulations`** (`ResultNode`)

Attribute containing the simulation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultResultTargetSimulationsNormalScore`** (`ResultNode`)

Attribute containing the simulation results in normal score space.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultResultTargetPointSimulations`** (`ResultNode`)

Attribute containing the point simulation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultResultTargetPointSimulationsNormalScore`** (`ResultNode`)

Attribute containing the point simulation results in normal score space.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultResultTargetLossCalculationAttribute`** (`ResultNode`)

Attribute containing the loss calculation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsConsimResultResultTargetProbabilityAboveCutoffAttributesItem`** (`ResultNode`)

Attribute containing the probability of being above a cutoff value.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |
| `cutoff` | `float` | — | The cutoff value that this attribute represents. |

**`GeostatisticsConsimResultResultTargetMeanAboveCutoffAttributesItem`** (`ResultNode`)

Attribute containing the mean of values above a cutoff.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |
| `cutoff` | `float` | — | The cutoff value that this attribute represents. |

**`GeostatisticsConsimResultResultTarget`** (`ResultNode`)

The target grid with simulation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `summary_attributes` | `GeostatisticsConsimResultResultTargetSummaryAttributes` | — | Attribute containing summary statistics of the simulation results. |
| `quantile_attributes` | `list[GeostatisticsConsimResultQuantile]` | `await node.quantile_attributes[i].load()`, `await node.quantile_attributes[i].to_dataframe()` | List of attributes containing quantiles of the simulation results. |
| `simulations` | `GeostatisticsConsimResultSimulations | None` | `await node.simulations.load()`, `await node.simulations.to_dataframe()` | Attribute containing the simulation results. |
| `simulations_normal_score` | `GeostatisticsConsimResultResultTargetSimulationsNormalScore | None` | `await node.simulations_normal_score.load()`, `await node.simulations_normal_score.to_dataframe()` | Attribute containing the simulation results in normal score space. |
| `point_simulations` | `GeostatisticsConsimResultResultTargetPointSimulations | None` | `await node.point_simulations.load()`, `await node.point_simulations.to_dataframe()` | Attribute containing the point simulation results. |
| `point_simulations_normal_score` | `GeostatisticsConsimResultResultTargetPointSimulationsNormalScore | None` | `await node.point_simulations_normal_score.load()`, `await node.point_simulations_normal_score.to_dataframe()` | Attribute containing the point simulation results in normal score space. |
| `loss_calculation_attribute` | `GeostatisticsConsimResultResultTargetLossCalculationAttribute | None` | `await node.loss_calculation_attribute.load()`, `await node.loss_calculation_attribute.to_dataframe()` | Attribute containing the loss calculation results. |
| `probability_above_cutoff_attributes` | `list[GeostatisticsConsimResultResultTargetProbabilityAboveCutoffAttributesItem]` | `await node.probability_above_cutoff_attributes[i].load()`, `await node.probability_above_cutoff_attributes[i].to_dataframe()` | List of attributes containing probability above cutoff values. |
| `mean_above_cutoff_attributes` | `list[GeostatisticsConsimResultResultTargetMeanAboveCutoffAttributesItem]` | `await node.mean_above_cutoff_attributes[i].load()`, `await node.mean_above_cutoff_attributes[i].to_dataframe()` | List of attributes containing mean above cutoff values. |

**`GeostatisticsConsimResultValidationSummary`** (`ResultNode`)

Summary of the validation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference_mean` | `float` | — | Mean of the input values. |
| `mean` | `float` | — | Mean of the simulated values. |

**`GeostatisticsConsimResultResultValidationReport`** (`ResultNode`)

Link to the validation report file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a file. |
| `name` | `str` | — | The name of the file, typically the terminal segment of the path. |

**`GeostatisticsConsimResultLinks`** (`ResultNode`)

Links related to the simulation task.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `dashboard` | `str | None` | — | Link to the validation dashboard. |

### continuous_distribution

`client.geostatistics.continuous_distribution` · Task: `geostatistics/continuous-distribution` · Version: `1.0.1` · Preview (`preview=True` required)

For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/continuous-distribution)

```python
await client.geostatistics.continuous_distribution.run(
    *,
    source: GeostatisticsContinuousDistributionSource | AnyTypedAttribute,
    target: GeostatisticsContinuousDistributionTarget,
    weights: GeostatisticsContinuousDistributionWeights | None | AnyTypedAttribute = ...,
    tail_extrapolation: GeostatisticsContinuousDistributionTailExtrapolation | None = ...,
    preview: bool = False,
) -> GeostatisticsContinuousDistributionResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsContinuousDistributionSource | AnyTypedAttribute` | Source data to create the distribution from. | _required_ |
| `target` | `GeostatisticsContinuousDistributionTarget` | Target specifying where to create the continuous distribution object. | _required_ |
| `weights` | `GeostatisticsContinuousDistributionWeights | None | AnyTypedAttribute` | Optional weights for creating a weighted (declustered) distribution. | `None` |
| `tail_extrapolation` | `GeostatisticsContinuousDistributionTailExtrapolation | None` | Optional tail extrapolation parameters for extending the distribution beyond the data range. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsContinuousDistributionResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message describing the result of the task. |
| `distribution` | `GeostatisticsContinuousDistributionResultDistribution` | `await result.distribution.load()`, `await result.distribution.to_dataframe(*keys)` | Reference to the created continuous distribution object. |

#### Nested Shapes

**`GeostatisticsContinuousDistributionSource`** (`TypedDict`)

Source data to create the distribution from.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object containing the data to create a distribution from. Supported schemas: `pointset/[>=1.2,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The attribute containing the values to create a distribution from. |

**`GeostatisticsContinuousDistributionTarget`** (`TypedDict`)

Target specifying where to create the continuous distribution object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `non-parametric-continuous-cumulative-distribution/[>=1.0,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeostatisticsContinuousDistributionWeights`** (`TypedDict`)

Optional weights for creating a weighted (declustered) distribution.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object containing the weights data. Supported schemas: `pointset/[>=1.2,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The attribute containing the weights values. |

**`GeostatisticsContinuousDistributionTailExtrapolationUpper`** (`TypedDict`)

Power model for extrapolating the upper tail of the distribution.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `power` | `float` | Yes | Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive). Bounds: `> 0.0`, `<= 1.0`. |
| `max` | `float` | Yes | Maximum extent of tail, must be greater than the maximum value in the data. |

**`GeostatisticsContinuousDistributionTailExtrapolationLower`** (`TypedDict`)

Power model for extrapolating the lower tail of the distribution. Optional.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `power` | `float` | Yes | Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive). Bounds: `> 0.0`, `<= 1.0`. |
| `min` | `float` | Yes | Minimum extent of tail, must be less than the minimum value in the data. |

**`GeostatisticsContinuousDistributionTailExtrapolation`** (`TypedDict`)

Optional tail extrapolation parameters for extending the distribution beyond the data range.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `upper` | `GeostatisticsContinuousDistributionTailExtrapolationUpper` | Yes | Power model for extrapolating the upper tail of the distribution. |
| `lower` | `GeostatisticsContinuousDistributionTailExtrapolationLower | None` | No | Power model for extrapolating the lower tail of the distribution. Optional. |

**`GeostatisticsContinuousDistributionResultDistribution`** (`ResultNode`)

Reference to the created continuous distribution object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `non-parametric-continuous-cumulative-distribution/[>=1.0,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### declustering

`client.geostatistics.declustering` · Task: `geostatistics/declustering` · Version: `1.0.1` · Preview (`preview=True` required)

Computes grid-based declustering weights by measuring each sample's influence on evaluation locations. Supports both KNN (arithmetic mean) and IDW (inverse-distance weighted) modes via an optional power parameter.

```python
await client.geostatistics.declustering.run(
    *,
    source: GeostatisticsDeclusteringSource | ObjectInput,
    grid: GeostatisticsDeclusteringGrid | ObjectInput,
    target: GeostatisticsDeclusteringTarget | AnyTypedAttribute,
    neighborhood: GeostatisticsDeclusteringExtendedNeighborhood | SearchNeighborhood,
    power: float | None = ...,
    preview: bool = False,
) -> GeostatisticsDeclusteringResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsDeclusteringSource | ObjectInput` | The source sample locations. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. | _required_ |
| `grid` | `GeostatisticsDeclusteringGrid | ObjectInput` | The evaluation grid for measuring sample influence. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. | _required_ |
| `target` | `GeostatisticsDeclusteringTarget | AnyTypedAttribute` | The target object and attribute to write declustering weights to. | _required_ |
| `neighborhood` | `GeostatisticsDeclusteringExtendedNeighborhood | SearchNeighborhood` | Search neighborhood parameters. The search ellipsoid and sample limits control which samples influence each grid location during the estimation used to compute declustering weights. | _required_ |
| `power` | `float | None` | Inverse-distance weighting power. Controls how strongly distance affects influence: higher values give nearer samples more weight. The default (2.0) uses standard inverse-distance-squared weighting. Set to null to use arithmetic-mean KNN (equal neighbour votes, no distance weighting). Bounds: `> 0.0`. | `2.0` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsDeclusteringResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message that says what happened in the task. |
| `target` | `GeostatisticsDeclusteringResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target that was created or updated with declustering weights. |

#### Nested Shapes

**`GeostatisticsDeclusteringSource`** (`TypedDict`)

The source sample locations.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Spatial locations of samples to compute declustering weights for. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |

**`GeostatisticsDeclusteringGrid`** (`TypedDict`)

The evaluation grid for measuring sample influence.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Regular grid or set of evaluation locations used to measure each sample's spatial influence. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |

**`GeostatisticsDeclusteringTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsDeclusteringTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsDeclusteringTarget`** (`TypedDict`)

The target object and attribute to write declustering weights to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object to write the declustering weight attribute onto. Typically the same object as the source. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `GeostatisticsDeclusteringTargetAttributeCreate | GeostatisticsDeclusteringTargetAttributeUpdate | AttributeInput` | Yes | Name of the attribute to create or update with the computed declustering weights. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |

**`GeostatisticsDeclusteringNeighborhoodEllipsoidEllipsoidRanges`** (`TypedDict`)

The ranges of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `major` | `float` | Yes | The major axis length of the ellipsoid Bounds: `> 0`. |
| `semi_major` | `float` | Yes | The semi major axis length of the ellipsoid Bounds: `> 0`. |
| `minor` | `float` | Yes | The minor axis length of the ellipsoid Bounds: `> 0`. |

**`GeostatisticsDeclusteringRotation`** (`TypedDict`)

The rotation of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dip_azimuth` | `float` | No | first rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |
| `dip` | `float` | No | second rotation, about the x-axis, in degrees Bounds: `>= 0`, `<= 180`. |
| `pitch` | `float` | No | third rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |

**`GeostatisticsDeclusteringEllipsoid`** (`TypedDict`)

The ellipsoid, to search for points within
Alias: anisotropy

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid_ranges` | `GeostatisticsDeclusteringNeighborhoodEllipsoidEllipsoidRanges` | Yes | The ranges of the ellipsoid |
| `rotation` | `GeostatisticsDeclusteringRotation` | Yes | The rotation of the ellipsoid |

**`GeostatisticsDeclusteringExtendedNeighborhood`** (`TypedDict`)

Search neighborhood parameters. The search ellipsoid and sample limits control which samples influence each grid location during the estimation used to compute declustering weights.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid` | `GeostatisticsDeclusteringEllipsoid` | Yes | The ellipsoid, to search for points within Alias: anisotropy |
| `max_samples` | `int` | Yes | The maximum number of samples to use for each evaluation point. Alias: max-samples Bounds: `>= 1`. |
| `min_samples` | `int` | No | The minimum number of samples to use for each evaluation point. Alias: min-samples Bounds: `>= 1`. |
| `max_empty_octants` | `int` | No | The maximum number of empty octants (aka sectors) allowed when searching for neighbors. Omitting this, or a value of 8, disables the octant check. Bounds: `>= 0`, `<= 8`. |
| `max_samples_per_octant` | `int | None` | No | The maximum number of samples to use from each octant. Bounds: `>= 1`. |
| `max_samples_per_drillhole` | `int | None` | No | The maximum number of samples to use from each drillhole (defined by drillhole ID) Bounds: `>= 1`. |
| `max_empty_quadrants` | `int | None` | No | The maximum number of empty quadrants (2D sectors, ignoring Z) allowed when searching for neighbors. If not specified, quadrant filtering is disabled. Bounds: `>= 0`, `<= 4`. |
| `max_samples_per_quadrant` | `int | None` | No | The maximum number of samples to use from each quadrant (2D sectors, ignoring Z). Bounds: `>= 1`. |
| `max_drillholes` | `int | None` | No | The maximum number of drillholes to be used in each estimate. Alias: max_drillholes_per_estimate Bounds: `>= 1`. |
| `min_drillholes` | `int` | No | The minimum number of distinct drillholes required in each estimate. 0 disables the check. Requires Downhole Intervals. Alias: min_drillholes_per_estimate Bounds: `>= 0`. |
| `drillhole_limit_scope` | `Literal["global", "quadrant", "octant"]` | No | Apply max_samples_per_drillhole globally (the default), independently in each quadrant (ignoring Z), or independently in each octant, in search-ellipsoid coordinates. Requires max_samples_per_drillhole. Limits on the number of distinct drillholes remain global. |

**`GeostatisticsDeclusteringResultAttribute`** (`ResultNode`)

Attribute containing the declustering weights.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsDeclusteringResultTarget`** (`ResultNode`)

The target that was created or updated with declustering weights.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeostatisticsDeclusteringResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Attribute containing the declustering weights. |

### experimental_variogram

`client.geostatistics.experimental_variogram` · Task: `geostatistics/experimental-variogram` · Version: `1.0.1` · Preview (`preview=True` required)

Computes one or more experimental (sample) variograms from a points or downhole intervals source object and writes them to a new experimental variogram object. A single call can mix omnidirectional, directional (azimuth/dip tolerance cone with optional bandwidth/bandheight) and downhole variograms, each with its own lag definition.

```python
await client.geostatistics.experimental_variogram.run(
    *,
    source: GeostatisticsExperimentalVariogramSource | AnyTypedAttribute,
    directions: list[GeostatisticsExperimentalVariogramOmnidirectionalVariogram | GeostatisticsExperimentalVariogramDirectionalVariogram | GeostatisticsExperimentalVariogramDownholeVariogram],
    target: GeostatisticsExperimentalVariogramTarget,
    preview: bool = False,
) -> GeostatisticsExperimentalVariogramResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsExperimentalVariogramSource | AnyTypedAttribute` | The source object and attribute providing the sample locations and values. | _required_ |
| `directions` | `list[GeostatisticsExperimentalVariogramOmnidirectionalVariogram | GeostatisticsExperimentalVariogramDirectionalVariogram | GeostatisticsExperimentalVariogramDownholeVariogram]` | One or more variogram specifications to compute in this call (at most 100). Each entry is an omnidirectional, directional or downhole variogram and produces one direction in the output object, except directional sweeps, which produce one direction per settings combination plus an optional average. A request may produce at most 100,000 curves and 20,000,000 lag bins in total. | _required_ |
| `target` | `GeostatisticsExperimentalVariogramTarget` | Target specifying where to create the experimental variogram object. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsExperimentalVariogramResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message describing the result of the task. |
| `experimental_variogram` | `GeostatisticsExperimentalVariogramResultResultExperimentalVariogram` | `await result.experimental_variogram.load()`, `await result.experimental_variogram.to_dataframe(*keys)` | Reference to the created experimental variogram object. |

#### Nested Shapes

**`GeostatisticsExperimentalVariogramSourceFilterCondition`** (`TypedDict`)

A leaf filter condition applied to one attribute of the filtered object.

Exactly one operator field must be provided.

- ``in``: Membership test. Provide string category names for
    ``CategoryAttribute`` objects or integers for integer-coded indices.
- ``not_in``: Exclusion test.
- ``equal``, ``not_equal``: Numeric equality/inequality.
- ``greater_than``, ``greater_than_or_equal_to``, ``less_than``,
    ``less_than_or_equal_to``: Numeric comparisons on numeric attributes.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operator` | `Literal["in", "not_in", "equal", "not_equal", "greater_than", "greater_than_or_equal_to", "less_than", "less_than_or_equal_to"]` | Yes | The filter operator. Use 'in' or 'not_in' for category/membership tests (pair with 'values'). Use 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to' for numeric comparisons (pair with 'threshold'). |
| `values` | `list[str] | None` | No | Category values to match. Required when operator is 'in' or 'not_in'. Use strings for string/category-name attributes; use integers for integer-coded attributes. |
| `threshold` | `float | None` | No | Numeric threshold for comparison. Required when operator is 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to'. |
| `type` | `Literal["condition"]` | No | Identifies this as a leaf filter condition. |
| `attribute` | `AttributeInput` | Yes | Attribute used to filter locations. |

**`GeostatisticsExperimentalVariogramSourceAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsExperimentalVariogramSourceFilterCondition | GeostatisticsExperimentalVariogramSourceAllOfFilter | GeostatisticsExperimentalVariogramSourceAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsExperimentalVariogramSourceAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsExperimentalVariogramSourceFilterCondition | GeostatisticsExperimentalVariogramSourceAllOfFilter | GeostatisticsExperimentalVariogramSourceAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsExperimentalVariogramSourceFilter`** (`TypedDict`)

Optional filter to apply to the source object. This restricts the variogram computation to a subset of the samples, defined by filtering an attribute on the source object (for example a domain or region code).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsExperimentalVariogramSourceFilterCondition | GeostatisticsExperimentalVariogramSourceAllOfFilter | GeostatisticsExperimentalVariogramSourceAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsExperimentalVariogramSource`** (`TypedDict`)

The source object and attribute providing the sample locations and values.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Geoscience object containing the sample locations and values. Must be a points or downhole intervals object. Supported schemas: `pointset/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The numeric attribute on the source object to compute the variogram on. |
| `drillhole_id` | `AttributeInput | None` | No | Optional category attribute identifying the drillhole each sample belongs to. Required for downhole variograms when the source is a points object. When the source is a downhole intervals object and this is omitted, the object's intrinsic hole identifier is used instead. |
| `filter` | `GeostatisticsExperimentalVariogramSourceFilter | None` | No | Optional filter to apply to the source object. This restricts the variogram computation to a subset of the samples, defined by filtering an attribute on the source object (for example a domain or region code). |
| `domain` | `str | None` | No | Optional label for the domain these samples represent, recorded on the created experimental variogram object. Typically describes the subset selected by 'filter', for example the name of an estimation domain. |

**`GeostatisticsExperimentalVariogramExplicitLags`** (`TypedDict`)

Explicit lag bins: the lower and upper bound of every bin, with starts in non-decreasing order.

Bins may overlap.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["explicit"]` | Yes | Method discriminator. |
| `starts` | `list[float]` | Yes | Lower bound (inclusive) of each lag bin, in non-decreasing order. |
| `ends` | `list[float]` | Yes | Upper bound (exclusive) of each lag bin. |

**`GeostatisticsExperimentalVariogramUniformLags`** (`TypedDict`)

Equal-width lag bins from zero to ``max_distance``.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["uniform"]` | Yes | Method discriminator. |
| `max_distance` | `float` | Yes | Largest separation distance the lag bins reach, in source coordinate units. Bounds: `> 0.0`. |
| `n_lags` | `int` | No | Number of equal-width lag bins. Bounds: `>= 1`, `<= 10000`. |

**`GeostatisticsExperimentalVariogramAutoDensityLags`** (`TypedDict`)

Variable-width lag bins derived from the sample density, each holding a comparable number of pairs.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["auto_density"]` | Yes | Method discriminator. |
| `max_distance` | `float` | Yes | Largest separation distance the lag bins reach, in source coordinate units. Bounds: `> 0.0`. |
| `min_lags` | `int` | No | Lower bound on the number of density-based lag bins. Bounds: `>= 1`, `<= 10000`. |
| `max_lags` | `int` | No | Upper bound on the number of density-based lag bins. Bounds: `>= 1`, `<= 10000`. |
| `min_pairs_fraction` | `float` | No | Target number of pairs per lag bin, as a fraction of the sample count. Bounds: `> 0.0`. |

**`GeostatisticsExperimentalVariogramOmnidirectionalVariogram`** (`TypedDict`)

An omnidirectional variogram — all sample pairs binned by separation distance.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["omnidirectional"]` | Yes | Variogram type discriminator. |
| `method` | `GeostatisticsExperimentalVariogramExplicitLags | GeostatisticsExperimentalVariogramUniformLags | GeostatisticsExperimentalVariogramAutoDensityLags` | Yes | How the lag bins are defined. |

**`GeostatisticsExperimentalVariogramDirectionalExplicit`** (`TypedDict`)

One directional variogram with explicit lag bins and a fixed search geometry.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["explicit"]` | Yes | Method discriminator. |
| `starts` | `list[float]` | Yes | Lower bound (inclusive) of each lag bin, in non-decreasing order. |
| `ends` | `list[float]` | Yes | Upper bound (exclusive) of each lag bin. |
| `azimuth_tolerance` | `float` | No | Half-width of the azimuth acceptance cone in degrees. Bounds: `> 0.0`, `<= 90.0`. |
| `dip_tolerance` | `float` | No | Half-width of the dip acceptance cone in degrees. Bounds: `> 0.0`, `<= 90.0`. |
| `bandwidth` | `float` | Yes | Maximum perpendicular horizontal distance from the search axis. Bounds: `> 0.0`. |
| `bandheight` | `float` | Yes | Maximum perpendicular vertical distance from the search axis. Bounds: `> 0.0`. |

**`GeostatisticsExperimentalVariogramDirectionalUniform`** (`TypedDict`)

One directional variogram with equal-width lag bins and a fixed search geometry.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["uniform"]` | Yes | Method discriminator. |
| `max_distance` | `float` | Yes | Largest separation distance the lag bins reach, in source coordinate units. Bounds: `> 0.0`. |
| `n_lags` | `int` | No | Number of equal-width lag bins. Bounds: `>= 1`, `<= 10000`. |
| `azimuth_tolerance` | `float` | No | Half-width of the azimuth acceptance cone in degrees. Bounds: `> 0.0`, `<= 90.0`. |
| `dip_tolerance` | `float` | No | Half-width of the dip acceptance cone in degrees. Bounds: `> 0.0`, `<= 90.0`. |
| `bandwidth` | `float` | Yes | Maximum perpendicular horizontal distance from the search axis. Bounds: `> 0.0`. |
| `bandheight` | `float` | Yes | Maximum perpendicular vertical distance from the search axis. Bounds: `> 0.0`. |

**`GeostatisticsExperimentalVariogramDirectionalAutoDensity`** (`TypedDict`)

One directional variogram with density-based lag bins; bandwidth and bandheight are derived.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["auto_density"]` | Yes | Method discriminator. |
| `max_distance` | `float` | Yes | Largest separation distance the lag bins reach, in source coordinate units. Bounds: `> 0.0`. |
| `min_lags` | `int` | No | Lower bound on the number of density-based lag bins. Bounds: `>= 1`, `<= 10000`. |
| `max_lags` | `int` | No | Upper bound on the number of density-based lag bins. Bounds: `>= 1`, `<= 10000`. |
| `min_pairs_fraction` | `float` | No | Target number of pairs per lag bin, as a fraction of the sample count. Bounds: `> 0.0`. |
| `azimuth_tolerance` | `float` | No | Half-width of the azimuth acceptance cone in degrees. Bounds: `> 0.0`, `<= 90.0`. |
| `dip_tolerance` | `float` | No | Half-width of the dip acceptance cone in degrees. Bounds: `> 0.0`, `<= 90.0`. |

**`GeostatisticsExperimentalVariogramRegularLagAveraging`** (`TypedDict`)

Also save one curve averaging this direction's curves. Omit to skip.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `lag_width` | `float` | Yes | Spacing of the regular bin centres (0, lag_width, 2*lag_width, ...). Bounds: `> 0.0`. |
| `max_distance` | `float` | Yes | Largest bin centre of the averaged curve. Bounds: `> 0.0`. |

**`GeostatisticsExperimentalVariogramDirectionalUniformSweep`** (`TypedDict`)

Directional variograms with equal-width lag bins, one for every combination of the listed settings.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["uniform_sweep"]` | Yes | Method discriminator. |
| `azimuth_tolerances` | `list[float]` | No | Azimuth tolerances in degrees to try. Defaults to [15]. |
| `dip_tolerances` | `list[float]` | No | Dip tolerances in degrees to try. Defaults to [15]. |
| `max_distances` | `list[float]` | Yes | Largest lag distances to try. |
| `lag_counts` | `list[int]` | No | Numbers of equal-width lag bins to try. Defaults to [20]. |
| `bandwidths` | `list[float]` | Yes | Bandwidths to try. |
| `bandheights` | `list[float]` | Yes | Bandheights to try. |
| `averaging` | `GeostatisticsExperimentalVariogramRegularLagAveraging | None` | No | Also save one curve averaging this direction's curves. Omit to skip. |

**`GeostatisticsExperimentalVariogramDirectionalAutoDensitySweep`** (`TypedDict`)

Directional variograms with density-based lag bins, one for every combination of the listed settings.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["auto_density_sweep"]` | Yes | Method discriminator. |
| `azimuth_tolerances` | `list[float]` | No | Azimuth tolerances in degrees to try. Defaults to [15]. |
| `dip_tolerances` | `list[float]` | No | Dip tolerances in degrees to try. Defaults to [15]. |
| `max_distances` | `list[float]` | Yes | Largest lag distances to try. |
| `min_lags` | `list[int]` | No | Lower bounds on the number of lag bins to try. Defaults to [20]. |
| `max_lags` | `list[int]` | No | Upper bounds on the number of lag bins to try. Defaults to [120]. |
| `min_pairs_fraction` | `float` | No | Target number of pairs per lag bin, as a fraction of the sample count. Bounds: `> 0.0`. |
| `averaging` | `GeostatisticsExperimentalVariogramRegularLagAveraging | None` | No | Also save one curve averaging this direction's curves. Omit to skip. |

**`GeostatisticsExperimentalVariogramDirectionalVariogram`** (`TypedDict`)

A directional variogram — pairs within an azimuth/dip search around one direction.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["directional"]` | Yes | Variogram type discriminator. |
| `azimuth` | `float` | Yes | Azimuth of the search direction in degrees, measured clockwise from North. Range [0, 360). Bounds: `>= 0.0`, `< 360.0`. |
| `dip` | `float` | Yes | Dip of the search direction in degrees, measured downward from horizontal. Range [-90, 90]. Bounds: `>= -90.0`, `<= 90.0`. |
| `rectangular_cone` | `bool` | No | When True, the acceptance region is a rectangular (trapezoidal) search where the azimuth and dip tolerances are applied independently, matching Leapfrog's default search. When False, an elliptical cone is used. |
| `method` | `GeostatisticsExperimentalVariogramDirectionalExplicit | GeostatisticsExperimentalVariogramDirectionalUniform | GeostatisticsExperimentalVariogramDirectionalAutoDensity | GeostatisticsExperimentalVariogramDirectionalUniformSweep | GeostatisticsExperimentalVariogramDirectionalAutoDensitySweep` | Yes | How the lag bins and search geometry are defined, for one curve or a sweep of curves. |

**`GeostatisticsExperimentalVariogramDownholeVariogram`** (`TypedDict`)

A downhole variogram — pairs within the same drillhole, binned by along-hole distance.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["downhole"]` | Yes | Variogram type discriminator. |
| `method` | `GeostatisticsExperimentalVariogramExplicitLags | GeostatisticsExperimentalVariogramUniformLags | GeostatisticsExperimentalVariogramAutoDensityLags` | Yes | How the lag bins are defined. |

**`GeostatisticsExperimentalVariogramTarget`** (`TypedDict`)

Target specifying where to create the experimental variogram object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `experimental-variogram/[>=1.0,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeostatisticsExperimentalVariogramResultResultExperimentalVariogram`** (`ResultNode`)

Reference to the created experimental variogram object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `experimental-variogram/[>=1.0,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### idw

`client.geostatistics.idw` · Task: `geostatistics/idw` · Version: `1.0.1`

Inverse distance weighting estimation using weighted average of nearby sample values, where weights are inversely proportional to distance raised to a specified power.

```python
await client.geostatistics.idw.run(
    *,
    source: GeostatisticsIdwSource | AnyTypedAttribute,
    target: GeostatisticsIdwTarget | AnyTypedAttribute,
    neighborhood: GeostatisticsIdwExtendedNeighborhood | SearchNeighborhood,
    power: float,
    preview: bool = False,
) -> GeostatisticsIdwResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsIdwSource | AnyTypedAttribute` | The source object and attribute containing known values. | _required_ |
| `target` | `GeostatisticsIdwTarget | AnyTypedAttribute` | The target object and attribute to create or update with IDW results. | _required_ |
| `neighborhood` | `GeostatisticsIdwExtendedNeighborhood | SearchNeighborhood` | In IDW estimation, the value of each evaluation point is determined by a weighted average of nearby points with known values, where weights are inversely proportional to distance raised to the specified power. The search parameters determine which nearby points to use. | _required_ |
| `power` | `float` | IDW power parameter. Controls how quickly influence decreases with distance. Common values: 1.0, 2.0. Bounds: `> 0.0`. | _required_ |
| `preview` | `bool` | Whether to include the preview opt-in header. | `False` |

Results (`GeostatisticsIdwResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message that says what happened in the task. |
| `target` | `GeostatisticsIdwResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target that was created or updated. |

#### Nested Shapes

**`GeostatisticsIdwSourceFilterCondition`** (`TypedDict`)

A leaf filter condition applied to one attribute of the filtered object.

Exactly one operator field must be provided.

- ``in``: Membership test. Provide string category names for
    ``CategoryAttribute`` objects or integers for integer-coded indices.
- ``not_in``: Exclusion test.
- ``equal``, ``not_equal``: Numeric equality/inequality.
- ``greater_than``, ``greater_than_or_equal_to``, ``less_than``,
    ``less_than_or_equal_to``: Numeric comparisons on numeric attributes.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operator` | `Literal["in", "not_in", "equal", "not_equal", "greater_than", "greater_than_or_equal_to", "less_than", "less_than_or_equal_to"]` | Yes | The filter operator. Use 'in' or 'not_in' for category/membership tests (pair with 'values'). Use 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to' for numeric comparisons (pair with 'threshold'). |
| `values` | `list[str] | None` | No | Category values to match. Required when operator is 'in' or 'not_in'. Use strings for string/category-name attributes; use integers for integer-coded attributes. |
| `threshold` | `float | None` | No | Numeric threshold for comparison. Required when operator is 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to'. |
| `type` | `Literal["condition"]` | No | Identifies this as a leaf filter condition. |
| `attribute` | `AttributeInput` | Yes | Attribute used to filter locations. |

**`GeostatisticsIdwSourceAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsIdwSourceFilterCondition | GeostatisticsIdwSourceAllOfFilter | GeostatisticsIdwSourceAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsIdwSourceAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsIdwSourceFilterCondition | GeostatisticsIdwSourceAllOfFilter | GeostatisticsIdwSourceAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsIdwSourceFilter`** (`TypedDict`)

Optional filter to apply to the source object. This allows IDW to use only a subset of the known data, defined by filtering an attribute on the source object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsIdwSourceFilterCondition | GeostatisticsIdwSourceAllOfFilter | GeostatisticsIdwSourceAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsIdwSource`** (`TypedDict`)

The source object and attribute containing known values.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Spatial locations of known values. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The known values, used to estimate values at unknown locations. |
| `filter` | `GeostatisticsIdwSourceFilter | None` | No | Optional filter to apply to the source object. This allows IDW to use only a subset of the known data, defined by filtering an attribute on the source object. |

**`GeostatisticsIdwTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsIdwTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsIdwAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsIdwSourceFilterCondition | GeostatisticsIdwAllOfFilter | GeostatisticsIdwAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsIdwAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsIdwSourceFilterCondition | GeostatisticsIdwAllOfFilter | GeostatisticsIdwAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsIdwFilter`** (`TypedDict`)

Optional filter to apply to the target object. This allows the estimation to only be applied to a subset of the target object, defined by filtering an attribute on the target object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsIdwSourceFilterCondition | GeostatisticsIdwAllOfFilter | GeostatisticsIdwAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsIdwTarget`** (`TypedDict`)

The target object and attribute to create or update with IDW results.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object to evaluate onto. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `GeostatisticsIdwTargetAttributeCreate | GeostatisticsIdwTargetAttributeUpdate | AttributeInput` | Yes | Name of the attribute to create or update on the target object Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |
| `filter` | `GeostatisticsIdwFilter | None` | No | Optional filter to apply to the target object. This allows the estimation to only be applied to a subset of the target object, defined by filtering an attribute on the target object. |
| `update_mode` | `Literal["auto", "merge", "replace"]` | No | How to treat target locations that this run did not estimate, such as locations excluded by the target filter. 'merge' keeps the values those locations already had, so running one domain at a time accumulates results into a single attribute. 'replace' clears them, so the attribute only ever holds the results of the latest run. 'auto' (the default) merges when a target filter is set and replaces otherwise. Merging only applies to attributes that already exist on the target object; a newly created attribute has nothing to merge with. Boolean and integer diagnostic attributes have no way to represent 'not estimated', so within the filter they are always written as produced. |

**`GeostatisticsIdwNeighborhoodEllipsoidEllipsoidRanges`** (`TypedDict`)

The ranges of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `major` | `float` | Yes | The major axis length of the ellipsoid Bounds: `> 0`. |
| `semi_major` | `float` | Yes | The semi major axis length of the ellipsoid Bounds: `> 0`. |
| `minor` | `float` | Yes | The minor axis length of the ellipsoid Bounds: `> 0`. |

**`GeostatisticsIdwRotation`** (`TypedDict`)

The rotation of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dip_azimuth` | `float` | No | first rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |
| `dip` | `float` | No | second rotation, about the x-axis, in degrees Bounds: `>= 0`, `<= 180`. |
| `pitch` | `float` | No | third rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |

**`GeostatisticsIdwEllipsoid`** (`TypedDict`)

The ellipsoid, to search for points within
Alias: anisotropy

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid_ranges` | `GeostatisticsIdwNeighborhoodEllipsoidEllipsoidRanges` | Yes | The ranges of the ellipsoid |
| `rotation` | `GeostatisticsIdwRotation` | Yes | The rotation of the ellipsoid |

**`GeostatisticsIdwExtendedNeighborhood`** (`TypedDict`)

In IDW estimation, the value of each evaluation point is determined by a weighted average of nearby points with known values, where weights are inversely proportional to distance raised to the specified power. The search parameters determine which nearby points to use.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid` | `GeostatisticsIdwEllipsoid` | Yes | The ellipsoid, to search for points within Alias: anisotropy |
| `max_samples` | `int` | Yes | The maximum number of samples to use for each evaluation point. Alias: max-samples Bounds: `>= 1`. |
| `min_samples` | `int` | No | The minimum number of samples to use for each evaluation point. Alias: min-samples Bounds: `>= 1`. |
| `max_empty_octants` | `int` | No | The maximum number of empty octants (aka sectors) allowed when searching for neighbors. Omitting this, or a value of 8, disables the octant check. Bounds: `>= 0`, `<= 8`. |
| `max_samples_per_octant` | `int | None` | No | The maximum number of samples to use from each octant. Bounds: `>= 1`. |
| `max_samples_per_drillhole` | `int | None` | No | The maximum number of samples to use from each drillhole (defined by drillhole ID) Bounds: `>= 1`. |
| `max_empty_quadrants` | `int | None` | No | The maximum number of empty quadrants (2D sectors, ignoring Z) allowed when searching for neighbors. If not specified, quadrant filtering is disabled. Bounds: `>= 0`, `<= 4`. |
| `max_samples_per_quadrant` | `int | None` | No | The maximum number of samples to use from each quadrant (2D sectors, ignoring Z). Bounds: `>= 1`. |
| `max_drillholes` | `int | None` | No | The maximum number of drillholes to be used in each estimate. Alias: max_drillholes_per_estimate Bounds: `>= 1`. |
| `min_drillholes` | `int` | No | The minimum number of distinct drillholes required in each estimate. 0 disables the check. Requires Downhole Intervals. Alias: min_drillholes_per_estimate Bounds: `>= 0`. |
| `drillhole_limit_scope` | `Literal["global", "quadrant", "octant"]` | No | Apply max_samples_per_drillhole globally (the default), independently in each quadrant (ignoring Z), or independently in each octant, in search-ellipsoid coordinates. Requires max_samples_per_drillhole. Limits on the number of distinct drillholes remain global. |

**`GeostatisticsIdwResultAttribute`** (`ResultNode`)

Attribute containing the IDW result.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsIdwResultTarget`** (`ResultNode`)

The target that was created or updated.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeostatisticsIdwResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Attribute containing the IDW result. |

### knn

`client.geostatistics.knn` · Task: `geostatistics/knn` · Version: `1.0.1`

Nearest neighbor estimation using arithmetic mean of nearby sample values.

```python
await client.geostatistics.knn.run(
    *,
    source: GeostatisticsKnnSource | AnyTypedAttribute,
    target: GeostatisticsKnnTarget | AnyTypedAttribute,
    neighborhood: GeostatisticsKnnExtendedNeighborhood | SearchNeighborhood,
    preview: bool = False,
) -> GeostatisticsKnnResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsKnnSource | AnyTypedAttribute` | The source object and attribute containing known values. | _required_ |
| `target` | `GeostatisticsKnnTarget | AnyTypedAttribute` | The target object and attribute to create or update with KNN results. | _required_ |
| `neighborhood` | `GeostatisticsKnnExtendedNeighborhood | SearchNeighborhood` | In KNN estimation, the value of each evaluation point is determined by the arithmetic mean of a set of nearby points with known values. The search parameters determines which nearby points to use. | _required_ |
| `preview` | `bool` | Whether to include the preview opt-in header. | `False` |

Results (`GeostatisticsKnnResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message that says what happened in the task. |
| `target` | `GeostatisticsKnnResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target that was created or updated. |

#### Nested Shapes

**`GeostatisticsKnnSourceFilterCondition`** (`TypedDict`)

A leaf filter condition applied to one attribute of the filtered object.

Exactly one operator field must be provided.

- ``in``: Membership test. Provide string category names for
    ``CategoryAttribute`` objects or integers for integer-coded indices.
- ``not_in``: Exclusion test.
- ``equal``, ``not_equal``: Numeric equality/inequality.
- ``greater_than``, ``greater_than_or_equal_to``, ``less_than``,
    ``less_than_or_equal_to``: Numeric comparisons on numeric attributes.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operator` | `Literal["in", "not_in", "equal", "not_equal", "greater_than", "greater_than_or_equal_to", "less_than", "less_than_or_equal_to"]` | Yes | The filter operator. Use 'in' or 'not_in' for category/membership tests (pair with 'values'). Use 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to' for numeric comparisons (pair with 'threshold'). |
| `values` | `list[str] | None` | No | Category values to match. Required when operator is 'in' or 'not_in'. Use strings for string/category-name attributes; use integers for integer-coded attributes. |
| `threshold` | `float | None` | No | Numeric threshold for comparison. Required when operator is 'equal', 'not_equal', 'greater_than', 'greater_than_or_equal_to', 'less_than', or 'less_than_or_equal_to'. |
| `type` | `Literal["condition"]` | No | Identifies this as a leaf filter condition. |
| `attribute` | `AttributeInput` | Yes | Attribute used to filter locations. |

**`GeostatisticsKnnSourceAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsKnnSourceFilterCondition | GeostatisticsKnnSourceAllOfFilter | GeostatisticsKnnSourceAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsKnnSourceAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsKnnSourceFilterCondition | GeostatisticsKnnSourceAllOfFilter | GeostatisticsKnnSourceAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsKnnSourceFilter`** (`TypedDict`)

Optional filter to apply to the source object. This allows KNN to use only a subset of the known data, defined by filtering an attribute on the source object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsKnnSourceFilterCondition | GeostatisticsKnnSourceAllOfFilter | GeostatisticsKnnSourceAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsKnnSource`** (`TypedDict`)

The source object and attribute containing known values.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Spatial locations of known values. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The known values, used to estimate values at unknown locations. |
| `filter` | `GeostatisticsKnnSourceFilter | None` | No | Optional filter to apply to the source object. This allows KNN to use only a subset of the known data, defined by filtering an attribute on the source object. |

**`GeostatisticsKnnTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsKnnTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsKnnAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeostatisticsKnnSourceFilterCondition | GeostatisticsKnnAllOfFilter | GeostatisticsKnnAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeostatisticsKnnAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeostatisticsKnnSourceFilterCondition | GeostatisticsKnnAllOfFilter | GeostatisticsKnnAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeostatisticsKnnFilter`** (`TypedDict`)

Optional filter to apply to the target object. This allows the estimation to only be applied to a subset of the target object, defined by filtering an attribute on the target object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeostatisticsKnnSourceFilterCondition | GeostatisticsKnnAllOfFilter | GeostatisticsKnnAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeostatisticsKnnTarget`** (`TypedDict`)

The target object and attribute to create or update with KNN results.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object to evaluate onto. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `GeostatisticsKnnTargetAttributeCreate | GeostatisticsKnnTargetAttributeUpdate | AttributeInput` | Yes | Name of the attribute to create or update on the target object Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |
| `filter` | `GeostatisticsKnnFilter | None` | No | Optional filter to apply to the target object. This allows the estimation to only be applied to a subset of the target object, defined by filtering an attribute on the target object. |
| `update_mode` | `Literal["auto", "merge", "replace"]` | No | How to treat target locations that this run did not estimate, such as locations excluded by the target filter. 'merge' keeps the values those locations already had, so running one domain at a time accumulates results into a single attribute. 'replace' clears them, so the attribute only ever holds the results of the latest run. 'auto' (the default) merges when a target filter is set and replaces otherwise. Merging only applies to attributes that already exist on the target object; a newly created attribute has nothing to merge with. Boolean and integer diagnostic attributes have no way to represent 'not estimated', so within the filter they are always written as produced. |

**`GeostatisticsKnnNeighborhoodEllipsoidEllipsoidRanges`** (`TypedDict`)

The ranges of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `major` | `float` | Yes | The major axis length of the ellipsoid Bounds: `> 0`. |
| `semi_major` | `float` | Yes | The semi major axis length of the ellipsoid Bounds: `> 0`. |
| `minor` | `float` | Yes | The minor axis length of the ellipsoid Bounds: `> 0`. |

**`GeostatisticsKnnRotation`** (`TypedDict`)

The rotation of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dip_azimuth` | `float` | No | first rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |
| `dip` | `float` | No | second rotation, about the x-axis, in degrees Bounds: `>= 0`, `<= 180`. |
| `pitch` | `float` | No | third rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |

**`GeostatisticsKnnEllipsoid`** (`TypedDict`)

The ellipsoid, to search for points within
Alias: anisotropy

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid_ranges` | `GeostatisticsKnnNeighborhoodEllipsoidEllipsoidRanges` | Yes | The ranges of the ellipsoid |
| `rotation` | `GeostatisticsKnnRotation` | Yes | The rotation of the ellipsoid |

**`GeostatisticsKnnExtendedNeighborhood`** (`TypedDict`)

In KNN estimation, the value of each evaluation point is determined by the arithmetic mean of a set of nearby points with known values. The search parameters determines which nearby points to use.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid` | `GeostatisticsKnnEllipsoid` | Yes | The ellipsoid, to search for points within Alias: anisotropy |
| `max_samples` | `int` | Yes | The maximum number of samples to use for each evaluation point. Alias: max-samples Bounds: `>= 1`. |
| `min_samples` | `int` | No | The minimum number of samples to use for each evaluation point. Alias: min-samples Bounds: `>= 1`. |
| `max_empty_octants` | `int` | No | The maximum number of empty octants (aka sectors) allowed when searching for neighbors. Omitting this, or a value of 8, disables the octant check. Bounds: `>= 0`, `<= 8`. |
| `max_samples_per_octant` | `int | None` | No | The maximum number of samples to use from each octant. Bounds: `>= 1`. |
| `max_samples_per_drillhole` | `int | None` | No | The maximum number of samples to use from each drillhole (defined by drillhole ID) Bounds: `>= 1`. |
| `max_empty_quadrants` | `int | None` | No | The maximum number of empty quadrants (2D sectors, ignoring Z) allowed when searching for neighbors. If not specified, quadrant filtering is disabled. Bounds: `>= 0`, `<= 4`. |
| `max_samples_per_quadrant` | `int | None` | No | The maximum number of samples to use from each quadrant (2D sectors, ignoring Z). Bounds: `>= 1`. |
| `max_drillholes` | `int | None` | No | The maximum number of drillholes to be used in each estimate. Alias: max_drillholes_per_estimate Bounds: `>= 1`. |
| `min_drillholes` | `int` | No | The minimum number of distinct drillholes required in each estimate. 0 disables the check. Requires Downhole Intervals. Alias: min_drillholes_per_estimate Bounds: `>= 0`. |
| `drillhole_limit_scope` | `Literal["global", "quadrant", "octant"]` | No | Apply max_samples_per_drillhole globally (the default), independently in each quadrant (ignoring Z), or independently in each octant, in search-ellipsoid coordinates. Requires max_samples_per_drillhole. Limits on the number of distinct drillholes remain global. |

**`GeostatisticsKnnResultAttribute`** (`ResultNode`)

Attribute containing the KNN result.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsKnnResultTarget`** (`ResultNode`)

The target that was created or updated.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeostatisticsKnnResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Attribute containing the KNN result. |

### kriging

`client.geostatistics.kriging` · Task: `geostatistics/kriging` · Custom runner: `KrigingOverride` · Version: `1.0.1`

For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/kriging)

```python
await client.geostatistics.kriging.run(
    *,
    source: Source | Attribute | BlockModelAttribute,
    target: Target | Attribute | PendingAttribute | BlockModelAttribute | BlockModelPendingAttribute,
    variogram: ObjectInput,
    neighborhood: SearchNeighborhood,
    kriging_method: SimpleKriging | OrdinaryKriging | None = None,
    source_filter: Filter | None = None,
    target_filter: Filter | None = None,
    block_discretization: BlockDiscretization | None = None,
    diagnostics: KrigingDiagnostics | None = None,
    preview: bool = False,
) -> KrigingResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `Source | Attribute | BlockModelAttribute` | The object and attribute holding the known values, or the typed attribute itself. | _required_ |
| `target` | `Target | Attribute | PendingAttribute | BlockModelAttribute | BlockModelPendingAttribute` | Where to write the estimate -- a ``Target``, or the typed attribute to create or update. | _required_ |
| `variogram` | `ObjectInput` | The variogram object, or a reference to it. Supported schemas: `variogram/[>=1.1,<2]`. | _required_ |
| `neighborhood` | `SearchNeighborhood` | The search neighbourhood: the ellipsoid to look in, and how many samples to use. | _required_ |
| `kriging_method` | `SimpleKriging | OrdinaryKriging | None` | Ordinary kriging by default; ``KrigingMethod.simple`` when the mean is known. | `None` |
| `source_filter` | `Filter | None` | Restrict the estimate to a subset of the source data. | `None` |
| `target_filter` | `Filter | None` | Restrict the estimate to a subset of the target object. | `None` |
| `block_discretization` | `BlockDiscretization | None` | Sub-block discretization for block kriging. Omitted means point kriging. | `None` |
| `diagnostics` | `KrigingDiagnostics | None` | Optional diagnostic attributes to write beside the estimate. | `None` |
| `preview` | `bool` | Opt in to the preview API. Defaults to ``False``. | `False` |

Results (`KrigingResult`):

| Member | Type | Access | Description |
| --- | --- | --- | --- |
| `attribute_name` | `str` | `result.attribute_name` | The name of the attribute that was created/updated. |
| `diagnostics` | `dict[str, TaskAttribute]` | `result.diagnostics` | The diagnostic attributes that were written, keyed by diagnostic name. |
| `get_target_object()` | `BaseObject` | `await result.get_target_object()` | Load and return the target geoscience object. |
| `message` | `str` | `result.message` | A message describing what happened in the task. |
| `schema` | `ObjectSchema` | `result.schema` | The schema type of the target object (e.g., 'regular-masked-3d-grid'). |
| `target_name` | `str` | `result.target_name` | The name of the target object. |
| `target_reference` | `str` | `result.target_reference` | Reference URL to the target object. |
| `to_dataframe()` | `pd.DataFrame` | `await result.to_dataframe()` | Get the task results as a DataFrame. |

### location_wise

`client.geostatistics.location_wise` · Task: `geostatistics/location-wise` · Version: `1.0.1` · Preview (`preview=True` required)

For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/location-wise)

```python
await client.geostatistics.location_wise.run(
    *,
    source: GeostatisticsLocationWiseSource | AnyTypedAttribute,
    target: GeostatisticsLocationWiseTarget | ObjectInput,
    summary: bool | None = ...,
    quantiles: list[float] | None = ...,
    probability_above_cutoff: GeostatisticsLocationWiseProbabilityAboveCutoff | None = ...,
    mean_above_cutoff: GeostatisticsLocationWiseMeanAboveCutoff | None = ...,
    preview: bool = False,
) -> GeostatisticsLocationWiseResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsLocationWiseSource | AnyTypedAttribute` | Source object and attribute containing ensemble values. | _required_ |
| `target` | `GeostatisticsLocationWiseTarget | ObjectInput` | Target object to save summary statistics. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`. | _required_ |
| `summary` | `bool | None` | If true, compute summary statistics (min, max, mean, variance). | `None` |
| `quantiles` | `list[float] | None` | List of quantiles between 0.0 and 1.0 to compute. | `None` |
| `probability_above_cutoff` | `GeostatisticsLocationWiseProbabilityAboveCutoff | None` | Compute P(value > cutoff) per location for each cutoff. | `None` |
| `mean_above_cutoff` | `GeostatisticsLocationWiseMeanAboveCutoff | None` | Compute E[value \| value > cutoff] per location for each cutoff. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsLocationWiseResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message that says what happened in the task. |
| `target` | `GeostatisticsLocationWiseResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target that was created or updated. |

#### Nested Shapes

**`GeostatisticsLocationWiseSource`** (`TypedDict`)

Source object and attribute containing ensemble values.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Spatial locations of known values. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | Reference to the ensemble to be processed. |

**`GeostatisticsLocationWiseTarget`** (`TypedDict`)

Target object to save summary statistics.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object to evaluate onto. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`. |

**`GeostatisticsLocationWiseProbabilityAboveCutoff`** (`TypedDict`)

Compute P(value > cutoff) per location for each cutoff.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `cutoffs` | `list[float]` | Yes | List of cutoff values to compute P(value > cutoff) per location. |

**`GeostatisticsLocationWiseMeanAboveCutoff`** (`TypedDict`)

Compute E[value | value > cutoff] per location for each cutoff.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `cutoffs` | `list[float]` | Yes | List of cutoff values to compute E[value \| value > cutoff] per location. |

**`GeostatisticsLocationWiseResultResultTargetAttributesItem`** (`ResultNode`)

Location-wise computed values.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsLocationWiseResultTarget`** (`ResultNode`)

The target that was created or updated.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attributes` | `list[GeostatisticsLocationWiseResultResultTargetAttributesItem]` | `await node.attributes[i].load()`, `await node.attributes[i].to_dataframe()` | Computed location-wise values. |

### loss_calculation

`client.geostatistics.loss_calculation` · Task: `geostatistics/loss-calculation` · Version: `1.0.1` · Preview (`preview=True` required)

Calculate loss given block model and economic parameters

```python
await client.geostatistics.loss_calculation.run(
    *,
    source: GeostatisticsLossCalculationSource | AnyTypedAttribute,
    target: GeostatisticsLossCalculationTarget | AnyTypedAttribute,
    material_categories: list[GeostatisticsLossCalculationMaterialCategories],
    processing_cost: float,
    mining_waste_cost: float,
    mining_ore_cost: float,
    metal_recovery_fraction: float,
    preview: bool = False,
) -> GeostatisticsLossCalculationResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsLossCalculationSource | AnyTypedAttribute` | The source object and attribute containing known values. | _required_ |
| `target` | `GeostatisticsLossCalculationTarget | AnyTypedAttribute` | The target object and attribute to create or update with grade calculation results. | _required_ |
| `material_categories` | `list[GeostatisticsLossCalculationMaterialCategories]` | List of material categories, sorted from lowest to highest cutoff grade | _required_ |
| `processing_cost` | `float` | Processing cost per unit of material Bounds: `>= 0.0`. | _required_ |
| `mining_waste_cost` | `float` | Mining cost per unit of waste material Bounds: `>= 0.0`. | _required_ |
| `mining_ore_cost` | `float` | Mining cost per unit of ore material Bounds: `>= 0.0`. | _required_ |
| `metal_recovery_fraction` | `float` | Fraction of metal expected to be recovered during processing Bounds: `>= 0.0`, `<= 1.0`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsLossCalculationResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | Description of the result |
| `target` | `GeostatisticsLossCalculationResultTarget | None` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target that was created or updated. |

#### Nested Shapes

**`GeostatisticsLossCalculationSource`** (`TypedDict`)

The source object and attribute containing known values.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | The location of a regular or masked 3D grid with known values. Should be one of Regular3DGrid_V1_2_0, Regular3DGrid_V1_3_0, RegularMasked3DGrid_V1_2_0, or RegularMasked3DGrid_V1_3_0. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The original values, used to calculate grade categories. |

**`GeostatisticsLossCalculationTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsLossCalculationTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsLossCalculationTarget`** (`TypedDict`)

The target object and attribute to create or update with grade calculation results.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object to evaluate onto. Can be the same as the source object. Should be one of Regular3DGrid_V1_2_0, Regular3DGrid_V1_3_0, RegularMasked3DGrid_V1_2_0, or RegularMasked3DGrid_V1_3_0. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. |
| `attribute` | `GeostatisticsLossCalculationTargetAttributeCreate | GeostatisticsLossCalculationTargetAttributeUpdate | AttributeInput` | Yes | Name of the attribute to create or update on the target object. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |

**`GeostatisticsLossCalculationMaterialCategories`** (`TypedDict`)

Definition of a material category used in grade calculations.
Each category is defined by a cutoff grade, metal price, and label.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `cutoff_grade` | `float` | No | The cutoff grade Bounds: `>= 0.0`. |
| `metal_price` | `float` | No | The metal price |
| `label` | `str` | No | The category label |

**`GeostatisticsLossCalculationResultAttribute`** (`ResultNode`)

Attribute containing the material category results from the grade calculation.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsLossCalculationResultTarget`** (`ResultNode`)

The target that was created or updated.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeostatisticsLossCalculationResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Attribute containing the material category results from the grade calculation. |

### normal_score

`client.geostatistics.normal_score` · Task: `geostatistics/normal-score` · Version: `1.0.1` · Preview (`preview=True` required)

For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/normal-score)

```python
await client.geostatistics.normal_score.run(
    *,
    method: Literal["forward", "backward"],
    source: GeostatisticsNormalScoreSource | AnyTypedAttribute,
    distribution: ObjectInput,
    target: GeostatisticsNormalScoreTarget | AnyTypedAttribute,
    preview: bool = False,
) -> GeostatisticsNormalScoreResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `method` | `Literal["forward", "backward"]` | The direction of the normal-score transformation. 'forward' transforms data from the source distribution to a standard Gaussian (mean=0, std=1). 'backward' transforms data from a standard Gaussian back to the source distribution. | _required_ |
| `source` | `GeostatisticsNormalScoreSource | AnyTypedAttribute` | The source object and attribute containing the values to transform. | _required_ |
| `distribution` | `ObjectInput` | A non-parametric continuous cumulative distribution that defines the source distribution for the transformation. Created by the continuous-distribution task. Supported schemas: `non-parametric-continuous-cumulative-distribution/[>=1.0,<2]`. | _required_ |
| `target` | `GeostatisticsNormalScoreTarget | AnyTypedAttribute` | The target object and attribute to write the transformed values to. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsNormalScoreResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | A message describing what happened in the task. |
| `target` | `GeostatisticsNormalScoreResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target object that was created or updated. |

#### Nested Shapes

**`GeostatisticsNormalScoreSource`** (`TypedDict`)

The source object and attribute containing the values to transform.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object containing the values to transform. Supported schemas: `pointset/[>=1.2,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The numeric attribute to apply the normal-score transformation to. |

**`GeostatisticsNormalScoreTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsNormalScoreTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsNormalScoreTarget`** (`TypedDict`)

The target object and attribute to write the transformed values to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object to write the transformed values to. Supported schemas: `pointset/[>=1.2,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `GeostatisticsNormalScoreTargetAttributeCreate | GeostatisticsNormalScoreTargetAttributeUpdate | AttributeInput` | Yes | Name of the attribute to create or update on the target object. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |

**`GeostatisticsNormalScoreResultAttribute`** (`ResultNode`)

Attribute containing the transformed values.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsNormalScoreResultTarget`** (`ResultNode`)

The target object that was created or updated.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeostatisticsNormalScoreResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Attribute containing the transformed values. |

### profit_calculation

`client.geostatistics.profit_calculation` · Task: `geostatistics/profit-calculation` · Version: `1.0.1` · Preview (`preview=True` required)

Calculate profit given block model and economic parameters

```python
await client.geostatistics.profit_calculation.run(
    *,
    source: GeostatisticsProfitCalculationSource | AnyTypedAttribute,
    target: GeostatisticsProfitCalculationTarget | AnyTypedAttribute,
    material_categories: list[GeostatisticsProfitCalculationMaterialCategories],
    processing_cost: float,
    mining_waste_cost: float,
    mining_ore_cost: float,
    metal_recovery_fraction: float,
    preview: bool = False,
) -> GeostatisticsProfitCalculationResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeostatisticsProfitCalculationSource | AnyTypedAttribute` | The source object and attribute containing known values. | _required_ |
| `target` | `GeostatisticsProfitCalculationTarget | AnyTypedAttribute` | The target object and attribute to create or update with grade calculation results. | _required_ |
| `material_categories` | `list[GeostatisticsProfitCalculationMaterialCategories]` | List of material categories, sorted from lowest to highest cutoff grade | _required_ |
| `processing_cost` | `float` | Processing cost per unit of material Bounds: `>= 0.0`. | _required_ |
| `mining_waste_cost` | `float` | Mining cost per unit of waste material Bounds: `>= 0.0`. | _required_ |
| `mining_ore_cost` | `float` | Mining cost per unit of ore material Bounds: `>= 0.0`. | _required_ |
| `metal_recovery_fraction` | `float` | Fraction of metal expected to be recovered during processing Bounds: `>= 0.0`, `<= 1.0`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsProfitCalculationResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `message` | `str` | — | Description of the result |
| `target` | `GeostatisticsProfitCalculationResultTarget | None` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The target that was created or updated. |

#### Nested Shapes

**`GeostatisticsProfitCalculationSource`** (`TypedDict`)

The source object and attribute containing known values.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | The location of a regular or masked 3D grid with known values. Should be one of Regular3DGrid_V1_2_0, Regular3DGrid_V1_3_0, RegularMasked3DGrid_V1_2_0, or RegularMasked3DGrid_V1_3_0. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | The original values, used to calculate grade categories. |

**`GeostatisticsProfitCalculationTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeostatisticsProfitCalculationTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeostatisticsProfitCalculationTarget`** (`TypedDict`)

The target object and attribute to create or update with grade calculation results.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Object to evaluate onto. Can be the same as the source object. Should be one of Regular3DGrid_V1_2_0, Regular3DGrid_V1_3_0, RegularMasked3DGrid_V1_2_0, or RegularMasked3DGrid_V1_3_0. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. |
| `attribute` | `GeostatisticsProfitCalculationTargetAttributeCreate | GeostatisticsProfitCalculationTargetAttributeUpdate | AttributeInput` | Yes | Name of the attribute to create or update on the target object. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |

**`GeostatisticsProfitCalculationMaterialCategories`** (`TypedDict`)

Definition of a material category used in grade calculations.
Each category is defined by a cutoff grade, metal price, and label.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `cutoff_grade` | `float` | No | The cutoff grade Bounds: `>= 0.0`. |
| `metal_price` | `float` | No | The metal price |
| `label` | `str` | No | The category label |

**`GeostatisticsProfitCalculationResultAttribute`** (`ResultNode`)

Attribute containing the material category results from the grade calculation.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeostatisticsProfitCalculationResultTarget`** (`ResultNode`)

The target that was created or updated.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeostatisticsProfitCalculationResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Attribute containing the material category results from the grade calculation. |

### simulation_report

`client.geostatistics.simulation_report` · Task: `geostatistics/simulation-report` · Version: `1.0.1` · Preview (`preview=True` required)

For more information, please read the [guide](https://developer.seequent.com/docs/guides/geostatistics-tasks/tasks/simulation-report)

```python
await client.geostatistics.simulation_report.run(
    *,
    simulation_source: ObjectInput,
    source_attribute: AttributeInput,
    simulation_target: ObjectInput,
    point_simulations: AttributeInput,
    point_simulations_normal_score: AttributeInput,
    block_simulations: AttributeInput,
    block_simulations_normal_score: AttributeInput,
    variogram_model: ObjectInput,
    neighborhood: GeostatisticsSimulationReportNeighborhood | SearchNeighborhood,
    block_discretization: GeostatisticsSimulationReportBlockDiscretization,
    number_of_simulations: int,
    distribution: GeostatisticsSimulationReportContinuousDistributionParameters | None = ...,
    kriging_method: Literal["simple", "ordinary"] = ...,
    number_of_lines: int = ...,
    random_seed: int = ...,
    material: GeostatisticsSimulationReportMaterial | None = ...,
    report_context: GeostatisticsSimulationReportReportContext | None = ...,
    report_mean_thresholds: GeostatisticsSimulationReportReportMeanThresholds | None = ...,
    preview: bool = False,
) -> GeostatisticsSimulationReportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `simulation_source` | `ObjectInput` | Reference to the Geoscience Object containing the source data used for conditioning the simulation. Supported object types are point sets, downhole intervals, 3D grids (regular, masked, or tensor) and block models. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. | _required_ |
| `source_attribute` | `AttributeInput` | Reference to the attribute containing the source values. | _required_ |
| `simulation_target` | `ObjectInput` | Reference to the Geoscience Object containing the grid that already has simulation results. The report will be placed next to this object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`. | _required_ |
| `point_simulations` | `AttributeInput` | Reference to the point-scale simulation results ensemble attribute on the target grid. | _required_ |
| `point_simulations_normal_score` | `AttributeInput` | Reference to the point-scale normal-score simulation results ensemble attribute on the target grid. | _required_ |
| `block_simulations` | `AttributeInput` | Reference to the block-scale simulation results ensemble attribute on the target grid. | _required_ |
| `block_simulations_normal_score` | `AttributeInput` | Reference to the block-scale normal-score simulation results ensemble attribute on the target grid. | _required_ |
| `variogram_model` | `ObjectInput` | Reference to the variogram model used to model the covariance within the domain. Supported schemas: `variogram/[>=1.1,<2]`. | _required_ |
| `neighborhood` | `GeostatisticsSimulationReportNeighborhood | SearchNeighborhood` | Search neighborhood for conditioning. | _required_ |
| `block_discretization` | `GeostatisticsSimulationReportBlockDiscretization` | Sub-block discretization used for support correction within each grid block. | _required_ |
| `number_of_simulations` | `int` | The total number of simulations that were performed. Bounds: `>= 1`, `<= 100`. | _required_ |
| `distribution` | `GeostatisticsSimulationReportContinuousDistributionParameters | None` | Parameters for a continuous distribution used for normal-score transformation. | `None` |
| `kriging_method` | `Literal["simple", "ordinary"]` | The kriging method used for the conditioning step. | `'simple'` |
| `number_of_lines` | `int` | Number of lines used for the turning-band simulation. Bounds: `>= 1`, `<= 1000`. | `500` |
| `random_seed` | `int` | Random seed for tie-breaking. | `38239342` |
| `material` | `GeostatisticsSimulationReportMaterial | None` | Material cost and category information for loss calculations. | `None` |
| `report_context` | `GeostatisticsSimulationReportReportContext | None` | Context for the validation report. | `None` |
| `report_mean_thresholds` | `GeostatisticsSimulationReportReportMeanThresholds | None` | Thresholds for the mean comparison in the validation report. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeostatisticsSimulationReportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `validation_summary` | `GeostatisticsSimulationReportResultValidationSummary` | — | Summary of the validation results. |
| `validation_report` | `GeostatisticsSimulationReportResultResultValidationReport` | — | Link to the validation report file. |
| `links` | `GeostatisticsSimulationReportResultLinks` | — | Links related to the simulation report task. |

#### Nested Shapes

**`GeostatisticsSimulationReportNeighborhoodEllipsoidEllipsoidRanges`** (`TypedDict`)

The ranges of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `major` | `float` | Yes | The major axis length of the ellipsoid Bounds: `> 0`. |
| `semi_major` | `float` | Yes | The semi major axis length of the ellipsoid Bounds: `> 0`. |
| `minor` | `float` | Yes | The minor axis length of the ellipsoid Bounds: `> 0`. |

**`GeostatisticsSimulationReportRotation`** (`TypedDict`)

The rotation of the ellipsoid

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dip_azimuth` | `float` | No | first rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |
| `dip` | `float` | No | second rotation, about the x-axis, in degrees Bounds: `>= 0`, `<= 180`. |
| `pitch` | `float` | No | third rotation, about the z-axis, in degrees Bounds: `>= 0`, `<= 360`. |

**`GeostatisticsSimulationReportEllipsoid`** (`TypedDict`)

The ellipsoid, to search for points within
Alias: anisotropy

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid_ranges` | `GeostatisticsSimulationReportNeighborhoodEllipsoidEllipsoidRanges` | Yes | The ranges of the ellipsoid |
| `rotation` | `GeostatisticsSimulationReportRotation` | Yes | The rotation of the ellipsoid |

**`GeostatisticsSimulationReportNeighborhood`** (`TypedDict`)

Search neighborhood for conditioning.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `ellipsoid` | `GeostatisticsSimulationReportEllipsoid` | Yes | The ellipsoid, to search for points within Alias: anisotropy |
| `max_samples` | `int` | Yes | The maximum number of samples to use for each evaluation point. Alias: max-samples Bounds: `>= 1`. |
| `min_samples` | `int` | No | The minimum number of samples to use for each evaluation point. Alias: min-samples Bounds: `>= 1`. |

**`GeostatisticsSimulationReportBlockDiscretization`** (`TypedDict`)

Sub-block discretization used for support correction within each grid block.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `nx` | `int` | No | Number of subdivisions in the x direction Bounds: `>= 1`, `<= 9`. |
| `ny` | `int` | No | Number of subdivisions in the y direction Bounds: `>= 1`, `<= 9`. |
| `nz` | `int` | No | Number of subdivisions in the z direction Bounds: `>= 1`, `<= 9`. |

**`GeostatisticsSimulationReportDistributionTailExtrapolationUpper`** (`TypedDict`)

Power model for extrapolating the upper tail of the distribution.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `power` | `float` | Yes | Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive). Bounds: `> 0.0`, `<= 1.0`. |
| `max` | `float` | Yes | Maximum extent of tail, must be greater than the maximum value in the data. |

**`GeostatisticsSimulationReportDistributionTailExtrapolationLower`** (`TypedDict`)

Power model for extrapolating the lower tail of the distribution. Optional.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `power` | `float` | Yes | Denominator of the exponent for the power model (between 0 exclusive and 1 inclusive). Bounds: `> 0.0`, `<= 1.0`. |
| `min` | `float` | Yes | Minimum extent of tail, must be less than the minimum value in the data. |

**`GeostatisticsSimulationReportTailExtrapolation`** (`TypedDict`)

Tail extrapolation

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `upper` | `GeostatisticsSimulationReportDistributionTailExtrapolationUpper` | Yes | Power model for extrapolating the upper tail of the distribution. |
| `lower` | `GeostatisticsSimulationReportDistributionTailExtrapolationLower | None` | No | Power model for extrapolating the lower tail of the distribution. Optional. |

**`GeostatisticsSimulationReportContinuousDistributionParameters`** (`TypedDict`)

Parameters for a continuous distribution used for normal-score transformation.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `tail_extrapolation` | `GeostatisticsSimulationReportTailExtrapolation | None` | No | Tail extrapolation |
| `weights` | `AttributeInput | None` | No | Optional weights attribute for weighted distribution |

**`GeostatisticsSimulationReportCost`** (`TypedDict`)

Material cost parameters.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `processing` | `float` | Yes | Processing cost per unit of material. |
| `mining_ore` | `float` | Yes | Mining cost per unit of ore material. |
| `mining_waste` | `float` | Yes | Mining cost per unit of waste material. |
| `recovery_percentage` | `float` | Yes | Fraction of metal expected to be recovered during processing. |

**`GeostatisticsSimulationReportValidationReportMaterialCategory`** (`TypedDict`)

A material category with cutoff grade and price information.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | `str` | Yes | Name of the material category. |
| `cutoff` | `float` | Yes | Cutoff grade for the material category. |
| `min_price` | `float` | Yes | Minimum metal price for the material category. |

**`GeostatisticsSimulationReportMaterial`** (`TypedDict`)

Material cost and category information for loss calculations.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `cost` | `GeostatisticsSimulationReportCost` | Yes | Material cost parameters. |
| `categories` | `list[GeostatisticsSimulationReportValidationReportMaterialCategory]` | Yes | List of material categories. |

**`GeostatisticsSimulationReportValidationReportContextItem`** (`TypedDict`)

A single item of contextual information to be included in the validation report.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `label` | `str` | Yes | Label of the context item, which will appear on the report. |
| `value` | `str` | Yes | Value of the context item. |

**`GeostatisticsSimulationReportReportContext`** (`TypedDict`)

Context for the validation report.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `title` | `str` | Yes | Title that will appear on the report. |
| `details` | `list[GeostatisticsSimulationReportValidationReportContextItem]` | Yes | Key-value pairs of details to appear on the report. |

**`GeostatisticsSimulationReportReportMeanThresholds`** (`TypedDict`)

Thresholds for the mean comparison in the validation report.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `acceptable` | `float` | Yes | Threshold for the percentage difference between the reference and simulated means that is considered acceptable. Bounds: `>= 0.0`. |
| `marginal` | `float` | Yes | Threshold for the percentage difference between the reference and simulated means that is considered marginal. Bounds: `>= 0.0`. |

**`GeostatisticsSimulationReportResultValidationSummary`** (`ResultNode`)

Summary of the validation results.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference_mean` | `float` | — | Mean of the input values. |
| `mean` | `float` | — | Mean of the simulated values. |

**`GeostatisticsSimulationReportResultResultValidationReport`** (`ResultNode`)

Link to the validation report file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a file. |
| `name` | `str` | — | The name of the file, typically the terminal segment of the path. |

**`GeostatisticsSimulationReportResultLinks`** (`ResultNode`)

Links related to the simulation report task.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `dashboard` | `str | None` | — | Link to the validation dashboard. |

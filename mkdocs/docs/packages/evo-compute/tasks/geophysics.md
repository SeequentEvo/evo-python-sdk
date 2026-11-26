[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/geophysics)

# Geophysics Tasks

Discoverable compute tasks in the `geophysics` topic. Call any task below through `ComputeClient` as `await client.geophysics.<task>.run(...)`, or by name via `await client.arun("geophysics", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |
| `AttributeInput` | `str | AnyTypedAttribute` | A typed attribute, a JMESPath expression, or a bare attribute name. |
| `AnyTypedAttribute` | `Attribute | BlockModelAttribute | PendingAttribute | BlockModelPendingAttribute` | A typed attribute handle carrying both its parent geoscience object and attribute name. |

## Tasks

### gravity_terrain_correction

`client.geophysics.gravity_terrain_correction` · Task: `geophysics/gravity-terrain-correction` · Version: `2.2.7` · Preview (`preview=True` required)

Gravity terrain correction compute task for geophysical surveys

```python
await client.geophysics.gravity_terrain_correction.run(
    *,
    surfaces: ObjectInput,
    observation_points: ObjectInput,
    correction_radius: float,
    terrain_layer: GeophysicsGravityTerrainCorrectionTerrainLayer,
    water_layer: GeophysicsGravityTerrainCorrectionWaterLayer | None = ...,
    preview: bool = False,
) -> GeophysicsGravityTerrainCorrectionResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `surfaces` | `ObjectInput` | A 2D regular grid with topography values as attributes and (optionally, if a water layer is present) elevations of the top of the solid earth (the base of any water layer) Supported schemas: `regular-2d-grid/[>=1.2,<2]`. | _required_ |
| `observation_points` | `ObjectInput` | A point set with gravity stations for which terrain corrections will be calculated. Supported schemas: `pointset/[>=1.2,<2]`. | _required_ |
| `correction_radius` | `float` | Radius used for terrain correction [m]. Should ideally be small enough so that a horizontal circle of this radius around any observation point fits within the grid area. | _required_ |
| `terrain_layer` | `GeophysicsGravityTerrainCorrectionTerrainLayer` | Parameters for the terrain layer, including density and reference to elevation attribute. In the absence of a specified water layer, the terrain represents the topography. If a water layer is provided, the terrain represents the elevation of the top of the solid earth (the base of the water layer). | _required_ |
| `water_layer` | `GeophysicsGravityTerrainCorrectionWaterLayer | None` | Optional: Parameters for a water layer to be used in the model. If provided, the terrain layer represents the elevation of the top of the solid earth (the base of the water layer). At locations without water, terrain and water elevations should be identical. If not provided, no water layer is modeled. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeophysicsGravityTerrainCorrectionResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `target` | `GeophysicsGravityTerrainCorrectionResultResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | The object containing the computed terrain correction values. |

#### Nested Shapes

**`GeophysicsGravityTerrainCorrectionTerrainLayer`** (`TypedDict`)

Parameters for the terrain layer, including density and reference to elevation attribute. In the absence of a specified water layer, the terrain represents the topography. If a water layer is provided, the terrain represents the elevation of the top of the solid earth (the base of the water layer).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `density` | `float` | Yes | Density of the layer [kg/m^3] |
| `elevation` | `AttributeInput` | Yes | Reference to an attribute of a 2D grid representing the elevations of a layer defining a stratigraphic surface. Elevations [m] correspond to the same third coordinate axis as used in the observation points. |

**`GeophysicsGravityTerrainCorrectionWaterLayer`** (`TypedDict`)

Optional: Parameters for a water layer to be used in the model. If provided, the terrain layer represents the elevation of the top of the solid earth (the base of the water layer). At locations without water, terrain and water elevations should be identical. If not provided, no water layer is modeled.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `density` | `float` | Yes | Density of the layer [kg/m^3] |
| `elevation` | `AttributeInput` | Yes | Reference to an attribute of a 2D grid representing the elevations of a layer defining a stratigraphic surface. Elevations [m] correspond to the same third coordinate axis as used in the observation points. |

**`GeophysicsGravityTerrainCorrectionResultResultTargetTerrainCorrection`** (`ResultNode`)

Attribute containing the terrain correction associated with each location at the input observation points. The terrain correction corresponds to the vertical gravity component. Unit is in m/s² and corresponds to the coordinate system of the observation points where z-axis is positive upwards. This means that the terrain correction values should be multiplied by -1 if applied to gravity observations assuming a z-axis positive downwards.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeophysicsGravityTerrainCorrectionResultResultTarget`** (`ResultNode`)

The object containing the computed terrain correction values.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `terrain_correction` | `GeophysicsGravityTerrainCorrectionResultResultTargetTerrainCorrection` | `await node.terrain_correction.load()`, `await node.terrain_correction.to_dataframe()` | Attribute containing the terrain correction associated with each location at the input observation points. The terrain correction corresponds to the vertical gravity component. Unit is in m/s² and corresponds to the coordinate system of the observation points where z-axis is positive upwards. This means that the terrain correction values should be multiplied by -1 if applied to gravity observations assuming a z-axis positive downwards. |

### regular_2d_grid_to_trimesh

`client.geophysics.regular_2d_grid_to_trimesh` · Task: `geophysics/regular-2d-grid-to-trimesh` · Version: `2.2.7` · Preview (`preview=True` required)

Convert a regular 2D grid to a triangle mesh using a cell-centered or vertex-centered scalar attribute as Z values. Supports optional NaN filtering.

```python
await client.geophysics.regular_2d_grid_to_trimesh.run(
    *,
    source: GeophysicsRegular2dGridToTrimeshSourceInput | AnyTypedAttribute,
    target: GeophysicsRegular2dGridToTrimeshTarget,
    settings: GeophysicsRegular2dGridToTrimeshSettings = ...,
    preview: bool = False,
) -> GeophysicsRegular2dGridToTrimeshResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeophysicsRegular2dGridToTrimeshSourceInput | AnyTypedAttribute` | Source grid and attribute | _required_ |
| `target` | `GeophysicsRegular2dGridToTrimeshTarget` | Where to create the output Triangle Mesh object | _required_ |
| `settings` | `GeophysicsRegular2dGridToTrimeshSettings` | Optional settings for mesh generation | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeophysicsRegular2dGridToTrimeshResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `output` | `GeophysicsRegular2dGridToTrimeshResultResultOutput | None` | `await result.output.load()`, `await result.output.to_dataframe(*keys)` | Reference to the created Triangle Mesh object. |
| `summary` | `GeophysicsRegular2dGridToTrimeshResultSummary | None` | — | Summary statistics for the generated triangle mesh. |
| `has_nan` | `bool` | — | True if the source attribute contains NaN values. |
| `nan_count` | `int` | — | Number of NaN values in the source attribute. |
| `is_rotated` | `bool` | — | True if the source grid has a non-identity rotation. |

#### Nested Shapes

**`GeophysicsRegular2dGridToTrimeshSourceInput`** (`TypedDict`)

Source grid and attribute

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Input regular 2D grid containing the scalar attribute to use as Z values Supported schemas: `regular-2d-grid/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | Scalar attribute to use as Z values for the triangle mesh |

**`GeophysicsRegular2dGridToTrimeshTarget`** (`TypedDict`)

Where to create the output Triangle Mesh object

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeophysicsRegular2dGridToTrimeshSettings`** (`TypedDict`)

Optional settings for mesh generation

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `flip_triangles` | `bool` | No | Reverse triangle winding order (flip normals) |
| `drop_nan_patches` | `bool` | No | Exclude triangles that touch NaN attribute values |
| `use_centroids` | `bool` | No | Add a centroid vertex to each quad and split into 4 triangles instead of 2, producing a smoother surface at the cost of twice as many triangles |
| `triMeshVersion` | `Literal["2.1", "2.2"]` | No | Triangle mesh schema version for output payload. Use 2.1 for Oasis Montaj compatibility or 2.2 for latest schema. |

**`GeophysicsRegular2dGridToTrimeshResultResultOutput`** (`ResultNode`)

Reference to the created Triangle Mesh object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`GeophysicsRegular2dGridToTrimeshResultSummary`** (`ResultNode`)

Summary statistics for the generated triangle mesh.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `num_triangles` | `int` | — | Number of triangles in the generated mesh |
| `num_vertices` | `int` | — | Number of vertices in the mesh vertex array |
| `has_unused_vertices` | `bool` | — | Whether the mesh has orphan vertices not referenced by any triangle |
| `has_non_manifold_edges` | `bool` | — | Always false — splitting a regular grid into triangles cannot produce edges shared by more than 2 triangles |

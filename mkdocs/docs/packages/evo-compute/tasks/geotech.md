[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/geotech)

# Geotechnical Tasks

Discoverable compute tasks in the `geotech` topic. Call any task below through `ComputeClient` as `await client.geotech.<task>.run(...)` (or `submit(...)` for a non-blocking `TaskJob`), or by name via `await client.arun("geotech", "<task>", parameters)` / `await client.asubmit("geotech", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |

## Tasks

### add_to_intersected_geometry

`client.geotech.add_to_intersected_geometry` · Task: `geotech/add-to-intersected-geometry` · Version: `1.6.3` · Preview (`preview=True` required)

Adds one or more triangle mesh objects to an existing intersected Geological Model Meshes object.

```python
await client.geotech.add_to_intersected_geometry.run(
    *,
    source: ObjectInput,
    meshes_to_add: list[ObjectInput],
    target: GeotechAddToIntersectedGeometryTarget,
    tolerance: float,
    min_point: GeotechAddToIntersectedGeometryPoint3D = ...,
    max_point: GeotechAddToIntersectedGeometryPoint3D2 = ...,
    rotation_angle: float = ...,
    coordinate_reference_system: str | None = ...,
    preview: bool = False,
) -> GeotechAddToIntersectedGeometryResult

await client.geotech.add_to_intersected_geometry.submit(
    *,
    source: ObjectInput,
    meshes_to_add: list[ObjectInput],
    target: GeotechAddToIntersectedGeometryTarget,
    tolerance: float,
    min_point: GeotechAddToIntersectedGeometryPoint3D = ...,
    max_point: GeotechAddToIntersectedGeometryPoint3D2 = ...,
    rotation_angle: float = ...,
    coordinate_reference_system: str | None = ...,
    preview: bool = False,
) -> TaskJob[GeotechAddToIntersectedGeometryResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The existing intersected geometry (Geological Model Meshes) to add meshes to. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `meshes_to_add` | `list[ObjectInput]` | List of triangle mesh objects to add to the intersected geometry. | _required_ |
| `target` | `GeotechAddToIntersectedGeometryTarget` | Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `tolerance` | `float` | Tolerance distance for the intersection operation. Bounds: `> 0`. | _required_ |
| `min_point` | `GeotechAddToIntersectedGeometryPoint3D` | Minimum bounding point to clip to. | `None` |
| `max_point` | `GeotechAddToIntersectedGeometryPoint3D2` | Maximum bounding point to clip to. | `None` |
| `rotation_angle` | `float` | Rotation angle around the bounds center. | `0.0` |
| `coordinate_reference_system` | `str | None` | The coordinate reference system that min_point and max_point are defined in. Specified as either an EPSG code string (e.g., 'EPSG:32611') or a WKT2 coordinate reference system definition string. Use the same CRS and bounds that were used to initialize the intersected geometry. If not provided, the CRS will be marked as unspecified. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechAddToIntersectedGeometryResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechAddToIntersectedGeometryResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new Geological Model Meshes object with added meshes. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechAddToIntersectedGeometryTarget`** (`TypedDict`)

Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechAddToIntersectedGeometryPoint3D`** (`TypedDict`)

Minimum bounding point to clip to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechAddToIntersectedGeometryPoint3D2`** (`TypedDict`)

Maximum bounding point to clip to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechAddToIntersectedGeometryResultResultObject`** (`ResultNode`)

The new Geological Model Meshes object with added meshes.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### calculate_mesh_quality_histogram

`client.geotech.calculate_mesh_quality_histogram` · Task: `geotech/calculate-mesh-quality-histogram` · Version: `1.6.3` · Preview (`preview=True` required)

Calculates a normalized inverse condition number histogram for triangles in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.calculate_mesh_quality_histogram.run(
    *,
    source: ObjectInput,
    bin_width: float = ...,
    preview: bool = False,
) -> GeotechCalculateMeshQualityHistogramResult

await client.geotech.calculate_mesh_quality_histogram.submit(
    *,
    source: ObjectInput,
    bin_width: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechCalculateMeshQualityHistogramResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to analyze. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `bin_width` | `float` | Width of each quality bin. Bins span the quality range from 0.0 to 1.0. Bounds: `> 0`, `<= 1`. | `0.1` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechCalculateMeshQualityHistogramResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `bins` | `list[GeotechCalculateMeshQualityHistogramResultResultBinsItem]` | — | Ordered quality bins spanning the range from 0.0 to 1.0. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechCalculateMeshQualityHistogramResultResultBinsItem`** (`ResultNode`)

A mesh quality range.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `lower_bound` | `float` | — | Inclusive lower bound of the quality range. |
| `upper_bound` | `float` | — | Upper bound of the quality range. |
| `triangle_count` | `int` | — | Number of triangles in this quality range. Bounds: `>= 0`. |

### check_triangle_orientation

`client.geotech.check_triangle_orientation` · Task: `geotech/check-triangle-orientation` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies triangles with inconsistent normal orientation relative to their neighbors in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.check_triangle_orientation.run(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> GeotechCheckTriangleOrientationResult

await client.geotech.check_triangle_orientation.submit(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> TaskJob[GeotechCheckTriangleOrientationResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for inconsistently oriented triangles. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechCheckTriangleOrientationResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `triangles` | `list[int]` | — | Indices of triangles with inconsistent orientation relative to their neighbors. |
| `message` | `str` | — | Description of the result of the operation. |

### copy_geological_model_meshes

`client.geotech.copy_geological_model_meshes` · Task: `geotech/copy-geological-model-meshes` · Version: `1.6.3` · Preview (`preview=True` required)

Copies an existing set of geological model meshes, optionally removing external volumes from the result.

```python
await client.geotech.copy_geological_model_meshes.run(
    *,
    source: ObjectInput,
    target: GeotechCopyGeologicalModelMeshesTarget,
    remove_external_volumes: bool = ...,
    preview: bool = False,
) -> GeotechCopyGeologicalModelMeshesResult

await client.geotech.copy_geological_model_meshes.submit(
    *,
    source: ObjectInput,
    target: GeotechCopyGeologicalModelMeshesTarget,
    remove_external_volumes: bool = ...,
    preview: bool = False,
) -> TaskJob[GeotechCopyGeologicalModelMeshesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The source Geological Model Meshes object to copy. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechCopyGeologicalModelMeshesTarget` | Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `remove_external_volumes` | `bool` | Whether to remove external volumes from the result. | `False` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechCopyGeologicalModelMeshesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechCopyGeologicalModelMeshesResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new Geological Model Meshes object. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechCopyGeologicalModelMeshesTarget`** (`TypedDict`)

Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechCopyGeologicalModelMeshesResultResultObject`** (`ResultNode`)

The new Geological Model Meshes object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### create_cross_section

`client.geotech.create_cross_section` · Task: `geotech/create-cross-section` · Version: `1.6.3` · Preview (`preview=True` required)

Intersects parametric volumes in a 3D Design Geometry with a vertical plane defined by a section origin and dip azimuth, producing a 2D Design Geometry with polyline boundaries.

```python
await client.geotech.create_cross_section.run(
    *,
    source: ObjectInput,
    target: GeotechCreateCrossSectionTarget,
    section_origin: GeotechCreateCrossSectionPoint3D,
    section_dip_azimuth_degrees: float,
    preview: bool = False,
) -> GeotechCreateCrossSectionResult

await client.geotech.create_cross_section.submit(
    *,
    source: ObjectInput,
    target: GeotechCreateCrossSectionTarget,
    section_origin: GeotechCreateCrossSectionPoint3D,
    section_dip_azimuth_degrees: float,
    preview: bool = False,
) -> TaskJob[GeotechCreateCrossSectionResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The input 3D Design Geometry (Domain3D) containing parameterized volumes. Supports Design Geometry schema version 1.1+. Supported schemas: `design-geometry/[>=1.0,<2]`. | _required_ |
| `target` | `GeotechCreateCrossSectionTarget` | Target specifying where to create the 2D cross-section Design Geometry result. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. | _required_ |
| `section_origin` | `GeotechCreateCrossSectionPoint3D` | Origin point of the cross-section plane. | _required_ |
| `section_dip_azimuth_degrees` | `float` | Azimuth angle in degrees for the cross-section plane. Bounds: `>= 0.0`, `< 360.0`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechCreateCrossSectionResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechCreateCrossSectionResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The 2D cross-section design geometry output. |
| `message` | `str` | — | Summary of the task outcome. |

#### Nested Shapes

**`GeotechCreateCrossSectionTarget`** (`TypedDict`)

Target specifying where to create the 2D cross-section Design Geometry result. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `design-geometry/[>=1.0,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechCreateCrossSectionPoint3D`** (`TypedDict`)

Origin point of the cross-section plane.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechCreateCrossSectionResultResultObject`** (`ResultNode`)

The 2D cross-section design geometry output.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `design-geometry/[>=1.0,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### create_geological_model_meshes

`client.geotech.create_geological_model_meshes` · Task: `geotech/create-geological-model-meshes` · Version: `1.6.3` · Preview (`preview=True` required)

Decomposes an intersected mesh or existing GMM into per-volume geological model meshes, optionally removing the top (external) volume.

```python
await client.geotech.create_geological_model_meshes.run(
    *,
    source: ObjectInput,
    target: GeotechCreateGeologicalModelMeshesTarget,
    preview: bool = False,
) -> GeotechCreateGeologicalModelMeshesResult

await client.geotech.create_geological_model_meshes.submit(
    *,
    source: ObjectInput,
    target: GeotechCreateGeologicalModelMeshesTarget,
    preview: bool = False,
) -> TaskJob[GeotechCreateGeologicalModelMeshesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The Geological Model Meshes object to process. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechCreateGeologicalModelMeshesTarget` | Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechCreateGeologicalModelMeshesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechCreateGeologicalModelMeshesResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new Geological Model Meshes object. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechCreateGeologicalModelMeshesTarget`** (`TypedDict`)

Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechCreateGeologicalModelMeshesResultResultObject`** (`ResultNode`)

The new Geological Model Meshes object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### detect_degenerate_triangles

`client.geotech.detect_degenerate_triangles` · Task: `geotech/detect-degenerate-triangles` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies needle and cap triangles in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_degenerate_triangles.run(
    *,
    source: ObjectInput,
    needle_threshold_ratio: float = ...,
    cap_min_angle_degrees: float = ...,
    preview: bool = False,
) -> GeotechDetectDegenerateTrianglesResult

await client.geotech.detect_degenerate_triangles.submit(
    *,
    source: ObjectInput,
    needle_threshold_ratio: float = ...,
    cap_min_angle_degrees: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechDetectDegenerateTrianglesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for degenerate triangles. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `needle_threshold_ratio` | `float` | Longest-to-shortest edge ratio threshold for needle detection. Bounds: `> 0`. | `1000.0` |
| `cap_min_angle_degrees` | `float` | Minimum angle in degrees for cap detection. Bounds: `> 0`, `< 180`. | `177.5` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectDegenerateTrianglesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `triangles` | `list[int]` | — | Indices of detected degenerate triangles. |
| `message` | `str` | — | Description of the result of the operation. |

### detect_duplicate_points

`client.geotech.detect_duplicate_points` · Task: `geotech/detect-duplicate-points` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies exact duplicate vertices in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_duplicate_points.run(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> GeotechDetectDuplicatePointsResult

await client.geotech.detect_duplicate_points.submit(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> TaskJob[GeotechDetectDuplicatePointsResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for duplicate points. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectDuplicatePointsResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `points` | `list[int]` | — | Indices of duplicate vertices (excluding first occurrence). |
| `message` | `str` | — | Description of the result of the operation. |

### detect_duplicate_triangles

`client.geotech.detect_duplicate_triangles` · Task: `geotech/detect-duplicate-triangles` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies duplicate triangles (same vertex indices regardless of order) in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_duplicate_triangles.run(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> GeotechDetectDuplicateTrianglesResult

await client.geotech.detect_duplicate_triangles.submit(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> TaskJob[GeotechDetectDuplicateTrianglesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for duplicate triangles. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectDuplicateTrianglesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `triangles` | `list[int]` | — | Indices of detected duplicate triangles (excluding first occurrence). |
| `message` | `str` | — | Description of the result of the operation. |

### detect_fins

`client.geotech.detect_fins` · Task: `geotech/detect-fins` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies fin-like connected components without closed boundaries in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_fins.run(
    *,
    source: ObjectInput,
    max_area_ratio: float = ...,
    preview: bool = False,
) -> GeotechDetectFinsResult

await client.geotech.detect_fins.submit(
    *,
    source: ObjectInput,
    max_area_ratio: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechDetectFinsResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for fins. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `max_area_ratio` | `float` | Components with surface area larger than the largest component times this ratio are ignored. Bounds: `> 0`. | `1.0` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectFinsResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `fins` | `list[list[int]]` | — | Each fin is a list of triangle indices forming a connected component without a closed boundary. |
| `message` | `str` | — | Description of the result of the operation. |

### detect_holes

`client.geotech.detect_holes` · Task: `geotech/detect-holes` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies holes in a Triangle Mesh or Geological Model Meshes object based on boundary edge analysis.

```python
await client.geotech.detect_holes.run(
    *,
    source: ObjectInput,
    hole_size_ratio_tolerance: float = ...,
    preview: bool = False,
) -> GeotechDetectHolesResult

await client.geotech.detect_holes.submit(
    *,
    source: ObjectInput,
    hole_size_ratio_tolerance: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechDetectHolesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for holes. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `hole_size_ratio_tolerance` | `float` | A boundary line is considered a hole if the ratio of the hole bounding box diagonal to the mesh bounding box diagonal is smaller than this tolerance. Bounds: `> 0`, `<= 1`. | `0.05` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectHolesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `holes` | `list[list[list[int]]]` | — | Each hole is a list of boundary edges, where each edge is a pair of vertex indices. |
| `message` | `str` | — | Description of the result of the operation. |

### detect_non_manifold_edges

`client.geotech.detect_non_manifold_edges` · Task: `geotech/detect-non-manifold-edges` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies edges shared by more than two triangles in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_non_manifold_edges.run(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> GeotechDetectNonManifoldEdgesResult

await client.geotech.detect_non_manifold_edges.submit(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> TaskJob[GeotechDetectNonManifoldEdgesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for non-manifold edges. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectNonManifoldEdgesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `edges` | `list[list[int]]` | — | Non-manifold edges, each as a pair of vertex indices. |
| `message` | `str` | — | Description of the result of the operation. |

### detect_non_manifold_vertices

`client.geotech.detect_non_manifold_vertices` · Task: `geotech/detect-non-manifold-vertices` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies vertices with non-manifold topology in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_non_manifold_vertices.run(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> GeotechDetectNonManifoldVerticesResult

await client.geotech.detect_non_manifold_vertices.submit(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> TaskJob[GeotechDetectNonManifoldVerticesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for non-manifold vertices. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectNonManifoldVerticesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `points` | `list[int]` | — | Indices of vertices with non-manifold topology. |
| `message` | `str` | — | Description of the result of the operation. |

### detect_non_partitioning_surfaces

`client.geotech.detect_non_partitioning_surfaces` · Task: `geotech/detect-non-partitioning-surfaces` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies non-partitioning surfaces that are neither fins nor surfaces with holes in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_non_partitioning_surfaces.run(
    *,
    source: ObjectInput,
    max_area_ratio: float = ...,
    hole_size_ratio_tolerance: float = ...,
    preview: bool = False,
) -> GeotechDetectNonPartitioningSurfacesResult

await client.geotech.detect_non_partitioning_surfaces.submit(
    *,
    source: ObjectInput,
    max_area_ratio: float = ...,
    hole_size_ratio_tolerance: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechDetectNonPartitioningSurfacesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for non-partitioning surfaces. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `max_area_ratio` | `float` | Components with surface area larger than the largest component times this ratio are ignored. Bounds: `> 0`. | `1.0` |
| `hole_size_ratio_tolerance` | `float` | A boundary line is considered a hole if the ratio of the hole bounding box diagonal to the mesh bounding box diagonal is smaller than this tolerance. Bounds: `> 0`, `<= 1`. | `0.05` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectNonPartitioningSurfacesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `surfaces` | `list[GeotechDetectNonPartitioningSurfacesResultNonPartitioningSurfaceResult]` | — | Detected non-partitioning surfaces with their triangles and edges. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechDetectNonPartitioningSurfacesResultNonPartitioningSurfaceResult`** (`ResultNode`)

A single non-partitioning surface.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `triangles` | `list[int]` | — | Indices of triangles forming this surface. |
| `edges` | `list[list[int]]` | — | Boundary edges of this surface, each as a pair of vertex indices. |

### detect_self_intersections

`client.geotech.detect_self_intersections` · Task: `geotech/detect-self-intersections` · Version: `1.6.3` · Preview (`preview=True` required)

Identifies self-intersecting triangles in a Triangle Mesh or Geological Model Meshes object.

```python
await client.geotech.detect_self_intersections.run(
    *,
    source: ObjectInput,
    tolerance: float = ...,
    preview: bool = False,
) -> GeotechDetectSelfIntersectionsResult

await client.geotech.detect_self_intersections.submit(
    *,
    source: ObjectInput,
    tolerance: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechDetectSelfIntersectionsResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to check for self-intersections. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `tolerance` | `float` | Tolerance for self-intersection detection. Bounds: `>= 0`. | `1e-06` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechDetectSelfIntersectionsResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `triangles` | `list[int]` | — | Indices of triangles involved in self-intersections. |
| `message` | `str` | — | Description of the result of the operation. |

### fill_holes

`client.geotech.fill_holes` · Task: `geotech/fill-holes` · Version: `1.6.3` · Preview (`preview=True` required)

Fills holes in the triangle mesh of a GMM object using specified edge identifiers and a configurable fill mode strategy.

```python
await client.geotech.fill_holes.run(
    *,
    source: ObjectInput,
    target: GeotechFillHolesTarget,
    edges: list[list[int]],
    fill_mode_selection: Literal["Fill", "FillAndRefine", "FillRefineAndFair"],
    preview: bool = False,
) -> GeotechFillHolesResult

await client.geotech.fill_holes.submit(
    *,
    source: ObjectInput,
    target: GeotechFillHolesTarget,
    edges: list[list[int]],
    fill_mode_selection: Literal["Fill", "FillAndRefine", "FillRefineAndFair"],
    preview: bool = False,
) -> TaskJob[GeotechFillHolesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to process. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechFillHolesTarget` | Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. | _required_ |
| `edges` | `list[list[int]]` | List of edges that identify holes to fill. Each edge is a pair of vertex indices. One edge per hole is sufficient to identify it. | _required_ |
| `fill_mode_selection` | `Literal["Fill", "FillAndRefine", "FillRefineAndFair"]` | The hole filling strategy to use. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechFillHolesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechFillHolesResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new mesh object with holes filled. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechFillHolesTarget`** (`TypedDict`)

Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechFillHolesResultResultObject`** (`ResultNode`)

The new mesh object with holes filled.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### fix_self_intersections

`client.geotech.fix_self_intersections` · Task: `geotech/fix-self-intersections` · Version: `1.6.3` · Preview (`preview=True` required)

Fixes self-intersections in the triangle mesh of a GMM object using configurable intersection detection settings.

```python
await client.geotech.fix_self_intersections.run(
    *,
    source: ObjectInput,
    target: GeotechFixSelfIntersectionsTarget,
    intersect_settings: GeotechFixSelfIntersectionsIntersectSettings,
    preview: bool = False,
) -> GeotechFixSelfIntersectionsResult

await client.geotech.fix_self_intersections.submit(
    *,
    source: ObjectInput,
    target: GeotechFixSelfIntersectionsTarget,
    intersect_settings: GeotechFixSelfIntersectionsIntersectSettings,
    preview: bool = False,
) -> TaskJob[GeotechFixSelfIntersectionsResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to process. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechFixSelfIntersectionsTarget` | Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. | _required_ |
| `intersect_settings` | `GeotechFixSelfIntersectionsIntersectSettings` | Settings that control the intersection detection behaviour. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechFixSelfIntersectionsResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechFixSelfIntersectionsResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new mesh object with self-intersections fixed. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechFixSelfIntersectionsTarget`** (`TypedDict`)

Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechFixSelfIntersectionsIntersectSettings`** (`TypedDict`)

Settings that control the intersection detection behaviour.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `assume_no_self_intersections_in_parts` | `bool` | No | When true, skip checking each part for self-intersections and only check between parts. |
| `tolerance` | `float` | No | Numerical tolerance used when detecting intersections. Bounds: `> 0`. |

**`GeotechFixSelfIntersectionsResultResultObject`** (`ResultNode`)

The new mesh object with self-intersections fixed.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### init_aggregate_geom_gmm

`client.geotech.init_aggregate_geom_gmm` · Task: `geotech/init-aggregate-geom-gmm` · Version: `1.6.3` · Preview (`preview=True` required)

Creates a rotated bounding box as a Geological Model Meshes object, used as the initial intersected geometry when creating a model.

```python
await client.geotech.init_aggregate_geom_gmm.run(
    *,
    target: GeotechInitAggregateGeomGmmTarget,
    min_point: GeotechInitAggregateGeomGmmPoint3D,
    max_point: GeotechInitAggregateGeomGmmPoint3D2,
    rotation_angle: float = ...,
    coordinate_reference_system: str | None = ...,
    preview: bool = False,
) -> GeotechInitAggregateGeomGmmResult

await client.geotech.init_aggregate_geom_gmm.submit(
    *,
    target: GeotechInitAggregateGeomGmmTarget,
    min_point: GeotechInitAggregateGeomGmmPoint3D,
    max_point: GeotechInitAggregateGeomGmmPoint3D2,
    rotation_angle: float = ...,
    coordinate_reference_system: str | None = ...,
    preview: bool = False,
) -> TaskJob[GeotechInitAggregateGeomGmmResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `target` | `GeotechInitAggregateGeomGmmTarget` | Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `min_point` | `GeotechInitAggregateGeomGmmPoint3D` | The minimum corner of the bounding box. | _required_ |
| `max_point` | `GeotechInitAggregateGeomGmmPoint3D2` | The maximum corner of the bounding box. | _required_ |
| `rotation_angle` | `float` | The rotation angle for the bounding box, in radians. | `0.0` |
| `coordinate_reference_system` | `str | None` | The coordinate reference system that the other parameters are defined in. Specified as either an EPSG code string (e.g., 'EPSG:32611') or a WKT2 coordinate reference system definition string. If not provided, the CRS will be marked as unspecified. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechInitAggregateGeomGmmResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechInitAggregateGeomGmmResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new Geological Model Meshes object. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechInitAggregateGeomGmmTarget`** (`TypedDict`)

Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechInitAggregateGeomGmmPoint3D`** (`TypedDict`)

The minimum corner of the bounding box.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechInitAggregateGeomGmmPoint3D2`** (`TypedDict`)

The maximum corner of the bounding box.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechInitAggregateGeomGmmResultResultObject`** (`ResultNode`)

The new Geological Model Meshes object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### local_remesh

`client.geotech.local_remesh` · Task: `geotech/local-remesh` · Version: `1.6.3` · Preview (`preview=True` required)

Locally remeshes the region around selected seed triangles in a Geological Model Meshes object using configurable remeshing parameters and radius-based region growth.

```python
await client.geotech.local_remesh.run(
    *,
    source: ObjectInput,
    target: GeotechLocalRemeshTarget,
    seed_triangle_indices: list[int],
    remesh_settings: GeotechLocalRemeshRemeshSettings = ...,
    radius: float = ...,
    preview: bool = False,
) -> GeotechLocalRemeshResult

await client.geotech.local_remesh.submit(
    *,
    source: ObjectInput,
    target: GeotechLocalRemeshTarget,
    seed_triangle_indices: list[int],
    remesh_settings: GeotechLocalRemeshRemeshSettings = ...,
    radius: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechLocalRemeshResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to process. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechLocalRemeshTarget` | Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. | _required_ |
| `seed_triangle_indices` | `list[int]` | Triangle indices used as seeds for the local remeshing region. | _required_ |
| `remesh_settings` | `GeotechLocalRemeshRemeshSettings` | Settings that control the underlying remeshing algorithm. | `None` |
| `radius` | `float` | Flood-fill radius around the seed triangles used to select the local remeshing region. Bounds: `>= 0`. | `0.0` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechLocalRemeshResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechLocalRemeshResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new mesh object with the local remeshing applied. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechLocalRemeshTarget`** (`TypedDict`)

Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechLocalRemeshRemeshSettings`** (`TypedDict`)

Settings that control the underlying remeshing algorithm.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `patch_angle_tolerance` | `float` | No | Angle change in degrees used to define a patch skeleton line. Bounds: `> 0`. |
| `max_chordal_error` | `float` | No | Maximum relative local error. -0.005 corresponds to 0.5%. |
| `strain_tolerance` | `float` | No | Relative quality difference before and after unfolding to a plane. Bounds: `>= 0`. |
| `is_closed` | `bool` | No | Whether the mesh is closed; should be true for manifold GMVs. |
| `initial_cleanup` | `bool` | No | Whether to run initial cleanup such as node merging, gap filling, and topological fixing. |
| `line_proximity_detection` | `bool` | No | Whether to consider distance between skeleton lines when sizing elements. |
| `optimize` | `bool` | No | Whether to perform final optimization. |
| `shape_quality_weight` | `float` | No | Size-versus-shape weight from 0 to 1. Bounds: `>= 0`. |
| `target_h` | `float` | No | Target element size. Zero bases the size on skeleton lines. Bounds: `>= 0`. |
| `regularize_mode` | `bool` | No | If true, do not remesh; only regularize the mesh. |
| `relax_chunk_interfaces` | `bool` | No | Whether to relax chunk interfaces during remeshing. |

**`GeotechLocalRemeshResultResultObject`** (`ResultNode`)

The new mesh object with the local remeshing applied.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### make_gmm_into_intersected_geometry

`client.geotech.make_gmm_into_intersected_geometry` · Task: `geotech/make-gmm-into-intersected-geometry` · Version: `1.6.3` · Preview (`preview=True` required)

Clips a Geological Model Meshes object and processes it to create an intersected geometry mesh.

```python
await client.geotech.make_gmm_into_intersected_geometry.run(
    *,
    source: ObjectInput,
    target: GeotechMakeGmmIntoIntersectedGeometryTarget,
    clip_tolerance: float,
    min_point: GeotechMakeGmmIntoIntersectedGeometryPoint3D = ...,
    max_point: GeotechMakeGmmIntoIntersectedGeometryPoint3D2 = ...,
    rotation_angle: float = ...,
    collapse_settings: GeotechMakeGmmIntoIntersectedGeometryCollapseOverlapsSettings = ...,
    attempt_conformalization: bool = ...,
    attempt_auto_fix_on_failure: bool = ...,
    intersection_tolerance_for_auto_fix: float = ...,
    remesh_settings: GeotechMakeGmmIntoIntersectedGeometryRemeshSettings = ...,
    preview: bool = False,
) -> GeotechMakeGmmIntoIntersectedGeometryResult

await client.geotech.make_gmm_into_intersected_geometry.submit(
    *,
    source: ObjectInput,
    target: GeotechMakeGmmIntoIntersectedGeometryTarget,
    clip_tolerance: float,
    min_point: GeotechMakeGmmIntoIntersectedGeometryPoint3D = ...,
    max_point: GeotechMakeGmmIntoIntersectedGeometryPoint3D2 = ...,
    rotation_angle: float = ...,
    collapse_settings: GeotechMakeGmmIntoIntersectedGeometryCollapseOverlapsSettings = ...,
    attempt_conformalization: bool = ...,
    attempt_auto_fix_on_failure: bool = ...,
    intersection_tolerance_for_auto_fix: float = ...,
    remesh_settings: GeotechMakeGmmIntoIntersectedGeometryRemeshSettings = ...,
    preview: bool = False,
) -> TaskJob[GeotechMakeGmmIntoIntersectedGeometryResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The Geological Model Meshes object to process. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechMakeGmmIntoIntersectedGeometryTarget` | Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `clip_tolerance` | `float` | Clip tolerance distance. Bounds: `> 0`. | _required_ |
| `min_point` | `GeotechMakeGmmIntoIntersectedGeometryPoint3D` | Minimum bounding point to clip to. | `None` |
| `max_point` | `GeotechMakeGmmIntoIntersectedGeometryPoint3D2` | Maximum bounding point to clip to. | `None` |
| `rotation_angle` | `float` | Rotation angle around the bounds center. | `0.0` |
| `collapse_settings` | `GeotechMakeGmmIntoIntersectedGeometryCollapseOverlapsSettings` | Settings for collapsing overlapping mesh regions. | `None` |
| `attempt_conformalization` | `bool` | Whether to attempt conformalization of the intersected geometry. | `False` |
| `attempt_auto_fix_on_failure` | `bool` | Whether to attempt automatic fix on failure. | `False` |
| `intersection_tolerance_for_auto_fix` | `float` | Intersection tolerance for automatic fix. Bounds: `> 0`. | `0.001` |
| `remesh_settings` | `GeotechMakeGmmIntoIntersectedGeometryRemeshSettings` | Settings for remeshing during automatic fix. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechMakeGmmIntoIntersectedGeometryResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechMakeGmmIntoIntersectedGeometryResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new Geological Model Meshes object with intersected geometry. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechMakeGmmIntoIntersectedGeometryTarget`** (`TypedDict`)

Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechMakeGmmIntoIntersectedGeometryPoint3D`** (`TypedDict`)

Minimum bounding point to clip to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechMakeGmmIntoIntersectedGeometryPoint3D2`** (`TypedDict`)

Maximum bounding point to clip to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechMakeGmmIntoIntersectedGeometryCollapseOverlapsSettings`** (`TypedDict`)

Settings for collapsing overlapping mesh regions.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `distance_tolerance` | `float` | No | Distance tolerance for collapse, relative to the plane of one of the triangles. Bounds: `> 0`. |
| `overlap_threshold` | `float` | No | Overlap threshold fraction. Bounds: `> 0`, `<= 1`. |

**`GeotechMakeGmmIntoIntersectedGeometryRemeshSettings`** (`TypedDict`)

Settings for remeshing during automatic fix.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `patch_angle_tolerance` | `float` | No | Angle change in degrees used to define a patch skeleton line. Bounds: `> 0`. |
| `max_chordal_error` | `float` | No | Maximum relative local error. -0.005 corresponds to 0.5%. |
| `strain_tolerance` | `float` | No | Relative quality difference before and after unfolding to a plane. Bounds: `>= 0`. |
| `is_closed` | `bool` | No | Whether the mesh is closed; should be true for manifold GMVs. |
| `initial_cleanup` | `bool` | No | Whether to run initial cleanup such as node merging, gap filling, and topological fixing. |
| `line_proximity_detection` | `bool` | No | Whether to consider distance between skeleton lines when sizing elements. |
| `optimize` | `bool` | No | Whether to perform final optimization. |
| `shape_quality_weight` | `float` | No | Size-versus-shape weight from 0 to 1. Bounds: `>= 0`. |
| `target_h` | `float` | No | Target element size. Zero bases the size on skeleton lines. Bounds: `>= 0`. |
| `regularize_mode` | `bool` | No | If true, do not remesh; only regularize the mesh. |
| `relax_chunk_interfaces` | `bool` | No | Whether to relax chunk interfaces during remeshing. |

**`GeotechMakeGmmIntoIntersectedGeometryResultResultObject`** (`ResultNode`)

The new Geological Model Meshes object with intersected geometry.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### make_into_model_input

`client.geotech.make_into_model_input` · Task: `geotech/make-into-model-input` · Version: `1.6.3` · Preview (`preview=True` required)

Clips a triangle mesh loosely to the specified bounds, merges duplicate points, and translates the result so it can be used as model input.

```python
await client.geotech.make_into_model_input.run(
    *,
    source: ObjectInput,
    target: GeotechMakeIntoModelInputTarget,
    min_point: GeotechMakeIntoModelInputMinPoint = ...,
    max_point: GeotechMakeIntoModelInputMaxPoint = ...,
    rotation_angle: float = ...,
    tolerance: float = ...,
    preview: bool = False,
) -> GeotechMakeIntoModelInputResult

await client.geotech.make_into_model_input.submit(
    *,
    source: ObjectInput,
    target: GeotechMakeIntoModelInputTarget,
    min_point: GeotechMakeIntoModelInputMinPoint = ...,
    max_point: GeotechMakeIntoModelInputMaxPoint = ...,
    rotation_angle: float = ...,
    tolerance: float = ...,
    preview: bool = False,
) -> TaskJob[GeotechMakeIntoModelInputResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The triangle mesh object to prepare for use in a model. Supported schemas: `triangle-mesh/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechMakeIntoModelInputTarget` | Target specifying where to create the new Triangle Mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `min_point` | `GeotechMakeIntoModelInputMinPoint` | Minimum bounding point to clip the triangle mesh to. | `None` |
| `max_point` | `GeotechMakeIntoModelInputMaxPoint` | Maximum bounding point to clip the triangle mesh to. | `None` |
| `rotation_angle` | `float` | Rotation angle, in radians, used when evaluating the clipping bounds. | `0.0` |
| `tolerance` | `float` | Tolerance used when clipping the mesh and merging duplicate points. Bounds: `> 0`. | `1e-12` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechMakeIntoModelInputResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechMakeIntoModelInputResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The prepared Triangle Mesh object. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechMakeIntoModelInputTarget`** (`TypedDict`)

Target specifying where to create the new Triangle Mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechMakeIntoModelInputMinPoint`** (`TypedDict`)

Minimum bounding point to clip the triangle mesh to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechMakeIntoModelInputMaxPoint`** (`TypedDict`)

Maximum bounding point to clip the triangle mesh to.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`GeotechMakeIntoModelInputResultResultObject`** (`ResultNode`)

The prepared Triangle Mesh object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### mark_volumes_for_removal

`client.geotech.mark_volumes_for_removal` · Task: `geotech/mark-volumes-for-removal` · Version: `1.6.3` · Preview (`preview=True` required)

Marks specified volumes in a Geological Model Meshes for removal by setting their material to external, or clears the marking.

```python
await client.geotech.mark_volumes_for_removal.run(
    *,
    source: ObjectInput,
    target: GeotechMarkVolumesForRemovalTarget,
    settings: GeotechMarkVolumesForRemovalSettings,
    preview: bool = False,
) -> GeotechMarkVolumesForRemovalResult

await client.geotech.mark_volumes_for_removal.submit(
    *,
    source: ObjectInput,
    target: GeotechMarkVolumesForRemovalTarget,
    settings: GeotechMarkVolumesForRemovalSettings,
    preview: bool = False,
) -> TaskJob[GeotechMarkVolumesForRemovalResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The source Geological Model Meshes object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechMarkVolumesForRemovalTarget` | Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `settings` | `GeotechMarkVolumesForRemovalSettings` | Settings for marking volumes for removal. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechMarkVolumesForRemovalResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechMarkVolumesForRemovalResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new Geological Model Meshes object with volumes marked for removal. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechMarkVolumesForRemovalTarget`** (`TypedDict`)

Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechMarkVolumesForRemovalSettings`** (`TypedDict`)

Settings for marking volumes for removal.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `volume_indices` | `list[int]` | Yes | Indices of volumes to mark or unmark for removal. |
| `should_remove` | `bool` | No | When true, mark the specified volumes for removal. When false, clear the removal marking. |

**`GeotechMarkVolumesForRemovalResultResultObject`** (`ResultNode`)

The new Geological Model Meshes object with volumes marked for removal.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### parameterize_volumes

`client.geotech.parameterize_volumes` · Task: `geotech/parameterize-volumes` · Version: `1.6.3` · Preview (`preview=True` required)

Parameterizes closed volumes from Triangle Mesh or Geological Model Meshes inputs and writes the result as a Design Geometry object.

```python
await client.geotech.parameterize_volumes.run(
    *,
    source: ObjectInput,
    target: GeotechParameterizeVolumesTarget,
    distance_unit: Literal["m", "angstrom", "chain", "chain[BnA]", "chain[BnB]", "chain[Cla]", "chain[Ind37]", "chain[Se]", "chain[SeT]", "chain[US]", "fathom", "ft", "ft[BnA]", "ft[BnB]", "ft[Br36]", "ft[Br65]", "ft[Cla]", "ft[GC]", "ft[Ind]", "ft[Ind37]", "ft[Ind62]", "ft[Ind75]", "ft[Se]", "ft[SeT]", "ft[US]", "fur[US]", "in", "in[US]", "link", "link[BnA]", "link[BnB]", "link[Cla]", "link[Se]", "link[SeT]", "link[US]", "m[Ger]", "mi", "mi[naut]", "mi[nautUK]", "mi[US]", "mil", "rod[US]", "yd", "yd[BnA]", "yd[BnB]", "yd[Cla]", "yd[Ind]", "yd[Ind37]", "yd[Ind62]", "yd[Ind75]", "yd[Se]", "yd[SeT]", "yd[US]", "m3/m2", "0.1 ft", "0.1 ft[US]", "0.1 in", "0.1 yd", "1/16 in", "1/2 ft", "1/32 in", "1/64 in", "10 ft", "10 in", "10 km", "100 ft", "100 km", "1000 ft", "30 ft", "30 m", "1E6 bbl/acre", "bbl/acre", "ft3/ft2", "cm", "dam", "dm", "Em", "fm", "Gm", "hm", "km", "Mm", "mm", "nm", "pm", "Tm", "um"],
    include_hard_edges: bool = ...,
    max_repair_iterations: int = ...,
    preview: bool = False,
) -> GeotechParameterizeVolumesResult

await client.geotech.parameterize_volumes.submit(
    *,
    source: ObjectInput,
    target: GeotechParameterizeVolumesTarget,
    distance_unit: Literal["m", "angstrom", "chain", "chain[BnA]", "chain[BnB]", "chain[Cla]", "chain[Ind37]", "chain[Se]", "chain[SeT]", "chain[US]", "fathom", "ft", "ft[BnA]", "ft[BnB]", "ft[Br36]", "ft[Br65]", "ft[Cla]", "ft[GC]", "ft[Ind]", "ft[Ind37]", "ft[Ind62]", "ft[Ind75]", "ft[Se]", "ft[SeT]", "ft[US]", "fur[US]", "in", "in[US]", "link", "link[BnA]", "link[BnB]", "link[Cla]", "link[Se]", "link[SeT]", "link[US]", "m[Ger]", "mi", "mi[naut]", "mi[nautUK]", "mi[US]", "mil", "rod[US]", "yd", "yd[BnA]", "yd[BnB]", "yd[Cla]", "yd[Ind]", "yd[Ind37]", "yd[Ind62]", "yd[Ind75]", "yd[Se]", "yd[SeT]", "yd[US]", "m3/m2", "0.1 ft", "0.1 ft[US]", "0.1 in", "0.1 yd", "1/16 in", "1/2 ft", "1/32 in", "1/64 in", "10 ft", "10 in", "10 km", "100 ft", "100 km", "1000 ft", "30 ft", "30 m", "1E6 bbl/acre", "bbl/acre", "ft3/ft2", "cm", "dam", "dm", "Em", "fm", "Gm", "hm", "km", "Mm", "mm", "nm", "pm", "Tm", "um"],
    include_hard_edges: bool = ...,
    max_repair_iterations: int = ...,
    preview: bool = False,
) -> TaskJob[GeotechParameterizeVolumesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The input mesh to parameterize. Supports Triangle Mesh and Geological Model Meshes schema version 2+. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechParameterizeVolumesTarget` | Target specifying where to create the Design Geometry result. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. | _required_ |
| `distance_unit` | `Literal["m", "angstrom", "chain", "chain[BnA]", "chain[BnB]", "chain[Cla]", "chain[Ind37]", "chain[Se]", "chain[SeT]", "chain[US]", "fathom", "ft", "ft[BnA]", "ft[BnB]", "ft[Br36]", "ft[Br65]", "ft[Cla]", "ft[GC]", "ft[Ind]", "ft[Ind37]", "ft[Ind62]", "ft[Ind75]", "ft[Se]", "ft[SeT]", "ft[US]", "fur[US]", "in", "in[US]", "link", "link[BnA]", "link[BnB]", "link[Cla]", "link[Se]", "link[SeT]", "link[US]", "m[Ger]", "mi", "mi[naut]", "mi[nautUK]", "mi[US]", "mil", "rod[US]", "yd", "yd[BnA]", "yd[BnB]", "yd[Cla]", "yd[Ind]", "yd[Ind37]", "yd[Ind62]", "yd[Ind75]", "yd[Se]", "yd[SeT]", "yd[US]", "m3/m2", "0.1 ft", "0.1 ft[US]", "0.1 in", "0.1 yd", "1/16 in", "1/2 ft", "1/32 in", "1/64 in", "10 ft", "10 in", "10 km", "100 ft", "100 km", "1000 ft", "30 ft", "30 m", "1E6 bbl/acre", "bbl/acre", "ft3/ft2", "cm", "dam", "dm", "Em", "fm", "Gm", "hm", "km", "Mm", "mm", "nm", "pm", "Tm", "um"]` | Distance unit for the created design geometry. | _required_ |
| `include_hard_edges` | `bool` | Whether to include hard edges, defined as edges where adjacent faces exceed the hard-edge angle threshold. | `False` |
| `max_repair_iterations` | `int` | Maximum number of parameterization repair iterations. Set to 0 to skip repair attempts. Bounds: `>= 0`. | `20` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechParameterizeVolumesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechParameterizeVolumesResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The parameterized design geometry output. |
| `message` | `str` | — | Summary of the task outcome. |

#### Nested Shapes

**`GeotechParameterizeVolumesTarget`** (`TypedDict`)

Target specifying where to create the Design Geometry result. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `design-geometry/[>=1.0,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechParameterizeVolumesResultResultObject`** (`ResultNode`)

The parameterized design geometry output.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `design-geometry/[>=1.0,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### remesh

`client.geotech.remesh` · Task: `geotech/remesh` · Version: `1.6.3` · Preview (`preview=True` required)

Remeshes the full Geological Model Meshes object using configurable remeshing parameters to improve triangle quality while preserving the model structure.

```python
await client.geotech.remesh.run(
    *,
    source: ObjectInput,
    target: GeotechRemeshTarget,
    patch_angle_tolerance: float = ...,
    max_chordal_error: float = ...,
    strain_tolerance: float = ...,
    is_closed: bool = ...,
    initial_cleanup: bool = ...,
    line_proximity_detection: bool = ...,
    optimize: bool = ...,
    shape_quality_weight: float = ...,
    target_h: float = ...,
    regularize_mode: bool = ...,
    relax_chunk_interfaces: bool = ...,
    preview: bool = False,
) -> GeotechRemeshResult

await client.geotech.remesh.submit(
    *,
    source: ObjectInput,
    target: GeotechRemeshTarget,
    patch_angle_tolerance: float = ...,
    max_chordal_error: float = ...,
    strain_tolerance: float = ...,
    is_closed: bool = ...,
    initial_cleanup: bool = ...,
    line_proximity_detection: bool = ...,
    optimize: bool = ...,
    shape_quality_weight: float = ...,
    target_h: float = ...,
    regularize_mode: bool = ...,
    relax_chunk_interfaces: bool = ...,
    preview: bool = False,
) -> TaskJob[GeotechRemeshResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to process. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechRemeshTarget` | Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. | _required_ |
| `patch_angle_tolerance` | `float` | Angle change in degrees used to define a patch skeleton line. Bounds: `> 0`. | `10.0` |
| `max_chordal_error` | `float` | Maximum relative local error. -0.005 corresponds to 0.5%. | `-0.005` |
| `strain_tolerance` | `float` | Relative quality difference before and after unfolding to a plane. Bounds: `>= 0`. | `0.3` |
| `is_closed` | `bool` | Whether the mesh is closed; should be true for manifold GMVs. | `False` |
| `initial_cleanup` | `bool` | Whether to run initial cleanup such as node merging, gap filling, and topological fixing. | `True` |
| `line_proximity_detection` | `bool` | Whether to consider distance between skeleton lines when sizing elements. | `False` |
| `optimize` | `bool` | Whether to perform final optimization. | `True` |
| `shape_quality_weight` | `float` | Size-versus-shape weight from 0 to 1. Bounds: `>= 0`. | `0.6` |
| `target_h` | `float` | Target element size. Zero bases the size on skeleton lines. Bounds: `>= 0`. | `0.0` |
| `regularize_mode` | `bool` | If true, do not remesh; only regularize the mesh. | `False` |
| `relax_chunk_interfaces` | `bool` | Whether to relax chunk interfaces during remeshing. | `False` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechRemeshResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechRemeshResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new mesh object with the remeshing applied. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechRemeshTarget`** (`TypedDict`)

Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechRemeshResultResultObject`** (`ResultNode`)

The new mesh object with the remeshing applied.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### remove_triangles

`client.geotech.remove_triangles` · Task: `geotech/remove-triangles` · Version: `1.6.3` · Preview (`preview=True` required)

Removes the selected triangle indices from a Geological Model Meshes object and uploads the updated mesh as a new object.

```python
await client.geotech.remove_triangles.run(
    *,
    source: ObjectInput,
    target: GeotechRemoveTrianglesTarget,
    triangle_indices: list[int],
    preview: bool = False,
) -> GeotechRemoveTrianglesResult

await client.geotech.remove_triangles.submit(
    *,
    source: ObjectInput,
    target: GeotechRemoveTrianglesTarget,
    triangle_indices: list[int],
    preview: bool = False,
) -> TaskJob[GeotechRemoveTrianglesResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The mesh to process. Supports Triangle Mesh and Geological Model Meshes. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechRemoveTrianglesTarget` | Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. | _required_ |
| `triangle_indices` | `list[int]` | Indices of triangles to remove from the mesh. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechRemoveTrianglesResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechRemoveTrianglesResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new mesh object with the selected triangles removed. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechRemoveTrianglesTarget`** (`TypedDict`)

Target specifying where to create the output mesh object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechRemoveTrianglesResultResultObject`** (`ResultNode`)

The new mesh object with the selected triangles removed.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### subsume_embedded_volume_in_gmm

`client.geotech.subsume_embedded_volume_in_gmm` · Task: `geotech/subsume-embedded-volume-in-gmm` · Version: `1.6.3` · Preview (`preview=True` required)

Merges a specified source embedded volume into a target volume in a GMM, either by explicit index or by automatic selection strategy.

```python
await client.geotech.subsume_embedded_volume_in_gmm.run(
    *,
    source: ObjectInput,
    target: GeotechSubsumeEmbeddedVolumeInGmmTarget,
    subsume_source: int,
    target_selection: GeotechSubsumeEmbeddedVolumeInGmmExplicitTargetSelection | GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringVolumeTargetSelection | GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringContactAreaTargetSelection,
    preview: bool = False,
) -> GeotechSubsumeEmbeddedVolumeInGmmResult

await client.geotech.subsume_embedded_volume_in_gmm.submit(
    *,
    source: ObjectInput,
    target: GeotechSubsumeEmbeddedVolumeInGmmTarget,
    subsume_source: int,
    target_selection: GeotechSubsumeEmbeddedVolumeInGmmExplicitTargetSelection | GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringVolumeTargetSelection | GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringContactAreaTargetSelection,
    preview: bool = False,
) -> TaskJob[GeotechSubsumeEmbeddedVolumeInGmmResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The Geological Model Meshes object to process. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `target` | `GeotechSubsumeEmbeddedVolumeInGmmTarget` | Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object. | _required_ |
| `subsume_source` | `int` | The index of the embedded volume to subsume (source). Bounds: `>= 0`. | _required_ |
| `target_selection` | `GeotechSubsumeEmbeddedVolumeInGmmExplicitTargetSelection | GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringVolumeTargetSelection | GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringContactAreaTargetSelection` | Target volume selection. Either an explicit volume index or an automatic selection strategy. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechSubsumeEmbeddedVolumeInGmmResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `GeotechSubsumeEmbeddedVolumeInGmmResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The new Geological Model Meshes object with the volume subsumed. |
| `message` | `str` | — | Description of the result of the operation. |

#### Nested Shapes

**`GeotechSubsumeEmbeddedVolumeInGmmTarget`** (`TypedDict`)

Target specifying where to create the new Geological Model Meshes object. Provide an object with a reference field containing a path-based geoscience object URL and an overwrite field containing a boolean value. Only path-based object references can be used for creating a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`GeotechSubsumeEmbeddedVolumeInGmmExplicitTargetSelection`** (`TypedDict`)

Select the target volume by its explicit index.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `selection_mode` | `Literal["Explicit"]` | Yes | Explicit target selection mode. |
| `volume_index` | `int` | Yes | The index of the target volume. Bounds: `>= 0`. |

**`GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringVolumeTargetSelection`** (`TypedDict`)

Automatically select the neighbouring volume with the largest volume.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `selection_mode` | `Literal["LargestNeighbouringVolume"]` | Yes | Largest neighbouring volume selection mode. |

**`GeotechSubsumeEmbeddedVolumeInGmmLargestNeighbouringContactAreaTargetSelection`** (`TypedDict`)

Automatically select the neighbouring volume with the largest shared contact area.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `selection_mode` | `Literal["LargestNeighbouringContactArea"]` | Yes | Largest neighbouring contact area selection mode. |

**`GeotechSubsumeEmbeddedVolumeInGmmResultResultObject`** (`ResultNode`)

The new Geological Model Meshes object with the volume subsumed.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### validate_gmm_as_intersected_geometry

`client.geotech.validate_gmm_as_intersected_geometry` · Task: `geotech/validate-gmm-as-intersected-geometry` · Version: `1.6.3` · Preview (`preview=True` required)

Checks whether a Geological Model Meshes object is valid as intersected geometry and returns human-readable descriptions of any issues found.

```python
await client.geotech.validate_gmm_as_intersected_geometry.run(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> GeotechValidateGmmAsIntersectedGeometryResult

await client.geotech.validate_gmm_as_intersected_geometry.submit(
    *,
    source: ObjectInput,
    preview: bool = False,
) -> TaskJob[GeotechValidateGmmAsIntersectedGeometryResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `ObjectInput` | The Geological Model Meshes object to validate as intersected geometry. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeotechValidateGmmAsIntersectedGeometryResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `issues` | `list[str]` | — | Descriptions of validation issues found. Empty if the geometry is valid. |
| `message` | `str` | — | Description of the result of the operation. |

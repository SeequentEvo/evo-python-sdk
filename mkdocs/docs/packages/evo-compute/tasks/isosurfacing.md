[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/isosurfacing)

# Isosurfacing Tasks

Discoverable compute tasks in the `isosurfacing` topic. Call any task below through `ComputeClient` as `await client.isosurfacing.<task>.run(...)` (or `submit(...)` for a non-blocking `TaskJob`), or by name via `await client.arun("isosurfacing", "<task>", parameters)` / `await client.asubmit("isosurfacing", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |
| `AttributeInput` | `str | AnyTypedAttribute` | A typed attribute, a JMESPath expression, or a bare attribute name. |
| `AnyTypedAttribute` | `Attribute | BlockModelAttribute | PendingAttribute | BlockModelPendingAttribute` | A typed attribute handle carrying both its parent geoscience object and attribute name. |

## Tasks

### category

`client.isosurfacing.category` · Task: `isosurfacing/category` · Version: `0.1.4` · Preview (`preview=True` required)

Generate closed volume meshes or boundary surfaces from category attributes on 3D grids. Supports single-category extraction.

```python
await client.isosurfacing.category.run(
    *,
    source: IsosurfacingCategoryCategorySourceInput | AnyTypedAttribute,
    category: str,
    target: IsosurfacingCategoryTarget,
    settings: IsosurfacingCategoryCategorySettings = ...,
    preview: bool = False,
) -> IsosurfacingCategoryResult

await client.isosurfacing.category.submit(
    *,
    source: IsosurfacingCategoryCategorySourceInput | AnyTypedAttribute,
    category: str,
    target: IsosurfacingCategoryTarget,
    settings: IsosurfacingCategoryCategorySettings = ...,
    preview: bool = False,
) -> TaskJob[IsosurfacingCategoryResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `IsosurfacingCategoryCategorySourceInput | AnyTypedAttribute` | Source grid and category attribute | _required_ |
| `category` | `str` | Category value to extract as a closed volume surface | _required_ |
| `target` | `IsosurfacingCategoryTarget` | Where to create the output Triangle Mesh object | _required_ |
| `settings` | `IsosurfacingCategoryCategorySettings` | Optional settings for surface generation | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`IsosurfacingCategoryResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `output` | `IsosurfacingCategoryResultResultOutput` | `await result.output.load()`, `await result.output.to_dataframe(*keys)` | Reference to the created Triangle Mesh object |
| `summary` | `IsosurfacingCategoryResultCategorySummary` | — | Summary statistics for the generated surface |

#### Nested Shapes

**`IsosurfacingCategoryCategorySourceInput`** (`TypedDict`)

Source grid and category attribute

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Input grid containing the category attribute to generate surface from Supported schemas: `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `unstructured-hex-grid/[>=1.2,<2]`, `unstructured-tet-grid/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | Category attribute to generate surface from |

**`IsosurfacingCategoryTarget`** (`TypedDict`)

Where to create the output Triangle Mesh object

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`IsosurfacingCategoryCategorySettings`** (`TypedDict`)

Optional settings for surface generation

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `flip_triangles` | `bool` | No | Reverse triangle winding order (flip normals) |

**`IsosurfacingCategoryResultResultOutput`** (`ResultNode`)

Reference to the created Triangle Mesh object

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`IsosurfacingCategoryResultCategorySummary`** (`ResultNode`)

Summary statistics for the generated surface

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `num_triangles` | `int` | — | Number of triangles in the generated surface |
| `num_vertices` | `int` | — | Number of unique vertices in the mesh |
| `has_unused_vertices` | `bool` | — | Whether the mesh has orphan vertices (quality check) |
| `has_non_manifold_edges` | `bool` | — | Whether the mesh has non-manifold edges (quality check) |
| `is_closed` | `bool` | — | Whether the mesh forms a closed volume |

### numeric

`client.isosurfacing.numeric` · Task: `isosurfacing/numeric` · Version: `0.1.4` · Preview (`preview=True` required)

Generate an isosurface at a specified threshold value from numeric attributes on 3D grids. Supports regular, tensor, masked, block model, and unstructured hexahedral and tetrahedral grids.

```python
await client.isosurfacing.numeric.run(
    *,
    source: IsosurfacingNumericNumericSourceInput | AnyTypedAttribute,
    threshold: float,
    target: IsosurfacingNumericTarget,
    settings: IsosurfacingNumericNumericSettings = ...,
    preview: bool = False,
) -> IsosurfacingNumericResult

await client.isosurfacing.numeric.submit(
    *,
    source: IsosurfacingNumericNumericSourceInput | AnyTypedAttribute,
    threshold: float,
    target: IsosurfacingNumericTarget,
    settings: IsosurfacingNumericNumericSettings = ...,
    preview: bool = False,
) -> TaskJob[IsosurfacingNumericResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `IsosurfacingNumericNumericSourceInput | AnyTypedAttribute` | Source grid and attribute | _required_ |
| `threshold` | `float` | Threshold value at which to extract the isosurface | _required_ |
| `target` | `IsosurfacingNumericTarget` | Where to create the output Triangle Mesh object | _required_ |
| `settings` | `IsosurfacingNumericNumericSettings` | Optional settings for isosurface generation | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`IsosurfacingNumericResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `output` | `IsosurfacingNumericResultResultOutput` | `await result.output.load()`, `await result.output.to_dataframe(*keys)` | Reference to the created Triangle Mesh object |
| `summary` | `IsosurfacingNumericResultNumericSummary` | — | Summary statistics for the generated isosurface |

#### Nested Shapes

**`IsosurfacingNumericNumericSourceInput`** (`TypedDict`)

Source grid and attribute

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Input grid containing the numeric attribute to generate isosurface from Supported schemas: `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `unstructured-hex-grid/[>=1.2,<2]`, `unstructured-tet-grid/[>=1.2,<2]`. |
| `attribute` | `AttributeInput` | Yes | Numeric attribute to generate isosurface from |

**`IsosurfacingNumericTarget`** (`TypedDict`)

Where to create the output Triangle Mesh object

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`IsosurfacingNumericNumericSettings`** (`TypedDict`)

Optional settings for isosurface generation

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `flip_triangles` | `bool` | No | Reverse triangle winding order (flip normals) |
| `cluster_vertices` | `bool` | No | Merge close vertices on intersected edges to improve surface quality (structured grids only) |

**`IsosurfacingNumericResultResultOutput`** (`ResultNode`)

Reference to the created Triangle Mesh object

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`IsosurfacingNumericResultNumericSummary`** (`ResultNode`)

Summary statistics for the generated isosurface

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `num_triangles` | `int` | — | Number of triangles in the generated surface |
| `num_vertices` | `int` | — | Number of unique vertices in the mesh |
| `threshold` | `float` | — | The threshold value used |
| `has_unused_vertices` | `bool` | — | Whether the mesh has orphan vertices (quality check) |
| `has_non_manifold_edges` | `bool` | — | Whether the mesh has non-manifold edges (quality check) |

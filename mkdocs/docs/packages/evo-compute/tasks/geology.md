[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/geology)

# Geology Tasks

Discoverable compute tasks in the `geology` topic. Call any task below through `ComputeClient` as `await client.geology.<task>.run(...)`, or by name via `await client.arun("geology", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |
| `AttributeInput` | `str | AnyTypedAttribute` | A typed attribute, a JMESPath expression, or a bare attribute name. |

## Tasks

### classify_by_domain

`client.geology.classify_by_domain` · Task: `geology/classify-by-domain` · Version: `0.1.0` · Preview (`preview=True` required)

Classifies locations from a source geometry object against an ordered list of domain entries (volumes and surfaces), writing a categorical attribute back to the source object.

```python
await client.geology.classify_by_domain.run(
    *,
    source: GeologyClassifyByDomainSource,
    domain_entries: list[GeologyClassifyByDomainVolumeEntry | GeologyClassifyByDomainSurfaceEntry],
    output: GeologyClassifyByDomainOutput,
    preview: bool = False,
) -> GeologyClassifyByDomainResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `GeologyClassifyByDomainSource` | Source geometry object whose locations will be classified. | _required_ |
| `domain_entries` | `list[GeologyClassifyByDomainVolumeEntry | GeologyClassifyByDomainSurfaceEntry]` | Ordered list of domain entries defining the classification hierarchy. Locations are evaluated against entries in order; the first matching entry wins. Each entry is either a VolumeEntry (inside/outside a closed mesh) or a SurfaceEntry (above/below an open mesh surface). | _required_ |
| `output` | `GeologyClassifyByDomainOutput` | Output attribute configuration. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`GeologyClassifyByDomainResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `source` | `GeologyClassifyByDomainResultSource` | `await result.source.load()`, `await result.source.to_dataframe(*keys)` | The source object updated with the domain classification attribute. |
| `summary` | `GeologyClassifyByDomainResultSummary` | — | Classification summary statistics. |

#### Nested Shapes

**`GeologyClassifyByDomainFilterCondition`** (`TypedDict`)

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

**`GeologyClassifyByDomainAnyOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ANY child filter (OR logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["any_of"]` | No | Identifies this as an OR composite filter. |
| `filters` | `list[GeologyClassifyByDomainFilterCondition | GeologyClassifyByDomainAllOfFilter | GeologyClassifyByDomainAnyOfFilter]` | Yes | At least one of these filters must be satisfied (OR logic). |

**`GeologyClassifyByDomainAllOfFilter`** (`TypedDict`)

A composite filter that passes locations satisfying ALL child filters (AND logic).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["all_of"]` | No | Identifies this as an AND composite filter. |
| `filters` | `list[GeologyClassifyByDomainFilterCondition | GeologyClassifyByDomainAllOfFilter | GeologyClassifyByDomainAnyOfFilter]` | Yes | All of these filters must be satisfied (AND logic). |

**`GeologyClassifyByDomainFilter`** (`TypedDict`)

Optional filter to restrict which locations are classified. Filtered-out locations are written as null in the output attribute.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `where` | `GeologyClassifyByDomainFilterCondition | GeologyClassifyByDomainAllOfFilter | GeologyClassifyByDomainAnyOfFilter` | Yes | A filter expression defining which locations to include. Use a FilterCondition for a single predicate, AllOfFilter for AND logic, AnyOfFilter for OR logic, or nest them to build compound predicates. |

**`GeologyClassifyByDomainSource`** (`TypedDict`)

Source geometry object whose locations will be classified.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | The source geometry object whose locations will be classified. Supports pointsets, block models, regular 3D grids, masked 3D grids, tensor 3D grids, and downhole intervals (midpoints). Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `filter` | `GeologyClassifyByDomainFilter | None` | No | Optional filter to restrict which locations are classified. Filtered-out locations are written as null in the output attribute. |

**`GeologyClassifyByDomainMeshDomainObject`** (`TypedDict`)

A standalone triangle mesh domain.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["mesh"]` | Yes | Discriminator identifying a standalone triangle mesh domain. |
| `object` | `ObjectInput` | Yes | Standalone triangle mesh used as the domain. Supported schemas: `triangle-mesh/[>=2.0,<3]`. |

**`GeologyClassifyByDomainGMMDomainObject`** (`TypedDict`)

A named volume or surface within a geological-model-meshes object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["gmm"]` | Yes | Discriminator identifying a geological-model-meshes domain. |
| `object` | `ObjectInput` | Yes | Object containing the named domain. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | Yes | Name of the volume or surface within the geological-model-meshes object. |

**`GeologyClassifyByDomainVolumeEntry`** (`TypedDict`)

A domain entry that classifies locations as inside or outside a closed mesh volume.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["volume"]` | Yes | Discriminator identifying this as a volume (inside/outside) entry. |
| `domain` | `GeologyClassifyByDomainMeshDomainObject | GeologyClassifyByDomainGMMDomainObject` | Yes | A closed triangle mesh, or a named volume within a geological-model-meshes object. When a triangle mesh is provided it must be closed; otherwise an error is raised at runtime. |
| `relationship` | `Literal["inside", "outside"]` | Yes | Spatial relationship: 'inside' matches locations enclosed by the volume; 'outside' matches the rest. |
| `category` | `str` | Yes | Category label assigned to locations matching this entry. |

**`GeologyClassifyByDomainSurfaceEntry`** (`TypedDict`)

A domain entry that classifies locations as above or below an open mesh surface.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["surface"]` | Yes | Discriminator identifying this as a surface (above/below) entry. |
| `domain` | `GeologyClassifyByDomainMeshDomainObject | GeologyClassifyByDomainGMMDomainObject` | Yes | An open triangle mesh surface, or a named surface within a geological-model-meshes object. When a triangle mesh is provided it must NOT be closed; otherwise an error is raised at runtime. |
| `relationship` | `Literal["above", "below"]` | Yes | Spatial relationship: 'above' matches locations with a positive Z distance to the surface; 'below' matches the other side. |
| `category` | `str` | Yes | Category label assigned to locations matching this entry. |

**`GeologyClassifyByDomainOutputAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`GeologyClassifyByDomainOutputAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`GeologyClassifyByDomainOutput`** (`TypedDict`)

Output attribute configuration.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `attribute` | `GeologyClassifyByDomainOutputAttributeCreate | GeologyClassifyByDomainOutputAttributeUpdate | AttributeInput` | Yes | Target category attribute on the source object. Provide a bare attribute path (e.g. "attributes") to create a new attribute, or a filtered expression (e.g. "attributes[?id=='uuid'] \| [0]") to replace an existing one. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |
| `background_label` | `str` | No | Category label assigned to locations that do not match any domain entry. |

**`GeologyClassifyByDomainResultAttribute`** (`ResultNode`)

The domain classification attribute added to the source object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`GeologyClassifyByDomainResultSource`** (`ResultNode`)

The source object updated with the domain classification attribute.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `GeologyClassifyByDomainResultAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | The domain classification attribute added to the source object. |

**`GeologyClassifyByDomainResultCategoryCount`** (`ResultNode`)

Location count for a single category.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `name` | `str` | — | Category label. |
| `count` | `int` | — | Number of locations assigned this label. Bounds: `>= 0`. |

**`GeologyClassifyByDomainResultSummary`** (`ResultNode`)

Classification summary statistics.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `total` | `int` | — | Total number of locations evaluated. Bounds: `>= 0`. |
| `categories` | `list[GeologyClassifyByDomainResultCategoryCount]` | — | Per-category location counts, including the background label. |

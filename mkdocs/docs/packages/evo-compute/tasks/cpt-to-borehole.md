[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/cpt-to-borehole)

# CPT to Borehole Tasks

Discoverable compute tasks in the `cpt-to-borehole` topic. Call any task below through `ComputeClient` as `await client.cpt_to_borehole.<task>.run(...)`, or by name via `await client.arun("cpt-to-borehole", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |
| `FileInput` | `str | FileHandle` | A File API URL or an SDK file/upload handle exposing `.url` or `.metadata.url`. |

## Tasks

### cpt_reader

`client.cpt_to_borehole.cpt_reader` · Task: `cpt-to-borehole/cpt-reader` · Version: `0.1.72` · Preview (`preview=True` required)

CPT reader task

```python
await client.cpt_to_borehole.cpt_reader.run(
    *,
    cpt_file: FileInput,
    target_downhole_collection: CptToBoreholeCptReaderTargetDownholeCollection,
    preview: bool = False,
) -> CptToBoreholeCptReaderResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cpt_file` | `FileInput` | Reference to the file to be read (AGS/GEF/BroXML). | _required_ |
| `target_downhole_collection` | `CptToBoreholeCptReaderTargetDownholeCollection` | DownHoleCollection object to which the parsed results will be written. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`CptToBoreholeCptReaderResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `downhole_collection` | `CptToBoreholeCptReaderResultResultDownholeCollection` | `await result.downhole_collection.load()`, `await result.downhole_collection.to_dataframe(*keys)` | The Downhole Collection object created or updated from the CPT file. |
| `messages` | `list[CptToBoreholeCptReaderResultResultMessagesItem]` | — | List of messages generated during the CPT reading process. |

#### Nested Shapes

**`CptToBoreholeCptReaderTargetDownholeCollection`** (`TypedDict`)

DownHoleCollection object to which the parsed results will be written.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `downhole-collection/[>=1.3,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`CptToBoreholeCptReaderResultResultDownholeCollection`** (`ResultNode`)

The Downhole Collection object created or updated from the CPT file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `downhole-collection/[>=1.3,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`CptToBoreholeCptReaderResultResultMessagesItem`** (`ResultNode`)

A message produced by the task during execution.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `code` | `Literal[100, 101, 102, 103, 105]` | — | Integer code that uniquely identifies this message type within its task. Allows consumers to handle specific messages without relying on text matching. |
| `level` | `Literal["info", "warning", "error"]` | — | The level of the message, e.g., 'info', 'warning', 'error'. |
| `message` | `str` | — | The content of the message. |
| `object_id` | `str | None` | — | Optional identifier of the object (e.g., hole ID) this message relates to. None for global task-level messages. |
| `field` | `str | None` | — | Name of the field this message relates to, such as 'net_area_ratio'. Null when no specific field applies. |

### create_clusters

`client.cpt_to_borehole.create_clusters` · Task: `cpt-to-borehole/create-clusters` · Version: `0.1.72` · Preview (`preview=True` required)

Create clusters of related soil regions based on the stratification performed by the stratify task.

```python
await client.cpt_to_borehole.create_clusters.run(
    *,
    downhole_collection: ObjectInput,
    number_of_clusters: int | None = ...,
    features_option: Literal["qtAndRf", "QtnAndFr"] = ...,
    hole_ids: list[str] | None = ...,
    uncluster: bool = ...,
    preview: bool = False,
) -> CptToBoreholeCreateClustersResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `downhole_collection` | `ObjectInput` | The DownholeCollection whose CPT/Hole data will be clustered. Supported schemas: `downhole-collection/[>=1.3,<2]`. | _required_ |
| `number_of_clusters` | `int | None` | The number of clusters to create. Will be determined automatically if not provided. | `None` |
| `features_option` | `Literal["qtAndRf", "QtnAndFr"]` | Which CPT/Hole parameters to use to determine clusters. The default (qtAndRf) will use the corrected total cone resistance and the corrected friction ratio. | `'qtAndRf'` |
| `hole_ids` | `list[str] | None` | List of Hole IDs to use for clustering. If not provided, all holes in the downhole collection will be used. | `None` |
| `uncluster` | `bool` | When set to true, removes the entire cluster collection and all stored clustering parameters from the downhole collection, then terminates without running the clustering algorithm. Only downhole_collection may be provided alongside this parameter. | `False` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`CptToBoreholeCreateClustersResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `downhole_collection` | `CptToBoreholeCreateClustersResultResultDownholeCollection` | `await result.downhole_collection.load()`, `await result.downhole_collection.to_dataframe(*keys)` | Reference to the DownholeCollection object that was created or updated which will hold the results of clusters creation. |
| `messages` | `list[CptToBoreholeCreateClustersResultResultMessagesItem]` | — | List of messages generated during the clustering process. |
| `inertia_silhouette_score` | `CptToBoreholeCreateClustersResultResultInertiaSilhouetteScore | None` | — | Chart data for the inertia and silhouette score clustering diagnostics. |

#### Nested Shapes

**`CptToBoreholeCreateClustersResultResultDownholeCollection`** (`ResultNode`)

Reference to the DownholeCollection object that was created or updated which will hold the results of clusters creation.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `downhole-collection/[>=1.3,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`CptToBoreholeCreateClustersResultResultMessagesItem`** (`ResultNode`)

A message produced by the task during execution.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `code` | `Literal[300, 301, 302, 305, 307, 309, 310]` | — | Integer code that uniquely identifies this message type within its task. Allows consumers to handle specific messages without relying on text matching. |
| `level` | `Literal["info", "warning", "error"]` | — | The level of the message, e.g., 'info', 'warning', 'error'. |
| `message` | `str` | — | The content of the message. |
| `object_id` | `str | None` | — | Optional identifier of the object (e.g., hole ID) this message relates to. None for global task-level messages. |
| `field` | `str | None` | — | Name of the field this message relates to, such as 'net_area_ratio'. Null when no specific field applies. |

**`CptToBoreholeCreateClustersResultResultInertiaSilhouetteScore`** (`ResultNode`)

Chart data for the inertia and silhouette score clustering diagnostics.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `optimal_number_of_clusters` | `int` | — | Optimal number of clusters selected by the clustering algorithm. |
| `cluster_counts` | `list[int]` | — | Candidate cluster counts evaluated by the clustering algorithm. |
| `inertias` | `list[float]` | — | KMeans inertia values for each candidate cluster count. |
| `silhouette_scores` | `list[float]` | — | Silhouette scores for each candidate cluster count. |

### publish

`client.cpt_to_borehole.publish` · Task: `cpt-to-borehole/publish` · Version: `0.1.72` · Preview (`preview=True` required)

Publish a finalized downhole collection as a new geoscience object for downstream consumption.

```python
await client.cpt_to_borehole.publish.run(
    *,
    downhole_collection: ObjectInput,
    target_downhole_collection: CptToBoreholePublishTargetDownholeCollection,
    include_clusters: bool,
    exclude_collections: list[Literal["measured_cpt_readings", "derived_cpt_readings", "geology_layers"]] = ...,
    cluster_metadata: list[CptToBoreholePublishClusterMetadata] | None = ...,
    preview: bool = False,
) -> CptToBoreholePublishResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `downhole_collection` | `ObjectInput` | The working clustered DownholeCollection to publish. Read-only; left unchanged. Supported schemas: `downhole-collection/[>=1.3,<2]`. | _required_ |
| `target_downhole_collection` | `CptToBoreholePublishTargetDownholeCollection` | The geoscience object where the published DownholeCollection is written as a new object. | _required_ |
| `include_clusters` | `bool` | When true, retain populated CLUSTER_ID columns (labels and colour map); when false, remove CLUSTER_ID columns entirely from the retained tables. | _required_ |
| `exclude_collections` | `list[Literal["measured_cpt_readings", "derived_cpt_readings", "geology_layers"]]` | Data collections to drop from the published object. Defaults to none. May not exclude all data collections, or all CLUSTER_ID data tables when including clusters. | `None` |
| `cluster_metadata` | `list[CptToBoreholePublishClusterMetadata] | None` | Name and colour for each cluster, aligned by index with the Clusters rows. Required if include_clusters is true; must be omitted otherwise. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`CptToBoreholePublishResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `downhole_collection` | `CptToBoreholePublishResultResultDownholeCollection` | `await result.downhole_collection.load()`, `await result.downhole_collection.to_dataframe(*keys)` | Reference to the newly published DownholeCollection geoscience object. |
| `messages` | `list[CptToBoreholePublishResultResultMessagesItem]` | — | List of messages generated during the publish process. |

#### Nested Shapes

**`CptToBoreholePublishTargetDownholeCollection`** (`TypedDict`)

The geoscience object where the published DownholeCollection is written as a new object.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `downhole-collection/[>=1.3,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`CptToBoreholePublishClusterMetadata`** (`TypedDict`)

Human-assigned name and colour for a single cluster.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | `str` | Yes | Human-assigned name for the cluster. |
| `color` | `list[int]` | Yes | RGB colour of the cluster as three integers in the range 0-255. |

**`CptToBoreholePublishResultResultDownholeCollection`** (`ResultNode`)

Reference to the newly published DownholeCollection geoscience object.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `downhole-collection/[>=1.3,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`CptToBoreholePublishResultResultMessagesItem`** (`ResultNode`)

A message produced by the task during execution.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `code` | `Literal[601, 602, 603, 604, 605]` | — | Integer code that uniquely identifies this message type within its task. Allows consumers to handle specific messages without relying on text matching. |
| `level` | `Literal["info", "warning", "error"]` | — | The level of the message, e.g., 'info', 'warning', 'error'. |
| `message` | `str` | — | The content of the message. |
| `object_id` | `str | None` | — | Optional identifier of the object (e.g., hole ID) this message relates to. None for global task-level messages. |
| `field` | `str | None` | — | Name of the field this message relates to, such as 'net_area_ratio'. Null when no specific field applies. |

### stratify

`client.cpt_to_borehole.stratify` · Task: `cpt-to-borehole/stratify` · Version: `0.1.72` · Preview (`preview=True` required)

Classifies and stratifies CPT data into geological layers based on predefined criteria and algorithms, facilitating geotechnical analysis and modeling.

```python
await client.cpt_to_borehole.stratify.run(
    *,
    input_downhole_collections: list[ObjectInput],
    target_downhole_collection: CptToBoreholeStratifyTargetDownholeCollection,
    cpt_parameters: list[CptToBoreholeStratifyPerHoleParameters] | None = ...,
    execution_mode: Literal["inline", "fan_out", "auto"] = ...,
    preview: bool = False,
) -> CptToBoreholeStratifyResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input_downhole_collections` | `list[ObjectInput]` | A list of DownHoleCollection objects to be used as input for stratification. | _required_ |
| `target_downhole_collection` | `CptToBoreholeStratifyTargetDownholeCollection` | DownHoleCollection object to which results are written. | _required_ |
| `cpt_parameters` | `list[CptToBoreholeStratifyPerHoleParameters] | None` | List of stratification and classification parameters per CPT/hole. | `None` |
| `execution_mode` | `Literal["inline", "fan_out", "auto"]` | The execution strategy for classify/stratify steps. When set to 'auto' the task decides automatically based on available memory. | `'auto'` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`CptToBoreholeStratifyResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `downhole_collection` | `CptToBoreholeStratifyResultResultDownholeCollection` | `await result.downhole_collection.load()`, `await result.downhole_collection.to_dataframe(*keys)` | Reference to the DownholeCollection object that was created or updated which will hold the results of the stratification and classification. |
| `messages` | `list[CptToBoreholeStratifyResultResultMessagesItem]` | — | List of messages generated during the stratification process. |
| `normalized_cost_variation` | `list[CptToBoreholeStratifyResultNormalizedCostVariationEntry]` | — | Per-hole normalized cost variation data for holes stratified with the machine learning method. |

#### Nested Shapes

**`CptToBoreholeStratifyTargetDownholeCollection`** (`TypedDict`)

DownHoleCollection object to which results are written.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `downhole-collection/[>=1.3,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`CptToBoreholeStratifyPerHoleParameters`** (`TypedDict`)

Parameters for each CPT/hole to be stratified.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `hole_id` | `str` | Yes | Identifier for the CPT/hole to be classified and stratified. |
| `name` | `str | None` | No | A new name for the CPT/hole (this will change the hole_id to the desired name). |
| `water_depth` | `float | None` | No | Water depth at the location of this CPT/hole. |
| `net_area_ratio` | `float | None` | No | Cone area ratio for this CPT/hole. |
| `cone_surface_area` | `float | None` | No | Cone surface area (mm²) for this CPT/hole. Required when an inverse filtering method is used and the source file does not provide it. |
| `filtering_method` | `Literal["no_filtering", "inverse_filtering", "nearfield_inverse_filtering"]` | No | Filtering method to use for the classification of the CPT/hole. Defaults to no filtering. |
| `stratification_method` | `Literal["average", "std_dev", "machine_learning"]` | No | Stratification method to use for the stratification of the CPT/hole. |
| `layering_tolerance` | `float` | No | Layering tolerance to use for the stratification of the CPT/hole. Bounds: `> 0.01`. |
| `minimum_layer_thickness` | `float` | No | Minimum layer thickness to use for the stratification of the CPT/hole. Bounds: `> 0.05`. |
| `maximum_number_of_layers` | `int` | No | Maximum number of layers to use for the stratification of the CPT/hole. Bounds: `>= 1`. |
| `custom_number_of_layers` | `int | None` | No | Custom number of layers to use for the stratification of the CPT/hole. |

**`CptToBoreholeStratifyResultResultDownholeCollection`** (`ResultNode`)

Reference to the DownholeCollection object that was created or updated which will hold the results of the stratification and classification.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `downhole-collection/[>=1.3,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`CptToBoreholeStratifyResultResultMessagesItem`** (`ResultNode`)

A message produced by the task during execution.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `code` | `Literal[200, 201, 203, 204, 205, 206, 208, 209, 210, 211, 212]` | — | Integer code that uniquely identifies this message type within its task. Allows consumers to handle specific messages without relying on text matching. |
| `level` | `Literal["info", "warning", "error"]` | — | The level of the message, e.g., 'info', 'warning', 'error'. |
| `message` | `str` | — | The content of the message. |
| `object_id` | `str | None` | — | Optional identifier of the object (e.g., hole ID) this message relates to. None for global task-level messages. |
| `field` | `str | None` | — | Name of the field this message relates to, such as 'net_area_ratio'. Null when no specific field applies. |

**`CptToBoreholeStratifyResultNormalizedCostPoint`** (`ResultNode`)

One point on the normalized cost variation chart.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `number_of_layers` | `int` | — | Candidate number of layers used for the normalized cost calculation. |
| `cost` | `float` | — | Normalized cost value for the candidate number of layers. |

**`CptToBoreholeStratifyResultNormalizedCostVariationEntry`** (`ResultNode`)

Normalized cost variation chart data for one hole.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object_id` | `str` | — | Identifier of the object (meaning the hole id, e.g. "CPT_1") this normalized cost variation entry relates to. |
| `optimal_number_of_layers` | `int` | — | Optimal number of layers selected by the machine learning stratification method. |
| `first_minimum_cost_number_of_layers` | `int | None` | — | Number of layers at the first local minimum of the normalized cost curve, if available. |
| `points` | `list[CptToBoreholeStratifyResultNormalizedCostPoint]` | — | Chart points for the normalized cost variation curve. |

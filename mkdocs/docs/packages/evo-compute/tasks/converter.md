[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/converter)

# Converter Tasks

Discoverable compute tasks in the `converter` topic. Call any task below through `ComputeClient` as `await client.converter.<task>.run(...)`, or by name via `await client.arun("converter", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |
| `FileInput` | `str | FileHandle` | A File API URL or an SDK file/upload handle exposing `.url` or `.metadata.url`. |

## Tasks

### detect_converter

`client.converter.detect_converter` · Task: `converter/detect-converter` · Version: `0.1.41` · Preview (`preview=True` required)

Accepts one or more file references, inspects each file's extension, and returns which converter task(s) and geoscience object schema(s) are available for importing each file.

```python
await client.converter.detect_converter.run(
    *,
    files: list[FileInput],
    preview: bool = False,
) -> ConverterDetectConverterResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `files` | `list[FileInput]` | Up to 50 Evo file references to analyse. | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterDetectConverterResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `groups` | `list[ConverterDetectConverterResultConversionGroup]` | — | Conversion assessments for the supplied files. |

#### Nested Shapes

**`ConverterDetectConverterResultSourceFileBinding`** (`ResultNode`)

A supplied file bound to the parameter expected by a converter task.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `parameter_name` | `str` | — | The receiving converter task's file parameter name. |
| `file_ref` | `str` | — | The supplied Evo file reference. |
| `filename` | `str` | — | The filename from file metadata. |
| `role` | `str` | — | The file's role in the conversion group. |

**`ConverterDetectConverterResultCompanionFileRequirement`** (`ResultNode`)

A companion-file parameter expected by a converter task.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `parameter_name` | `str` | — | The receiving converter task's file parameter name. |
| `extension` | `str` | — | The expected companion extension, or an empty string for an extensionless companion. |
| `required` | `bool` | — | Whether the converter requires this companion file. |
| `supplied` | `bool` | — | Whether a matching companion file was supplied. |

**`ConverterDetectConverterResultConverterOption`** (`ResultNode`)

A converter task that can handle a particular file extension.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `converter_task` | `str` | — | The compute task URL for the converter (e.g. 'https://example.com/compute/orgs/{org_id}/converter/obj-import'). |
| `description` | `str` | — | A human-readable description of what the converter task does. |
| `schema` | `str` | — | The geoscience object schema the converter produces (e.g. 'triangle-mesh'). |

**`ConverterDetectConverterResultValidationDiagnostic`** (`ResultNode`)

A non-fatal validation result for a conversion group.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `code` | `str` | — | A stable machine-readable diagnostic code. |
| `message` | `str` | — | A human-readable explanation of the diagnostic. |

**`ConverterDetectConverterResultConversionInvocationPlan`** (`ResultNode`)

A ready-to-submit converter plan when one converter and schema remain.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `converter_task` | `str` | — | The URL of the converter task to invoke. |
| `schema` | `str | None` | — | The output schema when there is exactly one; null for multiple output schemas. |
| `output_schemas` | `list[str]` | — | All geoscience object schemas produced by this invocation. |
| `files` | `list[ConverterDetectConverterResultSourceFileBinding]` | — | Resolved file parameters for the converter task. |
| `required_parameters` | `list[str]` | — | Required non-file converter parameters the caller must still provide. |

**`ConverterDetectConverterResultConversionGroup`** (`ResultNode`)

The actionable conversion assessment for one primary input file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `status` | `str` | — | Group status: ready_for_invocation, incomplete, invalid, unsupported, or orphan. |
| `files` | `list[ConverterDetectConverterResultSourceFileBinding]` | — | Files associated with this conversion group. |
| `companion_requirements` | `list[ConverterDetectConverterResultCompanionFileRequirement]` | — | Required and optional companion-file parameters expected by the converter. |
| `converters` | `list[ConverterDetectConverterResultConverterOption]` | — | Content-compatible converter alternatives. |
| `diagnostics` | `list[ConverterDetectConverterResultValidationDiagnostic]` | — | Non-fatal validation diagnostics. |
| `invocation_plan` | `ConverterDetectConverterResultConversionInvocationPlan | None` | — | A ready-to-submit converter plan when one converter and schema remain. |

### ers_import

`client.converter.ers_import` · Task: `converter/ers-import` · Version: `0.1.41` · Preview (`preview=True` required)

Import ER Mapper Raster (.ers) files into Evo as Regular-2D geoscience objects.

```python
await client.converter.ers_import.run(
    *,
    file: FileInput,
    file_data: FileInput,
    target: ConverterErsImportTarget,
    preview: bool = False,
) -> ConverterErsImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | ER Mapper Raster header file (.ers) to import. Supported file extensions: `.ers`. | _required_ |
| `file_data` | `FileInput` | ER Mapper Raster binary data file (companion file without extension). | _required_ |
| `target` | `ConverterErsImportTarget` | Target specifying where to create or update the Regular2DGrid object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterErsImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterErsImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The Regular2DGrid object created from the input file. |

#### Nested Shapes

**`ConverterErsImportTarget`** (`TypedDict`)

Target specifying where to create or update the Regular2DGrid object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `regular-2d-grid/[>=1.2,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterErsImportResultResultObject`** (`ResultNode`)

The Regular2DGrid object created from the input file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-2d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### geosoft_grid_import

`client.converter.geosoft_grid_import` · Task: `converter/geosoft-grid-import` · Version: `0.1.41` · Preview (`preview=True` required)

Import Geosoft Grid (.grd/.grd.gi) files into Evo as Regular-2D geoscience objects.

```python
await client.converter.geosoft_grid_import.run(
    *,
    file: FileInput,
    target: ConverterGeosoftGridImportTarget,
    file_gi: FileInput | None = ...,
    preview: bool = False,
) -> ConverterGeosoftGridImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | Geosoft Grid (.grd) file to import. Supported file extensions: `.grd`. | _required_ |
| `target` | `ConverterGeosoftGridImportTarget` | Target specifying where to create or update the Regular2DGrid object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `file_gi` | `FileInput | None` | Optional Geosoft Grid Information (.grd.gi) file containing projection data. Supported file extensions: `.gi`. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterGeosoftGridImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterGeosoftGridImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The Regular2DGrid object created from the input Geosoft Grid file. |

#### Nested Shapes

**`ConverterGeosoftGridImportTarget`** (`TypedDict`)

Target specifying where to create or update the Regular2DGrid object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `regular-2d-grid/[>=1.2,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterGeosoftGridImportResultResultObject`** (`ResultNode`)

The Regular2DGrid object created from the input Geosoft Grid file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-2d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### geosoft_surface_import

`client.converter.geosoft_surface_import` · Task: `converter/geosoft-surface-import` · Version: `0.1.41` · Preview (`preview=True` required)

Import Geosoft Surface (.geosoft_surface) files into Evo as GeologicalModelMeshes geoscience objects.

```python
await client.converter.geosoft_surface_import.run(
    *,
    file: FileInput,
    target: ConverterGeosoftSurfaceImportTarget,
    preview: bool = False,
) -> ConverterGeosoftSurfaceImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | Geosoft Surface file (.geosoft_surface) to import. Supported file extensions: `.geosoft_surface`. | _required_ |
| `target` | `ConverterGeosoftSurfaceImportTarget` | Target specifying where to create or update the GeologicalModelMeshes object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterGeosoftSurfaceImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterGeosoftSurfaceImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The GeologicalModelMeshes object created from the input file. |

#### Nested Shapes

**`ConverterGeosoftSurfaceImportTarget`** (`TypedDict`)

Target specifying where to create or update the GeologicalModelMeshes object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterGeosoftSurfaceImportResultResultObject`** (`ResultNode`)

The GeologicalModelMeshes object created from the input file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `geological-model-meshes/[>=2.0,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### geosoft_voxel_import

`client.converter.geosoft_voxel_import` · Task: `converter/geosoft-voxel-import` · Version: `0.1.41` · Preview (`preview=True` required)

Import Geosoft Voxel (.geosoft_voxel) files into Evo as either Regular-3D or Tensor-3D geoscience objects.

```python
await client.converter.geosoft_voxel_import.run(
    *,
    file: FileInput,
    target: ConverterGeosoftVoxelImportTarget,
    preview: bool = False,
) -> ConverterGeosoftVoxelImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | Geosoft Voxel (.geosoft_voxel) file to import. Supported file extensions: `.geosoft_voxel`. | _required_ |
| `target` | `ConverterGeosoftVoxelImportTarget` | Target specifying where to create or update the Regular3DGrid \| Tensor3DGrid object. Contains 'reference' (Geoscience Object API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterGeosoftVoxelImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `voxel_object` | `ConverterGeosoftVoxelImportResultResultVoxelObject` | `await result.voxel_object.load()`, `await result.voxel_object.to_dataframe(*keys)` | The Regular3DGrid or Tensor3DGrid object created from the input Geosoft Voxel file. |
| `colormap` | `ConverterGeosoftVoxelImportResultResultColormap | None` | — | The category colormap created from the input Geosoft Voxel file's thematic data, or null if the file contains no thematic data. |

#### Nested Shapes

**`ConverterGeosoftVoxelImportTarget`** (`TypedDict`)

Target specifying where to create or update the Regular3DGrid | Tensor3DGrid object. Contains 'reference' (Geoscience Object API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterGeosoftVoxelImportResultResultVoxelObject`** (`ResultNode`)

The Regular3DGrid or Tensor3DGrid object created from the input Geosoft Voxel file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

**`ConverterGeosoftVoxelImportResultResultColormap`** (`ResultNode`)

The category colormap created from the input Geosoft Voxel file's thematic data, or null if the file contains no thematic data.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `colormap_id` | `str` | — | Identifier of the CategoryColormap created in the Colormap API. |
| `name` | `str` | — | Name given to the created colormap. |
| `attribute_id` | `str` | — | Key of the voxel cell attribute the colormap was associated with. |

### image_import

`client.converter.image_import` · Task: `converter/image-import` · Version: `0.1.41` · Preview (`preview=True` required)

Import image files into Evo as Regular-2D geoscience objects.

```python
await client.converter.image_import.run(
    *,
    file: FileInput,
    target: ConverterImageImportTarget,
    coordinate_reference_system: str = ...,
    origin: list[float] | None = ...,
    cell_size: list[float] | None = ...,
    preview: bool = False,
) -> ConverterImageImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | Image file to import. Supported file extensions: `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.gif`. | _required_ |
| `target` | `ConverterImageImportTarget` | Target specifying where to create or update the Regular2DGrid object. Contains 'reference' (Geoscience Object API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `coordinate_reference_system` | `str` | The coordinate reference system for the grid coordinates, specified as an EPSG code string (e.g., 'EPSG:4326'), an OGC WKT string, or 'unspecified'. | `'unspecified'` |
| `origin` | `list[float] | None` | Optional grid origin in [x, y, z] format. | `None` |
| `cell_size` | `list[float] | None` | Optional grid cell size in [dx, dy] format. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterImageImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterImageImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The Regular2DGrid object created from the input file. |

#### Nested Shapes

**`ConverterImageImportTarget`** (`TypedDict`)

Target specifying where to create or update the Regular2DGrid object. Contains 'reference' (Geoscience Object API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `regular-2d-grid/[>=1.2,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterImageImportResultValues`** (`ResultNode`)

Continuous cell attribute containing imported image pixel values.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`ConverterImageImportResultResultObject`** (`ResultNode`)

The Regular2DGrid object created from the input file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-2d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `values` | `ConverterImageImportResultValues` | `await node.values.load()`, `await node.values.to_dataframe()` | Continuous cell attribute containing imported image pixel values. |

### kml_import

`client.converter.kml_import` · Task: `converter/kml-import` · Version: `0.1.41` · Preview (`preview=True` required)

Task to import KML/KMZ files into Evo as Pointset and LineSegments geoscience objects.

```python
await client.converter.kml_import.run(
    *,
    file: FileInput,
    target: ConverterKmlImportTarget,
    preview: bool = False,
) -> ConverterKmlImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | KML or KMZ file to import. Supported file extensions: `.kml`, `.kmz`. | _required_ |
| `target` | `ConverterKmlImportTarget` | Target specifying where to create or update the imported geoscience object(s). Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterKmlImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `objects` | `list[ConverterKmlImportResultResultObjectsItem]` | `await result.objects[i].load()`, `await result.objects[i].to_dataframe()` | The geoscience objects created from the input KML/KMZ file. |

#### Nested Shapes

**`ConverterKmlImportTarget`** (`TypedDict`)

Target specifying where to create or update the imported geoscience object(s). Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `pointset/[>=1.3,<2]`, `line-segments/[>=2.2,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterKmlImportResultResultObjectsItem`** (`ResultNode`)

A geoscience object extracted from the input KML/KMZ file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.3,<2]`, `line-segments/[>=2.2,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### obj_import

`client.converter.obj_import` · Task: `converter/obj-import` · Version: `0.1.41` · Preview (`preview=True` required)

Import OBJ (Wavefront) files into Evo as Triangle Mesh geoscience objects. Supports triangles, quads (auto-triangulated), and multiple objects within a single file.

```python
await client.converter.obj_import.run(
    *,
    file: FileInput,
    target: ConverterObjImportTarget,
    coordinate_reference_system: str | None = ...,
    preview: bool = False,
) -> ConverterObjImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | OBJ file to import. Supported file extensions: `.obj`. | _required_ |
| `target` | `ConverterObjImportTarget` | Target specifying where to create or update the Triangle Mesh object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `coordinate_reference_system` | `str | None` | The coordinate reference system for the mesh coordinates, specified as either an EPSG code string (e.g., 'EPSG:4326') or a WKT2 coordinate reference system definition string. If not provided, the CRS will be marked as unspecified. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterObjImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterObjImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The Triangle Mesh object created from the input file. |

#### Nested Shapes

**`ConverterObjImportTarget`** (`TypedDict`)

Target specifying where to create or update the Triangle Mesh object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.2,<3]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterObjImportResultResultObject`** (`ResultNode`)

The Triangle Mesh object created from the input file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.2,<3]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### shp_import

`client.converter.shp_import` · Task: `converter/shp-import` · Version: `0.1.41` · Preview (`preview=True` required)

Task to import SHP files into Evo as Triangle Mesh geoscience objects.

```python
await client.converter.shp_import.run(
    *,
    file: FileInput,
    file_shx: FileInput,
    file_dbf: FileInput,
    target: ConverterShpImportTarget,
    file_prj: FileInput | None = ...,
    file_cpg: FileInput | None = ...,
    file_sbn: FileInput | None = ...,
    file_sbx: FileInput | None = ...,
    file_xml: FileInput | None = ...,
    preview: bool = False,
) -> ConverterShpImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | SHP file to import. Supported file extensions: `.shp`. | _required_ |
| `file_shx` | `FileInput` | SHX spatial index file (required companion to the .shp file). Supported file extensions: `.shx`. | _required_ |
| `file_dbf` | `FileInput` | DBF attribute file (required companion to the .shp file). Supported file extensions: `.dbf`. | _required_ |
| `target` | `ConverterShpImportTarget` | Target specifying where to create or update the imported geoscience object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `file_prj` | `FileInput | None` | Optional PRJ projection file. Supported file extensions: `.prj`. | `None` |
| `file_cpg` | `FileInput | None` | Optional CPG code page file for character encoding. Supported file extensions: `.cpg`. | `None` |
| `file_sbn` | `FileInput | None` | Optional SBN spatial index file (not currently used). Supported file extensions: `.sbn`. | `None` |
| `file_sbx` | `FileInput | None` | Optional SBX spatial index file (not currently used). Supported file extensions: `.sbx`. | `None` |
| `file_xml` | `FileInput | None` | Optional XML metadata file (not currently used). Supported file extensions: `.xml`. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterShpImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterShpImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The geoscience object created from the input file(s). |

#### Nested Shapes

**`ConverterShpImportTarget`** (`TypedDict`)

Target specifying where to create or update the imported geoscience object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `line-segments/[>=2.1,<3]`, `pointset/[>=1.2,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterShpImportResultResultObject`** (`ResultNode`)

The geoscience object created from the input file(s).

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `triangle-mesh/[>=2.0,<3]`, `line-segments/[>=2.1,<3]`, `pointset/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### ubc_import

`client.converter.ubc_import` · Task: `converter/ubc-import` · Version: `0.1.41` · Preview (`preview=True` required)

Task to import UBC Grids into Evo as Tensor-3D geoscience objects.

```python
await client.converter.ubc_import.run(
    *,
    file: FileInput,
    target: ConverterUbcImportTarget,
    file_mod: FileInput | None = ...,
    coordinate_reference_system: str = ...,
    preview: bool = False,
) -> ConverterUbcImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | UBC file to import. Supported file extensions: `.msh`. | _required_ |
| `target` | `ConverterUbcImportTarget` | Target specifying where to create or update the Tensor Grid object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `file_mod` | `FileInput | None` | Optional numeric property file to import. Supported file extensions: `.mod`, `.sus`, `.den`. | `None` |
| `coordinate_reference_system` | `str` | The coordinate reference system for the grid coordinates, specified as an EPSG code string (e.g., 'EPSG:4326'), an OGC WKT string, or 'unspecified'. | `'unspecified'` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterUbcImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterUbcImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The Tensor Grid object created from the input file(s). |

#### Nested Shapes

**`ConverterUbcImportTarget`** (`TypedDict`)

Target specifying where to create or update the Tensor Grid object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `tensor-3d-grid/[>=1.2,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterUbcImportResultResultObject`** (`ResultNode`)

The Tensor Grid object created from the input file(s).

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `tensor-3d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### vtk_import

`client.converter.vtk_import` · Task: `converter/vtk-import` · Version: `0.1.41` · Preview (`preview=True` required)

VTK Import compute task

```python
await client.converter.vtk_import.run(
    *,
    files: list[FileInput],
    preview: bool = False,
) -> ConverterVtkImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `files` | `list[FileInput]` | List of files to convert | _required_ |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterVtkImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `objects` | `list[ConverterVtkImportResultResultObjectsItem]` | `await result.objects[i].load()`, `await result.objects[i].to_dataframe()` | List of Geoscience objects that was extracted from the input files. |

#### Nested Shapes

**`ConverterVtkImportResultResultObjectsItem`** (`ResultNode`)

A Geoscience Object that was extracted from the input files

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

### xyz_import

`client.converter.xyz_import` · Task: `converter/xyz-import` · Version: `0.1.41` · Preview (`preview=True` required)

Task to import XYZ files into Evo as Pointset geoscience objects.

```python
await client.converter.xyz_import.run(
    *,
    file: FileInput,
    target: ConverterXyzImportTarget,
    x_index: int | None = ...,
    y_index: int | None = ...,
    z_index: int | None = ...,
    epsg_code: int | None = ...,
    preview: bool = False,
) -> ConverterXyzImportResult
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `file` | `FileInput` | XYZ file to import. Supported file extensions: `.xyz`. | _required_ |
| `target` | `ConverterXyzImportTarget` | Target specifying where to create or update the Pointset object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references). | _required_ |
| `x_index` | `int | None` | Index of the X coordinate in the XYZ file. | `-1` |
| `y_index` | `int | None` | Index of the Y coordinate in the XYZ file. | `-1` |
| `z_index` | `int | None` | Index of the Z coordinate in the XYZ file. | `-1` |
| `epsg_code` | `int | None` | EPSG code for the coordinate reference system. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`ConverterXyzImportResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `object` | `ConverterXyzImportResultResultObject` | `await result.object.load()`, `await result.object.to_dataframe(*keys)` | The Pointset object created from the input file. |

#### Nested Shapes

**`ConverterXyzImportTarget`** (`TypedDict`)

Target specifying where to create or update the Pointset object. Contains 'reference' (File API URL) and optional 'overwrite' (boolean, only valid for path references).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `reference` | `ObjectInput` | Yes | Reference to the targeted object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `overwrite` | `bool` | No | Whether to overwrite the existing object. |
| `description` | `str | None` | No | Description to put into the target object. |
| `tags` | `dict[str, str]` | No | Tags to put into the target object. |

**`ConverterXyzImportResultResultObject`** (`ResultNode`)

The Pointset object created from the input file.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |

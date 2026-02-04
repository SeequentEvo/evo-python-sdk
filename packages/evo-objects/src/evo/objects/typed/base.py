#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import copy
import sys
import weakref
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Generic, TypeVar, get_args, get_origin, get_type_hints, overload
from uuid import UUID

from pydantic import TypeAdapter

from evo import jmespath
from evo.common import Environment, IContext, StaticContext
from evo.common.styles.html import STYLESHEET, build_nested_table, build_table_row, build_table_row_vtop, build_title
from evo.common.urls import get_portal_url_from_environment, get_viewer_url_from_environment
from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference, ObjectSchema, SchemaVersion

from ._model import DataLocation, ModelContext, SchemaList, SchemaLocation, SchemaModel, SchemaProperty, SubModelMetadata
from ._utils import (
    create_geoscience_object,
    download_geoscience_object,
    replace_geoscience_object,
)
from .types import BoundingBox, CoordinateReferenceSystem, EpsgCode

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "BaseObject",
    "BaseObjectData",
    "ConstructableObject",
    "object_from_path",
    "object_from_reference",
    "object_from_uuid",
]

_T = TypeVar("_T")



def _get_annotation_metadata(annotation: Any) -> tuple[Any, SchemaLocation | None, DataLocation | None]:
    """Extract the base type, SchemaLocation, and DataLocation from an annotation.

    :param annotation: The type annotation to process.
    :return: A tuple of (base_type, schema_location, data_location).
    """
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if len(args) < 2:
            return annotation, None, None

        schema_location: SchemaLocation | None = None
        data_location: DataLocation | None = None
        for item in args[1:]:
            if isinstance(item, SchemaLocation):
                schema_location = item
            elif isinstance(item, DataLocation):
                data_location = item

        return args[0], schema_location, data_location
    return annotation, None, None


def _get_url_prefix(environment: Environment) -> str:
    return f"{environment.hub_url.rstrip('/')}/geoscience-object/orgs/{environment.org_id}/workspaces/{environment.workspace_id}/objects"


async def object_from_reference(
    context: IContext,
    reference: ObjectReference | str,
) -> _BaseObject:
    """Download a GeoscienceObject from an ObjectReference and create the appropriate typed instance.

    This function downloads the object from a full ObjectReference (which can contain path, UUID,
    version, etc.) and automatically selects the correct typed class (e.g., PointSet, Regular3DGrid)
    based on the object's sub-classification.

    :param context: The context for connecting to Evo APIs.
    :param reference: The ObjectReference identifying the object to download.

    :return: A typed GeoscienceObject instance (PointSet, Regular3DGrid, etc.).

    :raises ValueError: If no typed class is found for the object's sub-classification.

    Example::

        from evo.objects.typed import object_from_reference
        from evo.objects import ObjectReference

        # Create reference from URL
        ref = ObjectReference("evo://org/workspace/object/b208a6c9-6881-4b97-b02d-acb5d81299bb")
        obj = await object_from_reference(context, ref)
        
        # obj will be a PointSet if the object is a pointset,
        # a Regular3DGrid if it's a regular-3d-grid, etc.
    """        
    # Context for the reference's workspace
    reference = ObjectReference(reference)
    reference_context = StaticContext(
        connector=context.get_connector(),
        cache=context.get_cache(),
        org_id=reference.org_id,
        workspace_id=reference.workspace_id,
    )
    obj = await download_geoscience_object(reference_context, reference)
    
    # Look up the class directly from the sub-classification
    selected_cls = _BaseObject._sub_classification_lookup.get(obj.metadata.schema_id.sub_classification)
    if selected_cls is None:
        raise ValueError(
            f"No typed class found for sub-classification '{obj.metadata.schema_id.sub_classification}'. "
            f"Available types: {list(_BaseObject._sub_classification_lookup.keys())}"
        )
    
    return selected_cls(reference_context, obj)


async def object_from_path(
    context: IContext,
    path: str,
    version: str | None = None,
) -> _BaseObject:
    """Download a GeoscienceObject by its path and create the appropriate typed instance.

    This function downloads the object using its path (the hierarchical location/name
    in the workspace) and automatically selects the correct typed class (e.g., PointSet,
    Regular3DGrid) based on the object's sub-classification.

    :param context: The context for connecting to Evo APIs.
    :param path: The object path (e.g., "my-folder/my-object.json" or "/my-folder/my-object.json").
    :param version: Optional version ID string to download a specific version.

    :return: A typed GeoscienceObject instance (PointSet, Regular3DGrid, etc.).

    :raises ValueError: If no typed class is found for the object's sub-classification.

    Example::

        from evo.objects.typed import object_from_path

        # Download latest version by path
        obj = await object_from_path(context, "my-folder/pointset.json")
        
        # Download specific version
        obj = await object_from_path(context, "my-folder/pointset.json", version="abc123")
    """
    version = "?version=" + version if version else ""
    reference = ObjectReference(_get_url_prefix(context.get_environment()) + f"/path/{path}{version}")
    return await object_from_reference(context, reference)


async def object_from_uuid(
    context: IContext,
    uuid: UUID | str,
    version: str | None = None,
) -> _BaseObject:
    """Download a GeoscienceObject by its UUID and create the appropriate typed instance.

    This function downloads the object using its unique identifier (UUID) and automatically
    selects the correct typed class (e.g., PointSet, Regular3DGrid) based on the object's
    sub-classification.

    :param context: The context for connecting to Evo APIs.
    :param uuid: The UUID of the object to download (as a UUID object or string).
    :param version: Optional version ID string to download a specific version.

    :return: A typed GeoscienceObject instance (PointSet, Regular3DGrid, etc.).

    :raises ValueError: If no typed class is found for the object's sub-classification.

    Example::

        from evo.objects.typed import object_from_uuid

        # Download latest version by UUID
        obj = await object_from_uuid(context, "b208a6c9-6881-4b97-b02d-acb5d81299bb")
        
        # Download specific version
        obj = await object_from_uuid(context, "b208a6c9-6881-4b97-b02d-acb5d81299bb", version="abc123")
    """
    version = "?version=" + version if version else ""
    reference = ObjectReference(_get_url_prefix(context.get_environment()) + f"/{uuid}{version}")
    return await object_from_reference(context, reference)


class _BaseObject:
    """Base class for high-level Geoscience Objects."""

    _sub_classification_lookup: ClassVar[weakref.WeakValueDictionary[str, type[_BaseObject]]] = (
        weakref.WeakValueDictionary()
    )
    _schema_properties: ClassVar[dict[str, SchemaProperty[Any]]] = {}
    _sub_models: ClassVar[dict[str, SubModelMetadata]] = {}

    _data_class: ClassVar[type[BaseObjectData] | None] = None
    _data_class_lookup: ClassVar[weakref.WeakValueDictionary[type[BaseObjectData], type[_BaseObject]]] = (
        weakref.WeakValueDictionary()
    )

    sub_classification: ClassVar[str | None] = None
    """The sub-classification of the Geoscience Object schema.

    If None, this class is considered abstract and cannot be instantiated directly.
    """

    creation_schema_version: ClassVar[SchemaVersion | None] = None
    """The version of the Geoscience Object schema to use when creating new objects of this type.

    If None, this class can't create a new Geoscience Object, but can still load an existing one.
    """

    def __init__(self, context: IContext, obj: DownloadedObject) -> None:
        """
        :param context: The context containing the environment, connector, and cache to use.
        :param obj: The DownloadedObject representing the Geoscience Object.
        """
        self._context = context
        self._obj = obj
        self._document = obj.as_dict()

        # Initialize ModelContext for annotation-based sub-models
        self._model_context = ModelContext(obj=obj, root_model=self)

        self._reset_from_object()

        # Check whether the object that was loaded is valid
        self.validate()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        if cls.sub_classification is not None:
            existing_cls = cls._sub_classification_lookup.get(cls.sub_classification)
            if existing_cls is not None:
                raise ValueError(
                    f"Duplicate sub_classification '{cls.sub_classification}' for {cls.__name__}; "
                    f"already registered by {existing_cls.__name__}"
                )
            cls._sub_classification_lookup[cls.sub_classification] = cls
        if cls._data_class is not None:
            existing_cls = cls._data_class_lookup.get(cls._data_class)
            if existing_cls is not None:
                raise ValueError(
                    f"Duplicate data class '{cls._data_class.__name__}' for {cls.__name__}; "
                    f"already registered by {existing_cls.__name__}"
                )
            cls._data_class_lookup[cls._data_class] = cls

        # Gather schema properties and sub-models from parent classes
        cls._schema_properties = {}
        cls._sub_models = {}
        for parent in cls.__bases__:
            if issubclass(parent, _BaseObject):
                cls._schema_properties.update(parent._schema_properties)
                cls._sub_models.update(parent._sub_models)

        # Process directly assigned SchemaProperty descriptors
        for key, prop in cls.__dict__.items():
            if isinstance(prop, SchemaProperty):
                cls._schema_properties[key] = prop

        # Process Annotated[..., SchemaLocation(...)] annotations for sub-models and properties
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            return

        for field_name, annotation in hints.items():
            # Skip if already processed as a direct property
            if field_name in cls._schema_properties:
                continue

            # Extract SchemaLocation and DataLocation from Annotated types
            base_type, schema_location, data_location = _get_annotation_metadata(annotation)

            if schema_location is None:
                continue

            # Check if base type is a SchemaModel or SchemaList (sub-model)
            bare_base_type = get_origin(base_type) or base_type
            try:
                is_sub_model = isinstance(bare_base_type, type) and issubclass(bare_base_type, (SchemaModel, SchemaList))
            except TypeError:
                is_sub_model = False

            if is_sub_model:
                data_field = data_location.field_path if data_location else None
                cls._sub_models[field_name] = SubModelMetadata(
                    model_type=base_type,
                    jmespath_expr=schema_location.jmespath_expr,
                    data_field=data_field,
                )
            else:
                # Create a SchemaProperty for simple annotated fields
                type_adapter = TypeAdapter(annotation)
                prop = SchemaProperty(
                    jmespath_expr=schema_location.jmespath_expr,
                    type_adapter=type_adapter,
                )
                setattr(cls, field_name, prop)
                cls._schema_properties[field_name] = prop

    @classmethod
    async def _data_to_dict(cls, data: BaseObjectData, context: IContext) -> dict[str, Any]:
        """Convert the provided data to a dictionary suitable for creating a Geoscience Object.

        :param data: The BaseObjectData to convert.
        :param context: The context used to upload any data required for the object.
        :return: The dictionary representation of the data.
        """

        if cls.sub_classification is None or cls.creation_schema_version is None:
            raise NotImplementedError(
                f"Class '{cls.__name__}' cannot create new objects; "
                "sub_classification and creation_schema_version must be defined by the subclass"
            )
        schema_id = ObjectSchema("objects", cls.sub_classification, cls.creation_schema_version)
        result: dict[str, Any] = {
            "schema": str(schema_id),
        }

        # Handle schema properties
        for key, prop in cls._schema_properties.items():
            value = getattr(data, key, None)
            if value is not None:
                prop.apply_to(result, value)

        # Handle annotation-based sub-models
        for name, metadata in cls._sub_models.items():
            if metadata.data_field:
                sub_data = getattr(data, metadata.data_field, None)
            else:
                sub_data = data
            if sub_data is not None:
                from ._utils import assign_jmespath_value
                sub_document = await metadata.model_type._data_to_schema(sub_data, context)
                if metadata.jmespath_expr:
                    assign_jmespath_value(result, metadata.jmespath_expr, sub_document)
                else:
                    result.update(sub_document)

        return result

    @classmethod
    async def _data_to_schema(cls, data: BaseObjectData, context: IContext) -> dict[str, Any]:
        """Convert the provided data to a dictionary suitable for creating a Geoscience Object.

        This is an alias for _data_to_dict, providing compatibility with the new SchemaModel pattern.
        """
        return await cls._data_to_dict(data, context)

    @classmethod
    async def _create(
        cls,
        context: IContext,
        data: BaseObjectData,
        parent: str | None = None,
        path: str | None = None,
    ) -> Self:
        """Create a new object.

        :param context: The context containing the environment, connector, and cache to use.
        :param data: The data that will be used to create the object.
        :param parent: Optional parent path for the object.
        :param path: Full path to the object, can't be used with parent.
        """
        if type(data) is not cls._data_class:
            raise TypeError(f"Data must be of type '{cls._data_class.__name__}' to create a '{cls.__name__}' object.")

        # Take a copy to avoid the context changes affecting the object
        context = StaticContext.create_copy(context)
        object_dict = await cls._data_to_schema(data, context)
        object_dict["uuid"] = None  # New UUID is generated by the service
        obj = await create_geoscience_object(context, object_dict, parent, path)
        return cls(context, obj)

    @classmethod
    async def _replace(
        cls,
        context: IContext,
        reference: str,
        data: BaseObjectData,
        create_if_missing: bool = False,
    ) -> Self:
        """Replace an existing object.

        :param context: The context containing the environment, connector, and cache to use.
        :param reference: The reference of the object to replace.
        :param data: The data that will be used to create the object.
        """
        if type(data) is not cls._data_class:
            raise TypeError(f"Data must be of type '{cls._data_class.__name__}' to replace a '{cls.__name__}' object.")

        reference = ObjectReference(reference)
        # Context for the reference's workspace
        reference_context = StaticContext(
            connector=context.get_connector(),
            cache=context.get_cache(),
            org_id=reference.org_id,
            workspace_id=reference.workspace_id,
        )

        object_dict = await cls._data_to_schema(data, reference_context)
        obj = await replace_geoscience_object(
            reference_context, reference, object_dict, create_if_missing=create_if_missing
        )
        return cls(reference_context, obj)

    @classmethod
    def _get_object_type_from_data(cls, data: BaseObjectData) -> type[Self]:
        object_type = cls._data_class_lookup.get(type(data))
        if object_type is None:
            raise TypeError(f"No Typed Geoscience Object class found for data of type '{type(data).__name__}'")
        if not issubclass(object_type, cls):
            raise TypeError(f"Data of type '{type(data).__name__}' cannot be used to create a '{cls.__name__}' object")
        return object_type

    @classmethod
    def _adapt(cls, context: IContext, obj: DownloadedObject) -> Self:
        selected_cls = cls._sub_classification_lookup.get(obj.metadata.schema_id.sub_classification)
        if selected_cls is None:
            raise ValueError(f"No class found for sub-classification '{obj.metadata.schema_id.sub_classification}'")

        if not issubclass(selected_cls, cls):
            raise ValueError(
                f"Referenced object with sub-classification '{obj.metadata.schema_id.sub_classification}' "
                f"cannot be adapted to '{cls.__name__}'"
            )
        return selected_cls(context, obj)

    @classmethod
    async def from_reference(
        cls,
        context: IContext,
        reference: ObjectReference | str,
    ) -> Self:
        """Download a GeoscienceObject from the given reference, adapting it to this GeoscienceObject type.

        :param context: The context for connecting to Evo APIs.
        :param reference: The ObjectReference (or its string ID) identifying the object to download.

        :return: A GeoscienceObject instance.

        :raises ValueError: If the referenced object cannot be adapted to this GeoscienceObject type.
        """
        reference = ObjectReference(reference)
        # Context for the reference's workspace
        reference_context = StaticContext(
            connector=context.get_connector(),
            cache=context.get_cache(),
            org_id=reference.org_id,
            workspace_id=reference.workspace_id,
        )
        obj = await download_geoscience_object(reference_context, reference)
        return cls._adapt(reference_context, obj)

    @property
    def metadata(self) -> ObjectMetadata:
        """The metadata of the Geoscience Object.

        This does not include any local changes since the object was last updated.
        """
        return self._obj.metadata


    @property
    def viewer_url(self) -> str:
        """The URL to view the object in the Evo Viewer.

        :return: The viewer URL.
        """
        environment = self._obj.metadata.environment
        return get_viewer_url_from_environment(environment, str(self._obj.metadata.id))

    @property
    def portal_url(self) -> str:
        """The URL to view the object in the Evo Portal.

        :return: The portal URL.
        """
        environment = self._obj.metadata.environment
        return get_portal_url_from_environment(environment, str(self._obj.metadata.id))

    def as_dict(self) -> dict[str, Any]:
        """Get the Geoscience Object as a dictionary.

        :return: The Geoscience Object as a dictionary.
        """
        return copy.deepcopy(self._document)

    async def refresh(self) -> Self:
        """Refresh this object with the latest data from the server.

        Use this after a remote operation has updated the object to see
        any newly added attributes or modified data.

        :return: A new instance with refreshed data.

        Example:
            >>> # After a remote operation modifies the object...
            >>> obj = await obj.refresh()
            >>> obj.attributes  # Now shows the latest attributes
        """
        return await self.from_reference(self._context, self.metadata.url)

    def search(self, expression: str) -> Any:
        """Search the object metadata using a JMESPath expression.

        :param expression: The JMESPath expression to use for the search.

        :return: The result of the search.
        """
        return jmespath.search(expression, self._document)

    async def update(self):
        """Update the object on the geoscience object service

        :raise ObjectValidationError: If the object isn't valid.
        """
        self.validate()
        self._obj = await self._obj.update(self._document)

        # Reset the ModelContext to clear modified flags after successful update
        self._model_context = ModelContext(obj=self._obj, root_model=self)

        self._reset_from_object()

    def _reset_from_object(self) -> None:
        """Reset the object state from the underlying document."""
        self._rebuild_sub_models()

    def _rebuild_models(self) -> None:
        """Alias for _rebuild_sub_models for compatibility with SchemaModel interface."""
        self._rebuild_sub_models()

    def _rebuild_sub_models(self) -> None:
        """Rebuild any annotation-based sub-models to reflect changes in the underlying document."""
        for sub_model_name, metadata in self._sub_models.items():
            if metadata.jmespath_expr:
                sub_document = jmespath.search(metadata.jmespath_expr, self._document)
                if sub_document is None:
                    # Initialize an empty list/dict for the sub-model if not present
                    if issubclass(metadata.model_type, SchemaList):
                        sub_document = []
                    else:
                        sub_document = {}
                    from ._utils import assign_jmespath_value
                    assign_jmespath_value(self._document, metadata.jmespath_expr, sub_document)
                else:
                    # Unwrap jmespath proxy to get raw data for mutation
                    if hasattr(sub_document, 'raw'):
                        sub_document = sub_document.raw
            else:
                sub_document = self._document

            # Compute full path for nested context
            if metadata.jmespath_expr:
                if self._model_context.schema_path:
                    full_path = f"{self._model_context.schema_path}.{metadata.jmespath_expr}"
                else:
                    full_path = metadata.jmespath_expr
            else:
                full_path = self._model_context.schema_path

            # Create nested context with updated schema_path
            nested_context = ModelContext(
                obj=self._model_context.obj,
                root_model=self._model_context.root_model,
                data_modified=self._model_context.data_modified,
                schema_path=full_path,
            )

            sub_model = metadata.model_type(nested_context, sub_document)
            # Set _schema_path on models that support it (e.g., Attributes)
            if hasattr(sub_model, "_schema_path"):
                sub_model._schema_path = full_path
            setattr(self, sub_model_name, sub_model)

    def validate(self) -> None:
        """Validate the object to check if it is in a valid state.

        :raises ObjectValidationError: If the object isn't valid.
        """
        # Validate sub-models
        for sub_model_name in self._sub_models:
            sub_model = getattr(self, sub_model_name, None)
            if sub_model is not None and hasattr(sub_model, 'validate'):
                sub_model.validate()

    def _metadata_repr_html_(self) -> tuple[str, list[tuple[str, str]]]:
        """Return an HTML representation of common metadata for Jupyter notebooks.

        This renders the title with Portal/Viewer links and builds metadata rows.
        Subclasses should call this method and extend the rows with their specific content.

        :return: Tuple of (HTML string with stylesheet, opening div, and title, list of (label, value) rows).
        """
        doc = self.as_dict()

        # Get basic info
        name = doc.get("name", "Unnamed")
        schema = doc.get("schema", "Unknown")
        obj_id = doc.get("uuid", "Unknown")

        # Build title links for viewer and portal
        title_links = [("Portal", self.portal_url), ("Viewer", self.viewer_url)]

        # Build metadata rows
        rows = [
            ("Object ID:", str(obj_id)),
            ("Schema:", schema),
        ]

        # Add tags if present
        if tags := doc.get("tags"):
            tags_str = ", ".join(f"{k}: {v}" for k, v in tags.items())
            rows.append(("Tags:", tags_str))

        # Return opening HTML with stylesheet, container div, title, and the rows
        opening_html = (
            f'{STYLESHEET}'
            f'<div class="evo">'
            f'{build_title(name, title_links)}'
        )
        return opening_html, rows

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter notebooks."""
        doc = self.as_dict()

        # Start with common metadata
        opening_html, rows = self._metadata_repr_html_()

        # Add bounding box if present (as nested table)
        if bbox := doc.get("bounding_box"):
            bbox_rows = [
                ["<strong>X:</strong>", bbox.get('min_x', 0), bbox.get('max_x', 0)],
                ["<strong>Y:</strong>", bbox.get('min_y', 0), bbox.get('max_y', 0)],
                ["<strong>Z:</strong>", bbox.get('min_z', 0), bbox.get('max_z', 0)],
            ]
            bbox_table = build_nested_table(["", "Min", "Max"], bbox_rows)
            rows.append(("Bounding box:", bbox_table))

        # Add CRS if present
        if crs := doc.get("coordinate_reference_system"):
            crs_str = f"EPSG:{crs.get('epsg_code')}" if isinstance(crs, dict) else str(crs)
            rows.append(("CRS:", crs_str))

        # Build datasets section - add as rows to the main table
        for dataset_name in self._sub_models:
            dataset = getattr(self, dataset_name, None)
            if dataset and hasattr(dataset, 'attributes') and len(dataset.attributes) > 0:
                # Build attribute rows
                attr_rows = []
                for attr in dataset.attributes:
                    attr_info = attr.as_dict()
                    attr_name = attr_info.get("name", "Unknown")
                    attr_type = attr_info.get("attribute_type", "Unknown")
                    attr_rows.append([attr_name, attr_type])

                attrs_table = build_nested_table(["Attribute", "Type"], attr_rows)
                rows.append((f"{dataset_name}:", attrs_table))

        # Build unified table with all rows
        table_rows = []
        for label, value in rows:
            if label in ("Bounding box:",) or label.endswith(":") and isinstance(value, str) and "<table" in value:
                table_rows.append(build_table_row_vtop(label, value))
            else:
                table_rows.append(build_table_row(label, value))

        html = opening_html
        if table_rows:
            html += f'<table>{"".join(table_rows)}</table>'

        # Close the container div
        html += '</div>'
        return html


@dataclass(kw_only=True, frozen=True)
class BaseObjectData:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    extensions: dict[str, Any] | None = None


class BaseObject(_BaseObject):
    """Base object for all Geoscience Objects, containing common properties."""

    name: Annotated[str, SchemaLocation("name")]
    description: Annotated[str | None, SchemaLocation("description")]
    tags: Annotated[dict[str, str], SchemaLocation("tags")] = {}
    extensions: Annotated[dict, SchemaLocation("extensions")] = {}

    @classmethod
    def create(
        cls,
        context: IContext,
        data: BaseObjectData,
        parent: str | None = None,
        path: str | None = None,
    ) -> BaseObject:
        """Create a new object.

        The type of Geoscience Object created is determined by the type of `data` provided.

        :param context: The context containing the environment, connector, and cache to use.
        :param data: The data that will be used to create the object.
        :param parent: Optional parent path for the object.
        :param path: Full path to the object, can't be used with parent.
        """
        object_type = cls._get_object_type_from_data(data)
        return object_type._create(context, data, parent, path)

    @classmethod
    async def replace(
        cls,
        context: IContext,
        reference: str,
        data: BaseObjectData,
    ) -> BaseObject:
        """Replace an existing object.

        The type of Geoscience Object that will be replaced is determined by the type of `data` provided. This must match
        the type of the existing object.

        :param context: The context containing the environment, connector, and cache to use.
        :param reference: The reference of the object to replace.
        :param data: The data that will be used to create the object.
        """
        object_type = cls._get_object_type_from_data(data)
        return await object_type._replace(context, reference, data)

    @classmethod
    async def create_or_replace(
        cls,
        context: IContext,
        reference: str,
        data: BaseObjectData,
    ) -> BaseObject:
        """Create or replace an existing object.

        If the object identified by `reference` exists, it will be replaced. Otherwise, a new object will be created.

        The type of Geoscience Object that will be created or replaced is determined by the type of `data` provided. This
        must match the type of the existing object if it already exists.

        :param context: The context containing the environment, connector, and cache to use.
        :param reference: The reference of the object to create or replace.
        :param data: The data that will be used to create the object.
        """
        object_type = cls._get_object_type_from_data(data)
        return await object_type._replace(context, reference, data, create_if_missing=True)


class ConstructableObject(BaseObject, Generic[_T]):
    # The class methods in this class technically violate Liskov Substitution Principle,
    # as they narrow the type of the 'data' parameter.
    #
    @classmethod
    async def create(
        cls,
        context: IContext,
        data: _T,
        parent: str | None = None,
        path: str | None = None,
    ) -> Self:
        """Create a new object.

        :param context: The context containing the environment, connector, and cache to use.
        :param data: The data that will be used to create the object.
        :param parent: Optional parent path for the object.
        :param path: Full path to the object, can't be used with parent.
        """
        return await cls._create(context, data, parent, path)

    @classmethod
    async def replace(
        cls,
        context: IContext,
        reference: str,
        data: _T,
    ) -> Self:
        """Replace an existing object.

        :param context: The context containing the environment, connector, and cache to use.
        :param reference: The reference of the object to replace.
        :param data: The data that will be used to create the object.
        """
        return await cls._replace(context, reference, data)

    @classmethod
    async def create_or_replace(
        cls,
        context: IContext,
        reference: str,
        data: _T,
    ) -> Self:
        """Create or replace an existing object.

        If the object identified by `reference` exists, it will be replaced. Otherwise, a new object will be created.

        :param context: The context containing the environment, connector, and cache to use.
        :param reference: The reference of the object to create or replace.
        :param data: The data that will be used to create the object.
        """
        return await cls._replace(context, reference, data, create_if_missing=True)


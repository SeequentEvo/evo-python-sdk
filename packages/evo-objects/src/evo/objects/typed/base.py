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
from typing import Any, ClassVar, Generic, TypeVar, overload

from pydantic import TypeAdapter

from evo import jmespath
from evo.common import EvoContext
from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference, ObjectSchema, SchemaVersion

from ._adapters import AttributesAdapter, CategoryTableAdapter, DatasetAdapter, TableAdapter
from ._property import SchemaProperty
from ._utils import (
    create_geoscience_object,
    download_geoscience_object,
    replace_geoscience_object,
)
from .dataset import Dataset
from .types import BoundingBox, CoordinateReferenceSystem, EpsgCode

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "BaseObject",
    "BaseObjectData",
    "BaseSpatialObject",
    "BaseSpatialObjectData",
    "DatasetProperty",
    "SchemaProperty",
]

_T = TypeVar("_T")


class DatasetProperty(Generic[_T]):
    """Descriptor for datasets within a Geoscience Object."""

    def __init__(
        self,
        dataset_class: type[Dataset],
        value_adapters: list[TableAdapter | CategoryTableAdapter],
        attributes_adapters: list[AttributesAdapter],
        data_attribute: str | None = None,
    ) -> None:
        self._name = None
        self.dataset_class = dataset_class
        self._value_adapters = value_adapters
        self._attributes_adapters = attributes_adapters
        self._data_attribute = data_attribute

    def __set_name__(self, owner: type[BaseObject], name: str):
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type[BaseObject]) -> SchemaProperty[_T]: ...

    @overload
    def __get__(self, instance: BaseObject, owner: type[BaseObject]) -> _T: ...

    def __get__(self, instance: BaseObject | None, owner: type[BaseObject]) -> Any:
        if instance is None:
            return self
        return instance.get_dataset_by_name(self._name)

    def get_adapter(self, schema_id: ObjectSchema) -> DatasetAdapter:
        return DatasetAdapter.from_adapter_lists(
            schema_id.version.major,
            self._value_adapters,
            self._attributes_adapters,
        )

    def get_data_attribute(self) -> str:
        if self._data_attribute is not None:
            return self._data_attribute
        return self._name


class _BaseObject:
    """Base class for high-level Geoscience Objects."""

    _sub_classification_lookup: ClassVar[weakref.WeakValueDictionary[str, type[_BaseObject]]] = (
        weakref.WeakValueDictionary()
    )
    _schema_properties: ClassVar[dict[str, SchemaProperty[Any]]] = {}
    _dataset_properties: ClassVar[dict[str, DatasetProperty[Any]]] = {}

    sub_classification: ClassVar[str | None] = None
    """The sub-classification of the Geoscience Object schema.

    If None, this class is considered abstract and cannot be instantiated directly.
    """

    creation_schema_version: ClassVar[SchemaVersion | None] = None
    """The version of the Geoscience Object schema to use when creating new objects of this type.

    If None, this class can't create a new Geoscience Object, but can still load an existing one.
    """

    def __init__(self, evo_context: EvoContext, obj: DownloadedObject) -> None:
        """
        :param evo_context: The context containing the environment, connector, and cache to use.
        :param obj: The DownloadedObject representing the Geoscience Object.
        """
        self._evo_context = evo_context
        self._obj = obj
        self._document = obj.as_dict()
        self._datasets = {}
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

        # Gather schema and dataset properties, both from this class and parent classes
        cls._schema_properties = {}
        cls._dataset_properties = {}
        for parent in cls.__bases__:
            if issubclass(parent, _BaseObject):
                cls._schema_properties.update(parent._schema_properties)
                cls._dataset_properties.update(parent._dataset_properties)
        for key, prop in cls.__dict__.items():
            if isinstance(prop, SchemaProperty):
                cls._schema_properties[key] = prop
            if isinstance(prop, DatasetProperty):
                cls._dataset_properties[key] = prop

    @classmethod
    async def _data_to_dict(cls, data: BaseObjectData, evo_context: EvoContext) -> dict[str, Any]:
        """Convert the provided data to a dictionary suitable for creating a Geoscience Object.

        :param data: The BaseObjectData to convert.
        :param evo_context: The context used to upload any data required for the object.
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
        for key, prop in cls._schema_properties.items():
            prop.apply_to(result, getattr(data, key))
        for key, prop in cls._dataset_properties.items():
            adapter = prop.get_adapter(schema_id)
            dataset = Dataset.create_empty(document=result, dataset_adapter=adapter, evo_context=evo_context)
            data_value = getattr(data, prop.get_data_attribute())
            if data_value is not None:
                await dataset.set_dataframe(getattr(data, prop.get_data_attribute()))
                # Ensure the object_dict is updated with any changes made during set_dataframe
                dataset.update_document()

        return result

    @classmethod
    async def _create(
        cls,
        evo_context: EvoContext,
        data: BaseObjectData,
        parent: str | None = None,
    ) -> Self:
        """Create a new object.

        :param evo_context: The context containing the environment, connector, and cache to use.
        :param data: The data that will be used to create the object.
        :param parent: Optional parent path for the object.
        """

        object_dict = await cls._data_to_dict(data, evo_context)
        object_dict["uuid"] = None  # New UUID is generated by the service
        obj = await create_geoscience_object(evo_context, object_dict, parent)
        return cls(evo_context, obj)

    @classmethod
    async def _replace(
        cls,
        evo_context: EvoContext,
        reference: str,
        data: BaseObjectData,
    ) -> Self:
        """Replace an existing object.

        :param evo_context: The context containing the environment, connector, and cache to use.
        :param reference: The reference of the object to replace.
        :param data: The data that will be used to create the object.
        """
        reference = ObjectReference(reference)
        # Context for the reference's workspace
        reference_context = EvoContext(
            connector=evo_context.get_connector(),
            cache=evo_context.cache,
            org_id=reference.org_id,
            workspace_id=reference.workspace_id,
        )

        object_dict = await cls._data_to_dict(data, reference_context)
        obj = await replace_geoscience_object(reference_context, reference, object_dict)
        return cls(reference_context, obj)

    @classmethod
    def _adapt(cls, evo_context: EvoContext, obj: DownloadedObject) -> Self:
        selected_cls = cls._sub_classification_lookup.get(obj.metadata.schema_id.sub_classification)
        if selected_cls is None:
            raise ValueError(f"No class found for sub-classification '{obj.metadata.schema_id.sub_classification}'")

        if not issubclass(selected_cls, cls):
            raise ValueError(
                f"Referenced object with sub-classification '{obj.metadata.schema_id.sub_classification}' "
                f"cannot be adapted to '{cls.__name__}'"
            )
        return selected_cls(evo_context, obj)

    @classmethod
    async def from_reference(
        cls,
        evo_context: EvoContext,
        reference: ObjectReference | str,
    ) -> Self:
        """Download a GeoscienceObject from the given reference, adapting it to this GeoscienceObject type.

        :param evo_context: The context for connecting to Evo APIs.
        :param reference: The ObjectReference (or its string ID) identifying the object to download.

        :return: A GeoscienceObject instance.

        :raises ValueError: If the referenced object cannot be adapted to this GeoscienceObject type.
        """
        reference = ObjectReference(reference)
        # Context for the reference's workspace
        reference_context = EvoContext(
            connector=evo_context.get_connector(),
            cache=evo_context.cache,
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

    def as_dict(self) -> dict[str, Any]:
        """Get the Geoscience Object as a dictionary.

        :return: The Geoscience Object as a dictionary.
        """
        for dataset in self._datasets.values():
            dataset.update_document()
        return copy.deepcopy(self._document)

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
        for dataset in self._datasets.values():
            dataset.update_document()
        self._obj = await self._obj.update(self._document)
        self._reset_from_object()

    def get_dataset_by_name(self, name: str) -> Dataset:
        """Get the dataset by its name.

        :param name: The name of the dataset.

        :return: The Dataset instance.

        :raises KeyError: If the dataset with the specified name does not exist.
        """
        return self._datasets[name]

    def _reset_from_object(self) -> None:
        self._datasets = {
            key: prop.dataset_class(
                document=self._document,
                dataset_adapter=prop.get_adapter(self._obj.metadata.schema_id),
                obj=self._obj,
                evo_context=self._evo_context,
            )
            for key, prop in self._dataset_properties.items()
        }

    def validate(self) -> None:
        """Validate the object to check if it is in a valid state.

        :raises ObjectValidationError: If the object isn't valid.
        """
        for dataset in self._datasets.values():
            dataset.validate()


@dataclass(kw_only=True, frozen=True)
class BaseObjectData:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    extensions: dict[str, Any] | None = None


class BaseObject(_BaseObject):
    """Base object for all Geoscience Objects, containing common properties."""

    name: str = SchemaProperty("name", TypeAdapter(str))
    description: str | None = SchemaProperty("description", TypeAdapter(str | None))
    tags: dict[str, str] = SchemaProperty("tags", TypeAdapter(dict[str, str]), default_factory=dict)
    extensions: dict = SchemaProperty("extensions", TypeAdapter(dict), default_factory=dict)


@dataclass(kw_only=True, frozen=True)
class BaseSpatialObjectData(BaseObjectData):
    coordinate_reference_system: EpsgCode | str | None = None

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box for the object based on its datasets.

        :return: The computed bounding box.

        :raises ValueError: If the bounding box cannot be computed from the datasets.
        """
        raise NotImplementedError("Subclasses must implement compute_bounding_box to derive bounding box from data.")


class BaseSpatialObject(BaseObject):
    """Base class for all Geoscience Objects with spatial data."""

    _bbox_typed_adapter: ClassVar[TypeAdapter[BoundingBox]] = TypeAdapter(BoundingBox)
    coordinate_reference_system: EpsgCode | str | None = SchemaProperty(
        "coordinate_reference_system", TypeAdapter(CoordinateReferenceSystem)
    )

    @classmethod
    async def _data_to_dict(cls, data: BaseSpatialObjectData, evo_context: EvoContext) -> dict[str, Any]:
        """Create a object dictionary suitable for creating a new Geoscience Object."""
        object_dict = await super()._data_to_dict(data, evo_context)
        object_dict["bounding_box"] = cls._bbox_typed_adapter.dump_python(data.compute_bounding_box())
        return object_dict

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box for the object based on its datasets.

        :return: The computed bounding box.

        :raises ValueError: If the bounding box cannot be computed from the datasets.
        """
        raise NotImplementedError("Subclasses must implement compute_bounding_box to derive bounding box from data.")

    @property
    def bounding_box(self) -> BoundingBox:
        return self.compute_bounding_box()

    @bounding_box.setter
    def bounding_box(self, value: BoundingBox) -> None:
        raise AttributeError("Cannot set bounding_box on this object, as it is dynamically derived from the data.")

    async def update(self):
        """Update the object on the geoscience object service, including recomputing the bounding box."""

        # Update the bounding box in the document
        self._document["bounding_box"] = self._bbox_typed_adapter.dump_python(self.compute_bounding_box())
        await super().update()

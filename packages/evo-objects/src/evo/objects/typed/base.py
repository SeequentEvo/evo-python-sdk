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
from typing import Any, Callable, ClassVar, Generic, TypeVar, overload

from pydantic import TypeAdapter

from evo import jmespath
from evo.common import IContext, StaticContext
from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference, ObjectSchema, SchemaVersion

from ._adapters import AttributesAdapter, CategoryTableAdapter, DatasetAdapter, TableAdapter
from ._property import SchemaProperty
from ._utils import (
    create_geoscience_object,
    download_geoscience_object,
    replace_geoscience_object,
)
from .dataset import Dataset

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "BaseObject",
    "BaseObjectData",
    "ConstructableObject",
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
        extract_data: Callable[[Any], Any] | None = None,
    ) -> None:
        self._name = None
        self.dataset_class = dataset_class
        self._value_adapters = value_adapters
        self._attributes_adapters = attributes_adapters
        self._extract_data = extract_data

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

    def extract_data(self, data: Any) -> Any:
        if self._extract_data is not None:
            return self._extract_data(data)
        return None


class _BaseObject:
    """Base class for high-level Geoscience Objects."""

    _sub_classification_lookup: ClassVar[weakref.WeakValueDictionary[str, type[_BaseObject]]] = (
        weakref.WeakValueDictionary()
    )
    _schema_properties: ClassVar[dict[str, SchemaProperty[Any]]] = {}
    _dataset_properties: ClassVar[dict[str, DatasetProperty[Any]]] = {}

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
        if cls._data_class is not None:
            existing_cls = cls._data_class_lookup.get(cls._data_class)
            if existing_cls is not None:
                raise ValueError(
                    f"Duplicate data class '{cls._data_class.__name__}' for {cls.__name__}; "
                    f"already registered by {existing_cls.__name__}"
                )
            cls._data_class_lookup[cls._data_class] = cls

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
        for key, prop in cls._schema_properties.items():
            prop.apply_to(result, getattr(data, key))
        for key, prop in cls._dataset_properties.items():
            adapter = prop.get_adapter(schema_id)
            dataset_data = prop.extract_data(data)
            dataset = await prop.dataset_class.create_from_data(
                document=result, data=dataset_data, dataset_adapter=adapter, context=context
            )
            # Ensure the object_dict is updated with any changes made during set_dataframe
            dataset.update_document()
        return result

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
        object_dict = await cls._data_to_dict(data, context)
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

        object_dict = await cls._data_to_dict(data, reference_context)
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
                context=self._context,
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

    @classmethod
    async def create(
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
        return await object_type._create(context, data, parent, path)

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

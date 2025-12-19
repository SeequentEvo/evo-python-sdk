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

import sys
import weakref
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from pydantic import Field

from evo.common import IContext, StaticContext
from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference, ObjectSchema, SchemaVersion

from ._model import SchemaLocation, SchemaModel
from ._utils import (
    create_geoscience_object,
    replace_geoscience_object,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "BaseObject",
    "BaseObjectData",
]


@dataclass(kw_only=True, frozen=True)
class BaseObjectData:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    extensions: dict[str, Any] | None = None


class _BaseObject(SchemaModel):
    """Base class for high-level Geoscience Objects."""

    _data_class: ClassVar[type[BaseObjectData] | None] = None
    """The data class associated with this object type."""

    _data_class_lookup: ClassVar[weakref.WeakValueDictionary[type[BaseObjectData], type[_BaseObject]]] = (
        weakref.WeakValueDictionary()
    )

    _sub_classification_lookup: ClassVar[weakref.WeakValueDictionary[str, type[_BaseObject]]] = (
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

    def __init__(self, obj: DownloadedObject) -> None:
        """
        :param context: The context containing the environment, connector, and cache to use.
        :param obj: The DownloadedObject representing the Geoscience Object.
        """
        self._document = obj.as_dict()
        super().__init__(obj, self._document)

        # Check whether the object that was loaded is valid
        self.validate()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        # Register data class
        data_class = cls.__dict__.get("_data_class")
        if data_class is not None:
            existing_cls = cls._data_class_lookup.get(data_class)
            if existing_cls is not None:
                raise ValueError(
                    f"Duplicate data class '{data_class.__name__}' for {cls.__name__}; "
                    f"already registered by {existing_cls.__name__}"
                )
            cls._data_class_lookup[data_class] = cls
        # Register sub-classification
        if cls.sub_classification is not None:
            existing_cls = cls._sub_classification_lookup.get(cls.sub_classification)
            if existing_cls is not None:
                raise ValueError(
                    f"Duplicate sub_classification '{cls.sub_classification}' for {cls.__name__}; "
                    f"already registered by {existing_cls.__name__}"
                )
            cls._sub_classification_lookup[cls.sub_classification] = cls

    @classmethod
    def _get_object_type_from_data(cls, data: BaseObjectData) -> type[Self]:
        object_type = cls._data_class_lookup.get(type(data))
        if object_type is None:
            raise TypeError(f"No Typed Geoscience Object class found for data of type '{type(data).__name__}'")
        if not issubclass(object_type, cls):
            raise TypeError(f"Data of type '{type(data).__name__}' cannot be used to create a '{cls.__name__}' object")
        return object_type

    @classmethod
    async def _create(
        cls,
        context: IContext,
        data: BaseObjectData,
        parent: str | None = None,
        path: str | None = None,
    ) -> Self:
        if type(data) is not cls._data_class:
            raise TypeError(f"Data must be of type '{cls._data_class.__name__}' to create a '{cls.__name__}' object.")

        # Take a copy to avoid the context changes affecting the object
        context = StaticContext.create_copy(context)
        object_dict = await cls._data_to_schema(data, context)
        object_dict["uuid"] = None  # New UUID is generated by the service
        obj = await create_geoscience_object(context, object_dict, parent, path)
        return cls(obj)

    @classmethod
    async def create(
        cls,
        context: IContext,
        data: BaseObjectData,
        parent: str | None = None,
        path: str | None = None,
    ) -> Self:
        """Create a new object.

        The type of Geoscience Object created is determined by the type of `data` provided.
        Though if this method is called on a subclass, the created object will always be of that subclass type.

        :param context: The context containing the environment, connector, and cache to use.
        :param data: The data that will be used to create the object.
        :param parent: Optional parent path for the object.
        :param path: Full path to the object, can't be used with parent.
        """
        object_type = cls._get_object_type_from_data(data)
        return await object_type._create(context, data, parent, path)

    @classmethod
    async def _replace(
        cls,
        context: IContext,
        reference: str,
        data: BaseObjectData,
        create_if_missing: bool = False,
    ) -> Self:
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
        return cls(obj)

    @classmethod
    async def replace(
        cls,
        context: IContext,
        reference: str,
        data: BaseObjectData,
    ) -> Self:
        """Replace an existing object.

        The type of Geoscience Object that will be replaced is determined by the type of `data` provided. This must match
        the type of the existing object.

        Though if this method is called on a subclass, the replaced object will always be of that subclass type.

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
    ) -> Self:
        """Create or replace an existing object.

        If the object identified by `reference` exists, it will be replaced. Otherwise, a new object will be created.

        The type of Geoscience Object that will be created or replaced is determined by the type of `data` provided. This
        must match the type of the existing object if it already exists.

        Though if this method is called on a subclass, the created or replaced object will always be of that subclass type.

        :param context: The context containing the environment, connector, and cache to use.
        :param reference: The reference of the object to create or replace.
        :param data: The data that will be used to create the object.
        """
        object_type = cls._get_object_type_from_data(data)
        return await object_type._replace(context, reference, data, create_if_missing=True)

    @classmethod
    async def _data_to_schema(cls, data: BaseObjectData, context: IContext) -> dict[str, Any]:
        if cls.sub_classification is None or cls.creation_schema_version is None:
            raise TypeError(
                f"Class '{cls.__name__}' cannot create new objects; "
                "sub_classification and creation_schema_version must be defined by the subclass"
            )
        result = await super()._data_to_schema(data, context)
        schema_id = ObjectSchema("objects", cls.sub_classification, cls.creation_schema_version)
        result["schema"] = str(schema_id)
        return result

    @classmethod
    def _adapt(cls, obj: DownloadedObject) -> Self:
        selected_cls = cls._sub_classification_lookup.get(obj.metadata.schema_id.sub_classification)
        if selected_cls is None:
            raise ValueError(f"No class found for sub-classification '{obj.metadata.schema_id.sub_classification}'")

        if not issubclass(selected_cls, cls):
            raise ValueError(
                f"Referenced object with sub-classification '{obj.metadata.schema_id.sub_classification}' "
                f"cannot be adapted to '{cls.__name__}'"
            )
        return selected_cls(obj)

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
        obj = await DownloadedObject.from_context(reference_context, reference)
        return cls._adapt(obj)

    @property
    def metadata(self) -> ObjectMetadata:
        """The metadata of the Geoscience Object.

        This does not include any local changes since the object was last updated.
        """
        return self._obj.metadata

    async def update(self):
        """Update the object on the geoscience object service

        :raise ObjectValidationError: If the object isn't valid.
        """
        self.validate()
        self._context.obj = await self._obj.update(self._document)
        self._context.data_modified.clear()
        self._rebuild_models()


class BaseObject(_BaseObject):
    """Base object for all Geoscience Objects, containing common properties."""

    name: Annotated[str, SchemaLocation("name")]
    description: Annotated[str | None, SchemaLocation("description")]
    tags: Annotated[dict[str, str], SchemaLocation("tags"), Field(default_factory=dict)]
    extensions: Annotated[dict, SchemaLocation("extensions"), Field(default_factory=dict)]

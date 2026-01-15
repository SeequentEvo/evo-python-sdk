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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Generic, TypeVar, get_args, get_origin, get_type_hints, overload

from pydantic import TypeAdapter

from evo import jmespath
from evo.common import IContext
from evo.objects import DownloadedObject

from ._utils import (
    assign_jmespath_value,
    delete_jmespath_value,
)

_T = TypeVar("_T")


@dataclass
class ModelContext:
    """Context for the schema models."""

    obj: DownloadedObject
    """The DownloadedObject associated with this context."""

    data_modified: set[str] = field(default_factory=set)
    """Flags indicating which data fields have been modified.
    """

    def mark_modified(self, data_ref: str) -> None:
        """Mark that a specific data field has been modified and should not be loaded."""
        self.data_modified.add(data_ref)

    def is_data_modified(self, data_ref: str) -> bool:
        """Check if a specific data field has been marked as modified."""
        return data_ref in self.data_modified


@dataclass
class SchemaLocation:
    """Metadata for annotating a field's location within a Geoscience Object schema."""

    jmespath_expr: str
    """The JMESPath expression to locate this field in the document."""


@dataclass
class DataLocation:
    """Metadata for annotating a field's location within a data classes used to generate Geoscience Object data."""

    field_path: str
    """The path to the field within the data class."""


@dataclass
class SubModelMetadata:
    """Metadata for sub-model fields within a SchemaModel."""

    model_type: type[SchemaModel]
    """The type of the sub-model."""

    jmespath_expr: str | None
    """The JMESPath expression locating the sub-model in the document."""

    data_field: str | None
    """The field name in the data class for this sub-model."""


class SchemaProperty(Generic[_T]):
    """Descriptor for data within a Geoscience Object schema.

    This can be used on either typed objects classes or dataset classes.
    """

    def __init__(
        self,
        jmespath_expr: str,
        type_adapter: TypeAdapter[_T],
    ) -> None:
        self._jmespath_expr = jmespath_expr
        self._type_adapter = type_adapter

    @overload
    def __get__(self, instance: None, owner: type[SchemaModel]) -> SchemaProperty[_T]: ...

    @overload
    def __get__(self, instance: SchemaModel, owner: type[SchemaModel]) -> _T: ...

    def __get__(self, instance: SchemaModel | None, owner: type[SchemaModel]) -> Any:
        if instance is None:
            return self

        value = instance.search(self._jmespath_expr)
        if isinstance(value, (jmespath.JMESPathArrayProxy, jmespath.JMESPathObjectProxy)):
            value = value.raw
        # Use TypeAdapter to validate and apply defaults from Field annotations
        return self._type_adapter.validate_python(value)

    def __set__(self, instance: SchemaModel, value: Any) -> None:
        self.apply_to(instance._document, value)

    def apply_to(self, document: dict[str, Any], value: _T) -> None:
        dumped_value = self._type_adapter.dump_python(value)

        if dumped_value is None:
            # Remove the property from the document if the value is None
            delete_jmespath_value(document, self._jmespath_expr)
        else:
            # Update the document with the new value
            assign_jmespath_value(document, self._jmespath_expr, dumped_value)


def _get_base_type(annotation: Any) -> tuple[Any, SchemaLocation | None, DataLocation | None]:
    """Extract the base type and SchemaLocation from an annotation.

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


class SchemaModel:
    """Base class for models backed by a Geoscience Object schema.

    The data is stored in the underlying document dictionary. Sub-models are
    automatically created for nested SchemaModel/SchemaList fields.
    """

    _schema_properties: dict[str, SchemaProperty[Any]] = {}
    _sub_models: dict[str, SubModelMetadata] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Initialize with inherited values (copy to avoid mutating parent)
        schema_properties: dict[str, SchemaProperty[Any]] = {}
        sub_models: dict[str, SubModelMetadata] = {}
        for base in cls.__mro__[1:]:
            if issubclass(base, SchemaModel):
                schema_properties.update(base._schema_properties)
                sub_models.update(base._sub_models)

        # Resolve string annotations using get_type_hints
        try:
            # include_extras=True preserves Annotated metadata
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            # If get_type_hints fails, use inherited values
            cls._schema_properties = schema_properties
            cls._sub_models = sub_models
            return

        # Process the resolved annotations
        for field_name, annotation in hints.items():
            base_type, schema_location, data_location = _get_base_type(annotation)

            # Skip fields without a SchemaLocation
            if schema_location is None:
                continue

            # To robustly check for SchemaModel/SchemaList, we need to strip any generic or Annotated wrappers
            bare_base_type = get_origin(base_type) or base_type
            if issubclass(bare_base_type, (SchemaModel, SchemaList)):
                data_field = data_location.field_path if data_location else None
                sub_models[field_name] = SubModelMetadata(
                    model_type=base_type,
                    jmespath_expr=schema_location.jmespath_expr,
                    data_field=data_field,
                )
            else:
                # Create a TypeAdapter for the full annotation (preserves Field defaults)
                type_adapter = TypeAdapter(annotation)

                # Create SchemaProperty descriptor and set it on the class
                prop = SchemaProperty(
                    jmespath_expr=schema_location.jmespath_expr,
                    type_adapter=type_adapter,
                )
                setattr(cls, field_name, prop)
                schema_properties[field_name] = prop

        cls._schema_properties = schema_properties
        cls._sub_models = sub_models

    def __init__(self, context: ModelContext | DownloadedObject, document: dict[str, Any]) -> None:
        """Initialize the SchemaModel.

        :param context: Either a ModelContext or a DownloadedObject this model is associated with.
        :param document: The document dictionary representing the Geoscience Object.
        """
        if isinstance(context, DownloadedObject):
            self._context = ModelContext(obj=context)
        else:
            self._context = context
        self._document = document

        self._rebuild_models()

    @property
    def _obj(self) -> DownloadedObject:
        """Get the DownloadedObject for this model.

        :raises DataLoaderError: If this model was created without a DownloadedObject.
        """
        return self._context.obj

    def _rebuild_models(self) -> None:
        """Rebuild any sub-models to reflect changes in the underlying document."""
        for sub_model_name, metadata in self._sub_models.items():
            if metadata.jmespath_expr:
                sub_document = jmespath.search(metadata.jmespath_expr, self._document)
                if sub_document is None:
                    # Initialize an empty list/dict for the sub-model if not present
                    if issubclass(metadata.model_type, SchemaList):
                        sub_document = []
                    else:
                        sub_document = {}
                    assign_jmespath_value(self._document, metadata.jmespath_expr, sub_document)
                else:
                    # Unwrap jmespath proxy to get raw data for mutation
                    sub_document = sub_document.raw
            else:
                sub_document = self._document
            setattr(self, sub_model_name, metadata.model_type(self._context, sub_document))

    def validate(self) -> None:
        """Validate the model is valid."""
        for sub_model_name in self._sub_models:
            sub_model = getattr(self, sub_model_name, None)
            if sub_model is not None:
                sub_model.validate()

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> Any:
        """Convert data to a dictionary by applying schema properties.

        This base implementation iterates over all schema properties defined on the class
        and applies their values from the data object to the result dictionary.

        :param data: The data object containing values to convert.
        :param context: The context used for any data upload operations.
        :return: The dictionary representation of the data.
        """
        result: dict[str, Any] = {}
        for key, prop in cls._schema_properties.items():
            value = getattr(data, key, None)
            prop.apply_to(result, value)
        for metadata in cls._sub_models.values():
            if metadata.data_field:
                sub_data = getattr(data, metadata.data_field, None)
            else:
                sub_data = data

            sub_dict = await metadata.model_type._data_to_schema(sub_data, context)
            if metadata.jmespath_expr:
                assign_jmespath_value(result, metadata.jmespath_expr, sub_dict)
            else:
                result.update(sub_dict)
        return result

    def search(self, expression: str) -> Any:
        """Search the model using a JMESPath expression.

        :param expression: The JMESPath expression to use for the search.

        :return: The result of the search.
        """
        return jmespath.search(expression, self._document)

    def as_dict(self) -> dict[str, Any]:
        """Get the model as a dictionary.

        :return: The model as a dictionary.
        """
        return copy.deepcopy(self._document)


_M = TypeVar("_M", bound=SchemaModel)


class SchemaList(Sequence[_M]):
    """A list of SchemaModel instances backed by a list in the document."""

    _item_type: type[_M]

    def __init__(self, context: ModelContext | DownloadedObject, document: list[Any] | None) -> None:
        if isinstance(context, DownloadedObject):
            self._context = ModelContext(obj=context)
        else:
            self._context = context
        self._document = document if document is not None else []

    @property
    def _obj(self) -> DownloadedObject:
        """Get the DownloadedObject for this model list."""
        return self._context.obj

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Extract the item type from the generic parameter
        for base in cls.__orig_bases__:
            if get_origin(base) is SchemaList:
                args = get_args(base)
                if args:
                    cls._item_type = args[0]
                    break

    def __getitem__(self, index: int) -> _M:
        return self._item_type(self._context, self._document[index])

    def __iter__(self):
        for item in self._document:
            yield self._item_type(self._context, item)

    def __len__(self) -> int:
        return len(self._document)

    def clear(self) -> None:
        """Clear all items from the list."""
        self._document.clear()

    def _append(self, value: _M) -> None:
        """Append an item to the list."""
        self._document.append(value.as_dict())

    def validate(self) -> None:
        """Validate all items in the list."""
        for item in self:
            item.validate()

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> list[Any]:
        if data is None:
            return []
        if not isinstance(data, Sequence):
            raise TypeError(f"Expected a sequence for SchemaList data, got {type(data).__name__}")
        return [await cls._item_type._data_to_schema(item, context) for item in data]

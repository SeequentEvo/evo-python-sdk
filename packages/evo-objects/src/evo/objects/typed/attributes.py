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

import uuid
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import pandas as pd

from evo import jmespath
from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback, iter_with_fb
from evo.objects import DownloadedObject
from evo.objects.utils.table_formats import (
    BOOL_ARRAY_1,
    FLOAT_ARRAY_1,
    INTEGER_ARRAY_1_INT32,
    INTEGER_ARRAY_1_INT64,
    STRING_ARRAY,
)

from ._model import SchemaList, SchemaLocation, SchemaModel
from ._utils import get_data_client
from .exceptions import DataLoaderError, ObjectValidationError

if TYPE_CHECKING:
    from .block_model_ref import BlockModel

__all__ = [
    "Attribute",
    "Attributes",
    "BlockModelAttribute",
    "BlockModelAttributes",
    "BlockModelPendingAttribute",
    "PendingAttribute",
]


class UnSupportedDataTypeError(Exception):
    """An unsupported data type was encountered while processing data."""


def _infer_attribute_type_from_series(series: pd.Series) -> str:
    """Infer the attribute type from a Pandas Series.

    :param series: The Pandas Series to infer the attribute type from.

    :return: The inferred attribute type.
    """
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    elif pd.api.types.is_float_dtype(series):
        return "scalar"
    elif pd.api.types.is_bool_dtype(series):
        return "bool"
    elif isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    elif pd.api.types.is_string_dtype(series):
        return "string"
    else:
        raise UnSupportedDataTypeError(f"Unsupported dtype for attribute: {series.dtype}")


_attribute_table_formats = {
    "scalar": [FLOAT_ARRAY_1],
    "integer": [INTEGER_ARRAY_1_INT32, INTEGER_ARRAY_1_INT64],
    "bool": [BOOL_ARRAY_1],
    "string": [STRING_ARRAY],
}


class Attribute(SchemaModel):
    """A Geoscience Object Attribute"""

    name: Annotated[str, SchemaLocation("name")]
    _attribute_type: Annotated[str, SchemaLocation("attribute_type")]
    _key: Annotated[str | None, SchemaLocation("key")]
    _data: Annotated[str, SchemaLocation("values.data")]

    @property
    def key(self) -> str:
        """The key used to identify this attribute.

        This is required to be unique within a group of attributes.
        """
        # Gracefully handle historical attributes without a key.
        return self._key or self.name

    @property
    def attribute_type(self) -> str:
        """The type of this attribute."""
        return self._attribute_type

    @property
    def exists(self) -> bool:
        """Whether this attribute exists on the object.

        :return: True for existing attributes.
        """
        return True

    async def to_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the values for this attribute from the object.

        :param fb: Optional feedback object to report download progress.

        :return: The loaded DataFrame with values for this attribute, applying lookup table and NaN values as specified.
            The column name will be updated to match the attribute name.
        """
        if self._context.is_data_modified(self._data):
            raise DataLoaderError("Data was modified since the object was downloaded")
        return await self._obj.download_attribute_dataframe(self.as_dict(), fb=fb)

    async def set_attribute_values(
        self, df: pd.DataFrame, infer_attribute_type: bool = False, fb: IFeedback = NoFeedback
    ) -> None:
        """Update the values of this attribute.

        :param df: DataFrame containing the new values for this attribute. The DataFrame should contain a single column.
        :param infer_attribute_type: Whether to infer the attribute type from the DataFrame. If False, the existing attribute type will be used.
        :param fb: Optional feedback object to report upload progress.
        """

        if infer_attribute_type:
            attribute_type = _infer_attribute_type_from_series(df.iloc[:, 0])
            self._attribute_type = attribute_type
        else:
            attribute_type = self.attribute_type

        data_client = get_data_client(self._obj)
        await self._upload_attribute_values(self._document, df, attribute_type, data_client, fb)

        # Mark the context as modified so loading data is not allowed
        self._context.mark_modified(self._data)

    @staticmethod
    async def _upload_attribute_values(
        attr_doc: dict[str, Any],
        df: pd.DataFrame,
        attribute_type: str,
        data_client: Any,
        fb: IFeedback = NoFeedback,
    ) -> None:
        """Core logic for uploading attribute values to a document."""
        if attribute_type == "category":
            attr_doc.update(await data_client.upload_category_dataframe(df, fb=fb))
        else:
            table_formats = _attribute_table_formats.get(attribute_type)
            attr_doc["values"] = await data_client.upload_dataframe(df, table_format=table_formats, fb=fb)

        if attribute_type in ["scalar", "integer", "category"]:
            attr_doc["nan_description"] = {"values": []}


class PendingAttribute:
    """A placeholder for an attribute that doesn't exist yet on a Geoscience Object.

    This is returned when accessing an attribute by name that doesn't exist.
    It can be used as a target for compute tasks, which will create the attribute.
    """

    def __init__(self, parent: "Attributes", name: str) -> None:
        """
        :param parent: The Attributes collection this pending attribute belongs to.
        :param name: The name of the attribute to create.
        """
        self._parent = parent
        self._name = name

    @property
    def name(self) -> str:
        """The name of this attribute."""
        return self._name

    @property
    def exists(self) -> bool:
        """Whether this attribute exists on the object.

        :return: False for pending attributes.
        """
        return False

    @property
    def _obj(self) -> "DownloadedObject | None":
        """The DownloadedObject containing this attribute's parent object.

        Delegates to the parent Attributes collection.
        """
        return self._parent._obj

    def __repr__(self) -> str:
        return f"PendingAttribute(name={self._name!r}, exists=False)"


class Attributes(SchemaList[Attribute]):
    """A collection of Geoscience Object Attributes"""

    _schema_path: str | None = None
    """The full JMESPath to this attributes list within the parent object schema."""

    def __getitem__(self, index_or_name: int | str) -> Attribute | PendingAttribute:
        """Get an attribute by index or name.

        :param index_or_name: Either an integer index or the name/key of the attribute.
        :return: The attribute at the specified index or with the specified name/key.
            If a string is provided and no matching attribute exists, returns a PendingAttribute
            that can be used as a target for compute tasks.
        :raises IndexError: If the integer index is out of range.
        """
        if isinstance(index_or_name, str):
            for attr in self:
                if attr.name == index_or_name or attr.key == index_or_name:
                    return attr
            # Return a PendingAttribute for non-existent attributes accessed by name
            return PendingAttribute(self, index_or_name)
        return super().__getitem__(index_or_name)

    @classmethod
    async def _data_to_schema(
        cls,
        data: Any,
        context: IContext | DownloadedObject,
    ) -> list[dict[str, Any]]:
        """Convert a DataFrame to a list of attribute dictionaries for object creation.

        :param df: The DataFrame with columns to convert to attributes, or None.
        :param context: The context used for data upload operations.
        :param fb: Optional feedback object to report progress.
        :return: A list of attribute dictionaries suitable for the object document.
        """
        result: list[dict[str, Any]] = []
        if data is not None:
            await cls._upload_attributes_to_list(result, data, context)
        return result

    @staticmethod
    async def _upload_attributes_to_list(
        attributes_list: list[dict[str, Any]],
        df: pd.DataFrame,
        context: IContext | DownloadedObject,
        fb: IFeedback = NoFeedback,
    ) -> None:
        """Upload DataFrame columns as attributes and append to the attributes list.

        This is a static operation that doesn't require a model instance.

        :param attributes_list: The list in the document to append attribute dicts to.
        :param df: The DataFrame with columns to upload as attributes.
        :param context: The context used for data upload operations.
        :param fb: Optional feedback object to report progress.
        """
        data_client = get_data_client(context)

        for col in df.columns:
            series = df[col]
            attribute_type = _infer_attribute_type_from_series(series)

            attr_doc: dict[str, Any] = {
                "name": str(col),
                "key": str(uuid.uuid4()),
                "attribute_type": attribute_type,
            }

            col_df = df[[col]]
            await Attribute._upload_attribute_values(attr_doc, col_df, attribute_type, data_client)

            attributes_list.append(attr_doc)

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the values from the specified attributes in the object.

        :param keys: Optional list of attribute keys to filter the attributes by. If no keys are provided, all
            attributes will be loaded.
        :param fb: Optional feedback object to report download progress.

        :return: A DataFrame containing the values from the specified attributes. Column name(s) will be updated
            based on the attribute names.
        """
        attributes = [self[key] for key in keys] if keys else list(self)
        parts = [await attribute.to_dataframe(fb=fb_part) for attribute, fb_part in iter_with_fb(attributes, fb)]
        return pd.concat(parts, axis=1) if len(parts) > 0 else pd.DataFrame()

    async def append_attribute(self, df: pd.DataFrame, fb: IFeedback = NoFeedback):
        """Add a new attribute to the object.

        :param df: DataFrame containing the values for the new attribute. The DataFrame should contain a single column.
        :param fb: Optional feedback object to report upload progress.

        :raises ValueError: If the DataFrame does not contain exactly one column.
        """

        if df.shape[1] != 1:
            raise ValueError("DataFrame must contain exactly one column to append as an attribute.")

        # Use the static method to upload and append to document
        await self._upload_attributes_to_list(self._document, df, self._obj, fb)

        # Mark context as modified and create the Attribute wrapper
        attribute = self[-1]
        self._context.mark_modified(attribute._data)

    async def append_attributes(self, df: pd.DataFrame, fb: IFeedback = NoFeedback):
        """Add new attributes to the object.

        :param df: DataFrame containing the values for the new attributes.
        :param fb: Optional feedback object to report upload progress.
        """
        for attribute in df.columns:
            attribute_df = df[[attribute]]
            await self.append_attribute(attribute_df, fb)

    async def set_attributes(self, df: pd.DataFrame, fb: IFeedback = NoFeedback):
        """Set the attributes of the object to match the provided DataFrame.

        :param df: DataFrame containing the values for the new attributes.
        :param fb: Optional feedback object to report upload progress.
        """

        attributes_by_name = {attr.name: attr for attr in self}
        self.clear()
        for col in df.columns:
            attribute_df = df[[col]]
            attribute = attributes_by_name.get(col)
            if attribute is not None:
                await attribute.set_attribute_values(attribute_df, fb=fb)
                self._append(attribute)
            else:
                await self.append_attribute(attribute_df, fb=fb)

    def validate_lengths(self, expected_length: int) -> None:
        """Validate that all attributes have the expected length.

        :param expected_length: The expected number of values for each attribute.
        :raises ObjectValidationError: If any attribute has a different length.
        """
        for attribute in self:
            attribute_length = jmespath.search("values.length", attribute.as_dict())
            if attribute_length is None:
                raise ObjectValidationError(f"Can't determine length of attribute '{attribute.name}'")
            if attribute_length != expected_length:
                raise ObjectValidationError(
                    f"Attribute '{attribute.name}' length ({attribute_length}) does not match expected length ({expected_length})"
                )


class BlockModelAttribute:
    """An attribute on a block model.

    This class represents an existing attribute on a block model. It stores a reference
    to the parent BlockModel via `_obj`, similar to how `Attribute` in dataset.py works.
    """

    def __init__(
        self,
        name: str,
        attribute_type: str,
        block_model_column_uuid: UUID | None = None,
        unit: str | None = None,
        obj: BlockModel | None = None,
    ):
        self._name = name
        self._attribute_type = attribute_type
        self._block_model_column_uuid = block_model_column_uuid
        self._unit = unit
        self._obj = obj  # Reference to parent BlockModel, similar to Attribute._obj

    @property
    def name(self) -> str:
        """The name of this attribute."""
        return self._name

    @property
    def attribute_type(self) -> str:
        """The type of this attribute."""
        return self._attribute_type

    @property
    def block_model_column_uuid(self) -> UUID | None:
        """The UUID of the column in the block model service."""
        return self._block_model_column_uuid

    @property
    def unit(self) -> str | None:
        """The unit of this attribute."""
        return self._unit

    @property
    def exists(self) -> bool:
        """Whether this attribute exists on the block model.

        :return: True for existing attributes.
        """
        return True

    def __repr__(self) -> str:
        return f"BlockModelAttribute(name={self._name!r}, attribute_type={self._attribute_type!r}, unit={self._unit!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockModelAttribute):
            return NotImplemented
        return (
            self._name == other._name
            and self._attribute_type == other._attribute_type
            and self._block_model_column_uuid == other._block_model_column_uuid
            and self._unit == other._unit
        )

    def __hash__(self) -> int:
        return hash((self._name, self._attribute_type, self._block_model_column_uuid, self._unit))


class BlockModelPendingAttribute:
    """A placeholder for an attribute that doesn't exist yet on a Block Model.

    This is returned when accessing an attribute by name that doesn't exist.
    It can be used as a target for compute tasks, which will create the attribute.

    Stores a reference to the parent BlockModel via `_obj`, similar to how
    `BlockModelAttribute` and `Attribute` (in dataset.py) work.
    """

    def __init__(self, obj: BlockModel, name: str) -> None:
        """
        :param obj: The BlockModel this pending attribute belongs to.
        :param name: The name of the attribute to create.
        """
        self._obj = obj  # Reference to parent BlockModel
        self._name = name

    @property
    def name(self) -> str:
        """The name of this attribute."""
        return self._name

    @property
    def exists(self) -> bool:
        """Whether this attribute exists on the block model.

        :return: False for pending attributes.
        """
        return False

    def __repr__(self) -> str:
        return f"BlockModelPendingAttribute(name={self._name!r}, exists=False)"


class BlockModelAttributes:
    """A collection of attributes on a block model with pretty-printing support."""

    def __init__(self, attributes: list[BlockModelAttribute], block_model: BlockModel | None = None):
        self._block_model = block_model
        # Set _obj reference on each attribute to the parent BlockModel
        self._attributes = []
        for attr in attributes:
            # Create a new attribute with _obj reference to the block model
            attr_with_obj = BlockModelAttribute(
                name=attr.name,
                attribute_type=attr.attribute_type,
                block_model_column_uuid=attr.block_model_column_uuid,
                unit=attr.unit,
                obj=block_model,
            )
            self._attributes.append(attr_with_obj)

    @property
    def exists(self) -> bool:
        """Whether this attribute exists on the block model.

        :return: True for existing attributes.
        """
        return True

    @classmethod
    def from_schema(cls, attributes_list: list[dict], block_model: BlockModel | None = None) -> BlockModelAttributes:
        """Parse block model attributes from the schema format.

        :param attributes_list: List of attribute dictionaries from the schema.
        :param block_model: Optional parent BlockModel for back-references.
        :return: A BlockModelAttributes collection.
        """
        parsed = []
        for attr in attributes_list:
            col_uuid = attr.get("block_model_column_uuid")
            # Try to parse as UUID, but handle invalid formats gracefully
            parsed_uuid = None
            if col_uuid:
                try:
                    parsed_uuid = UUID(col_uuid)
                except (ValueError, AttributeError):
                    # col_uuid is not a valid UUID format, skip it
                    pass
            parsed.append(
                BlockModelAttribute(
                    name=attr.get("name", ""),
                    attribute_type=attr.get("attribute_type", "Float64"),
                    block_model_column_uuid=parsed_uuid,
                    unit=attr.get("unit"),
                )
            )
        return cls(parsed, block_model=block_model)

    def __iter__(self):
        return iter(self._attributes)

    def __len__(self):
        return len(self._attributes)

    def __getitem__(self, index_or_name: int | str) -> BlockModelAttribute | BlockModelPendingAttribute:
        if isinstance(index_or_name, str):
            for attr in self._attributes:
                if attr.name == index_or_name:
                    return attr
            # Return a BlockModelPendingAttribute for non-existent attributes accessed by name
            # Pass the block model directly as _obj
            return BlockModelPendingAttribute(self._block_model, index_or_name)
        return self._attributes[index_or_name]

    def __repr__(self) -> str:
        names = [attr.name for attr in self._attributes]
        return f"BlockModelAttributes({names})"

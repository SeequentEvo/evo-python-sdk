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
from typing import Annotated, Any

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

__all__ = [
    "Attribute",
    "Attributes",
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

    async def as_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
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


class Attributes(SchemaList[Attribute]):
    """A collection of Geoscience Object Attributes"""

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

    async def as_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the values from the specified attributes in the object.

        :param keys: Optional list of attribute keys to filter the attributes by. If no keys are provided, all
            attributes will be loaded.
        :param fb: Optional feedback object to report download progress.

        :return: A DataFrame containing the values from the specified attributes. Column name(s) will be updated
            based on the attribute names.
        """
        parts = [await attribute.as_dataframe(fb=fb_part) for attribute, fb_part in iter_with_fb(self, fb)]
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
                # Skip attributes without length info (e.g., newly created ones)
                continue
            if attribute_length != expected_length:
                raise ObjectValidationError(
                    f"Attribute '{attribute.name}' length ({attribute_length}) does not match expected length ({expected_length})"
                )

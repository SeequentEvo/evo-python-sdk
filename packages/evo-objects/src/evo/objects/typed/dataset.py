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

import asyncio
import sys
import uuid
from collections.abc import Sequence
from typing import cast

import pandas as pd

from evo import jmespath
from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback, iter_with_fb, split_feedback
from evo.objects import DownloadedObject
from evo.objects.utils.table_formats import (
    BOOL_ARRAY_1,
    FLOAT_ARRAY_1,
    INTEGER_ARRAY_1_INT32,
    INTEGER_ARRAY_1_INT64,
    STRING_ARRAY,
)
from evo.objects.utils.types import AttributeInfo

from ._adapters import AttributesAdapter, BaseAdapter, CategoryTableAdapter, DatasetAdapter, TableAdapter
from ._utils import assign_jmespath_value, get_data_client
from .exceptions import ObjectValidationError

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    "Attribute",
    "Attributes",
    "DataLoaderError",
    "Dataset",
]


class DataLoaderError(Exception):
    """An error occurred while loading data from a Geoscience Object."""


class UnSupportedDataTypeError(Exception):
    """An unsupported data type was encountered while processing data."""


class _ValuesStore:
    """A store for loading and uploading values for a given dataset"""

    def __init__(
        self,
        document: dict,
        value_adapters: Sequence[TableAdapter | CategoryTableAdapter],
        context: IContext,
        obj: DownloadedObject | None = None,
    ):
        self._document = document
        self._value_adapters = value_adapters
        self._obj = obj
        self._context = context
        self.column_names = [name for adapter in value_adapters for name in adapter.column_names]

    async def _load_values(self, adapter: BaseAdapter, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        if isinstance(adapter, TableAdapter):
            return await self._obj.download_dataframe(
                table_info=adapter.values_path,
                nan_values=adapter.nan_values_path,
                column_names=adapter.column_names,
                fb=fb,
            )
        if isinstance(adapter, CategoryTableAdapter):
            return await self._obj.download_category_dataframe(
                category_info=adapter.category_data_path,
                nan_values=adapter.nan_values_path,
                column_names=adapter.column_names,
                fb=fb,
            )
        raise ValueError(f"Unsupported adapter type: {type(adapter)}")

    async def as_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Download a DataFrame containing the referenced values.

        :param fb: Optional feedback object to report download progress.

        :return: The downloaded DataFrame with values.
            The column name(s) will be updated to match the provided column names, if any.
        """
        if self._obj is None:
            raise DataLoaderError("Values were changed since the object was downloaded")
        if len(self._value_adapters) == 0:
            return pd.DataFrame()

        fbs = split_feedback(fb, [1.0] * len(self._value_adapters))
        parts = [self._load_values(adapter, fb) for adapter, fb in zip(self._value_adapters, fbs)]
        return pd.concat(await asyncio.gather(*parts), axis=1)

    async def _store_value(self, adapter: BaseAdapter, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        data_client = get_data_client(self._context)
        if isinstance(adapter, TableAdapter):
            table_info = await data_client.upload_dataframe(
                df[list(adapter.column_names)], table_format=adapter.table_formats
            )
            assign_jmespath_value(self._document, adapter.values_path, table_info)
        elif isinstance(adapter, CategoryTableAdapter):
            category_info = await data_client.upload_category_dataframe(
                df[list(adapter.column_names)],
            )
            assign_jmespath_value(self._document, adapter.category_data_path, category_info)
        else:
            raise ValueError(f"Unsupported adapter type: {type(adapter)}")

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        """Upload a DataFrame containing values to the given DownloadedObject.

        :param df: The DataFrame containing the values to upload.
        :param fb: Optional feedback object to report upload progress.
        """
        fbs = split_feedback(fb, [1.0] * len(self._value_adapters))
        parts = [self._store_value(adapter, df, fb) for adapter, fb in zip(self._value_adapters, fbs)]
        await asyncio.gather(*parts)

        # Clear the reference to the object after upload, as that is now stale
        self._obj = None

    def validate(self) -> int | None:
        """Validate that the values are of a consistent length.

        :return: The length of the values, or None if there are no values.
        """

        lengths = []
        for adapter in self._value_adapters:
            if isinstance(adapter, CategoryTableAdapter):
                length_path = adapter.category_data_path + ".values.length"
            else:
                length_path = adapter.values_path + ".length"
            length = jmespath.search(length_path, self._document)
            if not isinstance(length, int):
                raise ObjectValidationError("Can't determine length of values")
            lengths.append(length)
        if len(lengths) == 0:
            return None
        if not all(length == lengths[0] for length in lengths):
            raise ObjectValidationError("Values have inconsistent lengths")
        return lengths[0]


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


class Attribute:
    """A Geoscience Object Attribute"""

    def __init__(
        self,
        parent: Attributes,
        attribute: dict,
        context: IContext,
        obj: DownloadedObject | None = None,
    ) -> None:
        """
        :param attribute: The dictionary containing the attribute information.
        :param context: The context for uploading data to the Geoscience Object Service.
        :param obj: The DownloadedObject containing the attribute.
        """
        self._parent = parent
        self._attribute = attribute
        self._context = context
        self._obj = obj

    @property
    def expression(self) -> str:
        """The JMESPath expression to access this attribute from the object."""
        return f"{self._parent._attribute_adapter.attribute_list_path}[?key=='{self.key}']"

    @property
    def key(self) -> str:
        """The key used to identify this attribute.

        This is required to be unique within a group of attributes.
        """
        # Gracefully handle historical attributes without a key.
        return self._attribute.get("key") or self._attribute["name"]

    @property
    def name(self) -> str:
        """The name of this attribute."""
        return self._attribute["name"]

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of this attribute."""
        self._attribute["name"] = value

    @property
    def attribute_type(self) -> str:
        """The type of this attribute."""
        return self._attribute["attribute_type"]

    async def as_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the values for this attribute from the object.

        :param fb: Optional feedback object to report download progress.

        :return: The loaded DataFrame with values for this attribute, applying lookup table and NaN values as specified.
            The column name will be updated to match the attribute name.
        """
        if self._obj is None:
            raise DataLoaderError("Attribute values was changed since the object was downloaded")
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
            self._attribute["attribute_type"] = attribute_type
        else:
            attribute_type = self.attribute_type

        data_client = get_data_client(self._context)
        if attribute_type == "category":
            self._attribute.update(await data_client.upload_category_dataframe(df))
        else:
            table_formats = _attribute_table_formats.get(attribute_type)
            self._attribute["values"] = await data_client.upload_dataframe(df, table_format=table_formats)

        if attribute_type in ["scalar", "integer", "category"]:
            self._attribute["nan_description"] = {"values": []}

        # As the values have been updated, don't allow loading the old values again.
        self._obj = None

    def as_dict(self) -> AttributeInfo:
        """Get the attribute as a dictionary.

        :return: The attribute as a dictionary.
        """
        return cast(AttributeInfo, self._attribute)


class Attributes(Sequence[Attribute]):
    """A collection of Geoscience Object Attributes"""

    def __init__(
        self,
        document: dict,
        attribute_adapter: AttributesAdapter,
        context: IContext,
        obj: DownloadedObject | None = None,
    ) -> None:
        """
        :param document: The document containing the attributes.
        :param attribute_adapter: The AttributesAdapter to extract attributes from the document.
        :param context: The context for uploading data to the Geoscience Object Service.
        :param obj: The DownloadedObject, representing the object containing the attributes.
        """
        self._document = document
        self._attribute_adapter = attribute_adapter
        self._context = context

        attribute_list = jmespath.search(self._attribute_adapter.attribute_list_path, document)
        if not isinstance(attribute_list, jmespath.JMESPathArrayProxy):
            raise ValueError("Attribute list path did not resolve to a list")
        attribute_list = attribute_list.raw
        if obj is None:
            self._attributes = [Attribute(self, attr, context) for attr in attribute_list]
        else:
            self._attributes = [Attribute(self, attr, context, obj) for attr in attribute_list]

    def __getitem__(self, index_or_name: int | str) -> Attribute:
        if isinstance(index_or_name, str):
            for attribute in self._attributes:
                if attribute.name == index_or_name:
                    return attribute
            raise KeyError(f"Attribute with name '{index_or_name}' not found")
        return self._attributes[index_or_name]

    def __len__(self) -> int:
        return len(self._attributes)

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

        attribute = Attribute(
            self,
            {
                "name": str(df.columns[0]),
                "key": str(uuid.uuid4()),
            },
            self._context,
        )
        await attribute.set_attribute_values(df, infer_attribute_type=True, fb=fb)
        self._attributes.append(attribute)

    async def append_attributes(self, df: pd.DataFrame, fb: IFeedback = NoFeedback):
        """Add a new attribute to the object.

        :param df: DataFrame containing the values for the new attribute. The DataFrame should contain a single column.
        :param fb: Optional feedback object to report upload progress.

        :raises ValueError: If the DataFrame does not contain exactly one column.
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
        self._attributes = []
        for col in df.columns:
            attribute_df = df[[col]]
            attribute = attributes_by_name.get(col)
            if attribute is not None:
                await attribute.set_attribute_values(attribute_df, fb=fb)
                self._attributes.append(attribute)
            else:
                await self.append_attribute(attribute_df, fb=fb)

    def update_document(self) -> None:
        """Update the provided document with the current attributes."""
        assign_jmespath_value(
            self._document, self._attribute_adapter.attribute_list_path, [attr.as_dict() for attr in self]
        )

    def clear(self) -> None:
        """Clear all attributes from the collection."""
        self._attributes = []


class Dataset:
    """A tabular dataset containing:
    - a set of value columns, which defines the geometry or structure of the dataset
    - a set of attributes, which defines custom values for each element in the dataset
    """

    def __init__(
        self,
        document: dict,
        dataset_adapter: DatasetAdapter,
        context: IContext,
        obj: DownloadedObject | None = None,
    ):
        self._document = document
        self._values = _ValuesStore(document, dataset_adapter.value_adapters, obj=obj, context=context)
        if dataset_adapter.attributes_adapter is not None:
            self.attributes = Attributes(document, dataset_adapter.attributes_adapter, obj=obj, context=context)
        else:
            self.attributes = None

    def search(self, jmespath_expr: str):
        """Search the document that this dataset is based on using a JMESPath expression.

        :param jmespath_expr: The JMESPath expression to search with.

        :return: The result of the JMESPath search.
        """
        return jmespath.search(jmespath_expr, self._document)

    @classmethod
    async def create_from_data(
        cls, document: dict, data: pd.DataFrame | None, dataset_adapter: DatasetAdapter, context: IContext
    ) -> Self:
        # Create an empty attribute list
        assign_jmespath_value(document, dataset_adapter.attributes_adapter.attribute_list_path, [])

        dataset = cls(document, dataset_adapter, context)
        if data is not None:
            await dataset.set_dataframe(data)
        return dataset

    def _expected_length(self) -> int | None:
        """Get the expected length of the dataset based on other properties of the object.

        :return: The expected length of the dataset, or None if there is no expected length.
        """
        return None

    def validate(self) -> None:
        """Validate that the dataset is valid.

        Subclasses implement this to perform specific validation, like checking the attributes lengths match the expected length.

        :raises ObjectValidationError: If the dataset is not valid.
        """
        length = self._expected_length()
        values_length = self._values.validate()
        if values_length is not None and length is not None and values_length != length:
            raise ObjectValidationError(
                f"Values length ({values_length}) does not match the expected length ({length})"
            )
        if length is None:
            length = values_length

        if self.attributes is not None:
            for attribute in self.attributes:
                attribute_length = jmespath.search("values.length", attribute.as_dict())
                if not isinstance(attribute_length, int):
                    raise ObjectValidationError(f"Can't determine length of attribute '{attribute.name}'")
                if length is None:
                    length = attribute_length
                if attribute_length != length:
                    raise ObjectValidationError(
                        f"Attribute '{attribute.name}' length ({attribute_length}) does not match dataset length ({length})"
                    )

    async def as_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the datasets base values and the values from the specified attributes.

        :param keys: Optional list of attribute keys to filter the attributes by. If no keys are provided, all
            attributes will be loaded.
        :param fb: Optional feedback object to report download progress.

        :raise DataLoaderError: If the dataset can't be loaded, either due to the dataset being invalid or the
            data has already been changed since the object was downloaded.

        :return: The loaded DataFrame with values from all sources and the specified attributes, applying lookup tables
            and NaN values as specified. The column name(s) will be updated to match the column names provided in the
            ValuesAdapters and the attribute names.
        """
        # First validate the dataset is valid before loading any data
        try:
            self.validate()
        except ObjectValidationError as e:
            raise DataLoaderError(f"Dataset is not valid: {e}") from e
        values = await self._values.as_dataframe(fb=fb)
        if self.attributes is None:
            return values
        attributes = await self.attributes.as_dataframe(*keys, fb=fb)
        return pd.concat([values, attributes], axis=1)

    async def set_dataframe(self, df: pd.DataFrame, *, fb: IFeedback = NoFeedback) -> None:
        """Replaces the dataset with the data from the provided DataFrame.

        Any attributes that are not present in the DataFrame but exist on the object, will be deleted.

        This uploads the data to the Geoscience Object Service, ready for a ne version of the Geoscience Object to be
        created.

        :param df: The DataFrame containing the data to set on the object.
        :param fb: Optional feedback object to report progress.
        """
        value_columns = []
        attribute_columns = []
        for column in df.columns:
            if column in self._values.column_names:
                value_columns.append(column)
            else:
                attribute_columns.append(column)

        await self._values.set_dataframe(df[value_columns], fb)
        if self.attributes is None:
            if attribute_columns:
                raise DataLoaderError("Object can't store attributes, but additional columns were provided.")
        else:
            await self.attributes.set_attributes(df[attribute_columns], fb=fb)

    def update_document(self) -> None:
        """Update the underlying document with any changes made to the attributes."""
        if self.attributes is not None:
            self.attributes.update_document()

    def clear_attributes(self) -> None:
        """Clear all attributes from the dataset."""
        if self.attributes is not None:
            self.attributes.clear()

    def has_attributes(self) -> bool:
        """Check if the dataset has any attributes.

        :return: True if the dataset has attributes, False otherwise.
        """
        return self.attributes is not None and len(self.attributes) > 0

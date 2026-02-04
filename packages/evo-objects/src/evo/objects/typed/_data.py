from __future__ import annotations

from typing import Annotated, Any, ClassVar

import pandas as pd

from evo.common import IFeedback
from evo.common.interfaces import IContext
from evo.common.utils import NoFeedback
from evo.objects.typed.attributes import Attributes
from evo.objects.utils.table_formats import KnownTableFormat

from ._model import SchemaBuilder, SchemaLocation, SchemaModel
from ._utils import get_data_client
from .exceptions import DataLoaderError, ObjectValidationError


class DataTable(SchemaModel):
    length: Annotated[int, SchemaLocation("length")]
    _data: Annotated[str, SchemaLocation("data")]

    table_format: ClassVar[KnownTableFormat | None] = None
    data_columns: ClassVar[list[str]] = []

    async def get_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing value for this table.

        :param fb: Optional feedback object to report download progress.

        :return: The loaded DataFrame with values for this table.
        """
        if self._context.is_data_modified(self._data):
            raise DataLoaderError("Data was modified since the object was downloaded")
        return await self._obj.download_dataframe(self.as_dict(), fb=fb, column_names=self.data_columns)

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the values and attributes.

        This is an alias for get_dataframe for backwards compatibility.

        :param keys: Optional attribute keys to include. If not specified, all attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: DataFrame with data columns (e.g., X, Y, Z) and additional columns for attributes.
        """
        table_df = await self._table.get_dataframe(fb=fb)
        if self.attributes is not None and len(self.attributes) > 0:
            attr_df = await self.attributes.get_dataframe(*keys, fb=fb)
            combined_df = pd.concat([table_df, attr_df], axis=1)
            return combined_df
        else:
            return table_df

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        """Update the values of this table.

        :param df: DataFrame containing the new values for this table.
        :param fb: Optional feedback object to report upload progress.
        """
        self._document.update(await self._data_to_schema(df, self._context, fb=fb))

        # Mark the context as modified so loading data is not allowed
        self._context.mark_modified(self._data)

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext, fb: IFeedback = NoFeedback) -> Any:
        """Upload a DataFrame and return the schema dictionary for the DataTable."""
        if not isinstance(data, pd.DataFrame):
            raise ObjectValidationError(f"Input data must be a pandas DataFrame, but got {type(data)}")
        if list(data.columns) != cls.data_columns:
            raise ObjectValidationError(
                f"Input DataFrame must have columns {cls.data_columns}, but got {list(data.columns)}"
            )

        data_client = get_data_client(context)
        return await data_client.upload_dataframe(data, table_format=cls.table_format, fb=fb)


class DataTableAndAttributes(SchemaModel):
    """A dataset representing a table of data along with associated attributes.

    Subclasses should redefine the _table property to provide additional details about the data table like:
    1. the location of it within the schema using SchemaLocation
    2. the data columns that are expected in the table, which is done by creating a subclass of DataTable,
    3. the table format used for storing the data, which can also be done by creating a subclass of DataTable.

    e.g.,
    class LocationTable(DataTable):
        table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
        data_columns: ClassVar[list[str]] = ["x", "y", "z"]


    class Locations(DataTableAndAttributes):
        _table: Annotated[LocationTable, SchemaLocation("coordinates")]

    """

    attributes: Annotated[Attributes, SchemaLocation("attributes")]
    _table: DataTable

    @property
    def length(self) -> int:
        """The expected number of rows in the table and attributes."""
        return self._table.length

    async def get_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the values and attributes.

        :param fb: Optional feedback object to report download progress.
        :return: DataFrame with data columns (e.g., X, Y, Z) and additional columns for attributes.
        """
        table_df = await self._table.get_dataframe(fb=fb)
        if self.attributes is not None and len(self.attributes) > 0:
            attr_df = await self.attributes.get_dataframe(fb=fb)
            combined_df = pd.concat([table_df, attr_df], axis=1)
            return combined_df
        else:
            return table_df

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback) -> None:
        """Set the table data and attributes from a DataFrame.

        :param df: DataFrame containing data columns (e.g., X, Y, Z) and additional columns for attributes.
        :param fb: Optional feedback object to report upload progress.
        """
        table_df, attr_df = self._split_dataframe(df, self._table.data_columns)

        await self._table.set_dataframe(table_df, fb=fb)
        if attr_df is not None:
            await self.attributes.set_attributes(attr_df, fb=fb)
        else:
            await self.attributes.clear()

    def validate(self) -> None:
        """Validate that all attributes have the correct length."""
        self.attributes.validate_lengths(self.length)

    @classmethod
    def _split_dataframe(cls, data: pd.DataFrame, data_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Validate and split a DataFrame into table data and attribute data."""

        missing = set(data_columns) - set(data.columns)
        if missing:
            raise ObjectValidationError(f"Input DataFrame must have {data_columns} columns. Missing: {missing}")

        table_df = data[data_columns]
        attr_cols = [col for col in data.columns if col not in data_columns]
        attr_df = data[attr_cols] if attr_cols else None
        return table_df, attr_df

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> Any:
        if not isinstance(data, pd.DataFrame):
            raise ObjectValidationError(f"Input data must be a pandas DataFrame, but got {type(data)}")

        # Lookup the metadata of the _table sub-model, as sub-classes may redefine it
        table_metadata = cls._sub_models["_table"]
        table_type = table_metadata.model_type
        table_df, attr_df = cls._split_dataframe(data, table_type.data_columns)

        builder = SchemaBuilder(cls, context)
        await builder.set_sub_model_value("_table", table_df)
        await builder.set_sub_model_value("attributes", attr_df)
        return builder.document

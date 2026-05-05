#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Utility functions for typed block model access."""

from typing import Any

import pandas as pd
import pyarrow as pa

__all__ = [
    "dataframe_to_pyarrow",
]

# Geometry columns for regular block models
GEOMETRY_COLUMNS_IJK = {"i", "j", "k"}
GEOMETRY_COLUMNS_XYZ = {"x", "y", "z"}
GEOMETRY_COLUMNS = GEOMETRY_COLUMNS_IJK | GEOMETRY_COLUMNS_XYZ


def _get_schema_from_dataframe(dataframe: pd.DataFrame) -> pa.Schema:
    """Get a normalized pyarrow schema from a pandas dataframe.

    Converts unsupported Arrow types before table creation:
    - large_string -> string
    - dictionary<*, large_string> -> dictionary<*, string>
    """
    schema = pa.Schema.from_pandas(dataframe)
    fields = []
    for field in schema:
        if pa.types.is_large_string(field.type):
            fields.append(field.with_type(pa.string()))
        elif pa.types.is_dictionary(field.type) and pa.types.is_large_string(field.type.value_type):
            fields.append(field.with_type(pa.dictionary(field.type.index_type, pa.string())))
        else:
            fields.append(field)
    return pa.schema(fields)


def dataframe_to_pyarrow(df: pd.DataFrame) -> pa.Table:
    """Convert a pandas DataFrame to a PyArrow Table.

    Ensures geometry columns (i, j, k) are present and properly typed.
    The i, j, k columns are cast to uint32 as required by the Block Model Service.

    :param df: The pandas DataFrame to convert.
    :return: A PyArrow Table.
    :raises ValueError: If required geometry columns are missing.
    """

    # Check for required geometry columns
    columns = set(df.columns)
    has_ijk = GEOMETRY_COLUMNS_IJK.issubset(columns)
    has_xyz = GEOMETRY_COLUMNS_XYZ.issubset(columns)

    if not has_ijk and not has_xyz:
        raise ValueError("DataFrame must contain either (i, j, k) or (x, y, z) geometry columns")

    # Convert to PyArrow table with normalized schema
    schema = _get_schema_from_dataframe(df)
    table = pa.Table.from_pandas(df, schema=schema)

    # Cast i, j, k columns to uint32 as required by the Block Model Service
    if has_ijk:
        for col_name in GEOMETRY_COLUMNS_IJK:
            col_idx = table.schema.get_field_index(col_name)
            if col_idx >= 0:
                col = table.column(col_name)
                if col.type != pa.uint32():
                    table = table.set_column(col_idx, col_name, col.cast(pa.uint32()))

    return table


def get_attribute_columns(df: Any) -> list[str]:
    """Get the list of attribute (non-geometry) columns from a DataFrame.

    :param df: The DataFrame to extract attribute columns from.
    :return: A list of attribute column names.
    """
    return [col for col in df.columns if col not in GEOMETRY_COLUMNS]

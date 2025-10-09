import hashlib
import os
from uuid import UUID

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq


## Source: https://stackoverflow.com/questions/53847404/how-to-check-uuid-validity-in-python
def is_valid_uuid(uuid_to_test, version=4):
    """
    Check if uuid_to_test is a valid UUID.

     Parameters
    ----------
    uuid_to_test : str
    version : {1, 2, 3, 4}

     Returns
    -------
    `True` if uuid_to_test is a valid UUID, otherwise `False`.

     Examples
    --------
    >>> is_valid_uuid('c9bf9e57-1685-4c89-bafb-ff5af830be8a')
    True
    >>> is_valid_uuid('c9bf9e58')
    False
    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


def parquet_to_csv(parquet_file, csv_path):
    """
    Convert a Parquet file to a CSV file.

    Args:
        parquet_file (pyarrow.parquet.ParquetFile): The Parquet file object.
        csv_path (str): The path to save the CSV file.
    """
    row_groups = parquet_file.num_row_groups

    for grp in range(0, row_groups):
        table = parquet_file.read_row_group(grp)
        df = table.to_pandas()
        if grp == 0:
            df.to_csv(
                csv_path,
                sep=",",
                header=True,
                index=False,
                mode="w",
                lineterminator="\r\n",
            )
        else:
            df.to_csv(
                csv_path,
                sep=",",
                header=False,
                index=False,
                mode="a",
                lineterminator="\r\n",
            )

        print(f"Saved {csv_path}")


def remove_rogue_hole_ids(hole_id_table, obj_hole_id, data):
    """
    Remove rogue hole IDs from the data.

    Args:
        hole_id_table (pd.DataFrame): The hole ID lookup table.
        obj_hole_id (str): The column name of the hole ID in the data.
        data (pd.DataFrame): The data to remove rogue hole IDs from.

    Returns:
        data (pd.DataFrame): The data with rogue hole IDs removed.
    """
    print("Removing rogue hole IDs")
    values = hole_id_table["value"]
    keys = list(values)
    data.drop(data[~data[obj_hole_id].isin(keys)].index, inplace=True)
    return data.reset_index(drop=True)


def create_category_lookup_and_data(column):
    """
    Create a category lookup table and a data column with mapped values.

    Args:
        column (pd.Series): The column to create the lookup table and data column from.

    Returns:
        lookup_df (pd.DataFrame): The category lookup table.
        values_df (pd.DataFrame): The data column with mapped values.
    """
    # Replace NaN with empty string
    column.replace(np.nan, "", regex=True, inplace=True)
    set_obj = set(column["data"])
    list_obj = list(set_obj)
    list_obj.sort()
    num_unique_elements = len(list_obj)

    # Create lookup table
    lookup_df = pd.DataFrame([])
    lookup_df["key"] = list(range(1, num_unique_elements + 1))
    lookup_df["value"] = list_obj

    # Create data column
    values_df = pd.DataFrame([])
    values_df["data"] = column["data"].map(lookup_df.set_index("value")["key"])
    return lookup_df, values_df


def create_hole_id_mapping(hole_id_table, value_list):
    """
    Create a hole ID mapping table based on the hole ID table and the value list.

    Args:
        hole_id_table (pd.DataFrame): The hole ID lookup table.
        value_list (pd.DataFrame): The value list to create the mapping from.

    Returns:
        mapping_df (pd.DataFrame): The hole ID mapping table.
    """

    # print(hole_id_table)
    # print(value_list)
    num_keys = len(hole_id_table.index)

    mapping_df = pd.DataFrame(list())
    mapping_df["hole_index"] = hole_id_table["key"]
    mapping_df["offset"] = [0] * num_keys
    mapping_df["count"] = [0] * num_keys

    mapping_df["hole_index"] = mapping_df["hole_index"].astype("int32")
    mapping_df["offset"] = mapping_df["offset"].astype("uint64")
    mapping_df["count"] = mapping_df["count"].astype("uint64")

    prev_value = ""
    key = ""
    count = 0
    offset = 0

    for index, row in value_list.iterrows():
        new_value = row["data"]

        if new_value != prev_value:
            if prev_value != "":
                mapping_df.loc[mapping_df["hole_index"] == key, "count"] = count
                mapping_df.loc[mapping_df["hole_index"] == key, "offset"] = offset
                offset += count

            mask = hole_id_table["value"] == new_value
            masked_df = hole_id_table[mask]
            try:
                key_row = masked_df.iloc[[0]]
            except IndexError:
                print("Ignoring this hole ID")
                continue

            key = key_row["key"].iloc[0]
            count = 1
            prev_value = new_value
        else:
            count += 1

    mapping_df.loc[mapping_df["hole_index"] == key, "count"] = count
    mapping_df.loc[mapping_df["hole_index"] == key, "offset"] = offset

    return mapping_df


def write_csv(df, path):
    """
    Write a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to write.
        path (str): The path to save the CSV file.
    """
    with open(path, "wb") as outfile:
        df.to_csv(outfile, index=False)


def sha256sum(filename):
    """
    Calculate the SHA256 hash of a file.

    Args:
        filename (str): The path to the file.

    Returns:
        hash (str): The SHA256 hash of the file.
    """
    with open(filename, "rb", buffering=0) as f:
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        with open(filename, "rb", buffering=0) as f:
            while n := f.readinto(mv):
                h.update(mv[:n])
        return h.hexdigest()


def df_to_parquet(df, schema, path, include_csv=False):
    """
    Convert a DataFrame to a Parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        schema (pyarrow.Schema): The schema of the DataFrame.
        path (str): The path to save the Parquet file.

    Returns:
        hash (str): The SHA256 hash of the Parquet file.
    """

    # Write the csv to a temp file
    temp_path = "temp.csv"
    write_csv(df, temp_path)

    # Debug - save csv to the output folder
    if include_csv:
        debug_path = os.path.splitext(path)[0] + ".csv"
        write_csv(df, debug_path)

    writer = None
    with open(temp_path, "rb") as f:
        with pv.open_csv(
            f,
            read_options=pv.ReadOptions(block_size=100_000),
            parse_options=pv.ParseOptions(delimiter=","),
            convert_options=pv.ConvertOptions(column_types=schema, include_columns=schema.names),
        ) as reader:
            # Write the new Parquet file
            options = dict(compression="gzip", version="2.4", data_page_version="1.0")
            writer = pq.ParquetWriter(path, schema, **options)
            # with pq.ParquetWriter(path, reader.schema, **options) as writer:
            for batch in reader:
                chunk = pa.Table.from_batches([batch], reader.schema)
                writer.write_table(chunk, row_group_size=100_000)
        writer.close()

    # Clean up
    os.remove(temp_path)
    print(f"Saved {path}")

    return sha256sum(path)

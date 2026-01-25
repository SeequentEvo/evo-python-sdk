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

"""Shared test helpers for grid tests."""

from __future__ import annotations

import copy
import uuid
from unittest.mock import Mock

import pandas as pd

from evo.common import Environment, IContext
from evo.objects import DownloadedObject, ObjectReference, ObjectSchema


class MockDownloadedObject(DownloadedObject):
    def __init__(self, mock_client: MockClient, object_dict: dict, version_id: str = "1"):
        self.mock_client = mock_client
        self.object_dict = object_dict
        self._metadata = Mock()
        self._metadata.schema_id = ObjectSchema.from_id(object_dict["schema"])
        self._metadata.url = ObjectReference.new(
            environment=mock_client.environment,
            object_id=uuid.UUID(object_dict["uuid"]),
        )
        self._metadata.version_id = version_id

    @property
    def metadata(self):
        return self._metadata

    def as_dict(self):
        return self.object_dict

    async def download_attribute_dataframe(self, data: dict, fb) -> pd.DataFrame:
        return self.mock_client.get_dataframe(data["values"])

    async def download_array(self, jmespath_expr: str, fb=None):
        """Download an array from the object using a JMESPath expression."""

        from evo import jmespath as jp

        data_info = jp.search(jmespath_expr, self.object_dict)
        if data_info is None:
            raise ValueError(f"No data found at {jmespath_expr}")
        df = self.mock_client.get_dataframe(data_info)
        # Return the first column as a numpy array
        return df.iloc[:, 0].values

    async def update(self, object_dict):
        new_version_id = str(int(self.metadata.version_id) + 1)
        return MockDownloadedObject(self.mock_client, object_dict, new_version_id)


class MockClient:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.data = {}
        self.objects = {}

    def get_dataframe(self, data: dict) -> pd.DataFrame:
        return self.data[data["data"]]

    async def upload_dataframe(self, df: pd.DataFrame, *args, **kwargs) -> dict:
        data_id = str(uuid.uuid4())
        self.data[data_id] = df
        return {"data": data_id, "length": df.shape[0]}

    async def upload_table(self, table, *args, **kwargs) -> dict:
        """Upload a PyArrow table (used for masks and other array data)."""
        data_id = str(uuid.uuid4())
        # Convert PyArrow table to pandas for storage
        self.data[data_id] = table.to_pandas()
        # Return table info with length
        return {"data": data_id, "length": len(table)}

    async def upload_category_dataframe(self, df: pd.DataFrame, *args, **kwargs) -> dict:
        return {
            "values": await self.upload_dataframe(df),
            "category_data": True,
        }

    async def create_geoscience_object(
        self, context: IContext, object_dict: dict, parent: str | None = None, path: str | None = None
    ):
        object_dict = object_dict.copy()
        object_dict["uuid"] = str(uuid.uuid4())
        self.objects[object_dict["uuid"]] = copy.deepcopy(object_dict)
        return MockDownloadedObject(self, object_dict)

    async def replace_geoscience_object(
        self, context: IContext, reference: ObjectReference, object_dict: dict, create_if_missing=False
    ):
        object_dict = object_dict.copy()
        assert reference.object_id is not None, "Reference must have an object ID"
        object_dict["uuid"] = str(reference.object_id)
        self.objects[object_dict["uuid"]] = copy.deepcopy(object_dict)
        return MockDownloadedObject(self, object_dict)

    async def from_reference(self, context: IContext, reference: ObjectReference):
        assert reference.object_id is not None, "Reference must have an object ID"
        object_dict = copy.deepcopy(self.objects[str(reference.object_id)])
        return MockDownloadedObject(self, object_dict)

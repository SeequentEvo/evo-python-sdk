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

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pandas as pd

from evo.common.interfaces import IContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.utils.table_formats import FLOAT_ARRAY_3, INDEX_ARRAY_3, KnownTableFormat

from ._data import DataTable, DataTableAndAttributes
from ._model import DataLocation, SchemaBuilder, SchemaLocation, SchemaModel
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox

__all__ = [
    "Indices",
    "TriangleMesh",
    "TriangleMeshData",
    "Triangles",
    "Vertices",
]

_X = "x"
_Y = "y"
_Z = "z"
_VERTEX_COLUMNS = [_X, _Y, _Z]

_N0 = "n0"
_N1 = "n1"
_N2 = "n2"
_INDEX_COLUMNS = [_N0, _N1, _N2]


def _bounding_box_from_dataframe(df: pd.DataFrame) -> BoundingBox:
    return BoundingBox.from_points(
        df[_X].values,
        df[_Y].values,
        df[_Z].values,
    )


@dataclass(kw_only=True, frozen=True)
class TriangleMeshData(BaseSpatialObjectData):
    """Data class for creating a new TriangleMesh object.

    :param name: The name of the object.
    :param vertices: A DataFrame containing the vertex data. Must have 'x', 'y', 'z' columns for coordinates.
        Any additional columns will be treated as vertex attributes.
    :param triangles: A DataFrame containing the triangle indices. Must have 'n0', 'n1', 'n2' columns
        as 0-based indices into the vertices. Any additional columns will be treated as triangle attributes.
    :param coordinate_reference_system: Optional EPSG code or WKT string for the coordinate reference system.
    :param description: Optional description of the object.
    :param tags: Optional dictionary of tags for the object.
    :param extensions: Optional dictionary of extensions for the object.
    """

    vertices: pd.DataFrame
    triangles: pd.DataFrame

    def __post_init__(self):
        missing_vertex_cols = set(_VERTEX_COLUMNS) - set(self.vertices.columns)
        if missing_vertex_cols:
            raise ObjectValidationError(
                f"vertices DataFrame must have 'x', 'y', 'z' columns. Missing: {missing_vertex_cols}"
            )

        missing_index_cols = set(_INDEX_COLUMNS) - set(self.triangles.columns)
        if missing_index_cols:
            raise ObjectValidationError(
                f"triangles DataFrame must have 'n0', 'n1', 'n2' columns. Missing: {missing_index_cols}"
            )

        # Validate that triangle indices are within valid range
        max_index = self.triangles[_INDEX_COLUMNS].max().max()
        num_vertices = len(self.vertices)
        if max_index >= num_vertices:
            raise ObjectValidationError(
                f"Triangle indices reference vertex index {max_index}, but only {num_vertices} vertices exist."
            )

    def compute_bounding_box(self) -> BoundingBox:
        return _bounding_box_from_dataframe(self.vertices)


class VertexCoordinateTable(DataTable):
    """DataTable subclass for vertex coordinates with x, y, z columns."""

    table_format: ClassVar[KnownTableFormat] = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _VERTEX_COLUMNS

    async def set_dataframe(self, df: pd.DataFrame, fb: IFeedback = NoFeedback):
        """Update the vertex coordinate values and recalculate the bounding box.

        :param df: DataFrame containing x, y, z coordinate columns.
        :param fb: Optional feedback object to report upload progress.
        """
        await super().set_dataframe(df, fb)

        # Update the bounding box in the parent object context
        self._context.root_model.bounding_box = _bounding_box_from_dataframe(df)


class TriangleIndexTable(DataTable):
    """DataTable subclass for triangle indices with n0, n1, n2 columns."""

    table_format: ClassVar[KnownTableFormat] = INDEX_ARRAY_3
    data_columns: ClassVar[list[str]] = _INDEX_COLUMNS


class Vertices(DataTableAndAttributes):
    """A dataset representing the vertices of a TriangleMesh.

    Contains the coordinates of each vertex and optional attributes.
    """

    _table: Annotated[VertexCoordinateTable, SchemaLocation("")]


class Indices(DataTableAndAttributes):
    """A dataset representing the triangle indices of a TriangleMesh.

    Contains indices into the vertex list defining triangles and optional attributes.
    """

    _table: Annotated[TriangleIndexTable, SchemaLocation("")]


@dataclass(kw_only=True, frozen=True)
class _TrianglesData:
    """Internal data class for the triangles component."""

    vertices: pd.DataFrame
    triangles: pd.DataFrame


class Triangles(SchemaModel):
    """A dataset representing the triangles of a TriangleMesh.

    This is the top-level container for the triangles component of the mesh,
    containing both vertices and triangle indices.
    """

    vertices: Annotated[Vertices, SchemaLocation("vertices"), DataLocation("vertices")]
    indices: Annotated[Indices, SchemaLocation("indices"), DataLocation("triangles")]

    @property
    def num_vertices(self) -> int:
        """The number of vertices in this mesh."""
        return self.vertices.length

    @property
    def num_triangles(self) -> int:
        """The number of triangles in this mesh."""
        return self.indices.length

    async def get_vertices_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the vertex coordinates and attributes.

        :param fb: Optional feedback object to report download progress.
        :return: DataFrame with x, y, z coordinates and additional columns for attributes.
        """
        return await self.vertices.get_dataframe(fb=fb)

    async def get_indices_dataframe(self, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Load a DataFrame containing the triangle indices and attributes.

        :param fb: Optional feedback object to report download progress.
        :return: DataFrame with n0, n1, n2 indices and additional columns for attributes.
        """
        return await self.indices.get_dataframe(fb=fb)

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> Any:
        """Convert triangles data to schema format."""
        builder = SchemaBuilder(cls, context)
        await builder.set_sub_model_value("vertices", data.vertices)
        await builder.set_sub_model_value("indices", data.triangles)
        return builder.document


class TriangleMesh(BaseSpatialObject):
    """A GeoscienceObject representing a mesh composed of triangles.

    The triangles are defined by triplets of indices into a vertex list.
    The object contains a triangles dataset with vertices, indices, and optional attributes
    for both vertices and triangles.
    """

    _data_class = TriangleMeshData

    sub_classification = "triangle-mesh"
    creation_schema_version = SchemaVersion(major=2, minor=2, patch=0)

    triangles: Annotated[Triangles, SchemaLocation("triangles")]

    @classmethod
    async def _data_to_schema(cls, data: TriangleMeshData, context: IContext) -> dict[str, Any]:
        """Create an object dictionary suitable for creating a new Geoscience Object."""
        object_dict = await super()._data_to_schema(data, context)
        # Create the triangles data structure
        triangles_data = _TrianglesData(vertices=data.vertices, triangles=data.triangles)
        object_dict["triangles"] = await Triangles._data_to_schema(triangles_data, context)
        return object_dict

    @property
    def num_vertices(self) -> int:
        """The number of vertices in this mesh."""
        return self.triangles.num_vertices

    @property
    def num_triangles(self) -> int:
        """The number of triangles in this mesh."""
        return self.triangles.num_triangles

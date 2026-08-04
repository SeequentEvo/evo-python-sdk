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

"""Typed access for downhole-intervals objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pandas as pd

from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.utils.table_formats import FLOAT_ARRAY_3

from ._data import DataTable
from ._downhole import FromToModel, HoleIdCategory
from ._model import DataLocation, SchemaLocation
from .attributes import Attributes
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox

__all__ = [
    "DownholeIntervals",
    "DownholeIntervalsData",
]


_HOLE_ID_COL = "hole_id"
_FROM_COL = "from"
_TO_COL = "to"
_START_COLS: list[str] = ["x_start", "y_start", "z_start"]
_END_COLS: list[str] = ["x_end", "y_end", "z_end"]
_MID_COLS: list[str] = ["x_mid", "y_mid", "z_mid"]

_ALL_REQUIRED_COLS: frozenset[str] = frozenset([_HOLE_ID_COL, _FROM_COL, _TO_COL] + _START_COLS + _END_COLS + _MID_COLS)


@dataclass(kw_only=True, frozen=True)
class DownholeIntervalsData(BaseSpatialObjectData):
    """Data for creating a new DownholeIntervals object.

    :param name: The name of the object.
    :param intervals: DataFrame containing the interval data. Required columns:

        * `hole_id` — hole identifier (string or Categorical)
        * `from` — depth of the top of the interval
        * `to` — depth of the base of the interval
        * `x_start`, `y_start`, `z_start` — 3D start-point coordinates
        * `x_end`, `y_end`, `z_end` — 3D end-point coordinates
        * `x_mid`, `y_mid`, `z_mid` — 3D mid-point coordinates

        Any additional columns are uploaded as interval attributes. Per-attribute
        units can be specified via the DataFrame's `attrs` dictionary using
        :class:`~evo.objects.typed.attributes.AttributeDescription`::

            from evo.objects.typed.attributes import AttributeDescription
            df.attrs["attribute_descriptions"] = {"grade": AttributeDescription(unit="ppm")}

    :param is_composited: Whether the intervals have been composited.
    :param depth_unit: Optional unit identifier for the from/to depths (e.g. `"m"`).
    :param coordinate_reference_system: Optional EPSG code or OGC WKT string for the CRS.
    :param description: Optional description of the object.
    :param tags: Optional dictionary of tags for the object.
    :param extensions: Optional dictionary of extensions for the object.
    """

    intervals: pd.DataFrame
    is_composited: bool
    depth_unit: str | None = None

    def __post_init__(self) -> None:
        missing = _ALL_REQUIRED_COLS - set(self.intervals.columns)
        if missing:
            raise ObjectValidationError(f"intervals DataFrame is missing required columns: {sorted(missing)}")

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from all start, end, and mid-point coordinates."""
        df = self.intervals
        all_x = pd.concat([df["x_start"], df["x_end"], df["x_mid"]])
        all_y = pd.concat([df["y_start"], df["y_end"], df["y_mid"]])
        all_z = pd.concat([df["z_start"], df["z_end"], df["z_mid"]])
        return BoundingBox.from_points(all_x.values, all_y.values, all_z.values)


class StartCoordTable(DataTable):
    table_format: ClassVar = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _START_COLS

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        return await super()._data_to_schema(data[_START_COLS], context)


class EndCoordTable(DataTable):
    table_format: ClassVar = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _END_COLS

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        return await super()._data_to_schema(data[_END_COLS], context)


class MidCoordTable(DataTable):
    table_format: ClassVar = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _MID_COLS

    @classmethod
    async def _data_to_schema(cls, data: pd.DataFrame, context: IContext) -> Any:
        return await super()._data_to_schema(data[_MID_COLS], context)


class IntervalAttributes(Attributes):
    """Attributes sub-model that filters out required columns before upload."""

    @classmethod
    async def _data_to_schema(cls, data: Any, context: IContext) -> list[dict[str, Any]]:
        if data is not None:
            attr_cols = [c for c in data.columns if c not in _ALL_REQUIRED_COLS]
            data = data[attr_cols] if attr_cols else None
        return await super()._data_to_schema(data, context)


class DownholeIntervals(BaseSpatialObject):
    """A GeoscienceObject representing downhole intervals.

    Downhole intervals describe depth-ranged samples along drill holes.  Each
    interval is defined by a hole identifier, a from/to depth range, and the
    3D coordinates of the interval's start, end, and mid-point.  Optional
    attributes (assay values, lithology codes, etc.) may also be attached.

    Example usage::

        import pandas as pd
        from evo.objects.typed import DownholeIntervals, DownholeIntervalsData

        df = pd.DataFrame(
            {
                "hole_id": pd.Categorical(["DH001", "DH001", "DH002"]),
                "from":    [0.0,  5.0,  0.0],
                "to":      [5.0, 10.0,  3.0],
                "x_start": [100.0, 100.0, 200.0],
                "y_start": [200.0, 200.0, 300.0],
                "z_start": [  0.0,  -5.0,   0.0],
                "x_end":   [100.0, 100.0, 200.0],
                "y_end":   [200.0, 200.0, 300.0],
                "z_end":   [ -5.0, -10.0,  -3.0],
                "x_mid":   [100.0, 100.0, 200.0],
                "y_mid":   [200.0, 200.0, 300.0],
                "z_mid":   [ -2.5,  -7.5,  -1.5],
            }
        )
        data = DownholeIntervalsData(
            name="My Downhole Intervals",
            intervals=df,
            is_composited=False,
            depth_unit="m",
        )
        obj = await DownholeIntervals.create(context, data)

        # Download all data as a single DataFrame
        df = await obj.to_dataframe()
    """

    _data_class = DownholeIntervalsData

    sub_classification = "downhole-intervals"
    creation_schema_version = SchemaVersion(major=1, minor=3, patch=0)

    is_composited: Annotated[bool, SchemaLocation("is_composited")]

    start: Annotated[StartCoordTable, SchemaLocation("start.coordinates"), DataLocation("intervals")]
    end: Annotated[EndCoordTable, SchemaLocation("end.coordinates"), DataLocation("intervals")]
    mid_points: Annotated[MidCoordTable, SchemaLocation("mid_points.coordinates"), DataLocation("intervals")]
    from_to: Annotated[FromToModel, SchemaLocation("from_to")]
    hole_id: Annotated[HoleIdCategory, SchemaLocation("hole_id"), DataLocation("intervals")]
    attributes: Annotated[IntervalAttributes, SchemaLocation("attributes"), DataLocation("intervals")]

    @property
    def depth_unit(self) -> str | None:
        """The unit of the from/to depths, or None if not specified."""
        return self.from_to.unit

    @property
    def num_intervals(self) -> int:
        """The number of intervals in this object."""
        return self.from_to.intervals.length

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get all interval data as a single DataFrame.

        The returned DataFrame has the following columns, in order:

        * `hole_id` — hole identifier
        * `from`, `to` — depth interval
        * `x_start`, `y_start`, `z_start` — start-point coordinates
        * `x_end`, `y_end`, `z_end` — end-point coordinates
        * `x_mid`, `y_mid`, `z_mid` — mid-point coordinates
        * Any attribute columns

        :param keys: Optional attribute keys/names to include.  If omitted, all
            attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A combined DataFrame of all interval data and attributes.
        """
        hole_id_df = await self.hole_id.to_dataframe(fb=fb)
        depth_df = await self.from_to.intervals.to_dataframe(fb=fb)
        start_df = await self.start.to_dataframe(fb=fb)
        end_df = await self.end.to_dataframe(fb=fb)
        mid_df = await self.mid_points.to_dataframe(fb=fb)

        parts: list[pd.DataFrame] = [hole_id_df, depth_df, start_df, end_df, mid_df]

        if len(self.attributes) > 0:
            attr_df = await self.attributes.to_dataframe(*keys, fb=fb)
            parts.append(attr_df)

        return pd.concat(parts, axis=1)

    def validate(self) -> None:
        """Validate the object, checking that all interval tables have consistent lengths."""
        super().validate()
        expected_length = self.num_intervals
        for table_name, table in (
            ("hole_id", self.hole_id),
            ("start.coordinates", self.start),
            ("end.coordinates", self.end),
            ("mid_points.coordinates", self.mid_points),
        ):
            if table.length != expected_length:
                raise ObjectValidationError(
                    f"{table_name} length ({table.length}) does not match expected length ({expected_length})"
                )

        self.attributes.validate_lengths(expected_length)

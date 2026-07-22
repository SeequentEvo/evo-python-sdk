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

"""Typed access for downhole-intervals objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pandas as pd

from evo.common import IContext, IFeedback
from evo.common.utils import NoFeedback
from evo.objects import SchemaVersion
from evo.objects.utils.table_formats import FLOAT_ARRAY_2, FLOAT_ARRAY_3

from ._data import DataTable
from ._model import DataLocation, SchemaBuilder, SchemaLocation, SchemaModel
from ._utils import get_data_client
from .attributes import Attributes
from .exceptions import ObjectValidationError
from .spatial import BaseSpatialObject, BaseSpatialObjectData
from .types import BoundingBox

__all__ = [
    "DownholeIntervals",
    "DownholeIntervalsData",
]

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

_HOLE_ID_COL = "hole_id"
_FROM_COL = "from"
_TO_COL = "to"
_START_COLS: list[str] = ["x_start", "y_start", "z_start"]
_END_COLS: list[str] = ["x_end", "y_end", "z_end"]
_MID_COLS: list[str] = ["x_mid", "y_mid", "z_mid"]
_DEPTH_COLS: list[str] = [_FROM_COL, _TO_COL]

_ALL_REQUIRED_COLS: frozenset[str] = frozenset(
    [_HOLE_ID_COL, _FROM_COL, _TO_COL] + _START_COLS + _END_COLS + _MID_COLS
)


# ---------------------------------------------------------------------------
# Data class (used when creating new objects)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, frozen=True)
class DownholeIntervalsData(BaseSpatialObjectData):
    """Data for creating a new DownholeIntervals object.

    :param name: The name of the object.
    :param intervals: DataFrame containing the interval data. Required columns:

        * ``hole_id`` — hole identifier (string or Categorical)
        * ``from`` — depth of the top of the interval
        * ``to`` — depth of the base of the interval
        * ``x_start``, ``y_start``, ``z_start`` — 3D start-point coordinates
        * ``x_end``, ``y_end``, ``z_end`` — 3D end-point coordinates
        * ``x_mid``, ``y_mid``, ``z_mid`` — 3D mid-point coordinates

        Any additional columns are uploaded as interval attributes.

    :param is_composited: Whether the intervals have been composited.
    :param depth_unit: Optional unit identifier for the from/to depths (e.g. ``"m"``).
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
            raise ObjectValidationError(
                f"intervals DataFrame is missing required columns: {sorted(missing)}"
            )

    def compute_bounding_box(self) -> BoundingBox:
        """Compute the bounding box from all start, end, and mid-point coordinates."""
        df = self.intervals
        all_x = pd.concat([df["x_start"], df["x_end"], df["x_mid"]])
        all_y = pd.concat([df["y_start"], df["y_end"], df["y_mid"]])
        all_z = pd.concat([df["z_start"], df["z_end"], df["z_mid"]])
        return BoundingBox.from_points(all_x.values, all_y.values, all_z.values)


# ---------------------------------------------------------------------------
# Schema sub-models for reading
# ---------------------------------------------------------------------------


class _StartCoordTable(DataTable):
    table_format: ClassVar = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _START_COLS


class _EndCoordTable(DataTable):
    table_format: ClassVar = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _END_COLS


class _MidCoordTable(DataTable):
    table_format: ClassVar = FLOAT_ARRAY_3
    data_columns: ClassVar[list[str]] = _MID_COLS


class _StartLocations(SchemaModel):
    _coords: Annotated[_StartCoordTable, SchemaLocation("coordinates")]


class _EndLocations(SchemaModel):
    _coords: Annotated[_EndCoordTable, SchemaLocation("coordinates")]


class _MidLocations(SchemaModel):
    _coords: Annotated[_MidCoordTable, SchemaLocation("coordinates")]


class _DepthIntervalsTable(DataTable):
    table_format: ClassVar = FLOAT_ARRAY_2
    data_columns: ClassVar[list[str]] = _DEPTH_COLS


class _IntervalsWrapper(SchemaModel):
    """Wraps the 'intervals' sub-object inside 'from_to'."""

    _table: Annotated[_DepthIntervalsTable, SchemaLocation("start_and_end")]


class _FromToModel(SchemaModel):
    """Schema model for the from_to component of a downhole intervals object."""

    _intervals: Annotated[_IntervalsWrapper, SchemaLocation("intervals")]
    unit: Annotated[str | None, SchemaLocation("unit")]


# ---------------------------------------------------------------------------
# Typed object
# ---------------------------------------------------------------------------


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

    # --- schema properties ---
    is_composited: Annotated[bool, SchemaLocation("is_composited")]

    # --- sub-models (reading only; skipped in _data_to_schema) ---
    _start: Annotated[_StartLocations, SchemaLocation("start")]
    _end: Annotated[_EndLocations, SchemaLocation("end")]
    _mid_points: Annotated[_MidLocations, SchemaLocation("mid_points")]
    _from_to: Annotated[_FromToModel, SchemaLocation("from_to")]
    attributes: Annotated[Attributes, SchemaLocation("attributes"), DataLocation("intervals")]

    @property
    def depth_unit(self) -> str | None:
        """The unit of the from/to depths, or None if not specified."""
        return self._from_to.unit

    @property
    def num_intervals(self) -> int:
        """The number of intervals in this object."""
        return self._from_to._intervals._table.length

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------

    @classmethod
    async def _data_to_schema(cls, data: DownholeIntervalsData, context: IContext) -> dict[str, Any]:
        """Build the schema document for a new DownholeIntervals object.

        Handles the complex multi-table upload while delegating scalar
        properties (name, description, coordinate_reference_system, etc.) to
        the standard SchemaBuilder machinery.
        """
        from evo.objects import ObjectSchema

        schema_id = ObjectSchema("objects", cls.sub_classification, cls.creation_schema_version)

        # Build scalar properties from the full inheritance chain, skipping all
        # sub-models which are handled manually below.
        all_sub_models = set(cls._sub_models.keys())
        builder = SchemaBuilder(cls, context)
        result = await builder.build_from_data(data, skip_sub_models=all_sub_models)

        # schema ID (normally added by _BaseObject._data_to_schema)
        result["schema"] = str(schema_id)

        # bounding_box — _bounding_box is a schema property that resolves to None
        # from the data class (BaseSpatialObjectData doesn't store _bounding_box),
        # so we compute and set it explicitly here.
        result["bounding_box"] = cls._bbox_type_adapter.dump_python(data.compute_bounding_box())

        # Upload all table data
        df = data.intervals
        data_client = get_data_client(context)

        # Start coordinates (x_start, y_start, z_start) → start.coordinates
        start_info = await data_client.upload_dataframe(df[_START_COLS], table_format=FLOAT_ARRAY_3)
        result["start"] = {"coordinates": start_info}

        # End coordinates (x_end, y_end, z_end) → end.coordinates
        end_info = await data_client.upload_dataframe(df[_END_COLS], table_format=FLOAT_ARRAY_3)
        result["end"] = {"coordinates": end_info}

        # Mid-point coordinates (x_mid, y_mid, z_mid) → mid_points.coordinates
        mid_info = await data_client.upload_dataframe(df[_MID_COLS], table_format=FLOAT_ARRAY_3)
        result["mid_points"] = {"coordinates": mid_info}

        # From/to depths → from_to.intervals.start_and_end
        depth_info = await data_client.upload_dataframe(df[_DEPTH_COLS], table_format=FLOAT_ARRAY_2)
        result["from_to"] = {"intervals": {"start_and_end": depth_info}}
        if data.depth_unit is not None:
            result["from_to"]["unit"] = data.depth_unit

        # Hole IDs → hole_id (category data: lookup table + integer indices)
        hole_id_df = df[[_HOLE_ID_COL]]
        category_info = await data_client.upload_category_dataframe(hole_id_df)
        result["hole_id"] = category_info

        # Interval attributes (any extra columns beyond the required set)
        attr_cols = [c for c in df.columns if c not in _ALL_REQUIRED_COLS]
        result["attributes"] = []
        if attr_cols:
            await Attributes._upload_attributes_to_list(result["attributes"], df[attr_cols], context)

        return result

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    async def to_dataframe(self, *keys: str, fb: IFeedback = NoFeedback) -> pd.DataFrame:
        """Get all interval data as a single DataFrame.

        The returned DataFrame has the following columns, in order:

        * ``hole_id`` — hole identifier
        * ``from``, ``to`` — depth interval
        * ``x_start``, ``y_start``, ``z_start`` — start-point coordinates
        * ``x_end``, ``y_end``, ``z_end`` — end-point coordinates
        * ``x_mid``, ``y_mid``, ``z_mid`` — mid-point coordinates
        * Any attribute columns

        :param keys: Optional attribute keys/names to include.  If omitted, all
            attributes are included.
        :param fb: Optional feedback object to report download progress.
        :return: A combined DataFrame of all interval data and attributes.
        """
        hole_id_df = await self._obj.download_category_dataframe(
            "hole_id", column_names=[_HOLE_ID_COL]
        )
        depth_df = await self._from_to._intervals._table.to_dataframe(fb=fb)
        start_df = await self._start._coords.to_dataframe(fb=fb)
        end_df = await self._end._coords.to_dataframe(fb=fb)
        mid_df = await self._mid_points._coords.to_dataframe(fb=fb)

        parts: list[pd.DataFrame] = [hole_id_df, depth_df, start_df, end_df, mid_df]

        if len(self.attributes) > 0:
            attr_df = await self.attributes.to_dataframe(*keys, fb=fb)
            parts.append(attr_df)

        return pd.concat(parts, axis=1)

    def validate(self) -> None:
        """Validate the object, checking that all attribute lengths match num_intervals."""
        super().validate()
        self.attributes.validate_lengths(self.num_intervals)

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

import contextlib
import dataclasses
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
from parameterized import parameterized

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import BoundingBox, DownholeIntervals, DownholeIntervalsData
from evo.objects.typed.base import BaseObject
from evo.objects.typed.exceptions import ObjectValidationError

from .helpers import MockClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 4  # number of test intervals


def _make_intervals(**overrides) -> pd.DataFrame:
    """Return a minimal valid intervals DataFrame."""
    base = {
        "hole_id": pd.Categorical(["DH001", "DH001", "DH002", "DH002"]),
        "from": [0.0, 5.0, 0.0, 3.0],
        "to": [5.0, 10.0, 3.0, 6.0],
        "x_start": [100.0, 100.0, 200.0, 200.0],
        "y_start": [200.0, 200.0, 300.0, 300.0],
        "z_start": [0.0, -5.0, 0.0, -3.0],
        "x_end": [100.0, 100.0, 200.0, 200.0],
        "y_end": [200.0, 200.0, 300.0, 300.0],
        "z_end": [-5.0, -10.0, -3.0, -6.0],
        "x_mid": [100.0, 100.0, 200.0, 200.0],
        "y_mid": [200.0, 200.0, 300.0, 300.0],
        "z_mid": [-2.5, -7.5, -1.5, -4.5],
    }
    base.update(overrides)
    return pd.DataFrame(base)


_BBOX = BoundingBox(
    min_x=100.0, min_y=200.0, max_x=200.0, max_y=300.0, min_z=-10.0, max_z=0.0
)


class TestDownholeIntervals(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
        )

    @contextlib.contextmanager
    def _mock_geoscience_objects(self):
        mock_client = MockClient(self.environment)
        with (
            patch("evo.objects.typed.attributes.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed._data.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.downhole_intervals.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.DownloadedObject.from_context", mock_client.from_reference),
        ):
            yield mock_client

    example_data = DownholeIntervalsData(
        name="Test Downhole Intervals",
        intervals=_make_intervals(),
        is_composited=False,
        depth_unit="m",
    )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    @parameterized.expand([BaseObject, DownholeIntervals])
    async def test_create(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=self.example_data)
        self.assertIsInstance(result, DownholeIntervals)
        self.assertEqual(result.name, "Test Downhole Intervals")
        self.assertFalse(result.is_composited)
        self.assertEqual(result.num_intervals, _N)
        self.assertEqual(result.depth_unit, "m")

    @parameterized.expand([BaseObject, DownholeIntervals])
    async def test_replace(self, class_to_call):
        data = dataclasses.replace(self.example_data, name="Replaced Intervals")
        with self._mock_geoscience_objects():
            result = await class_to_call.replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        self.assertIsInstance(result, DownholeIntervals)
        self.assertEqual(result.name, "Replaced Intervals")

    @parameterized.expand([BaseObject, DownholeIntervals])
    async def test_create_or_replace(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create_or_replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=self.example_data,
            )
        self.assertIsInstance(result, DownholeIntervals)
        self.assertEqual(result.name, "Test Downhole Intervals")

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await DownholeIntervals.create(context=self.context, data=self.example_data)
            result = await DownholeIntervals.from_reference(
                context=self.context, reference=original.metadata.url
            )
        self.assertIsInstance(result, DownholeIntervals)
        self.assertEqual(result.name, "Test Downhole Intervals")
        self.assertEqual(result.num_intervals, _N)

    async def test_update(self):
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
            self.assertEqual(obj.metadata.version_id, "1")
            obj.name = "Updated Intervals"
            obj.is_composited = True
            await obj.update()
        self.assertEqual(obj.name, "Updated Intervals")
        self.assertTrue(obj.is_composited)
        self.assertEqual(obj.metadata.version_id, "2")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    async def test_to_dataframe_columns(self):
        """to_dataframe() returns all required columns."""
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
            df = await obj.to_dataframe()

        expected_cols = [
            "hole_id",
            "from", "to",
            "x_start", "y_start", "z_start",
            "x_end", "y_end", "z_end",
            "x_mid", "y_mid", "z_mid",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertEqual(len(df), _N)

    async def test_to_dataframe_depth_values(self):
        """from/to columns contain the expected depth values."""
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
            df = await obj.to_dataframe()

        source = self.example_data.intervals
        pd.testing.assert_series_equal(
            df["from"].reset_index(drop=True),
            source["from"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df["to"].reset_index(drop=True),
            source["to"].reset_index(drop=True),
            check_names=False,
        )

    async def test_to_dataframe_coordinates_roundtrip(self):
        """Coordinate columns survive a create → from_reference → to_dataframe round-trip."""
        with self._mock_geoscience_objects():
            created = await DownholeIntervals.create(context=self.context, data=self.example_data)
            obj = await DownholeIntervals.from_reference(
                context=self.context, reference=created.metadata.url
            )
            df = await obj.to_dataframe()

        source = self.example_data.intervals
        for col in ["x_start", "y_start", "z_start", "x_end", "y_end", "z_end", "x_mid", "y_mid", "z_mid"]:
            np.testing.assert_array_almost_equal(
                df[col].values, source[col].values, err_msg=f"Mismatch in column {col}"
            )

    async def test_to_dataframe_with_attributes(self):
        """Extra columns in the intervals DataFrame are stored as attributes and included in to_dataframe()."""
        df = _make_intervals(grade=pd.Series([1.2, 3.4, 5.6, 7.8]))
        data = dataclasses.replace(self.example_data, intervals=df)
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=data)
            result = await obj.to_dataframe()

        self.assertIn("grade", result.columns)
        pd.testing.assert_series_equal(
            result["grade"].reset_index(drop=True),
            df["grade"].reset_index(drop=True),
            check_names=False,
        )

    async def test_hole_id_roundtrip(self):
        """hole_id strings survive a round-trip."""
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
            df = await obj.to_dataframe()

        source = self.example_data.intervals["hole_id"]
        result = df["hole_id"]
        # Compare as plain strings (mock may not preserve Categorical dtype)
        self.assertEqual(list(result.astype(str)), list(source.astype(str)))

    # ------------------------------------------------------------------
    # Individual coordinate access
    # ------------------------------------------------------------------

    async def test_start_coordinates(self):
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
            start_df = await obj._start._coords.to_dataframe()

        self.assertEqual(list(start_df.columns), ["x_start", "y_start", "z_start"])
        self.assertEqual(len(start_df), _N)

    async def test_end_coordinates(self):
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
            end_df = await obj._end._coords.to_dataframe()

        self.assertEqual(list(end_df.columns), ["x_end", "y_end", "z_end"])

    async def test_mid_coordinates(self):
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
            mid_df = await obj._mid_points._coords.to_dataframe()

        self.assertEqual(list(mid_df.columns), ["x_mid", "y_mid", "z_mid"])

    # ------------------------------------------------------------------
    # Metadata / properties
    # ------------------------------------------------------------------

    async def test_depth_unit_none_when_not_set(self):
        data = dataclasses.replace(self.example_data, depth_unit=None)
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=data)
        self.assertIsNone(obj.depth_unit)

    async def test_is_composited_true(self):
        data = dataclasses.replace(self.example_data, is_composited=True)
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=data)
        self.assertTrue(obj.is_composited)

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def test_compute_bounding_box(self):
        bbox = self.example_data.compute_bounding_box()
        self.assertAlmostEqual(bbox.min_x, 100.0)
        self.assertAlmostEqual(bbox.max_x, 200.0)
        self.assertAlmostEqual(bbox.min_y, 200.0)
        self.assertAlmostEqual(bbox.max_y, 300.0)
        self.assertAlmostEqual(bbox.min_z, -10.0)
        self.assertAlmostEqual(bbox.max_z, 0.0)

    async def test_bounding_box_stored_in_schema(self):
        with self._mock_geoscience_objects():
            obj = await DownholeIntervals.create(context=self.context, data=self.example_data)
        bbox = obj.bounding_box
        self.assertAlmostEqual(bbox.min_x, 100.0)
        self.assertAlmostEqual(bbox.max_x, 200.0)
        self.assertAlmostEqual(bbox.min_z, -10.0)
        self.assertAlmostEqual(bbox.max_z, 0.0)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_validation_missing_columns(self):
        """DownholeIntervalsData rejects DataFrames missing required columns."""
        df = pd.DataFrame({"hole_id": ["A"], "from": [0.0], "to": [1.0]})
        with self.assertRaises(ObjectValidationError):
            DownholeIntervalsData(
                name="Bad",
                intervals=df,
                is_composited=False,
            )

    def test_validation_extra_columns_allowed(self):
        """Extra columns beyond the required set are accepted as attributes."""
        df = _make_intervals(grade=pd.Series([1.0, 2.0, 3.0, 4.0]))
        data = DownholeIntervalsData(name="Extra cols", intervals=df, is_composited=False)
        self.assertIn("grade", data.intervals.columns)

    def test_string_hole_id_accepted(self):
        """Plain string dtype for hole_id is accepted."""
        df = _make_intervals()
        df["hole_id"] = df["hole_id"].astype(str)
        data = DownholeIntervalsData(name="String hole_id", intervals=df, is_composited=False)
        # Verify that hole_id is a string-like column (object or StringDtype depending on pandas version)
        self.assertTrue(
            pd.api.types.is_string_dtype(data.intervals["hole_id"]),
            f"Expected string dtype, got {data.intervals['hole_id'].dtype}",
        )

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

from __future__ import annotations

from unittest import TestCase

import numpy as np
import numpy.testing as npt
from parameterized import parameterized
from pydantic import TypeAdapter, ValidationError

from evo.common import EpsgCode as CommonEpsgCode
from evo.objects.typed import (
    BoundingBox,
    CoordinateReferenceSystem,
    Ellipsoid,
    EllipsoidRanges,
    EpsgCode,
    Point3,
    Rotation,
    Size3d,
    Size3i,
)
from evo.objects.typed.spatial import BaseSpatialObject
from evo.objects.typed.types import EpsgCode as TypesEpsgCode


class TestTypes(TestCase):
    @parameterized.expand(
        [
            ("minimum", 0, -90, 0),
            ("maximum", 360, 90, 180),
        ]
    )
    def test_rotation_angle_boundaries(self, _name, dip_azimuth, dip, pitch):
        rotation = Rotation(dip_azimuth, dip, pitch)
        self.assertEqual((rotation.dip_azimuth, rotation.dip, rotation.pitch), (dip_azimuth, dip, pitch))

    @parameterized.expand(
        [
            ("azimuth_below", -1, 0, 0, "dip_azimuth"),
            ("azimuth_above", 361, 0, 0, "dip_azimuth"),
            ("dip_below", 0, -91, 0, "dip"),
            ("dip_above", 0, 91, 0, "dip"),
            ("pitch_below", 0, 0, -1, "pitch"),
            ("pitch_above", 0, 0, 181, "pitch"),
            ("nan", np.nan, 0, 0, "dip_azimuth"),
            ("infinity", 0, np.inf, 0, "dip"),
        ]
    )
    def test_invalid_rotation_angles(self, _name, dip_azimuth, dip, pitch, field):
        with self.assertRaisesRegex(ValueError, field):
            Rotation(dip_azimuth, dip, pitch)

    @parameterized.expand(
        [
            (0, 0, 0, [2, 5, 2.5]),
            (90, 0, 0, [5, -2, 2.5]),
            (0, 90, 0, [2, 2.5, -5]),
            (0, 0, 90, [5, -2, 2.5]),
            (124, 63.5, 22.1, [1.2020506, -5.31502659, -2.35702497]),
        ]
    )
    def test_rotation_matrix(self, dip_azimuth, dip, pitch, expected):
        rotation = Rotation(dip_azimuth, dip, pitch)
        matrix = rotation.as_rotation_matrix()
        npt.assert_array_almost_equal(matrix @ np.array([2, 5, 2.5]), expected)

    def test_bounding_box(self):
        box = BoundingBox.from_points(np.array([[0, -1, 5], [1, 2, 4]]))
        self.assertEqual(box.min, Point3(0, -1, 4))
        self.assertEqual(box.max, Point3(1, 2, 5))
        self.assertEqual(box.min_x, 0)
        self.assertEqual(box.max_x, 1)
        self.assertEqual(box.min_y, -1)
        self.assertEqual(box.max_y, 2)
        self.assertEqual(box.min_z, 4)
        self.assertEqual(box.max_z, 5)

        box = BoundingBox.from_points([0, 1], [-1, 2], [4, 5])
        self.assertEqual(box.min, Point3(0, -1, 4))
        self.assertEqual(box.max, Point3(1, 2, 5))

    def test_crs(self):
        type_adapter = TypeAdapter(CoordinateReferenceSystem)
        crs1 = type_adapter.validate_python({"epsg_code": 4326})
        self.assertEqual(crs1, EpsgCode(4326))

        crs2 = type_adapter.validate_python({"ogc_wkt": "WKT_STRING"})
        self.assertEqual(crs2, "WKT_STRING")

        crs3 = type_adapter.validate_python("unspecified")
        self.assertIsNone(crs3)

        self.assertEqual(type_adapter.dump_python(crs1), {"epsg_code": 4326})
        self.assertEqual(type_adapter.dump_python(crs2), {"ogc_wkt": "WKT_STRING"})
        self.assertEqual(type_adapter.dump_python(crs3), "unspecified")

    def test_epsg_code_reexport_is_the_same_class(self):
        self.assertIs(CommonEpsgCode, EpsgCode)
        self.assertIs(CommonEpsgCode, TypesEpsgCode)

    def test_spatial_crs_serializer_uses_schema_wire_shape(self):
        adapter = BaseSpatialObject._crs_type_adapter
        self.assertEqual(adapter.dump_python(EpsgCode(2193)), {"epsg_code": 2193})
        self.assertEqual(adapter.dump_python("WKT_STRING"), {"ogc_wkt": "WKT_STRING"})
        self.assertEqual(adapter.dump_python(None), "unspecified")

    @parameterized.expand(
        [
            ("both", {"epsg_code": 4326, "ogc_wkt": "WKT_STRING"}),
            ("extra", {"epsg_code": 4326, "extra": True}),
            ("wrong_wkt_type", {"ogc_wkt": 42}),
            ("unexpected", {"unexpected": 4326}),
        ]
    )
    def test_invalid_crs_wire_shapes_raise(self, _name, value):
        with self.assertRaises(ValidationError):
            TypeAdapter(CoordinateReferenceSystem).validate_python(value)

    def test_bounding_box_from_extent_no_rotation(self):
        """Test BoundingBox.from_extent without rotation."""
        origin = Point3(10, 20, 30)
        extent = Size3d(5, 10, 15)

        box = BoundingBox.from_extent(origin, extent)

        self.assertAlmostEqual(box.min_x, 10.0)
        self.assertAlmostEqual(box.min_y, 20.0)
        self.assertAlmostEqual(box.min_z, 30.0)
        self.assertAlmostEqual(box.max_x, 15.0)
        self.assertAlmostEqual(box.max_y, 30.0)
        self.assertAlmostEqual(box.max_z, 45.0)

    def test_bounding_box_from_extent_with_rotation(self):
        """Test BoundingBox.from_extent with 90 degree rotation around Z."""
        origin = Point3(0, 0, 0)
        extent = Size3d(10, 20, 5)
        rotation = Rotation(90, 0, 0)  # 90 degrees around Z

        box = BoundingBox.from_extent(origin, extent, rotation)

        # After 90 degree rotation: x becomes y, y becomes -x
        self.assertAlmostEqual(box.min_x, 0.0)
        self.assertAlmostEqual(box.min_y, -10.0)
        self.assertAlmostEqual(box.min_z, 0.0)
        self.assertAlmostEqual(box.max_x, 20.0)
        self.assertAlmostEqual(box.max_y, 0.0)
        self.assertAlmostEqual(box.max_z, 5.0)

    def test_bounding_box_from_regular_grid_no_rotation(self):
        """Test BoundingBox.from_regular_grid without rotation."""
        origin = Point3(100, 200, 0)
        size = Size3i(10, 8, 5)
        cell_size = Size3d(2.5, 5.0, 4.0)

        box = BoundingBox.from_regular_grid(origin, size, cell_size)

        # Extent should be: (10*2.5, 8*5.0, 5*4.0) = (25, 40, 20)
        self.assertAlmostEqual(box.min_x, 100.0)
        self.assertAlmostEqual(box.min_y, 200.0)
        self.assertAlmostEqual(box.min_z, 0.0)
        self.assertAlmostEqual(box.max_x, 125.0)
        self.assertAlmostEqual(box.max_y, 240.0)
        self.assertAlmostEqual(box.max_z, 20.0)

    def test_bounding_box_from_regular_grid_with_rotation(self):
        """Test BoundingBox.from_regular_grid with rotation."""
        origin = Point3(0, 0, 0)
        size = Size3i(10, 10, 5)
        cell_size = Size3d(2.5, 5.0, 5.0)
        rotation = Rotation(90, 0, 0)

        box = BoundingBox.from_regular_grid(origin, size, cell_size, rotation)

        # Extent is (25, 50, 25), rotated 90 degrees around Z
        self.assertAlmostEqual(box.min_x, 0.0)
        self.assertAlmostEqual(box.min_y, -25.0)
        self.assertAlmostEqual(box.min_z, 0.0)
        self.assertAlmostEqual(box.max_x, 50.0)
        self.assertAlmostEqual(box.max_y, 0.0)
        self.assertAlmostEqual(box.max_z, 25.0)

    def test_combine_boxes(self):
        box1 = BoundingBox(min_x=-10.0, max_x=5.0, min_y=-5.0, max_y=10.0, min_z=-15.0, max_z=25.0)
        box2 = BoundingBox(min_x=1.0, max_x=14.0, min_y=1.0, max_y=22.0, min_z=1.0, max_z=33.0)
        combined = BoundingBox.combine([box1, box2])
        self.assertAlmostEqual(combined.min_x, -10.0)
        self.assertAlmostEqual(combined.max_x, 14.0)
        self.assertAlmostEqual(combined.min_y, -5.0)
        self.assertAlmostEqual(combined.max_y, 22.0)
        self.assertAlmostEqual(combined.min_z, -15.0)
        self.assertAlmostEqual(combined.max_z, 33.0)


class TestEllipsoidRanges(TestCase):
    """Tests for EllipsoidRanges class."""

    def test_creation(self):
        """Should create ellipsoid ranges."""
        ranges = EllipsoidRanges(100, 50, 25)
        self.assertEqual(ranges.major, 100)
        self.assertEqual(ranges.semi_major, 50)
        self.assertEqual(ranges.minor, 25)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        ranges = EllipsoidRanges(100, 50, 25)
        d = ranges.to_dict()
        self.assertEqual(d, {"major": 100, "semi_major": 50, "minor": 25})

    def test_scaled(self):
        """Should create scaled ranges."""
        ranges = EllipsoidRanges(100, 50, 25)
        scaled = ranges.scaled(2.0)
        self.assertEqual(scaled.major, 200)
        self.assertEqual(scaled.semi_major, 100)
        self.assertEqual(scaled.minor, 50)


class TestEllipsoid(TestCase):
    """Tests for the Ellipsoid class."""

    def test_basic_creation(self):
        """Should create ellipsoid with ranges and rotation."""
        ell = Ellipsoid(
            ranges=EllipsoidRanges(100, 50, 25),
            rotation=Rotation(45, 30, 0),
        )
        self.assertEqual(ell.ranges.major, 100)
        self.assertEqual(ell.ranges.semi_major, 50)
        self.assertEqual(ell.ranges.minor, 25)
        self.assertEqual(ell.rotation.dip_azimuth, 45)

    @parameterized.expand(
        [
            ("list", [100, 50, 25]),
            ("tuple", (100, 50, 25)),
            ("numpy_array", np.array([100, 50, 25])),
        ]
    )
    def test_creation_from_sequence(self, _name, ranges):
        ell = Ellipsoid(ranges=ranges, rotation=Rotation(45, 30, 0))
        self.assertIsInstance(ell.ranges, EllipsoidRanges)
        self.assertEqual(ell.ranges.to_dict(), {"major": 100, "semi_major": 50, "minor": 25})
        self.assertEqual(ell.rotation.dip_azimuth, 45)
        self.assertEqual(ell.to_dict()["ellipsoid_ranges"], ell.ranges.to_dict())
        self.assertEqual(ell.scaled(2).ranges.to_dict(), {"major": 200, "semi_major": 100, "minor": 50})

    @parameterized.expand(
        [
            ("short_list", [100, 50]),
            ("long_tuple", (100, 50, 25, 10)),
            ("matrix", np.array([[100, 50, 25]])),
        ]
    )
    def test_invalid_sequence_shape(self, _name, ranges):
        with self.assertRaisesRegex(ValueError, "one-dimensional sequence of three values"):
            Ellipsoid(ranges=ranges)

    def test_invalid_ranges_type(self):
        with self.assertRaises(TypeError):
            Ellipsoid(ranges="100, 50, 25")

    def test_invalid_range_values(self):
        with self.assertRaisesRegex(TypeError, "three real numbers"):
            Ellipsoid(ranges=[[100], [50], [25]])

    @parameterized.expand(
        [
            ("nan_list", [100, np.nan, 25]),
            ("infinity_tuple", (100, 50, np.inf)),
            ("negative_infinity_array", np.array([-np.inf, 50, 25])),
        ]
    )
    def test_nonfinite_range_values(self, _name, ranges):
        with self.assertRaisesRegex(ValueError, "ranges must contain three finite real numbers"):
            Ellipsoid(ranges=ranges)

    @parameterized.expand(
        [
            ("list", [45, 30, 15]),
            ("tuple", (45, 30, 15)),
            ("numpy_array", np.array([45, 30, 15])),
        ]
    )
    def test_creation_from_rotation_sequence(self, _name, rotation):
        ell = Ellipsoid(ranges=[100, 50, 25], rotation=rotation)
        self.assertIsInstance(ell.rotation, Rotation)
        self.assertEqual(ell.rotation, Rotation(45, 30, 15))
        self.assertEqual(ell.to_dict()["rotation"], {"dip_azimuth": 45, "dip": 30, "pitch": 15})
        self.assertEqual(ell.scaled(2).rotation, ell.rotation)

    @parameterized.expand(
        [
            ("short_list", [45, 30]),
            ("long_tuple", (45, 30, 15, 0)),
            ("matrix", np.array([[45, 30, 15]])),
        ]
    )
    def test_invalid_rotation_shape(self, _name, rotation):
        with self.assertRaisesRegex(ValueError, "one-dimensional sequence of three values"):
            Ellipsoid(ranges=[100, 50, 25], rotation=rotation)

    def test_invalid_rotation_values(self):
        with self.assertRaisesRegex(TypeError, "three real numbers"):
            Ellipsoid(ranges=[100, 50, 25], rotation=[45, "30", 15])

    @parameterized.expand(
        [
            ("nan_list", [45, np.nan, 15]),
            ("infinity_tuple", (45, 30, np.inf)),
            ("negative_infinity_array", np.array([-np.inf, 30, 15])),
        ]
    )
    def test_nonfinite_rotation_values(self, _name, rotation):
        with self.assertRaisesRegex(ValueError, "rotation must contain three finite real numbers"):
            Ellipsoid(ranges=[100, 50, 25], rotation=rotation)

    def test_invalid_rotation_type(self):
        with self.assertRaises(TypeError):
            Ellipsoid(ranges=[100, 50, 25], rotation="45, 30, 15")

    def test_invalid_rotation_sequence_angle(self):
        with self.assertRaisesRegex(ValueError, "dip_azimuth"):
            Ellipsoid(ranges=[100, 50, 25], rotation=[361, 0, 0])

    def test_default_rotation(self):
        """Should use default rotation when not specified."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        self.assertEqual(ell.rotation.dip_azimuth, 0)
        self.assertEqual(ell.rotation.dip, 0)
        self.assertEqual(ell.rotation.pitch, 0)

    def test_scaled(self):
        """Should create scaled ellipsoid."""
        ell = Ellipsoid(
            ranges=EllipsoidRanges(100, 50, 25),
            rotation=Rotation(45, 30, 0),
        )
        scaled = ell.scaled(2.0)
        self.assertEqual(scaled.ranges.major, 200)
        self.assertEqual(scaled.ranges.semi_major, 100)
        self.assertEqual(scaled.ranges.minor, 50)
        # Rotation should be preserved
        self.assertEqual(scaled.rotation.dip_azimuth, 45)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        ell = Ellipsoid(
            ranges=EllipsoidRanges(100, 50, 25),
            rotation=Rotation(45, 30, 0),
        )
        d = ell.to_dict()
        self.assertEqual(d["ellipsoid_ranges"]["major"], 100)
        self.assertEqual(d["rotation"]["dip_azimuth"], 45)

    def test_surface_points(self):
        """Should generate surface points as 1D arrays."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.surface_points(center=(0, 0, 0), n_points=10)
        self.assertEqual(len(x), 100)  # 10 x 10
        self.assertEqual(len(y), 100)
        self.assertEqual(len(z), 100)

    def test_surface_points_with_center(self):
        """Should offset surface points by center."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x1, y1, z1 = ell.surface_points(center=(0, 0, 0))
        x2, y2, z2 = ell.surface_points(center=(100, 200, 50))
        # Second ellipsoid should be offset
        self.assertAlmostEqual(np.mean(x2) - np.mean(x1), 100, places=1)
        self.assertAlmostEqual(np.mean(y2) - np.mean(y1), 200, places=1)
        self.assertAlmostEqual(np.mean(z2) - np.mean(z1), 50, places=1)

    def test_wireframe_points(self):
        """Should generate wireframe points with NaN separators."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.wireframe_points(center=(0, 0, 0))
        # Should have NaN separators
        self.assertTrue(np.any(np.isnan(x)))
        self.assertTrue(np.any(np.isnan(y)))
        self.assertTrue(np.any(np.isnan(z)))

    def test_wireframe_bounds(self):
        """Wireframe points should be within ellipsoid bounds."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.wireframe_points(n_points=30)
        # Filter out NaN separators
        valid = ~np.isnan(x)
        self.assertTrue(np.all(np.abs(x[valid]) <= 100 * 1.01))
        self.assertTrue(np.all(np.abs(y[valid]) <= 50 * 1.01))
        self.assertTrue(np.all(np.abs(z[valid]) <= 25 * 1.01))

    def test_wireframe_with_rotation(self):
        """Wireframe should apply rotation."""
        ell_no_rot = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25), rotation=Rotation(0, 0, 0))
        ell_rot = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25), rotation=Rotation(45, 30, 0))
        x1, y1, z1 = ell_no_rot.wireframe_points()
        x2, y2, z2 = ell_rot.wireframe_points()
        # Rotated should have different coordinates
        self.assertFalse(np.allclose(x1, x2, equal_nan=True))

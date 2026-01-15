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

from unittest import TestCase

import numpy as np
import numpy.testing as npt
from parameterized import parameterized
from pydantic import TypeAdapter

from evo.objects.typed import BoundingBox, CoordinateReferenceSystem, EpsgCode, Point3, Rotation


class TestTypes(TestCase):
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

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

import unittest

from parameterized import parameterized

from evo.common import EpsgCode, parse_crs
from evo.common.crs import MAX_EPSG_CODE, MIN_EPSG_CODE, UNSPECIFIED_EPSG_CODE

WKT = (
    'PROJCS["NZGD2000 / New Zealand Transverse Mercator 2000",'
    'GEOGCS["NZGD2000",DATUM["New_Zealand_Geodetic_Datum_2000"]]]'
)


class TestEpsgCode(unittest.TestCase):
    def test_from_int(self) -> None:
        code = EpsgCode(2193)
        self.assertIsInstance(code, int)
        self.assertEqual(2193, code)

    def test_from_str(self) -> None:
        self.assertEqual(2193, EpsgCode("2193"))

    def test_repr_and_str(self) -> None:
        code = EpsgCode(2193)
        self.assertEqual("EpsgCode(2193)", repr(code))
        self.assertEqual("EPSG:2193", str(code))

    @parameterized.expand(
        [
            ("below_min", MIN_EPSG_CODE - 1),
            ("above_max", MAX_EPSG_CODE + 1),
            ("zero", 0),
            ("negative", -1),
            ("unspecified_sentinel", UNSPECIFIED_EPSG_CODE),
        ]
    )
    def test_out_of_range_raises(self, _name: str, value: int) -> None:
        with self.assertRaises(ValueError):
            EpsgCode(value)

    def test_non_numeric_str_raises(self) -> None:
        with self.assertRaises(ValueError):
            EpsgCode("not-a-number")


class TestParseCrs(unittest.TestCase):
    @parameterized.expand(
        [
            ("int", 2193),
            ("bare_numeric_str", "2193"),
            ("epsg_prefixed", "EPSG:2193"),
            ("epsg_prefixed_lowercase", "epsg:2193"),
            ("epsg_double_colon", "EPSG::2193"),
            ("epsg_spaced", "EPSG: 2193"),
            ("epsg_space_separated", "EPSG 2193"),
            ("urn", "urn:ogc:def:crs:EPSG::2193"),
            ("urn_versioned", "urn:ogc:def:crs:EPSG:6.6:2193"),
            ("urn_x_ogc", "urn:x-ogc:def:crs:EPSG::2193"),
            ("surrounding_whitespace", "  EPSG:2193  "),
        ]
    )
    def test_epsg_forms(self, _name: str, value: int | str) -> None:
        result = parse_crs(value)
        self.assertIsInstance(result, EpsgCode)
        self.assertEqual(2193, result)

    @parameterized.expand(
        [
            ("none", None),
            ("empty_str", ""),
            ("whitespace_only", "   "),
            ("unspecified", "unspecified"),
            ("unspecified_mixed_case", "Unspecified"),
            ("sentinel_int", UNSPECIFIED_EPSG_CODE),
            ("sentinel_str", "404000"),
            ("sentinel_epsg", "EPSG:404000"),
            ("sentinel_urn", "urn:ogc:def:crs:EPSG::404000"),
        ]
    )
    def test_unspecified_maps_to_none(self, _name: str, value: int | str | None) -> None:
        self.assertIsNone(parse_crs(value))

    def test_wkt_passthrough_unchanged(self) -> None:
        result = parse_crs(WKT)
        self.assertIsInstance(result, str)
        self.assertNotIsInstance(result, EpsgCode)
        self.assertEqual(WKT, result)

    def test_wkt_with_whitespace_is_not_stripped(self) -> None:
        value = f"  {WKT}  "
        self.assertEqual(value, parse_crs(value))

    @parameterized.expand([("unknown", "unknown"), ("none", "none")])
    def test_unrecognised_strings_are_preserved(self, _name: str, value: str) -> None:
        self.assertEqual(value, parse_crs(value))

    @parameterized.expand(
        [
            ("below_min_int", MIN_EPSG_CODE - 1),
            ("above_max_int", MAX_EPSG_CODE + 1),
            ("below_min_str", "EPSG:1023"),
            ("above_max_str", "32768"),
            ("above_max_urn", "urn:ogc:def:crs:EPSG::99999"),
            ("negative_bare_str", "-1"),
            ("negative_prefixed_str", "EPSG:-1"),
            ("negative_urn", "urn:ogc:def:crs:EPSG::-1"),
        ]
    )
    def test_out_of_range_raises(self, _name: str, value: int | str) -> None:
        with self.assertRaises(ValueError):
            parse_crs(value)

    @parameterized.expand(
        [
            ("float", 2193.0),
            ("bool_true", True),
            ("bool_false", False),
            ("list", [2193]),
            ("dict", {"epsg_code": 2193}),
            ("bytes", b"EPSG:2193"),
        ]
    )
    def test_bad_type_raises_type_error(self, _name: str, value: object) -> None:
        with self.assertRaises(TypeError):
            parse_crs(value)  # type: ignore[arg-type]

    def test_idempotent(self) -> None:
        for value in (2193, "EPSG:2193", WKT, None, "unspecified"):
            once = parse_crs(value)  # type: ignore[arg-type]
            self.assertEqual(once, parse_crs(once))


if __name__ == "__main__":
    unittest.main()

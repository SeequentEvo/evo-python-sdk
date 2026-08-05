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

import pandas as pd

from evo.objects.utils.downhole import expand_hole_index, hole_chunks_from_ids


class TestDownholeUtils(unittest.TestCase):
    def _chunks(self, values, categories):
        return hole_chunks_from_ids(pd.Series(values), dtype=pd.CategoricalDtype(categories=categories))

    def test_single_hole_uses_zero_based_code(self):
        chunks = self._chunks(["A"], ["A"])
        self.assertListEqual(chunks.to_dict("records"), [{"hole_index": 0, "offset": 0, "count": 1}])

    def test_multiple_rows_for_one_hole_are_one_chunk(self):
        chunks = self._chunks(["A", "A", "A"], ["A"])
        self.assertListEqual(chunks.to_dict("records"), [{"hole_index": 0, "offset": 0, "count": 3}])

    def test_contiguous_holes_have_consecutive_offsets(self):
        chunks = self._chunks(["A", "A", "B", "B"], ["A", "B"])
        self.assertListEqual(
            chunks.to_dict("records"),
            [{"hole_index": 0, "offset": 0, "count": 2}, {"hole_index": 1, "offset": 2, "count": 2}],
        )

    def test_varying_hole_counts_are_preserved(self):
        chunks = self._chunks(["A", "B", "B", "B", "C", "C"], ["A", "B", "C"])
        self.assertListEqual(
            chunks.to_dict("records"),
            [
                {"hole_index": 0, "offset": 0, "count": 1},
                {"hole_index": 1, "offset": 1, "count": 3},
                {"hole_index": 2, "offset": 4, "count": 2},
            ],
        )

    def test_category_absent_from_data_emits_zero_count_chunk(self):
        chunks = self._chunks(["A", "A"], ["A", "B"])
        self.assertListEqual(
            chunks.to_dict("records"),
            [{"hole_index": 0, "offset": 0, "count": 2}, {"hole_index": 1, "offset": 0, "count": 0}],
        )

    def test_id_absent_from_categories_raises(self):
        with self.assertRaises(ValueError):
            self._chunks(["A", "C"], ["A", "B"])

    def test_empty_input_emits_all_zero_count_chunks(self):
        chunks = self._chunks([], ["A", "B"])
        self.assertListEqual(
            chunks.to_dict("records"),
            [{"hole_index": 0, "offset": 0, "count": 0}, {"hole_index": 1, "offset": 0, "count": 0}],
        )

    def test_hole_index_dtype_is_int32(self):
        self.assertEqual(str(self._chunks(["A"], ["A"])["hole_index"].dtype), "int32")

    def test_offset_and_count_dtypes_are_uint64(self):
        chunks = self._chunks(["A"], ["A"])
        self.assertEqual(str(chunks["offset"].dtype), "uint64")
        self.assertEqual(str(chunks["count"].dtype), "uint64")

    def test_fifty_holes_with_two_hundred_rows_each_round_trip(self):
        categories = [f"H{index:02d}" for index in range(50)]
        values = [hole_id for hole_id in categories for _ in range(200)]
        chunks = self._chunks(values, categories)
        self.assertEqual(len(chunks), 50)
        self.assertListEqual(chunks["offset"].tolist(), list(range(0, 10_000, 200)))
        self.assertListEqual(chunks["count"].tolist(), [200] * 50)
        self.assertListEqual(
            expand_hole_index(chunks, len(values)).tolist(), [index for index in range(50) for _ in range(200)]
        )

    def test_category_order_is_preserved_not_lexicographically_sorted(self):
        chunks = self._chunks(["M", "M", "Z"], ["Z", "A", "M"])
        self.assertListEqual(
            chunks.to_dict("records"),
            [
                {"hole_index": 0, "offset": 2, "count": 1},
                {"hole_index": 1, "offset": 0, "count": 0},
                {"hole_index": 2, "offset": 0, "count": 2},
            ],
        )

    def test_chunks_use_dtype_codes_and_round_trip(self):
        dtype = pd.CategoricalDtype(categories=["Z", "A", "M"])
        values = pd.Series(["M", "M", "Z"])
        chunks = hole_chunks_from_ids(values, dtype=dtype)
        self.assertListEqual(chunks["hole_index"].tolist(), [0, 1, 2])
        self.assertListEqual(chunks["count"].tolist(), [1, 0, 2])
        self.assertListEqual(expand_hole_index(chunks, len(values)).tolist(), [2, 2, 0])

    def test_non_contiguous_and_unknown_ids_raise(self):
        dtype = pd.CategoricalDtype(categories=["A", "B"])
        with self.assertRaises(ValueError):
            hole_chunks_from_ids(pd.Series(["A", "B", "A"]), dtype=dtype)
        with self.assertRaises(ValueError):
            hole_chunks_from_ids(pd.Series(["C"]), dtype=dtype)

    def test_empty_input_emits_zero_count_entries_with_schema_dtypes(self):
        chunks = hole_chunks_from_ids(pd.Series([], dtype="string"), dtype=pd.CategoricalDtype(categories=["A", "B"]))
        self.assertListEqual(chunks["hole_index"].tolist(), [0, 1])
        self.assertListEqual(chunks["offset"].tolist(), [0, 0])
        self.assertListEqual(chunks["count"].tolist(), [0, 0])
        self.assertEqual(str(chunks["hole_index"].dtype), "int32")
        self.assertEqual(str(chunks["offset"].dtype), "uint64")
        self.assertEqual(str(chunks["count"].dtype), "uint64")

    def test_expand_rejects_out_of_bounds_chunks(self):
        with self.assertRaises(ValueError):
            expand_hole_index(pd.DataFrame({"hole_index": [0], "offset": [1], "count": [2]}), 2)

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

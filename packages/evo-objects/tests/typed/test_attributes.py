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

import pandas as pd
from parameterized import parameterized

from evo.objects.typed.attributes import UnSupportedDataTypeError, _infer_attribute_type_from_series


class TestAttributeTypeInference(TestCase):
    """Tests for attribute type inference from pandas Series."""

    @parameterized.expand(
        [
            ("integer_int64", pd.Series([1, 2, 3], dtype="int64"), "integer"),
            ("integer_int32", pd.Series([1, 2, 3], dtype="int32"), "integer"),
            ("scalar_float64", pd.Series([1.0, 2.0, 3.0], dtype="float64"), "scalar"),
            ("scalar_float32", pd.Series([1.0, 2.0, 3.0], dtype="float32"), "scalar"),
            ("bool", pd.Series([True, False, True], dtype="bool"), "bool"),
            ("string", pd.Series(["a", "b", "c"], dtype="string"), "string"),
            ("category", pd.Categorical(["a", "b", "a"]), "category"),
        ]
    )
    def test_infer_attribute_type(self, _name, series, expected_type):
        """Test that attribute types are correctly inferred from series dtype."""
        if isinstance(series, pd.Categorical):
            series = pd.Series(series)
        result = _infer_attribute_type_from_series(series)
        self.assertEqual(result, expected_type)

    def test_unsupported_dtype(self):
        """Test that unsupported dtypes raise an error."""
        # Complex numbers are not supported
        series = pd.Series([1 + 2j, 3 + 4j], dtype="complex128")
        with self.assertRaises(UnSupportedDataTypeError):
            _infer_attribute_type_from_series(series)

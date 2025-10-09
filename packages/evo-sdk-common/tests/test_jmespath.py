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

import json
import unittest
from typing import Any

import jmespath.parser
from parameterized import parameterized

from evo import jmespath as evo_jmespath


class TestJMESPath(unittest.TestCase):
    def test_parsed_result(self) -> None:
        """Test that our ParsedResult is distinct from jmespath's but is a subclass of it."""
        self.assertIsNot(
            evo_jmespath.ParsedResult,
            jmespath.parser.ParsedResult,
            "evo.jmespath.ParsedResult should not be jmespath.parser.ParsedResult",
        )
        self.assertTrue(
            issubclass(evo_jmespath.ParsedResult, jmespath.parser.ParsedResult),
            "evo.jmespath.ParsedResult should be a subclass of jmespath.parser.ParsedResult",
        )

    @parameterized.expand(
        [
            ("JMESPathArrayProxy", [1, 2, 3], evo_jmespath.JMESPathArrayProxy),
            ("JMESPathArrayProxy", (1, 2, 3), evo_jmespath.JMESPathArrayProxy),
            ("JMESPathObjectProxy", {"a": 1, "b": 2}, evo_jmespath.JMESPathObjectProxy),
            ("JMESPathObjectProxy", {"a": 1, "b": {"c": 3}}, evo_jmespath.JMESPathObjectProxy),
            ("JMESPathObjectProxy", {}, evo_jmespath.JMESPathObjectProxy),
            ("JMESPathArrayProxy", [], evo_jmespath.JMESPathArrayProxy),
            ("int", 123, int),
            ("str", "hello", str),
            ("float", 3.14, float),
            ("bool", True, bool),
            ("NoneType", None, type(None)),
        ]
    )
    def test_proxy_returns(self, _label: str, value: Any, expected_type: type) -> None:
        """Test that our proxy function returns the expected proxy type."""
        result = evo_jmespath.proxy(value)
        self.assertIsInstance(result, expected_type, f"Expected {expected_type} from proxy")

    def test_compile(self) -> None:
        """Test that our compile function returns our ParsedResult."""
        result = evo_jmespath.compile("foo.bar")
        self.assertIsInstance(result, evo_jmespath.ParsedResult, "Expected custom ParsedResult from compile")

    def test_search_returns_array_proxy(self) -> None:
        """Test that searching for an array returns our JMESPathArrayProxy."""
        data = {"foo": {"bar": [1, 2, 3]}}
        result = evo_jmespath.search("foo.bar", data)
        self.assertIsInstance(result, evo_jmespath.JMESPathArrayProxy, "Expected JMESPathArrayProxy from search")
        self.assertEqual(list(result), [1, 2, 3], "JMESPathArrayProxy did not contain expected data")

    def test_search_returns_object_proxy(self) -> None:
        """Test that searching for an object returns our JMESPathObjectProxy."""
        data = {"foo": {"bar": {"baz": 42}}}
        result = evo_jmespath.search("foo.bar", data)
        self.assertIsInstance(result, evo_jmespath.JMESPathObjectProxy, "Expected JMESPathObjectProxy from search")
        self.assertEqual(dict(result), {"baz": 42}, "JMESPathObjectProxy did not contain expected data")

    @parameterized.expand(
        [
            ("int", 123),
            ("str", "hello"),
            ("float", 3.14),
            ("bool", True),
            ("null", None),
            ("missing", None),
        ]
    )
    def test_search_returns_primitive(self, expression: str, expected_value: Any) -> None:
        """Test that searching for a primitive returns the primitive itself."""
        data = {"int": 123, "str": "hello", "float": 3.14, "bool": True, "null": None}
        result = evo_jmespath.search(expression, data)
        self.assertIsInstance(result, type(expected_value), f"Expected {type(expected_value)} from search")
        self.assertEqual(result, expected_value, "Primitive value did not match expected")


class TestJMESPathArrayProxy(unittest.TestCase):
    def setUp(self) -> None:
        self.data = [10, 20, 30]
        self.proxy = evo_jmespath.JMESPathArrayProxy(self.data)

    def test_len(self) -> None:
        """Test that len() works on JMESPathArrayProxy."""
        self.assertEqual(len(self.proxy), 3, "Length of JMESPathArrayProxy should be 3")

    def test_iter(self) -> None:
        """Test that iteration works on JMESPathArrayProxy."""
        self.assertEqual(list(iter(self.proxy)), self.data, "Iterating JMESPathArrayProxy did not yield expected data")

    def test_getitem_integer_indexing(self) -> None:
        """Test that integer indexing works on JMESPathArrayProxy."""
        self.assertEqual(self.proxy[1], 20, "JMESPathArrayProxy[1] should be 20")
        with self.assertRaises(IndexError):
            _ = self.proxy[3]

    @parameterized.expand(
        [
            ("array index syntax", "[0]", evo_jmespath.JMESPathObjectProxy({"name": "Alice", "age": 30})),
            ("array filter syntax", "[?age > `30`]", evo_jmespath.JMESPathArrayProxy([{"name": "Charlie", "age": 35}])),
            ("array projection syntax", "[].name", evo_jmespath.JMESPathArrayProxy(["Alice", "Bob", "Charlie"])),
        ]
    )
    def test_getitem_jmespath(self, _label: str, expression: str, expected_result: Any) -> None:
        """Test that JMESPath expressions work on JMESPathArrayProxy."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        proxy = evo_jmespath.JMESPathArrayProxy(data)
        result = proxy[expression]
        self.assertIsInstance(result, type(expected_result), f"Expected {type(expected_result)} from {expression}")
        self.assertEqual(result, expected_result, "JMESPath search did not yield expected result")

    @parameterized.expand(
        [
            ("array index syntax", "[0]", evo_jmespath.JMESPathObjectProxy({"name": "Alice", "age": 30})),
            ("array filter syntax", "[?age > `30`]", evo_jmespath.JMESPathArrayProxy([{"name": "Charlie", "age": 35}])),
            ("array projection syntax", "[].name", evo_jmespath.JMESPathArrayProxy(["Alice", "Bob", "Charlie"])),
        ]
    )
    def test_search(self, _label: str, expression: str, expected_result: Any) -> None:
        """Test the search method on JMESPathArrayProxy."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        proxy = evo_jmespath.JMESPathArrayProxy(data)
        result = proxy.search(expression)
        self.assertIsInstance(result, type(expected_result), f"Expected {type(expected_result)} from {expression}")
        self.assertEqual(result, expected_result, "JMESPath search did not yield expected result")

    def test_repr(self) -> None:
        """Test that repr() works on JMESPathArrayProxy."""
        self.assertEqual(
            repr(self.proxy),
            f"JMESPathArrayProxy({json.dumps([10, 20, 30], indent=2)})",
            "Unexpected repr for JMESPathArrayProxy",
        )

    def test_raw(self) -> None:
        """Test that the raw property returns the original data."""
        self.assertEqual(self.proxy.raw, self.data, "JMESPathArrayProxy.raw did not return the raw data")
        self.assertEqual(
            type(self.proxy.raw), type(self.data), "JMESPathArrayProxy.raw did not return the correct type"
        )


class TestJMESPathObjectProxy(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {"a": 1, "b": 2, "c": 3}
        self.proxy = evo_jmespath.JMESPathObjectProxy(self.data)

    def test_len(self) -> None:
        """Test that len() works on JMESPathObjectProxy."""
        self.assertEqual(len(self.proxy), 3, "Length of JMESPathObjectProxy should be 3")

    def test_iter(self) -> None:
        """Test that iteration works on JMESPathObjectProxy."""
        self.assertEqual(
            set(iter(self.proxy)), set(self.data.keys()), "Iterating JMESPathObjectProxy did not yield expected keys"
        )

    @parameterized.expand(
        [
            ("direct key access", "person1", evo_jmespath.JMESPathObjectProxy({"name": "Alice", "age": 30})),
            ("nested key access", "person2.name", "Bob"),
            ("missing key", "person4", None),
            ("array projection", "*.name", evo_jmespath.JMESPathArrayProxy(["Alice", "Bob", "Charlie"])),
            (
                "object projection",
                "{names: *.name}",
                evo_jmespath.JMESPathObjectProxy({"names": ["Alice", "Bob", "Charlie"]}),
            ),
        ]
    )
    def test_getitem_jmespath(self, _label: str, expression: str, expected_result: Any) -> None:
        """Test that JMESPath expressions work on JMESPathObjectProxy."""
        data = {
            "person1": {"name": "Alice", "age": 30},
            "person2": {"name": "Bob", "age": 25},
            "person3": {"name": "Charlie", "age": 35},
        }
        proxy = evo_jmespath.JMESPathObjectProxy(data)
        result = proxy[expression]
        self.assertIsInstance(result, type(expected_result), f"Expected {type(expected_result)} from {expression}")
        self.assertEqual(result, expected_result, "JMESPath search did not yield expected result")

    @parameterized.expand(
        [
            ("direct key access", "person1", evo_jmespath.JMESPathObjectProxy({"name": "Alice", "age": 30})),
            ("nested key access", "person2.name", "Bob"),
            ("missing key", "person4", None),
            ("array projection", "*.name", evo_jmespath.JMESPathArrayProxy(["Alice", "Bob", "Charlie"])),
            (
                "object projection",
                "{names: *.name}",
                evo_jmespath.JMESPathObjectProxy({"names": ["Alice", "Bob", "Charlie"]}),
            ),
        ]
    )
    def test_search(self, _label: str, expression: str, expected_result: Any) -> None:
        """Test the search method on JMESPathObjectProxy."""
        data = {
            "person1": {"name": "Alice", "age": 30},
            "person2": {"name": "Bob", "age": 25},
            "person3": {"name": "Charlie", "age": 35},
        }
        proxy = evo_jmespath.JMESPathObjectProxy(data)
        result = proxy.search(expression)
        self.assertIsInstance(result, type(expected_result), f"Expected {type(expected_result)} from {expression}")
        self.assertEqual(result, expected_result, "JMESPath search did not yield expected result")

    def test_repr(self) -> None:
        """Test that repr() works on JMESPathObjectProxy."""
        self.assertEqual(
            repr(self.proxy),
            f"JMESPathObjectProxy({json.dumps(self.data, indent=2)})",
            "Unexpected repr for JMESPathObjectProxy",
        )

    def test_raw(self) -> None:
        """Test that the raw property returns the original data."""
        self.assertEqual(self.proxy.raw, self.data, "JMESPathObjectProxy.raw did not return the raw data")
        self.assertEqual(
            type(self.proxy.raw), type(self.data), "JMESPathObjectProxy.raw did not return the correct type"
        )

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

import copy
import dataclasses
import unittest
from unittest import mock
from uuid import UUID

from parameterized import parameterized

from evo.common.data import EmptyResponse, Environment, HTTPHeaderDict, HTTPResponse, Page, ResourceMetadata
from evo.common.test_tools import BASE_URL, utc_datetime

SAMPLE_HEADER_DICT = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 123",
    "Cookie": "session=123",
    "Set-Cookie": "session=123",
}
ALTERNATE_HEADER_VALUES = {
    "Content-Type": "application/xml",
    "Authorization": "Bearer 456",
    "Cookie": "session=456",
    "Set-Cookie": "session=456",
}


# Uncomment the following import statement to compare header values that are usually hidden.
# from evo.common.test_tools import TestHTTPHeaderDict as HTTPHeaderDict


class TestHTTPHeaderDict(unittest.TestCase):
    def test_init_empty(self) -> None:
        """Test that the HTTPHeaderDict is empty when initialized with no arguments."""
        headers = HTTPHeaderDict()
        self.assertEqual(len(headers), 0)

    def assert_matches_sample_dict(self, headers: HTTPHeaderDict) -> None:
        """Assert that the given headers match the sample dict."""
        self.assertEqual(len(SAMPLE_HEADER_DICT), len(headers))
        for key, value in SAMPLE_HEADER_DICT.items():
            self.assertEqual(value, headers[key])

    def test_init_mapping(self) -> None:
        """Test that the HTTPHeaderDict is initialized correctly with a mapping."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assert_matches_sample_dict(headers)

    def test_init_sequence(self) -> None:
        """Test that the HTTPHeaderDict is initialized correctly with a sequence."""
        headers = HTTPHeaderDict(list(SAMPLE_HEADER_DICT.items()))
        self.assert_matches_sample_dict(headers)

    def test_init_kwargs(self) -> None:
        """Test that the HTTPHeaderDict is initialized correctly with keyword arguments."""
        headers = HTTPHeaderDict(**SAMPLE_HEADER_DICT)
        self.assert_matches_sample_dict(headers)

    def test_update_mapping(self) -> None:
        """Test that updating the headers with a mapping works correctly."""
        headers = HTTPHeaderDict()
        headers.update(SAMPLE_HEADER_DICT)
        self.assert_matches_sample_dict(headers)

    def test_update_sequence(self) -> None:
        """Test that updating the headers with a sequence works correctly."""
        headers = HTTPHeaderDict()
        headers.update(list(SAMPLE_HEADER_DICT.items()))
        self.assert_matches_sample_dict(headers)

    def test_update_kwargs(self) -> None:
        """Test that updating the headers with keyword arguments works correctly."""
        headers = HTTPHeaderDict()
        headers.update(**SAMPLE_HEADER_DICT)
        self.assert_matches_sample_dict(headers)

    def test_update_appends(self) -> None:
        """Test that updating an existing field appends the new value."""
        # RFC 7230 section 3.2.2: Field Order.
        # A recipient MAY combine multiple header fields with the same field
        # name into one "field-name: field-value" pair, without changing the
        # semantics of the message, by appending each subsequent field value to
        # the combined field value in order, separated by a comma.  The order
        # in which header fields with the same field name are received is
        # therefore significant to the interpretation of the combined field
        # value;
        # https://www.rfc-editor.org/rfc/rfc7230#section-3.2.2
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        headers.update({"Authorization": "Bearer 456"})
        self.assertEqual("Bearer 123,Bearer 456", headers["Authorization"])

    def test_update_set_cookie_overwrites(self) -> None:
        """Test that updating a "Set-Cookie" field overwrites the existing value."""
        # In practice, the "Set-Cookie" header field ([RFC6265]) often
        # appears multiple times in a response message and does not use the
        # list syntax, violating the above requirements on multiple header
        # fields with the same name.  Since it cannot be combined into a
        # single field-value, recipients ought to handle "Set-Cookie" as a
        # special case while processing header fields.
        # https://www.rfc-editor.org/rfc/rfc7230#section-3.2.2
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        headers.update({"Set-Cookie": "session=456"})
        self.assertEqual("session=456", headers["Set-Cookie"])

    @parameterized.expand(
        [(field, value, ALTERNATE_HEADER_VALUES[field]) for field, value in SAMPLE_HEADER_DICT.items()]
    )
    def test_setitem(self, field: str, value: str, alternate: str) -> None:
        """Test that setting an item works correctly."""
        headers = HTTPHeaderDict()
        headers[field] = value
        self.assertEqual(value, headers[field])

        with self.subTest("Updating an existing field"):
            headers[field] = alternate
            if field == "Set-Cookie":
                self.assertEqual(alternate, headers[field])
            else:
                self.assertEqual(f"{value},{alternate}", headers[field])

        headers = HTTPHeaderDict()  # Reset the headers.
        with self.subTest("Field names are case-insensitive"):
            headers[field.lower()] = value
            self.assertEqual(value, headers[field.upper()])

    @parameterized.expand(SAMPLE_HEADER_DICT.items())
    def test_getitem(self, field: str, value: str) -> None:
        """Test that getting an item works correctly."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertEqual(value, headers[field])

        with self.subTest("Field names are case-insensitive"):
            for key in (field.lower(), field.upper(), field.title()):
                self.assertEqual(value, headers[key])

    @parameterized.expand(SAMPLE_HEADER_DICT.items())
    def test_get(self, field: str, value: str) -> None:
        """Test that getting an item works correctly."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertEqual(value, headers.get(field))

        with self.subTest("Field names are case-insensitive"):
            for key in (field.lower(), field.upper(), field.title()):
                self.assertEqual(value, headers.get(key))

    def test_get_default(self) -> None:
        """Test that the default value is returned when a header is not found."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertNotIn("Non-Existent-Header", headers)
        self.assertIsNone(headers.get("Non-Existent-Header"))
        self.assertEqual("default", headers.get("Non-Existent-Header", "default"))

    def test_get_no_default(self) -> None:
        """Test that `None` is returned when a header is not found and no default is provided."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertNotIn("Non-Existent-Header", headers)
        self.assertIsNone(headers.get("Non-Existent-Header"))

    @parameterized.expand(SAMPLE_HEADER_DICT.items())
    def test_delitem(self, field: str, value: str) -> None:
        """Test that deleting an item works correctly."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertIn(field, headers)
        self.assertEqual(len(headers), len(SAMPLE_HEADER_DICT))

        del headers[field]
        self.assertNotIn(field, headers)
        self.assertEqual(len(headers), len(SAMPLE_HEADER_DICT) - 1)

        with self.subTest("Field names are case-insensitive"):
            for key in (field.lower(), field.upper(), field.title()):
                headers[key] = value
                self.assertIn(field, headers)
                del headers[key]
                self.assertNotIn(field, headers)

    def test_contains(self) -> None:
        """Test that the headers can be checked for containment."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        for key in SAMPLE_HEADER_DICT:
            self.assertIn(key, headers)

    def test_keys(self) -> None:
        """Test that the keys method returns the keys in lowercase."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertListEqual(list(SAMPLE_HEADER_DICT.keys()), list(headers.keys()))

    def test_values(self) -> None:
        """Test that the values method returns the correct values."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertListEqual(list(SAMPLE_HEADER_DICT.values()), list(headers.values()))

    def test_items(self) -> None:
        """Test that the items method returns the correct items."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertListEqual(list(SAMPLE_HEADER_DICT.items()), list(headers.items()))

    def test_iter(self) -> None:
        """Test that the header fieldnames can be iterated over."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertListEqual(list(SAMPLE_HEADER_DICT), list(headers))

    def test_clear(self) -> None:
        """Test that the headers can be cleared."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        self.assertEqual(len(headers), len(SAMPLE_HEADER_DICT))
        headers.clear()
        self.assertEqual(len(headers), 0)

    @parameterized.expand(SAMPLE_HEADER_DICT)
    def test_copy(self, field: str) -> None:
        """Test that the headers can be copied."""
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        copy = headers.copy()
        self.assertEqual(headers, copy)

        with self.subTest("Updating the copy does not affect the original headers"):
            copy[field] = "bleh"
            self.assertNotEqual(headers, copy)

        with self.subTest("Updating the original headers does not affect the copy"):
            copy = headers.copy()
            self.assertEqual(headers, copy)
            headers[field] = "bleh"
            self.assertNotEqual(headers, copy)

    @parameterized.expand(SAMPLE_HEADER_DICT.items())
    def test_repr(self, field: str, expected_value: str) -> None:
        """Test that the repr method returns the expected string.

        Sensitive information should be hidden.
        """
        headers = HTTPHeaderDict(SAMPLE_HEADER_DICT)
        repr_headers = repr(headers)
        self.assertIsInstance(repr_headers, str)

        self.assertEqual(expected_value, headers[field])
        eval_headers = eval(repr_headers, globals(), {"TestHTTPHeaderDict": HTTPHeaderDict})

        if field in ("Authorization", "Proxy-Authorization", "Cookie", "Set-Cookie"):
            expected_value = "*****"
        self.assertEqual(expected_value, eval_headers[field], "Sensitive information should be hidden.")
        self.assertIn(f"{field.title()!r}: {expected_value!r}", repr_headers)


SAMPLE_RESPONSE_DICT = {
    "status": 200,
    "data": b"Hello, World!",
    "reason": "OK",
    "headers": SAMPLE_HEADER_DICT.copy(),
}


class TestEmptyResponse(unittest.TestCase):
    def setUp(self) -> None:
        self.response = EmptyResponse(
            status=SAMPLE_RESPONSE_DICT["status"],
            reason=SAMPLE_RESPONSE_DICT["reason"],
            headers=HTTPHeaderDict(SAMPLE_RESPONSE_DICT["headers"]),
        )

    def test_init_default_reason(self) -> None:
        """Test that the response can be initialized without a reason."""
        response = EmptyResponse(
            status=SAMPLE_RESPONSE_DICT["status"],
            headers=HTTPHeaderDict(SAMPLE_RESPONSE_DICT["headers"]),
        )
        self.assertIsNone(response.reason)

    def test_init_default_headers(self) -> None:
        """Test that the response can be initialized without headers."""
        response = EmptyResponse(
            status=SAMPLE_RESPONSE_DICT["status"],
            reason=SAMPLE_RESPONSE_DICT["reason"],
        )
        self.assertEqual(HTTPHeaderDict(), response.headers)

    def test_getheaders(self) -> None:
        """Test that getheaders returns a copy of the headers."""
        headers = self.response.getheaders()
        self.assertIsInstance(headers, HTTPHeaderDict)
        self.assertEqual(self.response.headers, headers)
        self.assertEqual(HTTPHeaderDict(SAMPLE_RESPONSE_DICT["headers"]), headers)

        with self.subTest("Updating the headers does not affect the original response"):
            headers["Content-Type"] = "application/xml"
            self.assertNotEqual(headers, self.response.headers)

    def test_getheader(self) -> None:
        """Test that getheader returns the correct header value."""
        for field, value in SAMPLE_RESPONSE_DICT["headers"].items():
            self.assertEqual(value, self.response.getheader(field))

            with self.subTest("Field names are case-insensitive"):
                for key in (field.lower(), field.upper(), field.title()):
                    self.assertEqual(value, self.response.getheader(key))

    def test_getheader_default(self) -> None:
        """Test that getheader returns the default value when the header is not found."""
        self.assertNotIn("Non-Existent-Header", self.response.headers)
        self.assertIsNone(self.response.getheader("Non-Existent-Header"))
        self.assertEqual("default", self.response.getheader("Non-Existent-Header", "default"))

    def test_getheader_no_default(self) -> None:
        """Test that getheader returns `None` when the header is not found and no default is provided."""
        self.assertNotIn("Non-Existent-Header", self.response.headers)
        self.assertIsNone(self.response.getheader("Non-Existent-Header"))


class TestHTTPResponse(TestEmptyResponse):
    def setUp(self) -> None:
        self.response = HTTPResponse(
            status=SAMPLE_RESPONSE_DICT["status"],
            reason=SAMPLE_RESPONSE_DICT["reason"],
            headers=HTTPHeaderDict(SAMPLE_RESPONSE_DICT["headers"]),
            data=SAMPLE_RESPONSE_DICT["data"],
        )

    def test_init_default_reason(self) -> None:
        """Test that the response can be initialized without a reason."""
        response = HTTPResponse(
            status=SAMPLE_RESPONSE_DICT["status"],
            headers=HTTPHeaderDict(SAMPLE_RESPONSE_DICT["headers"]),
            data=SAMPLE_RESPONSE_DICT["data"],
        )
        self.assertIsNone(response.reason)

    def test_init_default_headers(self) -> None:
        """Test that the response can be initialized without headers."""
        response = HTTPResponse(
            status=SAMPLE_RESPONSE_DICT["status"],
            reason=SAMPLE_RESPONSE_DICT["reason"],
            data=SAMPLE_RESPONSE_DICT["data"],
        )
        self.assertEqual(HTTPHeaderDict(), response.headers)


OFFSET = 0
LIMIT = 10
TOTAL = 200

# Test with one item less than the limit, because the limit is not guaranteed to be the page size.
COUNT = LIMIT - 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class SomeItem(ResourceMetadata):
    extras: list = dataclasses.field(default_factory=list)

    @property
    def url(self) -> str:
        return f"{BASE_URL}/items/{self.id}"

    def __deepcopy__(self, memo: object) -> SomeItem:
        return dataclasses.replace(self, extras=copy.deepcopy(self.extras, memo))


def get_items(count: int, offset: int = 0) -> list[SomeItem]:
    mock_env = mock.Mock(spec=Environment)
    return [
        SomeItem(environment=mock_env, id=UUID(int=i), name=f"item-{i + offset}", created_at=utc_datetime(2024))
        for i in range(count)
    ]


class TestPage(unittest.TestCase):
    def setUp(self) -> None:
        self.items = get_items(COUNT, OFFSET)
        self.page = Page(offset=OFFSET, limit=LIMIT, total=TOTAL, items=self.items)

    def test_offset(self) -> None:
        self.assertEqual(OFFSET, self.page.offset)
        with self.assertRaises(AttributeError):
            self.page.offset = 1

    def test_limit(self) -> None:
        self.assertEqual(LIMIT, self.page.limit)
        with self.assertRaises(AttributeError):
            self.page.limit = 1

    def test_total(self) -> None:
        self.assertEqual(TOTAL, self.page.total)
        with self.assertRaises(AttributeError):
            self.page.total = 1

    def test_items(self) -> None:
        self.assertEqual(self.items, self.page.items())

    def assert_deepcopied(self, left: list[SomeItem], right: list[SomeItem]) -> None:
        self.assertEqual(left, right)
        for left_item, right_item in zip(left, right):
            self.assertIsNot(left_item, right_item)
            self.assertEqual(left_item.extras, right_item.extras)
            right_item.extras.append("extra")
            self.assertNotEqual(left_item.extras, right_item.extras)

    def test_items_copies_items(self) -> None:
        self.assert_deepcopied(self.items, self.page.items())

    def test_getitem(self) -> None:
        for i in range(COUNT):
            self.assertEqual(self.items[i], self.page[i])

    def test_getitem_copies_items(self) -> None:
        left = [self.items[0]]
        right = [self.page[0]]
        self.assert_deepcopied(left, right)

    def test_getitem_slice(self) -> None:
        left = self.items[1:5]
        right = self.page[1:5]
        self.assertEqual(left, right)

    def test_getitem_slice_copies_items(self) -> None:
        left = self.items[1:5]
        right = self.page[1:5]
        self.assert_deepcopied(left, right)

    def test_len(self) -> None:
        self.assertEqual(COUNT, len(self.page))

    def test_size(self) -> None:
        self.assertEqual(COUNT, self.page.size)

    @parameterized.expand(
        [
            ("first page", 0),
            ("some page", 42),  # The meaning of life, the universe, and everything.
            ("last page", TOTAL - COUNT),
        ]
    )
    def test_next_offset(self, _label: str, offset: int) -> None:
        page = Page(offset=offset, limit=LIMIT, total=TOTAL, items=get_items(COUNT, offset))
        expected_next_offset = offset + COUNT
        self.assertEqual(expected_next_offset, page.next_offset)

    @parameterized.expand(
        [
            ("first page", 0, False),
            ("some page", 42, False),
            ("last page", TOTAL - COUNT, True),
        ]
    )
    def test_is_last(self, _label: str, offset: int, expect_last: bool) -> None:
        page = Page(offset=offset, limit=LIMIT, total=TOTAL, items=get_items(COUNT, offset))
        self.assertEqual(expect_last, page.is_last)

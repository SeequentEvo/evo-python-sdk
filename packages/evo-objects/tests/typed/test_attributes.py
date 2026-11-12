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

import asyncio
import threading
from copy import deepcopy
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
from parameterized import parameterized

from evo.common import IFeedback
from evo.objects.typed._model import ModelContext
from evo.objects.typed.attributes import (
    Attribute,
    AttributeDescription,
    Attributes,
    PendingAttribute,
    UnSupportedDataTypeError,
    _infer_attribute_type_from_series,
)
from evo.objects.utils._concurrency import get_upload_semaphore, upload_concurrency


class TestConcurrentAttributeUploads(IsolatedAsyncioTestCase):
    def setUp(self):
        self.documents = [
            {
                "name": "first",
                "key": "stable-key",
                "attribute_type": "scalar",
                "attribute_description": {"discipline": "geology", "type": "length", "unit": "m"},
                "values": {"data": "old-first", "length": 1},
            },
            {"name": "removed", "attribute_type": "integer", "values": {"data": "old-removed", "length": 1}},
        ]
        self.context = ModelContext(obj=Mock(), root_model=None, data_modified={"previous-change"})
        self.attributes = Attributes(self.context, self.documents)

    async def test_attribute_order_and_descriptions_survive_out_of_order_completion(self):
        second_uploaded = asyncio.Event()
        frame = pd.DataFrame({"first": [1.0], "second": [2.0]})
        frame.attrs["attribute_descriptions"] = {"first": AttributeDescription(unit="m")}

        async def upload(document, *_args, **_kwargs):
            if document["name"] == "first":
                await second_uploaded.wait()
            else:
                second_uploaded.set()
            document["values"] = {"data": document["name"], "length": 1}

        with (
            patch("evo.objects.typed.attributes.get_data_client"),
            patch.object(Attribute, "_upload_attribute_values", side_effect=upload),
        ):
            result = await asyncio.wait_for(Attributes._data_to_schema(frame, Mock()), timeout=2)

        self.assertEqual(["first", "second"], [document["name"] for document in result])
        self.assertEqual("m", result[0]["attribute_description"]["unit"])
        self.assertNotIn("attribute_description", result[1])

    async def test_failure_cancels_and_waits_for_other_columns(self):
        started = asyncio.Event()
        release = asyncio.Event()
        cleaned_up = asyncio.Event()

        async def upload(document, *_args, **_kwargs):
            if document["name"] == "slow":
                started.set()
                try:
                    await release.wait()
                finally:
                    await asyncio.sleep(0)
                    cleaned_up.set()
            else:
                await started.wait()
                raise RuntimeError("attribute upload failed")

        with (
            patch("evo.objects.typed.attributes.get_data_client"),
            patch.object(Attribute, "_upload_attribute_values", side_effect=upload),
        ):
            try:
                with self.assertRaisesRegex(RuntimeError, "attribute upload failed"):
                    await Attributes._data_to_schema(pd.DataFrame({"slow": [1.0], "failed": [2.0]}), Mock())
                self.assertTrue(cleaned_up.is_set(), "A failed upload left another column running")
            finally:
                release.set()
                await asyncio.wait_for(cleaned_up.wait(), timeout=2)

    @parameterized.expand([("append_attribute",), ("append_attributes",), ("set_attributes",)])
    async def test_mutations_forward_aggregated_feedback(self, method):
        frame = pd.DataFrame({"first": [2.0]})
        if method != "append_attribute":
            frame["second"] = pd.Categorical(["sand"])
        feedback = Mock(spec=IFeedback)
        client = Mock()

        async def upload(df, *, fb, **_kwargs):
            fb.progress(0.5)
            await asyncio.sleep(0)
            fb.progress(1)
            return {"data": f"new-{df.columns[0]}", "length": 1}

        async def upload_category(df, *, fb):
            values = await upload(df, fb=fb)
            return {"values": values, "table": {"data": "lookup"}}

        client.upload_dataframe = AsyncMock(side_effect=upload)
        client.upload_category_dataframe = AsyncMock(side_effect=upload_category)
        with patch("evo.objects.typed.attributes.get_data_client", return_value=client):
            await getattr(self.attributes, method)(frame, fb=feedback)

        progress = [call.args[0] for call in feedback.progress.call_args_list]
        self.assertEqual(1, progress[-1])
        self.assertEqual(sorted(progress), progress)
        self.assertTrue(all(0 <= value <= 1 for value in progress))
        self.assertIn("new-first", self.context.data_modified)
        if method != "append_attribute":
            self.assertIn("new-second", self.context.data_modified)
            self.assertEqual(0.25, progress[0])

    @parameterized.expand([("create",), ("append_attributes",), ("set_attributes",)])
    async def test_invalid_description_fails_before_any_upload(self, method):
        frame = pd.DataFrame({"first": [1.0], "invalid": [2.0]})
        frame.attrs["attribute_descriptions"] = {"invalid": "not an AttributeDescription"}
        original = deepcopy(self.documents)
        with (
            patch("evo.objects.typed.attributes.get_data_client") as get_client,
            patch.object(Attribute, "_upload_attribute_values") as upload,
        ):
            with self.assertRaisesRegex(TypeError, "attribute description"):
                if method == "create":
                    await Attributes._data_to_schema(frame, Mock())
                else:
                    await getattr(self.attributes, method)(frame)
        get_client.assert_not_called()
        upload.assert_not_awaited()
        self.assertEqual(original, self.documents)
        self.assertEqual({"previous-change"}, self.context.data_modified)

    @parameterized.expand([("append_attributes",), ("set_attributes",)])
    async def test_mutations_overlap_and_preserve_column_order(self, method):
        first_finished = asyncio.Event()
        second_started = asyncio.Event()
        existing_attribute = self.attributes["first"]
        frame = pd.DataFrame({"second": [3.0], "first": [2.0]})
        frame.attrs["attribute_descriptions"] = {"second": AttributeDescription(unit="s")}

        async def upload(document, *_args, **_kwargs):
            if document["name"] == "second":
                second_started.set()
                await first_finished.wait()
            else:
                await second_started.wait()
                first_finished.set()
            document["values"] = {"data": f"new-{document['name']}", "length": 1}

        with (
            patch("evo.objects.typed.attributes.get_data_client"),
            patch.object(Attribute, "_upload_attribute_values", side_effect=upload),
        ):
            await asyncio.wait_for(getattr(self.attributes, method)(frame), timeout=2)

        self.assertEqual(["second", "first"], [document["name"] for document in self.documents[-2:]])
        self.assertEqual("s", self.documents[-2]["attribute_description"]["unit"])
        self.assertTrue({"new-first", "new-second"} <= self.context.data_modified)
        if method == "set_attributes":
            self.assertEqual(2, len(self.documents))
            self.assertEqual("stable-key", self.documents[1]["key"])
            self.assertEqual("m", self.documents[1]["attribute_description"]["unit"])
            self.assertEqual("scalar", self.documents[1]["attribute_type"])
            self.assertEqual("new-first", existing_attribute._data)

    @parameterized.expand(
        [
            ("append_failure", "append_attributes", False),
            ("set_failure", "set_attributes", False),
            ("append_cancel", "append_attributes", True),
            ("set_cancel", "set_attributes", True),
        ]
    )
    async def test_failed_or_cancelled_mutations_preserve_live_state(self, _name, method, cancel):
        original = deepcopy(self.documents)
        existing_attribute = self.attributes["first"]
        first_finished = asyncio.Event()
        second_started = asyncio.Event()
        release = asyncio.Event()
        frame = pd.DataFrame({"first": [2.0], "second": [3.0]})

        async def upload(document, *_args, **_kwargs):
            document["values"] = {"data": f"new-{document['name']}", "length": 1}
            if document["name"] == "first":
                first_finished.set()
            else:
                await first_finished.wait()
                second_started.set()
                if cancel:
                    await release.wait()
                else:
                    raise RuntimeError("upload failed")

        with (
            patch("evo.objects.typed.attributes.get_data_client"),
            patch.object(Attribute, "_upload_attribute_values", side_effect=upload),
        ):
            task = asyncio.create_task(getattr(self.attributes, method)(frame))
            try:
                await asyncio.wait_for(second_started.wait(), timeout=2)
                if cancel:
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError if cancel else RuntimeError):
                    await task
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertEqual(original, self.documents)
        self.assertEqual("old-first", existing_attribute._data)
        self.assertEqual({"previous-change"}, self.context.data_modified)

    async def test_empty_mutations_do_not_upload(self):
        with patch("evo.objects.typed.attributes.get_data_client") as get_client:
            await self.attributes.append_attributes(pd.DataFrame())
            self.assertEqual(2, len(self.documents))
            await self.attributes.set_attributes(pd.DataFrame())
            self.assertEqual([], self.documents)
        get_client.assert_not_called()
        self.assertEqual({"previous-change"}, self.context.data_modified)

    async def test_unsupported_type_is_rejected_before_upload(self):
        frame = pd.DataFrame({"first": [1.0], "unsupported": [1 + 2j]})
        with patch.object(Attribute, "_upload_attribute_values") as upload:
            with self.assertRaises(UnSupportedDataTypeError):
                await self.attributes.set_attributes(frame)
        upload.assert_not_awaited()
        self.assertEqual("old-first", self.documents[0]["values"]["data"])

    async def test_type_inference_runs_in_worker_with_upload_slot_held(self):
        started = threading.Event()
        release = threading.Event()
        worker_threads = []
        infer = _infer_attribute_type_from_series

        def blocking_infer(series):
            worker_threads.append(threading.get_ident())
            started.set()
            release.wait()
            return infer(series)

        with (
            upload_concurrency(1),
            patch("evo.objects.typed.attributes._infer_attribute_type_from_series", side_effect=blocking_infer),
            patch("evo.objects.typed.attributes.get_data_client"),
            patch.object(Attribute, "_upload_attribute_values", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(Attributes._data_to_schema(pd.DataFrame({"new": [2.0]}), Mock()))
            try:
                await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)
                self.assertEqual(0, get_upload_semaphore()._value)
                self.assertNotEqual(threading.get_ident(), worker_threads[0])
            finally:
                release.set()
                await task

    async def test_set_attributes_rejects_concurrent_document_edits(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def upload(*_args, **_kwargs):
            started.set()
            await release.wait()

        with patch.object(Attribute, "_upload_attribute_values", side_effect=upload):
            task = asyncio.create_task(self.attributes.set_attributes(pd.DataFrame({"first": [2.0]})))
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                self.documents[0]["external_edit"] = True
                release.set()
                with self.assertRaisesRegex(RuntimeError, "changed while replacement"):
                    await task
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertEqual("old-first", self.documents[0]["values"]["data"])
        self.assertTrue(self.documents[0]["external_edit"])

    @parameterized.expand([("string_label", "0"), ("integer_label", 0)])
    async def test_replacement_matches_stringified_names_and_preserves_existing_type(self, _name, label):
        self.documents[:] = [
            {
                "name": "0",
                "key": "stable-key",
                "attribute_type": "category",
                "table": {"data": "old-lookup"},
                "values": {"data": "old-values", "length": 1},
            }
        ]
        original = self.documents[0]
        attribute = self.attributes["0"]
        client = Mock()
        client.upload_category_dataframe = AsyncMock(
            return_value={
                "table": {"data": "new-lookup"},
                "values": {"data": "new-values", "length": 1},
            }
        )
        client.upload_dataframe = AsyncMock()
        with patch("evo.objects.typed.attributes.get_data_client", return_value=client):
            # A float column would infer scalar for a new attribute, but replacements keep their existing type.
            await self.attributes.set_attributes(pd.DataFrame({label: [2.0]}))

        self.assertIs(original, self.documents[0])
        self.assertEqual("category", attribute.attribute_type)
        self.assertEqual("stable-key", attribute.key)
        self.assertEqual("new-values", attribute._data)
        self.assertEqual({"data": "new-lookup"}, original["table"])
        client.upload_category_dataframe.assert_awaited_once()
        client.upload_dataframe.assert_not_awaited()
        self.assertEqual({"previous-change", "new-values"}, self.context.data_modified)


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


class TestPendingAttribute(TestCase):
    """Tests for PendingAttribute class."""

    def test_pending_attribute_name(self):
        """Test that PendingAttribute stores the name correctly."""
        pending = PendingAttribute(None, "test_attr")
        self.assertEqual(pending.name, "test_attr")

    def test_pending_attribute_exists_is_false(self):
        """Test that PendingAttribute.exists returns False."""
        pending = PendingAttribute(None, "test_attr")
        self.assertFalse(pending.exists)

    def test_pending_attribute_repr(self):
        """Test that PendingAttribute has a useful repr."""
        pending = PendingAttribute(None, "test_attr")
        self.assertEqual(repr(pending), "PendingAttribute(name='test_attr', exists=False)")


class TestAttributeDescription(TestCase):
    def test_default_description_preserves_required_fields(self):
        self.assertEqual(AttributeDescription().to_schema(), {"discipline": "", "type": ""})

    def test_unitless_description_omits_unit(self):
        description = AttributeDescription(discipline="Geology", type="Azimuth")
        self.assertEqual(description.to_schema(), {"discipline": "Geology", "type": "Azimuth"})

    def test_description_preserves_string_units(self):
        description = AttributeDescription(discipline="geology", type="length", unit="m")
        self.assertEqual(description.to_schema(), {"discipline": "geology", "type": "length", "unit": "m"})

    def test_description_normalizes_value_units(self):
        class Unit:
            value = "m"

        description = AttributeDescription(discipline="geology", type="length", unit=Unit())
        self.assertEqual(description.to_schema(), {"discipline": "geology", "type": "length", "unit": "m"})

    def test_description_rejects_invalid_units(self):
        with self.assertRaisesRegex(TypeError, "unit must be"):
            AttributeDescription(unit=cast(Any, object()))

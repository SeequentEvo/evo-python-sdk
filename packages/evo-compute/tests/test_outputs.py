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

"""Hydration of a task's results against the ``results`` schema it publishes.

The payload stays exactly what the platform sent -- these are ``dict`` subclasses, so a
caller who was indexing the raw JSON keeps working. What is added is attribute access and,
for the nodes the schema marks ``output: geoscience-object`` or ``output: attribute``, a
loader that turns the reference back into a typed object or attribute.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock

from evo.common.test_tools import TestWithConnector
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject

from data import load_test_data
from evo import jmespath
from evo.compute import ComputeClient, ResultNode, TaskResult
from test_resolution import (
    BLOCK_MODEL,
    BLOCK_MODEL_URL,
    POINTSET,
    POINTSET_URL,
    SCHEMAS_BY_URL,
    TARGET_URL,
    FakeContext,
)

RESULTS = {
    "message": "estimated 1000 blocks",
    "target": {
        "reference": TARGET_URL,
        "name": "estimated grid",
        "description": None,
        "schema_id": "/objects/pointset/1.2.0/pointset.schema.json",
        "attribute": {"reference": "locations.attributes[?name=='estimate']", "name": "estimate"},
    },
    "extras": [{"reference": POINTSET_URL, "name": "extra"}],
    "report": None,
}

# The two families the fixture's ``attribute_path`` knows about, each keeping its attributes
# somewhere else -- which is what makes an expression written for one wrong for the other.
POINTSET_DOCUMENT = {"locations": {"attributes": [{"name": "estimate", "key": "abc"}]}}
BLOCK_MODEL_DOCUMENT = {"attributes": [{"name": "estimate", "key": "abc"}]}


def loaded_object(document: dict, schema: ObjectSchema = POINTSET) -> mock.MagicMock:
    """Stands in for the typed object a result reference downloads to."""
    obj = mock.MagicMock(spec=BaseObject)
    obj.metadata.schema_id = schema
    obj.to_dataframe = mock.AsyncMock(return_value="dataframe")
    obj.search = lambda expression: jmespath.search(expression, document)
    obj.attributes = _FakeAttributes(jmespath.search("attributes || locations.attributes", document))
    return obj


class _FakeAttributes:
    """The name-or-key lookup every typed attribute collection offers."""

    def __init__(self, attributes: Sequence[Mapping[str, Any]]) -> None:
        self._attributes = attributes

    def __getitem__(self, name: str) -> mock.Mock:
        for attribute in self._attributes:
            if name in (attribute["name"], attribute["key"]):
                found = mock.Mock(exists=True, to_dataframe=mock.AsyncMock(return_value="attribute dataframe"))
                found.name = attribute["name"]
                return found
        return mock.Mock(exists=False)


class HydrationTestCase(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        catalogue = load_test_data("discovery-resolution.json")
        self.schema = catalogue["results"][0]["results"]
        self.context = FakeContext()
        self.result = TaskResult(RESULTS, self.schema, self.context)

    @contextmanager
    def loaded_object(self) -> Iterator[mock.AsyncMock]:
        """Stub the typed-object download the object loaders delegate to."""
        obj = mock.Mock()
        obj.to_dataframe = mock.AsyncMock(return_value="dataframe")
        loader = mock.AsyncMock(return_value=obj)
        with mock.patch("evo.compute.outputs.object_from_reference", loader):
            yield loader

    @contextmanager
    def objects_service(self, **by_url: mock.MagicMock) -> Iterator[mock.AsyncMock]:
        """Serve a different typed object per reference, as the objects service would."""
        loader = mock.AsyncMock(side_effect=lambda _context, reference: by_url[reference])
        with mock.patch("evo.compute.outputs.object_from_reference", loader):
            yield loader


class TestRawPayloadIsPreserved(HydrationTestCase):
    """Hydration adds to the payload; it never rewrites it."""

    def test_the_result_still_equals_the_payload_the_platform_sent(self) -> None:
        self.assertEqual(RESULTS, self.result)
        self.assertIsInstance(self.result, dict)

    def test_it_still_serialises_as_the_same_json(self) -> None:
        self.assertEqual(json.loads(json.dumps(RESULTS)), json.loads(json.dumps(self.result)))

    def test_index_access_reaches_every_level(self) -> None:
        self.assertEqual(TARGET_URL, self.result["target"]["reference"])
        self.assertEqual("estimate", self.result["target"]["attribute"]["name"])


class TestAttributeAccess(HydrationTestCase):
    """The schema shapes the payload, so it can be read as attributes."""

    def test_nested_objects_become_nodes(self) -> None:
        self.assertIsInstance(self.result.target, ResultNode)
        self.assertEqual("estimated grid", self.result.target.name)
        self.assertEqual("estimate", self.result.target.attribute.name)

    def test_arrays_of_outputs_become_lists_of_nodes(self) -> None:
        self.assertEqual(1, len(self.result.extras))
        self.assertIsInstance(self.result.extras[0], ResultNode)
        self.assertEqual(POINTSET_URL, self.result.extras[0].reference)

    def test_a_nullable_output_that_came_back_null_stays_none(self) -> None:
        self.assertIsNone(self.result.report)

    def test_an_unknown_field_raises_attribute_error(self) -> None:
        with self.assertRaises(AttributeError):
            _ = self.result.nonexistent

    def test_tab_completion_lists_the_fields_that_came_back(self) -> None:
        self.assertLessEqual({"message", "target", "extras"}, set(dir(self.result)))


class TestObjectOutputs(HydrationTestCase):
    """``output: geoscience-object`` closes the DataFrame -> task -> DataFrame loop."""

    async def test_load_downloads_the_object_the_task_wrote(self) -> None:
        with self.loaded_object() as loader:
            loaded = await self.result.target.load()
        loader.assert_awaited_once_with(self.context, TARGET_URL)
        self.assertIs(loader.return_value, loaded)

    async def test_to_dataframe_reads_the_loaded_object(self) -> None:
        with self.loaded_object() as loader:
            frame = await self.result.target.to_dataframe()
        self.assertEqual("dataframe", frame)
        loader.return_value.to_dataframe.assert_awaited_once_with()

    async def test_an_item_of_an_output_array_loads_too(self) -> None:
        with self.loaded_object() as loader:
            await self.result.extras[0].load()
        loader.assert_awaited_once_with(self.context, POINTSET_URL)

    async def test_loading_something_that_is_no_kind_of_reference_is_refused(self) -> None:
        """A file output is a reference the SDK has no loader for; it is not guessed at."""
        result = TaskResult({**RESULTS, "report": {"reference": "/file/v2/x", "name": "log"}}, self.schema, None)
        with self.assertRaises(TypeError):
            await result.report.load()

    def test_a_file_output_is_left_as_its_reference(self) -> None:
        """File outputs are surfaced as references until the File API layer lands."""
        result = TaskResult({**RESULTS, "report": {"reference": "/file/v2/x", "name": "log"}}, self.schema, None)
        self.assertEqual("/file/v2/x", result.report.reference)


class TestAttributeOutputs(HydrationTestCase):
    """``output: attribute`` names an attribute on a sibling object, not an object."""

    async def test_sdk_array_and_object_search_proxies_load(self) -> None:
        owner = loaded_object(POINTSET_DOCUMENT)
        for expression, proxy_type in (
            ("locations.attributes[?name=='estimate']", jmespath.JMESPathArrayProxy),
            ("locations.attributes[0]", jmespath.JMESPathObjectProxy),
        ):
            with self.subTest(expression=expression):
                self.assertIsInstance(owner.search(expression), proxy_type)
                results = {
                    **RESULTS,
                    "target": {**RESULTS["target"], "attribute": {"reference": expression, "name": "estimate"}},
                }
                result = TaskResult(results, self.schema, self.context)
                with self.objects_service(**{TARGET_URL: owner}):
                    attribute = await result.target.attribute.load()
                self.assertEqual("estimate", attribute.name)

    async def test_the_attribute_is_loaded_from_the_object_attribute_from_points_at(self) -> None:
        with self.objects_service(**{TARGET_URL: loaded_object(POINTSET_DOCUMENT)}) as loader:
            attribute = await self.result.target.attribute.load()
        # ``2/reference`` climbs out of the attribute node to the target that owns it, so the
        # other object in the payload is not what gets downloaded.
        loader.assert_awaited_once_with(self.context, TARGET_URL)
        self.assertEqual("estimate", attribute.name)

    async def test_an_expression_written_for_another_family_is_healed(self) -> None:
        """One expression is published for every family a task supports; only one can be right."""
        results = {**RESULTS, "target": {**RESULTS["target"], "reference": BLOCK_MODEL_URL}}
        result = TaskResult(results, self.schema, self.context)
        block_model = loaded_object(BLOCK_MODEL_DOCUMENT, BLOCK_MODEL)
        # Without healing there is nothing to find: the published expression names a
        # container this family does not have.
        self.assertIsNone(block_model.search(results["target"]["attribute"]["reference"]))
        with self.objects_service(**{BLOCK_MODEL_URL: block_model}):
            attribute = await result.target.attribute.load()
        self.assertEqual("estimate", attribute.name)

    async def test_to_dataframe_reads_the_attribute_alone(self) -> None:
        with self.objects_service(**{TARGET_URL: loaded_object(POINTSET_DOCUMENT)}):
            frame = await self.result.target.attribute.to_dataframe()
        self.assertEqual("attribute dataframe", frame)

    async def test_an_attribute_is_one_column_and_takes_no_keys(self) -> None:
        with self.objects_service(**{TARGET_URL: loaded_object(POINTSET_DOCUMENT)}):
            with self.assertRaises(TypeError):
                await self.result.target.attribute.to_dataframe("estimate")

    async def test_an_attribute_the_object_does_not_have_is_reported(self) -> None:
        results = {**RESULTS, "target": {**RESULTS["target"], "attribute": {"reference": "attributes[?name=='x']"}}}
        result = TaskResult(results, self.schema, self.context)
        with self.objects_service(**{TARGET_URL: loaded_object(POINTSET_DOCUMENT)}):
            with self.assertRaises(ValueError) as context:
                await result.target.attribute.load()
        self.assertIn("matches no attribute", str(context.exception))

    async def test_an_attribute_with_no_pointer_cannot_name_its_object(self) -> None:
        node = ResultNode(
            {"reference": "attributes[?name=='estimate']", "name": "estimate"},
            {"output": "attribute", "properties": {"reference": {"reference_to": "attribute"}}},
            self.context,
        )
        with self.assertRaises(ValueError) as context:
            await node.load()
        self.assertIn("which object this attribute belongs to", str(context.exception))


class TestTasksWithoutAResultSchema(HydrationTestCase):
    """A task that advertises no ``results`` block still yields a usable payload."""

    def test_the_payload_is_returned_as_it_came(self) -> None:
        result = TaskResult({"message": "done"}, None, self.context)
        self.assertEqual({"message": "done"}, result)
        self.assertEqual("done", result.message)

    async def test_nothing_claims_to_be_an_object(self) -> None:
        result = TaskResult({"target": {"reference": TARGET_URL}}, None, self.context)
        with self.assertRaises(TypeError):
            await result.target.load()


class TestEngineHydration(TestWithConnector):
    """``run(...)`` returns the hydrated result, not the bare payload."""

    def setUp(self) -> None:
        super().setUp()
        self.context = FakeContext(self.connector)
        self.catalogue = load_test_data("discovery-resolution.json")

    @contextmanager
    def catalogue_response(self) -> Iterator[None]:
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(self.catalogue),
            headers={"Content-Type": "application/json"},
        ):
            yield

    @contextmanager
    def mock_job_client(self) -> Iterator[None]:
        job = mock.Mock()
        job.wait_for_results = mock.AsyncMock(return_value=RESULTS)
        with mock.patch("evo.compute.engine.JobClient") as mock_job_client:
            mock_job_client.submit = mock.AsyncMock(return_value=job)
            yield

    @contextmanager
    def objects_service(self) -> Iterator[None]:
        async def load(_context: Any, reference: str) -> mock.Mock:
            return mock.Mock(metadata=mock.Mock(schema_id=SCHEMAS_BY_URL[str(reference)]))

        with mock.patch("evo.compute.resolution.DownloadedObject.from_context", mock.AsyncMock(side_effect=load)):
            yield

    async def run_task(self) -> Any:
        client = ComputeClient(self.context)
        with self.catalogue_response(), self.objects_service(), self.mock_job_client():
            return await client.demo.estimate.run(
                source={"object": POINTSET_URL, "attribute": "locations.attributes[?name=='grade']"},
                target={"object": TARGET_URL, "attribute": "estimate"},
            )

    async def test_the_result_is_hydrated_against_the_result_schema(self) -> None:
        result = await self.run_task()
        self.assertIsInstance(result, TaskResult)
        self.assertEqual("estimated grid", result.target.name)

    async def test_the_result_is_still_the_payload_the_platform_sent(self) -> None:
        self.assertEqual(RESULTS, await self.run_task())

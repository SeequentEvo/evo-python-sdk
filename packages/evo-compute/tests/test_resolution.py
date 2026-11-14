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

"""Reference resolution against a task schema shaped like the ones the platform publishes.

``tests/data/discovery-resolution.json`` mirrors the real catalogue rather than a convenient
simplification: reference leaves are declared as URL *strings*, attribute leaves carry the
``attribute_from``/``attribute_path`` pair, the filter union sits behind ``$ref``, and the
target attribute is a create-or-update ``oneOf``. Resolution is only as good as the shapes it
is tested against, so the fixture has to be the awkward one.

The objects service is stubbed at :meth:`DownloadedObject.from_context`, which is the single
point where resolution reaches the platform.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock
from uuid import UUID

from evo.common import Environment
from evo.common.test_tools import TestWithConnector
from evo.objects import ObjectMetadata, ObjectReference, ObjectSchema, SchemaVersion
from evo.objects.typed import Attribute, PendingAttribute
from evo.objects.typed.base import BaseObject

from data import load_test_data
from evo.compute import ComputeClient, ParameterValidationError, ReferenceResolver, TaskResource
from evo.compute.tasks import Source, Target
from evo.compute.tasks.geostatistics.kriging import BlockDiscretization

HUB_URL = "https://unittest.localhost/"
ORG_ID = UUID(int=1)
WORKSPACE_ID = UUID(int=2)


def _object_url(suffix: str) -> str:
    return (
        f"{HUB_URL.rstrip('/')}/geoscience-object"
        f"/orgs/{ORG_ID}/workspaces/{WORKSPACE_ID}"
        f"/objects/00000000-0000-0000-0000-0000000000{suffix}"
    )


POINTSET_URL = _object_url("10")
BLOCK_MODEL_URL = _object_url("20")
TARGET_URL = _object_url("30")

POINTSET = ObjectSchema("objects", "pointset", SchemaVersion(1, 2, 0))
BLOCK_MODEL = ObjectSchema("objects", "block-model", SchemaVersion(1, 0, 0))

SCHEMAS_BY_URL = {POINTSET_URL: POINTSET, BLOCK_MODEL_URL: BLOCK_MODEL, TARGET_URL: POINTSET}


class FakeContext:
    """Minimal duck-typed IContext exposing just what reference resolution uses."""

    def __init__(self, connector: Any = None) -> None:
        self._connector = connector

    def get_environment(self) -> Environment:
        return Environment(hub_url=HUB_URL, org_id=ORG_ID, workspace_id=WORKSPACE_ID)

    def get_connector(self) -> Any:
        return self._connector

    def get_org_id(self) -> UUID:
        return ORG_ID

    def get_cache(self) -> None:
        return None


def typed_object(url: str, schema: ObjectSchema | None = POINTSET) -> mock.MagicMock:
    """Stands in for a typed geoscience object the caller already holds."""
    obj = mock.MagicMock(spec=BaseObject)
    obj.metadata.url = ObjectReference(url)
    obj.metadata.schema_id = schema
    return obj


def existing_attribute(key: str, url: str, schema_path: str = "", schema: ObjectSchema = POINTSET) -> mock.MagicMock:
    """An attribute already on an object, which resolves to a key-based expression."""
    attribute = mock.MagicMock(spec=Attribute)
    attribute.key = key
    attribute.exists = True
    attribute._context = mock.MagicMock(schema_path=schema_path)
    attribute._obj = typed_object(url, schema)
    return attribute


def pending_attribute(name: str, url: str, schema: ObjectSchema = POINTSET) -> PendingAttribute:
    """An attribute that does not exist yet, which resolves to a create operation."""
    parent = mock.MagicMock()
    parent._obj = typed_object(url, schema)
    return PendingAttribute(parent, name)


class ResolutionTestCase(IsolatedAsyncioTestCase):
    """A resolver over the ``demo/estimate`` spec, with the objects service stubbed out."""

    def setUp(self) -> None:
        catalogue = load_test_data("discovery-resolution.json")
        self.spec = TaskResource.model_validate(catalogue["results"][0])
        self.import_spec = TaskResource.model_validate(catalogue["results"][1])
        self.resolver = ReferenceResolver(FakeContext())

    @contextmanager
    def objects_service(self, *, unavailable: bool = False) -> Iterator[mock.AsyncMock]:
        """Stub the metadata download resolution falls back to for a bare reference."""

        async def load(_context: Any, reference: str) -> mock.Mock:
            if unavailable:
                raise RuntimeError("objects service unavailable")
            return mock.Mock(metadata=mock.Mock(schema_id=SCHEMAS_BY_URL[str(reference)]))

        loader = mock.AsyncMock(side_effect=load)
        with mock.patch("evo.compute.resolution.DownloadedObject.from_context", loader):
            yield loader

    async def resolve(self, **parameters: Any) -> dict[str, Any]:
        return await self.resolver.resolve(self.spec, parameters)


class TestObjectReferences(ResolutionTestCase):
    """``reference_to: geoscience-object`` -- any handle in, one validated URL out."""

    async def frame(self, source_object: Any) -> Any:
        with self.objects_service():
            resolved = await self.resolve(source={"object": source_object, "attribute": "grade"})
        return resolved["source"]["object"]

    async def test_a_url_string_is_passed_through_verbatim(self) -> None:
        """The caller's own reference is submitted as written, version pin and all."""
        self.assertEqual(POINTSET_URL, await self.frame(POINTSET_URL))

    async def test_an_object_reference_resolves_to_its_string(self) -> None:
        self.assertEqual(POINTSET_URL, await self.frame(ObjectReference(POINTSET_URL)))

    async def test_a_typed_object_resolves_to_its_url(self) -> None:
        self.assertEqual(POINTSET_URL, await self.frame(typed_object(POINTSET_URL)))

    async def test_object_metadata_resolves_to_its_url(self) -> None:
        metadata = mock.MagicMock(spec=ObjectMetadata)
        metadata.url = ObjectReference(POINTSET_URL)
        metadata.schema_id = POINTSET
        self.assertEqual(POINTSET_URL, await self.frame(metadata))

    async def test_a_bare_uuid_is_expanded_in_the_client_workspace(self) -> None:
        """A caller who only has an object id gets the URL built for them."""
        object_id = UUID("00000000-0000-0000-0000-000000000010")
        self.assertEqual(POINTSET_URL, await self.frame(object_id))
        self.assertEqual(POINTSET_URL, await self.frame(str(object_id)))

    async def test_an_unusable_reference_is_reported_against_its_location(self) -> None:
        with self.assertRaises(ParameterValidationError) as ctx:
            await self.frame("not-a-reference")
        self.assertIn("source.object", str(ctx.exception))
        self.assertEqual(1, len(ctx.exception.errors))

    async def test_an_unsupported_handle_type_is_rejected(self) -> None:
        with self.assertRaises(ParameterValidationError) as ctx:
            await self.frame(12345)
        self.assertIn("source.object", str(ctx.exception))


class TestSupportedSchemas(ResolutionTestCase):
    """``supported_schemas`` -- a hard error when it can be checked, a warning when it cannot."""

    async def test_a_typed_object_is_checked_without_touching_the_network(self) -> None:
        with self.objects_service() as loader:
            await self.resolve(
                source={"object": typed_object(POINTSET_URL), "attribute": "locations.attributes[?name=='grade']"},
                target={"object": typed_object(TARGET_URL), "attribute": "estimate"},
            )
        loader.assert_not_awaited()

    async def test_an_object_outside_the_supported_set_is_rejected(self) -> None:
        unsupported = ObjectSchema("objects", "triangle-mesh", SchemaVersion(2, 1, 0))
        with self.objects_service():
            with self.assertRaises(ParameterValidationError) as ctx:
                await self.resolve(source={"object": typed_object(POINTSET_URL, unsupported), "attribute": "grade"})
        self.assertIn("source.object", str(ctx.exception))
        self.assertIn("triangle-mesh", str(ctx.exception))

    async def test_a_bare_reference_is_checked_by_loading_the_object_once(self) -> None:
        """The same object named twice costs one metadata request, not two."""
        with self.objects_service() as loader:
            await self.resolve(
                source={"object": POINTSET_URL, "attribute": "grade"},
                target={"object": POINTSET_URL, "attribute": "estimate"},
            )
        self.assertEqual(1, loader.await_count)

    async def test_an_unloadable_object_warns_instead_of_blocking(self) -> None:
        """An objects-service outage must not stop a job the platform would accept."""
        with self.objects_service(unavailable=True):
            with self.assertLogs("compute.resolution", level="WARNING") as logs:
                resolved = await self.resolve(source={"object": POINTSET_URL, "attribute": "grade"})
        self.assertEqual(POINTSET_URL, resolved["source"]["object"])
        self.assertIn("supported_schemas", logs.output[0])

    async def test_the_check_is_skipped_when_validation_is_off(self) -> None:
        unsupported = ObjectSchema("objects", "triangle-mesh", SchemaVersion(2, 1, 0))
        with self.objects_service() as loader:
            resolved = await self.resolver.resolve(
                self.spec,
                {"source": {"object": typed_object(POINTSET_URL, unsupported), "attribute": "grade"}},
                check_schemas=False,
            )
        self.assertEqual(POINTSET_URL, resolved["source"]["object"])
        loader.assert_not_awaited()

    async def test_the_declared_version_range_is_honoured(self) -> None:
        """``pointset/[>=1.2,<2]`` admits 1.9.9 but neither 1.1.0 nor 2.0.0."""
        accepted = [SchemaVersion(1, 2, 0), SchemaVersion(1, 9, 9)]
        rejected = [SchemaVersion(1, 1, 0), SchemaVersion(2, 0, 0)]
        for version in accepted:
            with self.subTest(version=str(version)), self.objects_service():
                await self.resolve(
                    source={"object": typed_object(POINTSET_URL, ObjectSchema("objects", "pointset", version))}
                )
        for version in rejected:
            with self.subTest(version=str(version)), self.objects_service():
                with self.assertRaises(ParameterValidationError):
                    await self.resolve(
                        source={"object": typed_object(POINTSET_URL, ObjectSchema("objects", "pointset", version))}
                    )


class TestAttributeReferences(ResolutionTestCase):
    """``reference_to: attribute`` -- a name or a typed attribute in, JMESPath out."""

    async def source_attribute(self, attribute: Any, object_: Any = POINTSET_URL) -> Any:
        with self.objects_service():
            resolved = await self.resolve(source={"object": object_, "attribute": attribute})
        return resolved["source"]["attribute"]

    async def test_an_existing_typed_attribute_resolves_by_key(self) -> None:
        attribute = existing_attribute("abc", POINTSET_URL, schema_path="locations.attributes")
        self.assertEqual("locations.attributes[?key=='abc']", await self.source_attribute(attribute))

    async def test_a_pending_typed_attribute_resolves_by_name(self) -> None:
        self.assertEqual(
            "attributes[?name=='grade']", await self.source_attribute(pending_attribute("grade", POINTSET_URL))
        )

    async def test_a_ready_expression_is_left_alone(self) -> None:
        """A caller who already speaks JMESPath is not second-guessed."""
        expression = "locations.attributes[?name=='grade']"
        self.assertEqual(expression, await self.source_attribute(expression))

    async def test_a_bare_name_is_placed_by_the_owning_objects_family(self) -> None:
        """A pointset keeps attributes under ``locations``; a block model does not."""
        self.assertEqual("locations.attributes[?name=='grade']", await self.source_attribute("grade"))
        self.assertEqual("attributes[?name=='grade']", await self.source_attribute("grade", object_=BLOCK_MODEL_URL))

    async def test_a_bare_name_falls_back_when_the_object_is_unknown(self) -> None:
        with self.objects_service(unavailable=True), self.assertLogs("compute.resolution", level="WARNING"):
            resolved = await self.resolve(source={"object": POINTSET_URL, "attribute": "grade"})
        self.assertEqual("attributes[?name=='grade']", resolved["source"]["attribute"])

    async def test_a_value_that_is_not_an_attribute_at_all_is_rejected(self) -> None:
        """Deep validation relaxes reference leaves, so this is the only place it can be caught."""
        with self.objects_service():
            with self.assertRaises(ParameterValidationError) as ctx:
                await self.source_attribute(12345)
        self.assertIn("source.attribute", str(ctx.exception))
        self.assertIn("int", str(ctx.exception))

    async def test_a_filter_attribute_is_placed_by_an_absolute_pointer(self) -> None:
        """``attribute_from: /source/object`` reaches across the payload, not just up it."""
        with self.objects_service():
            resolved = await self.resolve(
                source={
                    "object": POINTSET_URL,
                    "attribute": "grade",
                    "filter": {"where": {"type": "condition", "operator": "equal", "attribute": "domain"}},
                }
            )
        condition = resolved["source"]["filter"]["where"]
        self.assertEqual("locations.attributes[?name=='domain']", condition["attribute"])
        self.assertEqual("equal", condition["operator"])

    async def test_a_nested_filter_union_resolves_every_branch(self) -> None:
        """The union is recursive and behind ``$ref``; every leaf still gets resolved."""
        with self.objects_service():
            resolved = await self.resolve(
                source={
                    "object": BLOCK_MODEL_URL,
                    "attribute": "grade",
                    "filter": {
                        "where": {
                            "type": "all_of",
                            "filters": [
                                {"type": "condition", "operator": "equal", "attribute": "domain"},
                                {"type": "condition", "operator": "greater_than", "attribute": "zone"},
                            ],
                        }
                    },
                }
            )
        filters = resolved["source"]["filter"]["where"]["filters"]
        self.assertEqual(
            ["attributes[?name=='domain']", "attributes[?name=='zone']"],
            [condition["attribute"] for condition in filters],
        )

    async def test_an_optional_filter_written_as_a_union_is_walked_too(self) -> None:
        """``anyOf: [Filter, null]`` describes the same node as a nullable inline object.

        The catalogue inlines an optional sub-object and adds ``null`` to its ``type``, but a
        schema generated from a model writes the union instead. Both have to be descended
        into, or a reference inside an optional filter reaches the wire unresolved.
        """
        with self.objects_service():
            resolved = await self.resolve(
                source={"object": POINTSET_URL, "attribute": "grade"},
                target={
                    "object": TARGET_URL,
                    "attribute": {"operation": "create", "name": "estimate"},
                    "filter": {"where": {"type": "condition", "operator": "equal", "attribute": "domain"}},
                },
            )
        self.assertEqual("locations.attributes[?name=='domain']", resolved["target"]["filter"]["where"]["attribute"])


class TestTargetAttributes(ResolutionTestCase):
    """``target: attribute`` -- create a new attribute, or update an existing one."""

    async def target_attribute(self, attribute: Any, object_: Any = TARGET_URL) -> Any:
        with self.objects_service():
            resolved = await self.resolve(target={"object": object_, "attribute": attribute})
        return resolved["target"]["attribute"]

    async def test_a_bare_name_creates_the_attribute(self) -> None:
        self.assertEqual({"operation": "create", "name": "estimate"}, await self.target_attribute("estimate"))

    async def test_a_pending_typed_attribute_creates_the_attribute(self) -> None:
        self.assertEqual(
            {"operation": "create", "name": "estimate"},
            await self.target_attribute(pending_attribute("estimate", TARGET_URL)),
        )

    async def test_an_existing_typed_attribute_updates_it(self) -> None:
        attribute = existing_attribute("xyz", TARGET_URL, schema_path="locations.attributes")
        self.assertEqual(
            {"operation": "update", "reference": "locations.attributes[?key=='xyz']"},
            await self.target_attribute(attribute),
        )

    async def test_an_update_written_out_still_resolves_its_reference(self) -> None:
        """Choosing the branch by hand does not opt out of resolving what is inside it."""
        self.assertEqual(
            {"operation": "update", "reference": "locations.attributes[?name=='estimate']"},
            await self.target_attribute({"operation": "update", "reference": "estimate"}),
        )

    async def test_a_create_written_out_is_left_alone(self) -> None:
        wire = {"operation": "create", "name": "estimate"}
        self.assertEqual(wire, await self.target_attribute(wire))


class TestTypedAttributeShorthand(ResolutionTestCase):
    """A bare typed attribute names its object too, exactly as the typed tasks accept."""

    async def test_a_source_attribute_fills_in_the_whole_frame(self) -> None:
        attribute = existing_attribute("abc", POINTSET_URL, schema_path="locations.attributes")
        with self.objects_service():
            resolved = await self.resolve(source=attribute)
        self.assertEqual({"object": POINTSET_URL, "attribute": "locations.attributes[?key=='abc']"}, resolved["source"])

    async def test_a_target_attribute_fills_in_the_whole_frame(self) -> None:
        with self.objects_service():
            resolved = await self.resolve(target=pending_attribute("estimate", TARGET_URL))
        self.assertEqual(
            {"object": TARGET_URL, "attribute": {"operation": "create", "name": "estimate"}}, resolved["target"]
        )

    async def test_a_missing_source_attribute_keeps_the_typed_tasks_error(self) -> None:
        """The mistake and its wording are the same whichever path the caller took."""
        with self.objects_service():
            with self.assertRaises(ParameterValidationError) as ctx:
                await self.resolve(source=pending_attribute("graed", POINTSET_URL))
        self.assertIn("does not exist on the source object", str(ctx.exception))

    async def test_the_frame_object_still_faces_supported_schemas(self) -> None:
        """The shorthand is a shorter way to say the same thing, not a way past the checks."""
        unsupported = ObjectSchema("objects", "triangle-mesh", SchemaVersion(2, 1, 0))
        for parameter, attribute in (
            ("source", existing_attribute("abc", POINTSET_URL, schema=unsupported)),
            ("target", pending_attribute("estimate", TARGET_URL, schema=unsupported)),
        ):
            with self.subTest(parameter=parameter), self.objects_service() as loader:
                with self.assertRaises(ParameterValidationError) as ctx:
                    await self.resolve(**{parameter: attribute})
                self.assertIn(f"{parameter}.object", str(ctx.exception))
                # The attribute brought its object's schema along, so checking it is free.
                loader.assert_not_awaited()


class TestFileReferences(ResolutionTestCase):
    """``reference_to: file`` -- a File API URL, or a handle that carries one."""

    async def test_a_url_string_is_passed_through(self) -> None:
        resolved = await self.resolver.resolve(self.import_spec, {"file": "https://unittest.localhost/file/v2/x"})
        self.assertEqual("https://unittest.localhost/file/v2/x", resolved["file"])

    async def test_a_service_relative_path_is_accepted(self) -> None:
        resolved = await self.resolver.resolve(self.import_spec, {"file": "/file/v2/files/abc"})
        self.assertEqual("/file/v2/files/abc", resolved["file"])

    async def test_a_handle_carrying_a_url_resolves_to_it(self) -> None:
        handle = mock.Mock(url="https://unittest.localhost/file/v2/y")
        resolved = await self.resolver.resolve(self.import_spec, {"file": handle})
        self.assertEqual("https://unittest.localhost/file/v2/y", resolved["file"])

    async def test_a_local_filename_is_rejected(self) -> None:
        """Deep validation relaxes reference leaves, so this is the only place it can be caught."""
        with self.assertRaises(ParameterValidationError) as ctx:
            await self.resolver.resolve(self.import_spec, {"file": "mesh.obj"})
        self.assertIn("file", str(ctx.exception))
        self.assertIn("mesh.obj", str(ctx.exception))

    async def test_a_handle_without_a_url_is_reported(self) -> None:
        with self.assertRaises(ParameterValidationError) as ctx:
            await self.resolver.resolve(self.import_spec, {"file": object()})
        self.assertIn("file", str(ctx.exception))


class TestNonReferenceValues(ResolutionTestCase):
    """Everything the annotations say nothing about survives the walk untouched."""

    async def test_a_typed_parameter_model_is_accepted_and_resolved_through(self) -> None:
        """The SDK's own ``Source``/``Target`` models are valid engine inputs too."""
        with self.objects_service():
            resolved = await self.resolve(
                source=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']"),
                target=Target.new_attribute(TARGET_URL, "estimate"),
            )
        self.assertEqual(
            {"object": POINTSET_URL, "attribute": "locations.attributes[?name=='grade']"}, resolved["source"]
        )
        self.assertEqual(
            {"object": TARGET_URL, "attribute": {"operation": "create", "name": "estimate"}}, resolved["target"]
        )

    async def test_scalars_arrays_and_plain_objects_are_unchanged(self) -> None:
        with self.objects_service():
            resolved = await self.resolve(
                source={"object": POINTSET_URL, "attribute": "grade"},
                seeds=[3, 1, 4],
                options={"label": "run-1", "retries": 2},
            )
        self.assertEqual([3, 1, 4], resolved["seeds"])
        self.assertEqual({"label": "run-1", "retries": 2}, resolved["options"])

    async def test_an_explicit_none_survives_resolution(self) -> None:
        """A nullable reference set to ``None`` stays ``None`` rather than being resolved."""
        with self.objects_service():
            resolved = await self.resolve(source={"object": POINTSET_URL, "attribute": "grade"}, report=None)
        self.assertIsNone(resolved["report"])

    async def test_a_closed_object_rejects_undeclared_parameters(self) -> None:
        with self.assertRaisesRegex(ParameterValidationError, "mystery"):
            await self.resolver.resolve(self.spec, {"mystery": {"a": 1}})

    async def test_a_typed_model_in_an_optional_slot_is_dumped(self) -> None:
        """An ``X | None`` slot has no ``properties`` of its own, but a model there is still ours."""
        with self.objects_service():
            resolved = await self.resolve(
                source={"object": POINTSET_URL, "attribute": "grade"},
                discretisation=BlockDiscretization(nx=3, ny=3, nz=2),
            )
        self.assertEqual({"nx": 3, "ny": 3, "nz": 2}, resolved["discretisation"])


class TestBareObjectFrames(ResolutionTestCase):
    """A parameter that is an object and nothing else accepts the object itself."""

    async def resolve_grid(self, grid: Any) -> Any:
        with self.objects_service():
            resolved = await self.resolve(source={"object": POINTSET_URL, "attribute": "grade"}, grid=grid)
        return resolved["grid"]

    async def test_a_typed_object_fills_in_the_frame(self) -> None:
        """``run(grid=block_model)`` is how the typed tasks are called, so it works here too."""
        grid = typed_object(BLOCK_MODEL_URL, BLOCK_MODEL)
        self.assertEqual({"object": BLOCK_MODEL_URL}, await self.resolve_grid(grid))

    async def test_a_url_fills_in_the_frame(self) -> None:
        self.assertEqual({"object": BLOCK_MODEL_URL}, await self.resolve_grid(BLOCK_MODEL_URL))

    async def test_the_written_out_frame_is_left_alone(self) -> None:
        self.assertEqual({"object": BLOCK_MODEL_URL}, await self.resolve_grid({"object": BLOCK_MODEL_URL}))


class EngineResolutionTestCase(TestWithConnector):
    """A client over the ``demo`` catalogue, with the objects service and job client stubbed out."""

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
    def mock_job_client(self, results: dict | None = None) -> Iterator[mock.AsyncMock]:
        job = mock.Mock()
        job.wait_for_results = mock.AsyncMock(return_value={} if results is None else results)
        submit = mock.AsyncMock(return_value=job)
        with mock.patch("evo.compute.engine.JobClient") as mock_job_client:
            mock_job_client.submit = submit
            yield submit

    @contextmanager
    def objects_service(self) -> Iterator[mock.AsyncMock]:
        async def load(_context: Any, reference: str) -> mock.Mock:
            return mock.Mock(metadata=mock.Mock(schema_id=SCHEMAS_BY_URL[str(reference)]))

        with mock.patch("evo.compute.resolution.DownloadedObject.from_context", mock.AsyncMock(side_effect=load)):
            yield


class TestEngineResolution(EngineResolutionTestCase):
    """The engine resolves between its two validation passes, then submits the resolved payload."""

    async def test_run_submits_the_resolved_payload(self) -> None:
        client = ComputeClient(self.context)
        with self.catalogue_response(), self.objects_service(), self.mock_job_client() as submit:
            await client.demo.estimate.run(
                source=existing_attribute("abc", POINTSET_URL, schema_path="locations.attributes"),
                target=pending_attribute("estimate", TARGET_URL),
            )

        self.assertEqual(
            {
                "source": {"object": POINTSET_URL, "attribute": "locations.attributes[?key=='abc']"},
                "target": {"object": TARGET_URL, "attribute": {"operation": "create", "name": "estimate"}},
            },
            submit.await_args.kwargs["parameters"],
        )

    async def test_deep_validation_judges_the_resolved_payload(self) -> None:
        """Typed objects would fail a schema that asks for URL strings; resolved ones pass."""
        client = ComputeClient(self.context, deep_validation=True)
        with self.catalogue_response(), self.objects_service(), self.mock_job_client() as submit:
            await client.demo.estimate.run(
                source={"object": typed_object(POINTSET_URL), "attribute": "grade"},
                target={"object": typed_object(TARGET_URL), "attribute": "estimate"},
            )
        submit.assert_awaited_once()

    async def test_a_missing_required_parameter_is_caught_before_any_object_is_loaded(self) -> None:
        """Nothing is worth a request until the call itself is known to be well formed."""
        client = ComputeClient(self.context)
        with self.catalogue_response(), self.objects_service() as _, self.mock_job_client() as submit:
            with mock.patch("evo.compute.resolution.DownloadedObject.from_context", mock.AsyncMock()) as loader:
                with self.assertRaises(ParameterValidationError):
                    await client.demo.estimate.run(source={"object": POINTSET_URL, "attribute": "grade"})
        loader.assert_not_awaited()
        submit.assert_not_awaited()


class TestSupportedSchemaSwitch(EngineResolutionTestCase):
    """``check_schemas`` is independent of ``validate``, defaulting to follow it."""

    UNSUPPORTED = ObjectSchema("objects", "triangle-mesh", SchemaVersion(2, 1, 0))

    async def run_with_an_unsupported_source(self, client: ComputeClient, **overrides: Any) -> mock.AsyncMock:
        with self.catalogue_response(), self.objects_service(), self.mock_job_client() as submit:
            await client.arun(
                "demo",
                "estimate",
                {
                    "source": {"object": typed_object(POINTSET_URL, self.UNSUPPORTED), "attribute": "grade"},
                    "target": {"object": typed_object(TARGET_URL), "attribute": "estimate"},
                },
                **overrides,
            )
        return submit

    async def test_the_check_follows_validation_by_default(self) -> None:
        for validate, expect_rejection in ((True, True), (False, False)):
            with self.subTest(validate=validate):
                client = ComputeClient(self.context, validate=validate)
                if expect_rejection:
                    with self.assertRaises(ParameterValidationError):
                        await self.run_with_an_unsupported_source(client)
                else:
                    submit = await self.run_with_an_unsupported_source(client)
                    submit.assert_awaited_once()

    async def test_the_check_survives_validation_being_turned_off(self) -> None:
        """The guard that catches an object the task cannot read is worth keeping on its own."""
        client = ComputeClient(self.context, validate=False, check_schemas=True)
        with self.assertRaises(ParameterValidationError) as ctx:
            await self.run_with_an_unsupported_source(client)
        self.assertIn("triangle-mesh", str(ctx.exception))

    async def test_the_check_can_be_dropped_while_validation_stays_on(self) -> None:
        """Skipping it costs the request each referenced object would otherwise need."""
        client = ComputeClient(self.context, check_schemas=False)
        submit = await self.run_with_an_unsupported_source(client)
        submit.assert_awaited_once()

    async def test_the_client_setting_is_overridden_per_call(self) -> None:
        client = ComputeClient(self.context, check_schemas=False)
        with self.assertRaises(ParameterValidationError):
            await self.run_with_an_unsupported_source(client, check_schemas=True)

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

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Optional
from unittest import mock

from evo.common.test_tools import ORG as TEST_ORG
from evo.common.test_tools import TestWithConnector

from data import load_test_data
from evo.compute import ComputeClient, ParameterValidationError


def _object_url(suffix: str) -> str:
    """A structurally valid geoscience object URL, which reference resolution insists on."""
    return (
        "https://unittest.localhost/geoscience-object"
        "/orgs/00000000-0000-0000-0000-000000000001"
        "/workspaces/00000000-0000-0000-0000-000000000002"
        f"/objects/00000000-0000-0000-0000-0000000000{suffix}"
    )


SOURCE_URL = _object_url("10")
FILE_URL = "https://unittest.localhost/file/v2/orgs/00000000-0000-0000-0000-000000000001/files/mesh"


def _task(topic: str, name: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """A catalogue entry for a task the shared fixture does not describe."""
    return {
        "key": f"{topic}/{name}",
        "topic": topic,
        "name": name,
        "version": "0.1.0",
        "parameters": {"type": "object", "properties": properties, "required": required or []},
    }


class FakeContext:
    """Minimal duck-typed IContext exposing just what ComputeClient uses."""

    def __init__(self, connector, org_id) -> None:
        self._connector = connector
        self._org_id = org_id

    def get_connector(self):
        return self._connector

    def get_org_id(self):
        return self._org_id

    def get_cache(self):
        return None


class TestComputeClient(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.context = FakeContext(self.connector, TEST_ORG.id)
        self.catalogue = load_test_data("discovery-tasks.json")
        self.client = ComputeClient(self.context)

    @contextmanager
    def catalogue_response(self) -> Iterator[None]:
        """Serve the discovery catalogue for any GET /tasks issued while active."""
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(self.catalogue),
            headers={"Content-Type": "application/json"},
        ):
            yield

    @contextmanager
    def mock_job_client(self, results: dict | None = None) -> Iterator[mock.AsyncMock]:
        """Patch JobClient.submit so tests exercise the engine, not the job lifecycle."""
        job = mock.Mock()
        job.wait_for_results = mock.AsyncMock(return_value={"ok": True} if results is None else results)
        submit = mock.AsyncMock(return_value=job)
        with mock.patch("evo.compute.engine.JobClient") as mock_job_client:
            mock_job_client.submit = submit
            yield submit

    # -- namespace / discovery timing -------------------------------------- #

    def test_attribute_access_does_not_trigger_discovery(self) -> None:
        """Reaching a topic/task never hits the network; only run() does."""
        _ = self.client.geostatistics.declustering
        self.transport.request.assert_not_called()

    async def test_run_discovers_then_submits(self) -> None:
        """run() fetches the catalogue, then submits the resolved task."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            result = await self.client.geostatistics.declustering.run(source=SOURCE_URL)

        submit.assert_awaited_once()
        kwargs = submit.await_args.kwargs
        self.assertEqual("geostatistics", kwargs["topic"])
        self.assertEqual("declustering", kwargs["task"])
        self.assertEqual({"source": SOURCE_URL}, kwargs["parameters"])
        self.assertFalse(kwargs["preview"])
        self.assertEqual({"ok": True}, result)

    async def test_hyphenated_task_names_are_reachable_with_underscores(self) -> None:
        """``normal_score`` resolves to the platform's ``normal-score``."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.normal_score.run(distribution=SOURCE_URL)
        self.assertEqual("normal-score", submit.await_args.kwargs["task"])

    async def test_hyphenated_topics_and_parameters_are_reachable_with_underscores(self) -> None:
        """Parameters are submitted under the names the schema uses."""
        self.catalogue["results"].append(
            _task(
                "mesh-tools",
                "simplify",
                {"mesh": {"type": "string"}, "target-face-count": {"type": "integer"}},
                required=["mesh", "target-face-count"],
            )
        )
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.mesh_tools.simplify.run(mesh="mesh-1", target_face_count=1000)

        self.assertEqual("mesh-tools", submit.await_args.kwargs["topic"])
        self.assertEqual("simplify", submit.await_args.kwargs["task"])
        self.assertEqual({"mesh": "mesh-1", "target-face-count": 1000}, submit.await_args.kwargs["parameters"])
        self.assertIn("mesh_tools", dir(self.client))
        self.assertIn("target_face_count", inspect.signature(self.client.mesh_tools.simplify.run).parameters)

    async def test_parameters_that_cannot_all_be_keywords_are_taken_as_given(self) -> None:
        """A reserved word, or two names that normalise alike, leave ``run`` taking ``**parameters``."""
        self.catalogue["results"] += [
            _task("mesh-tools", "export", {"from": {"type": "string"}}, required=["from"]),
            _task("mesh-tools", "merge", {"user-info": {"type": "string"}, "user_info": {"type": "string"}}),
        ]
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.mesh_tools.export.run(**{"from": "mesh-1"})
            self.assertEqual({"from": "mesh-1"}, submit.await_args.kwargs["parameters"])

            await self.client.arun("mesh-tools", "merge", {"user-info": "a", "user_info": "b", "preview": True})
            self.assertEqual({"user-info": "a", "user_info": "b"}, submit.await_args.kwargs["parameters"])
            self.assertTrue(submit.await_args.kwargs["preview"])

        parameters = inspect.signature(self.client.mesh_tools.export.run).parameters
        self.assertEqual(["preview", "parameters"], list(parameters))
        self.assertEqual(inspect.Parameter.VAR_KEYWORD, parameters["parameters"].kind)

    async def test_fallback_parameters_still_reject_missing_and_unexpected_names(self) -> None:
        """Fallback ``**parameters`` calls still check required and closed-object schema keys."""
        export_spec = _task("mesh-tools", "export", {"from": {"type": "string"}}, required=["from"])
        export_spec["parameters"]["additionalProperties"] = False
        merge_spec = _task(
            "mesh-tools",
            "merge",
            {"user-info": {"type": "string"}, "user_info": {"type": "string"}},
            required=["user-info"],
        )
        merge_spec["parameters"]["additionalProperties"] = False
        self.catalogue["results"] += [export_spec, merge_spec]
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaisesRegex(ParameterValidationError, r"mesh_tools\.export\.run\(\):.*'from'"):
                await self.client.mesh_tools.export.run()
            with self.assertRaisesRegex(ParameterValidationError, r"mesh-tools\.export\.run\(\):.*'from'"):
                await self.client.arun("mesh-tools", "export", {}, validate=False)
            with self.assertRaisesRegex(ParameterValidationError, r"mesh_tools\.export\.run\(\):.*'bogus'"):
                await self.client.mesh_tools.export.run(**{"from": "mesh-1", "bogus": 1})
            with self.assertRaisesRegex(ParameterValidationError, r"mesh-tools\.merge\.run\(\):.*'user-info'"):
                await self.client.arun("mesh-tools", "merge", {"user_info": "b"})
            with self.assertRaisesRegex(ParameterValidationError, r"mesh-tools\.merge\.run\(\):.*'bogus'"):
                await self.client.arun("mesh-tools", "merge", {"user-info": "a", "bogus": 1})

        submit.assert_not_awaited()

    async def test_arun_takes_the_names_the_platform_publishes(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.arun("geostatistics", "normal-score", {"distribution": SOURCE_URL})
        self.assertEqual("normal-score", submit.await_args.kwargs["task"])
        self.assertEqual({"distribution": SOURCE_URL}, submit.await_args.kwargs["parameters"])

    async def test_catalogue_is_cached_across_runs(self) -> None:
        """The catalogue is fetched once and reused for later runs (any task/topic)."""
        with self.catalogue_response(), self.mock_job_client():
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)
            await self.client.geostatistics.normal_score.run(distribution=SOURCE_URL)
        self.assertEqual(1, self.transport.request.call_count)

    async def test_catalogue_is_shared_across_topics(self) -> None:
        """One catalogue fetch serves every topic, not just the first one touched."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)
            await self.client.converter.obj_import.run(file=FILE_URL)

        self.assertEqual(1, self.transport.request.call_count)
        self.assertEqual("converter", submit.await_args.kwargs["topic"])
        self.assertEqual("obj-import", submit.await_args.kwargs["task"])

    async def test_a_run_reads_the_catalogue_once(self) -> None:
        with (
            self.catalogue_response(),
            self.mock_job_client(),
            mock.patch.object(
                self.client._discovery, "list_tasks", wraps=self.client._discovery.list_tasks
            ) as list_tasks,
        ):
            await self.client.geostatistics.normal_score.run(distribution=SOURCE_URL)
        list_tasks.assert_awaited_once()

    # -- parameter forwarding ---------------------------------------------- #

    async def test_schema_defaults_are_not_forwarded(self) -> None:
        """Unset optionals are omitted so the platform applies its own defaults."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)

        parameters = submit.await_args.kwargs["parameters"]
        self.assertNotIn("method", parameters)  # schema default "cell" must not be sent
        self.assertNotIn("power", parameters)

    async def test_explicit_falsy_values_are_forwarded(self) -> None:
        """A supplied ``0``/``False`` is a real value, not an omission."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.declustering.run(source=SOURCE_URL, power=0.0)

        self.assertEqual({"source": SOURCE_URL, "power": 0.0}, submit.await_args.kwargs["parameters"])

    async def test_explicit_none_is_forwarded(self) -> None:
        """An explicit ``None`` reaches the wire for a nullable parameter."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.declustering.run(source=SOURCE_URL, power=None)

        self.assertEqual({"source": SOURCE_URL, "power": None}, submit.await_args.kwargs["parameters"])

    # -- preview flag ------------------------------------------------------ #

    async def test_preview_is_off_by_default_even_for_feature_flagged_tasks(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.converter.obj_import.run(file=FILE_URL)
        self.assertFalse(submit.await_args.kwargs["preview"])

    async def test_preview_can_be_overridden(self) -> None:
        """Explicit preview settings do not become task parameters."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.converter.obj_import.run(file=FILE_URL, preview=False)
        self.assertFalse(submit.await_args.kwargs["preview"])
        self.assertNotIn("preview", submit.await_args.kwargs["parameters"])

    async def test_preview_can_be_explicitly_enabled(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.converter.obj_import.run(file=FILE_URL, preview=True)
        self.assertTrue(submit.await_args.kwargs["preview"])

    # -- errors ------------------------------------------------------------ #

    async def test_unknown_task_raises_attribute_error(self) -> None:
        """Running a task the topic doesn't advertise reports what is available."""
        with self.catalogue_response(), self.mock_job_client():
            with self.assertRaises(AttributeError) as ctx:
                await self.client.geostatistics.does_not_exist.run()
        self.assertIn("declustering", str(ctx.exception))

    async def test_missing_required_parameter_raises_validation_error(self) -> None:
        """A missing required parameter is rejected before submission."""
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaises(ParameterValidationError) as ctx:
                await self.client.geostatistics.declustering.run()
        self.assertIn("source", str(ctx.exception))
        submit.assert_not_awaited()

    async def test_unexpected_parameter_raises_validation_error(self) -> None:
        spec = next(task for task in self.catalogue["results"] if task["name"] == "declustering")
        spec["parameters"]["additionalProperties"] = False
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaises(ParameterValidationError) as ctx:
                await self.client.geostatistics.declustering.run(source=SOURCE_URL, bogus=1)

        self.assertIn("geostatistics.declustering.run():", str(ctx.exception))
        self.assertIn("bogus", str(ctx.exception))
        self.assertEqual(["parameters: unexpected parameter(s): 'bogus'"], ctx.exception.errors)
        submit.assert_not_awaited()

    async def test_unspecified_extra_field_policy_warns_and_preserves_values(self) -> None:
        spec = next(task for task in self.catalogue["results"] if task["name"] == "declustering")
        spec["parameters"].pop("additionalProperties", None)
        extra = {"new_option": [1, None, {"enabled": True}]}
        with (
            self.catalogue_response(),
            self.mock_job_client() as submit,
            self.assertLogs("compute.validation", level="WARNING") as logs,
        ):
            await self.client.geostatistics.declustering.run(source=SOURCE_URL, new_parameter=extra)
        self.assertEqual(extra, submit.await_args.kwargs["parameters"]["new_parameter"])
        self.assertEqual(1, len(logs.output))
        self.assertIn("to be submitted as is", logs.output[0])

    async def test_explicitly_open_extra_fields_do_not_warn(self) -> None:
        spec = next(task for task in self.catalogue["results"] if task["name"] == "declustering")
        spec["parameters"]["additionalProperties"] = True
        with (
            self.catalogue_response(),
            self.mock_job_client() as submit,
            self.assertNoLogs("compute.validation", level="WARNING"),
        ):
            await self.client.geostatistics.declustering.run(source=SOURCE_URL, new_parameter=2)
        self.assertEqual(2, submit.await_args.kwargs["parameters"]["new_parameter"])

    async def test_typed_extra_fields_are_validated_without_deep_validation(self) -> None:
        spec = next(task for task in self.catalogue["results"] if task["name"] == "declustering")
        spec["parameters"]["additionalProperties"] = {"type": "integer", "minimum": 1}
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaisesRegex(ParameterValidationError, "new_count"):
                await self.client.geostatistics.declustering.run(source=SOURCE_URL, new_count="invalid")
            submit.assert_not_awaited()
            await self.client.geostatistics.declustering.run(source=SOURCE_URL, new_count=2)
        self.assertEqual(2, submit.await_args.kwargs["parameters"]["new_count"])

    async def test_unsupported_optional_neighborhood_settings_are_rejected_locally(self) -> None:
        from evo.compute import _stubgen
        from test_payload_parity import _search

        captured = next(
            task for task in _stubgen.load_snapshot(_stubgen.DEFAULT_SNAPSHOT_DIR) if task.name == "declustering"
        )
        neighborhood = captured.parameters["properties"]["neighborhood"]
        neighborhood["additionalProperties"] = False
        neighborhood["properties"].pop("max_samples_per_octant", None)
        spec = next(task for task in self.catalogue["results"] if task["name"] == "declustering")
        spec["parameters"]["properties"]["neighborhood"] = neighborhood
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaisesRegex(ParameterValidationError, "neighborhood.*max_samples_per_octant"):
                await self.client.geostatistics.declustering.run(
                    source=SOURCE_URL, neighborhood=_search().model_copy(update={"max_samples_per_octant": 3})
                )
        submit.assert_not_awaited()

    # -- progressive ergonomics (cache-backed, no forced discovery) -------- #

    async def test_signature_is_synthesised_after_caching(self) -> None:
        """Once a task is cached, run() advertises a schema-shaped signature."""
        with self.catalogue_response(), self.mock_job_client():
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)

        run = self.client.geostatistics.declustering.run
        signature = inspect.signature(run)
        self.assertIn("source", signature.parameters)
        self.assertIn("power", signature.parameters)
        self.assertIn("preview", signature.parameters)
        self.assertIs(inspect.Parameter.empty, signature.parameters["source"].default)
        self.assertIsNone(signature.parameters["power"].default)
        self.assertFalse(signature.parameters["preview"].default)
        self.assertEqual("Compute declustering weights for a pointset.", run.__doc__)

    async def test_signature_annotations_follow_the_schema(self) -> None:
        """Parameter annotations are derived from each property's JSON Schema type."""
        with self.catalogue_response(), self.mock_job_client():
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)

        parameters = inspect.signature(self.client.geostatistics.declustering.run).parameters
        self.assertEqual(Literal["cell", "polygon"], parameters["method"].annotation)  # enum
        self.assertEqual(Optional[float], parameters["power"].annotation)  # ["number", "null"]
        self.assertEqual(str, parameters["source"].annotation)  # reference leaf
        self.assertEqual(dict, parameters["settings"].annotation)  # object
        self.assertEqual(Any, parameters["options"].annotation)  # no declared type
        self.assertEqual(bool, parameters["preview"].annotation)

    async def test_signature_is_refreshed_on_a_reused_proxy(self) -> None:
        """A proxy (and its ``run``) held from before the first call still picks up the schema."""
        task_proxy = self.client.geostatistics.declustering
        run = task_proxy.run
        self.assertNotIn("source", inspect.signature(run).parameters)

        with self.catalogue_response(), self.mock_job_client():
            await run(source=SOURCE_URL)

        self.assertIn("source", inspect.signature(run).parameters)
        self.assertIn("source", inspect.signature(task_proxy.run).parameters)

    async def test_dir_lists_cached_topics_and_tasks(self) -> None:
        """After a run, tab-completion surfaces the cached topics/tasks."""
        with self.catalogue_response(), self.mock_job_client():
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)

        self.assertIn("geostatistics", dir(self.client))
        self.assertIn("converter", dir(self.client))
        topic_tasks = dir(self.client.geostatistics)
        self.assertIn("declustering", topic_tasks)
        self.assertIn("normal_score", topic_tasks)
        self.assertIn("kriging", topic_tasks)
        self.assertNotIn("obj_import", topic_tasks)

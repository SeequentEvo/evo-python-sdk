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

"""The override seam and the Kriging runner that uses it.

Three things are worth holding an override to, and they are the three sections below.

It has to be reached. The module path is the only registration, so what matters is that
``client.geostatistics.kriging`` is the hand-written runner while every other task --
and the same task through ``arun`` -- use the same runner.

It has to earn its place. An override exists to do what the generic path cannot, so each
addition is pinned by a test that fails if the check or the typed result were dropped.

And it has to send the same job. That is the part an override can quietly get wrong, so the
last section runs one set of inputs through all three paths -- the hand-written task runner,
the override, and the generic engine -- and requires ``JobClient.submit`` to have been
handed the same payload by each, equal in value and in concrete type. The
:mod:`test_payload_parity` harness already makes exactly that comparison for the runner and
the engine; this file adds the override as a third path through the same machinery rather
than restating it.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock
from uuid import UUID

from evo.compute import ComputeClient, JobClient, ParameterValidationError, TaskJob, TaskResource, _stubgen
from evo.compute.discovery import DiscoveryClient
from evo.compute.engine import _TaskProxy
from evo.compute.overrides import load_override
from evo.compute.overrides.geostatistics.kriging import KrigingOverride
from evo.compute.tasks import CreateAttribute, Source, Target, UpdateAttribute
from evo.compute.tasks.common import Filter, FilterCondition
from evo.compute.tasks.geostatistics.kriging import (
    BlockDiscretization,
    KrigingMethod,
    KrigingResult,
    KrigingRunner,
)
from test_payload_parity import (
    GRADE_ATTRIBUTE,
    POINTSET_URL,
    TARGET_URL,
    VARIOGRAM_URL,
    PayloadParityTestCase,
    _capture_submit,
    _pending_attribute,
    _search,
    _SubmitCaptured,
    payload_differences,
    task_spec,
)

TASK_RESULT = {
    "message": "Kriging complete.",
    "target": {
        "reference": TARGET_URL,
        "name": "Block model",
        "schema_id": "/objects/block-model/1.2.0/block-model.schema.json",
        "attribute": {"reference": "attributes[?name=='kriged_grade']", "name": "kriged_grade"},
    },
}


def kriging_catalogue(**overrides: Any) -> list[TaskResource]:
    """The kriging schema as the catalogue advertises it.

    The runner, the override and the catalogue all name this task ``kriging``, so the
    comparison is about the payload rather than about which name each path happens to use.
    """
    return [task_spec(KrigingRunner).model_copy(update=overrides)]


def published_catalogue(node: str | None = None, rule: Any = None) -> list[TaskResource]:
    """Kriging as the captured catalogue publishes it, with the extra-field rule at ``node`` set to ``rule``.

    ``node`` is a dotted path through ``properties``; a ``rule`` of ``None`` removes
    ``additionalProperties`` there, leaving the rule unspecified.
    """
    path = _stubgen.DEFAULT_SNAPSHOT_DIR / "geostatistics" / "kriging.json"
    task = TaskResource.model_validate(
        {**json.loads(path.read_text(encoding="utf-8")), "topic": "geostatistics", "name": "kriging"}
    )
    if node:
        schema = task.parameters
        for part in node.split("."):
            schema = schema["properties"][part]
        if rule is None:
            schema.pop("additionalProperties", None)
        else:
            schema["additionalProperties"] = rule
    return [task]


class OverrideTestCase(PayloadParityTestCase):
    """A client whose discovery is mocked, and the inputs every case here starts from."""

    def setUp(self) -> None:
        super().setUp()
        self.catalogue = mock.patch.object(
            DiscoveryClient, "list_tasks", mock.AsyncMock(return_value=kriging_catalogue())
        )

    def inputs(self, **overrides: Any) -> dict[str, Any]:
        """One set of inputs, named as the override names them."""
        inputs: dict[str, Any] = dict(
            source=Source(object=POINTSET_URL, attribute=GRADE_ATTRIBUTE),
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="kriged_grade")),
            variogram=VARIOGRAM_URL,
            neighborhood=_search(),
            kriging_method=KrigingMethod.ORDINARY,
        )
        inputs.update(overrides)
        return inputs

    @contextmanager
    def completed_job(self, results: dict) -> Iterator[mock.AsyncMock]:
        """Let a submission through and hand back ``results``, so the typed result gets built."""
        job = mock.Mock(wait_for_results=mock.AsyncMock(return_value=results))
        submit = mock.AsyncMock(return_value=job)
        with self.catalogue, mock.patch("evo.compute.engine.JobClient") as job_client:
            job_client.submit = submit
            yield submit

    async def override_payload(self, **inputs: Any) -> dict[str, Any]:
        """The payload the override submits for ``inputs``."""
        with self.catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaises(_SubmitCaptured):
                await self.client.geostatistics.kriging.run(**inputs)
        return submit.await_args.kwargs["parameters"]


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


class TestTheSeam(OverrideTestCase):
    def test_an_overridden_task_is_served_by_its_own_runner(self) -> None:
        """The module at ``overrides/geostatistics/kriging.py`` claims the task."""
        self.assertIsInstance(self.client.geostatistics.kriging, KrigingOverride)

    async def test_published_kriging_spelling_is_used_on_the_wire(self) -> None:
        catalogue = mock.AsyncMock(return_value=published_catalogue())
        with (
            mock.patch.object(DiscoveryClient, "list_tasks", catalogue),
            _capture_submit("evo.compute.engine") as submit,
        ):
            with self.assertRaises(_SubmitCaptured):
                await self.client.geostatistics.kriging.run(
                    **self.inputs(block_discretization=BlockDiscretization(nx=2, ny=3, nz=4))
                )

        payload = submit.await_args.kwargs["parameters"]
        self.assertEqual({"nx": 2, "ny": 3, "nz": 4}, payload["block_discretisation"])
        self.assertNotIn("block_discretization", payload)
        catalogue.assert_awaited_once()

    def test_the_override_remains_async_without_a_blocking_wrapper(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(self.client.geostatistics.kriging.run))

    def test_the_override_publishes_typed_async_submission(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(self.client.geostatistics.kriging.submit))
        run = inspect.signature(self.client.geostatistics.kriging.run)
        submit = inspect.signature(self.client.geostatistics.kriging.submit)
        self.assertEqual(run.parameters, submit.parameters)

    def test_every_other_task_stays_generic(self) -> None:
        """An override is one task opting out, not a change to how tasks are reached."""
        self.assertIsInstance(self.client.geostatistics.declustering, _TaskProxy)
        self.assertIsInstance(self.client.geostatistics.normal_score, _TaskProxy)
        self.assertIsInstance(self.client.converter.obj_import, _TaskProxy)

    def test_reaching_a_task_still_costs_no_discovery(self) -> None:
        """``bind`` is handed the client, not a spec, so the seam keeps attribute access free."""
        list_tasks = mock.AsyncMock(return_value=kriging_catalogue())
        with mock.patch.object(DiscoveryClient, "list_tasks", list_tasks):
            _ = self.client.geostatistics.kriging
        list_tasks.assert_not_awaited()

    def test_a_task_is_keyed_as_the_caller_spells_it(self) -> None:
        """A hyphenated task is reached with underscores, so that is where its module lives."""
        self.assertIsNotNone(load_override("geostatistics", "kriging"))
        self.assertIsNone(load_override("geostatistics", "normal-score"))
        self.assertIsNone(load_override("geostatistics", "declustering"))
        self.assertIsNone(load_override("no-such-topic", "no_such_task"))

    def test_a_module_without_bind_does_not_claim_its_task(self) -> None:
        """``bind`` is the contract; a module that does not expose it is not an override."""
        with mock.patch("importlib.import_module", return_value=SimpleNamespace()):
            self.assertIsNone(load_override.__wrapped__("geostatistics", "declustering"))

    def test_a_broken_override_is_not_silently_skipped(self) -> None:
        """An override whose own imports fail is a bug, not an absent override."""
        error = ModuleNotFoundError("No module named 'somethingelse'", name="somethingelse")
        with mock.patch("importlib.import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError):
                load_override.__wrapped__("geostatistics", "kriging")

    async def test_arun_cannot_bypass_the_override_checks(self) -> None:
        with self.catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaises(ParameterValidationError):
                await self.client.arun(
                    "geostatistics",
                    "kriging",
                    self.inputs(neighborhood=_search(min_samples=99)),
                )
        submit.assert_not_awaited()

    async def test_asubmit_cannot_bypass_the_override_checks(self) -> None:
        with self.catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaisesRegex(ParameterValidationError, r"kriging\.submit\(\)"):
                await self.client.asubmit("geostatistics", "kriging", self.inputs(neighborhood=_search(min_samples=99)))
        submit.assert_not_awaited()

    async def test_run_and_submit_keep_the_same_native_result_and_payload(self) -> None:
        with self.completed_job(TASK_RESULT) as submit:
            from_run = await self.client.geostatistics.kriging.run(**self.inputs())
            run_payload = submit.await_args.kwargs["parameters"]
        for named in (False, True):
            with self.subTest(named=named), self.completed_job(TASK_RESULT) as submit:
                if named:
                    job = await self.client.asubmit("geostatistics", "kriging", self.inputs())
                else:
                    job = await self.client.geostatistics.kriging.submit(**self.inputs())
                self.assertIsInstance(job, TaskJob)
                submit.return_value.wait_for_results.assert_not_awaited()
                from_submit = await job.results()
                self.assertIsInstance(from_submit, KrigingResult)
                self.assertEqual(from_run.target_name, from_submit.target_name)
                self.assertEqual(from_run.attribute_name, from_submit.attribute_name)
                self.assertTrue(inspect.iscoroutinefunction(from_submit.to_dataframe))
                self.assertTrue(inspect.iscoroutinefunction(from_submit.get_target_object))
                self.assertEqual(run_payload, submit.await_args.kwargs["parameters"])
                self.assertIs(self.context, from_submit._context)

    async def test_restored_kriging_job_keeps_native_result_conversion(self) -> None:
        self.context._connector.base_url = "https://unittest.localhost/"
        job_id = UUID(int=1234)
        url = (
            f"{self.context._connector.base_url}compute/orgs/{self.context._org_id}/"
            f"geostatistics/kriging/{job_id}/status"
        )
        with (
            self.catalogue,
            mock.patch.object(JobClient, "wait_for_results", mock.AsyncMock(return_value=TASK_RESULT)),
        ):
            job = await TaskJob.from_url(self.context, url)
            result = await job.results()
        self.assertEqual(job_id, job.id)
        self.assertIsInstance(result, KrigingResult)
        self.assertEqual("kriged_grade", result.attribute_name)

    async def test_discretization_uses_the_spelling_in_the_live_catalogue(self) -> None:
        for spelling in ("block_discretization", "block_discretisation"):
            spec = kriging_catalogue()[0]
            properties = spec.parameters["properties"]
            properties[spelling] = properties.pop("block_discretization")
            with (
                self.subTest(spelling=spelling),
                mock.patch.object(DiscoveryClient, "list_tasks", mock.AsyncMock(return_value=[spec])) as discover,
                _capture_submit("evo.compute.engine") as submit,
            ):
                with self.assertRaises(_SubmitCaptured):
                    await self.client.geostatistics.kriging.submit(
                        **self.inputs(block_discretization=BlockDiscretization(nx=2, ny=3, nz=4))
                    )
            payload = submit.await_args.kwargs["parameters"]
            self.assertEqual({"nx": 2, "ny": 3, "nz": 4}, payload[spelling])
            discover.assert_awaited_once()

    async def test_discretization_cannot_be_supplied_under_both_spellings(self) -> None:
        with self.catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaisesRegex(ParameterValidationError, "supplied both"):
                await self.client.asubmit(
                    "geostatistics",
                    "kriging",
                    self.inputs(
                        block_discretization=BlockDiscretization(nx=2, ny=2, nz=2),
                        block_discretisation={"nx": 3, "ny": 3, "nz": 3},
                    ),
                )
        submit.assert_not_awaited()


# --------------------------------------------------------------------------- #
# What the override adds
# --------------------------------------------------------------------------- #


class TestWhatTheOverrideAdds(OverrideTestCase):
    async def test_named_extra_fields_are_preserved_when_the_schema_leaves_the_rule_unspecified(self) -> None:
        catalogue = kriging_catalogue()
        catalogue[0].parameters.pop("additionalProperties", None)
        self.catalogue = mock.patch.object(DiscoveryClient, "list_tasks", mock.AsyncMock(return_value=catalogue))
        with self.completed_job(TASK_RESULT) as submit, self.assertLogs("compute.validation", level="WARNING") as logs:
            await self.client.arun("geostatistics", "kriging", {**self.inputs(), "new_input": {"enabled": True}})
        self.assertEqual({"enabled": True}, submit.await_args.kwargs["parameters"]["new_input"])
        self.assertTrue(any("'new_input'" in message for message in logs.output))

    async def test_named_extra_fields_are_rejected_when_the_schema_closes_the_object(self) -> None:
        with self.completed_job(TASK_RESULT) as submit:
            with self.assertRaisesRegex(ParameterValidationError, "new_input"):
                await self.client.arun("geostatistics", "kriging", {**self.inputs(), "new_input": True})
        submit.assert_not_awaited()

    async def test_arun_returns_the_same_native_result_as_run(self) -> None:
        parameters = KrigingRunner.params_type(
            source=self.inputs()["source"],
            target=self.inputs()["target"],
            variogram=VARIOGRAM_URL,
            neighborhood=_search(),
        )
        with self.completed_job(TASK_RESULT) as submit:
            result = await self.client.arun(
                "geostatistics",
                "kriging",
                parameters.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
        self.assertIsInstance(result, KrigingResult)
        self.assertEqual(
            parameters.model_dump(mode="json", by_alias=True, exclude_none=True)["neighborhood"],
            submit.await_args.kwargs["parameters"]["neighborhood"],
        )

    def test_neighborhood_reads_its_serialized_shape_without_losing_rotation(self) -> None:
        neighborhood = _search().model_copy(update={"max_samples_per_octant": 3})
        restored = type(neighborhood).model_validate(neighborhood.model_dump(mode="json"))
        self.assertEqual(neighborhood, restored)

    async def test_valid_parameters_are_accepted(self) -> None:
        """Anti-vacuous guard: the checks below must reject their own case, not every case."""
        with self.completed_job(TASK_RESULT) as submit:
            await self.client.geostatistics.kriging.run(**self.inputs())
        submit.assert_awaited_once()

    async def test_a_search_that_can_never_be_satisfied_is_refused(self) -> None:
        """Each bound is legal on its own; the schema has no way to say they must agree."""
        with self.catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaises(ParameterValidationError) as caught:
                await self.client.geostatistics.kriging.run(**self.inputs(neighborhood=_search(min_samples=99)))

        self.assertIn("min_samples (99) exceeds max_samples (20)", str(caught.exception))
        self.assertEqual("geostatistics.kriging", caught.exception.task)
        submit.assert_not_awaited()

    async def test_estimating_an_attribute_from_itself_is_refused(self) -> None:
        """Source and target are each valid references; pointing them at one attribute is not."""
        target = Target(object=POINTSET_URL, attribute=UpdateAttribute(reference=GRADE_ATTRIBUTE))
        with self.catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaises(ParameterValidationError) as caught:
                await self.client.geostatistics.kriging.run(**self.inputs(target=target))

        self.assertIn("also the source attribute", str(caught.exception))
        submit.assert_not_awaited()

    async def test_updating_a_different_attribute_on_the_source_object_is_allowed(self) -> None:
        """The check is attribute-level: writing an estimate back onto the source object is fine."""
        target = Target(object=POINTSET_URL, attribute=UpdateAttribute(reference="attributes[?name=='estimate']"))
        with self.completed_job(TASK_RESULT) as submit:
            await self.client.geostatistics.kriging.run(**self.inputs(target=target))
        submit.assert_awaited_once()

    async def test_a_versioned_source_url_is_still_the_same_object(self) -> None:
        """``?version=`` names a moment in an object's history, not a different object."""
        source = Source(object=f"{POINTSET_URL}?version=42", attribute=GRADE_ATTRIBUTE)
        target = Target(object=POINTSET_URL, attribute=UpdateAttribute(reference=GRADE_ATTRIBUTE))
        with self.catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaises(ParameterValidationError):
                await self.client.geostatistics.kriging.run(**self.inputs(source=source, target=target))
        submit.assert_not_awaited()

    async def test_the_same_object_written_a_different_way_is_still_the_same_object(self) -> None:
        """``ObjectReference`` keeps the caller's spelling, so the guard cannot compare text.

        The reference pattern is case-insensitive about the UUIDs it matches, and a fragment
        never reaches the path at all, so both of these are the source object under another name.
        """
        object_id = "0123abcd-ef00-0000-0000-00000000beef"
        canonical = f"{POINTSET_URL.rsplit('/', 1)[0]}/{object_id}"
        for spelling in (canonical.replace(object_id, object_id.upper()), f"{canonical}#section"):
            with self.subTest(spelling=spelling):
                source = Source(object=spelling, attribute=GRADE_ATTRIBUTE)
                target = Target(object=canonical, attribute=UpdateAttribute(reference=GRADE_ATTRIBUTE))
                with self.catalogue, _capture_submit("evo.compute.engine") as submit:
                    with self.assertRaises(ParameterValidationError):
                        await self.client.geostatistics.kriging.run(**self.inputs(source=source, target=target))
                submit.assert_not_awaited()

    async def test_the_same_attribute_on_a_different_object_is_allowed(self) -> None:
        """The other direction: the same expression against another object is not a collision."""
        target = Target(object=TARGET_URL, attribute=UpdateAttribute(reference=GRADE_ATTRIBUTE))
        with self.completed_job(TASK_RESULT) as submit:
            await self.client.geostatistics.kriging.run(**self.inputs(target=target))
        submit.assert_awaited_once()

    async def test_the_result_is_the_typed_one(self) -> None:
        """A hand-curated result with helpers the generic ``TaskResult`` could not synthesise."""
        with self.completed_job(TASK_RESULT):
            result = await self.client.geostatistics.kriging.run(**self.inputs())

        self.assertIsInstance(result, KrigingResult)
        self.assertEqual("Kriging complete.", result.message)
        self.assertEqual("Block model", result.target_name)
        self.assertEqual("kriged_grade", result.attribute_name)
        self.assertEqual("block-model", result.schema.sub_classification)

    def test_the_signature_is_the_hand_written_one(self) -> None:
        """Not synthesised from the schema: the arguments read as the typed SDK names them."""
        parameters = inspect.signature(self.client.geostatistics.kriging.run).parameters
        self.assertIn("neighborhood", parameters)
        self.assertIn("kriging_method", parameters)
        self.assertNotIn("search", parameters)
        self.assertNotIn("method", parameters)
        self.assertIn("source_filter", parameters)  # folded into ``source``; not a task parameter
        self.assertNotIn("self", parameters)


class TestFoldedInputsFollowTheExtraFieldRules(OverrideTestCase):
    """Filters and diagnostics are folded into ``source`` and ``target``, so their extra fields must survive the fold.

    The typed models drop a key they do not declare, which would leave the schema's rule for
    that object nothing to act on. These run against the captured catalogue, with the rule of
    the object that carries the folded input switched in turn.
    """

    CONDITION = {"type": "condition", "attribute": GRADE_ATTRIBUTE, "operator": "greater_than", "threshold": 0.5}

    RULES = {
        "closed": (False, "rejected"),
        "open": (True, "forwarded"),
        "typed, matching value": ({"type": "integer"}, "forwarded"),
        "typed, mismatching value": ({"type": "string"}, "rejected"),
        "unspecified": (None, "warned"),
    }

    ENTRY_POINTS = {"run": "run", "submit": "submit", "arun": "run", "asubmit": "submit"}

    def setUp(self) -> None:
        super().setUp()
        self.client = ComputeClient(self.context, check_schemas=False)

    def folded_inputs(self) -> dict[str, tuple[str, dict[str, Any]]]:
        """Each way of supplying a folded input, by the object whose rule applies to it."""
        target_condition = {"type": "condition", "attribute": "attributes[?name=='domain']", "operator": "in"}
        source_frame = {"object": POINTSET_URL, "attribute": GRADE_ATTRIBUTE}
        return {
            "source_filter": ("source.filter", {"source_filter": {"where": self.CONDITION, "extra": 7}}),
            "source.filter": (
                "source.filter",
                {"source": {**source_frame, "filter": {"where": self.CONDITION, "extra": 7}}},
            ),
            "target_filter": (
                "target.filter",
                {"target_filter": {"where": {**target_condition, "values": ["LMS1"]}, "extra": 7}},
            ),
            "diagnostics": ("target.diagnostics", {"diagnostics": {"kriging_variance": True, "extra": 7}}),
        }

    async def call(self, entry_point: str, inputs: dict[str, Any]) -> None:
        if entry_point in ("run", "submit"):
            await getattr(self.client.geostatistics.kriging, entry_point)(**inputs)
        else:
            await getattr(self.client, entry_point)("geostatistics", "kriging", inputs)

    async def test_an_extra_field_in_a_folded_input_follows_its_objects_rule(self) -> None:
        for name, (node, overrides) in self.folded_inputs().items():
            for rule_name, (rule, outcome) in self.RULES.items():
                for entry_point, operation in self.ENTRY_POINTS.items():
                    with self.subTest(input=name, rule=rule_name, entry_point=entry_point):
                        catalogue = mock.AsyncMock(return_value=published_catalogue(node, rule))
                        with mock.patch.object(DiscoveryClient, "list_tasks", catalogue):
                            with _capture_submit("evo.compute.engine") as submit:
                                if outcome == "rejected":
                                    with self.assertRaises(ParameterValidationError) as caught:
                                        await self.call(entry_point, self.inputs(**overrides))
                                else:
                                    logs = (
                                        self.assertLogs("compute.validation", level="WARNING")
                                        if outcome == "warned"
                                        else self.assertNoLogs("compute.validation", level="WARNING")
                                    )
                                    with logs as captured, self.assertRaises(_SubmitCaptured):
                                        await self.call(entry_point, self.inputs(**overrides))
                        if outcome == "rejected":
                            self.assertTrue(
                                str(caught.exception).startswith(f"geostatistics.kriging.{operation}(): "),
                                caught.exception,
                            )
                            self.assertTrue(
                                any(node in error and "extra" in error for error in caught.exception.errors),
                                caught.exception.errors,
                            )
                            submit.assert_not_awaited()
                            continue
                        if outcome == "warned":
                            self.assertTrue(any(f"at {node}: 'extra'" in line for line in captured.output))
                        wire = submit.await_args.kwargs["parameters"]
                        for part in node.split("."):
                            wire = wire[part]
                        self.assertEqual(7, wire["extra"])

    async def test_an_extra_field_deep_inside_a_filter_is_kept_where_it_was_written(self) -> None:
        """The captured catalogue closes every filter node, so the field is refused at its own location."""
        cases = {
            "source.filter.where": {**self.CONDITION, "extra": 7},
            "source.filter.where.filters.1": {
                "type": "all_of",
                "filters": [self.CONDITION, {**self.CONDITION, "extra": 7}],
            },
        }
        for location, where in cases.items():
            with self.subTest(location=location):
                catalogue = mock.AsyncMock(return_value=published_catalogue())
                with mock.patch.object(DiscoveryClient, "list_tasks", catalogue):
                    with _capture_submit("evo.compute.engine") as submit:
                        with self.assertRaises(ParameterValidationError) as caught:
                            await self.client.arun(
                                "geostatistics", "kriging", self.inputs(source_filter={"where": where})
                            )
                self.assertEqual([f"{location}: unexpected parameter(s): 'extra'"], caught.exception.errors)
                submit.assert_not_awaited()

    async def test_fields_the_typed_models_read_are_not_taken_for_extras(self) -> None:
        """``ranges`` is renamed on the way out and ``False`` turns a diagnostic off; neither is unknown."""
        neighborhood = {
            "ellipsoid": {
                "ranges": {"major": 200.0, "semi_major": 150.0, "minor": 100.0},
                "rotation": {"dip_azimuth": 45.0, "dip": 10.0, "pitch": 5.0},
            },
            "max_samples": 20,
        }
        diagnostics = {"kriging_variance": True, "kriging_mean": False}
        catalogue = mock.AsyncMock(return_value=published_catalogue())
        with (
            mock.patch.object(DiscoveryClient, "list_tasks", catalogue),
            _capture_submit("evo.compute.engine") as submit,
        ):
            with self.assertRaises(_SubmitCaptured):
                await self.client.arun(
                    "geostatistics", "kriging", self.inputs(neighborhood=neighborhood, diagnostics=diagnostics)
                )
        payload = submit.await_args.kwargs["parameters"]
        self.assertEqual({"ellipsoid_ranges", "rotation"}, set(payload["neighborhood"]["ellipsoid"]))
        self.assertEqual({"kriging_variance"}, set(payload["target"]["diagnostics"]))


# --------------------------------------------------------------------------- #
# Payload fidelity
# --------------------------------------------------------------------------- #


class TestPayloadFidelity(OverrideTestCase):
    """One set of inputs, three paths, one payload."""

    async def assertThreeWayParity(self, **inputs: Any) -> None:
        """Require the runner, the override and the generic engine to submit the same payload."""
        runner_payload = await self.runner_payload(KrigingRunner, KrigingRunner.params_type(**inputs))
        candidates = {
            "override": await self.override_payload(**inputs),
            "engine": await self.engine_payload(KrigingRunner, **inputs),
        }
        for label, payload in candidates.items():
            if differences := payload_differences(runner_payload, payload):
                self.fail(f"{label} payload diverges from the runner payload:\n  " + "\n  ".join(differences))

    async def test_the_everyday_call(self) -> None:
        """Source, target, variogram and a search, with the method left at ordinary."""
        await self.assertThreeWayParity(**self.inputs())

    async def test_a_tagged_method_object(self) -> None:
        """Simple kriging carries a mean, and the tag that says which branch of the union it is."""
        await self.assertThreeWayParity(**self.inputs(kriging_method=KrigingMethod.simple(mean=12.5)))

    async def test_a_typed_attribute_target_becomes_a_create_operation(self) -> None:
        """The override takes the typed inputs the runner takes, and expands them the same way."""
        await self.assertThreeWayParity(**self.inputs(target=_pending_attribute("kriged_grade", TARGET_URL)))

    async def test_a_folded_filter_lands_where_the_wire_carries_it(self) -> None:
        """``source_filter`` is a model input the serializer writes into ``source``.

        It is an argument the generic engine does not have -- the catalogue never advertises
        it -- which is one of the things a hand-written surface is for.
        """
        source_filter = Filter(where=FilterCondition(attribute=GRADE_ATTRIBUTE, operator="greater_than", threshold=0.5))
        payload = await self.override_payload(**self.inputs(source_filter=source_filter))
        self.assertNotIn("source_filter", payload)
        self.assertEqual(source_filter.model_dump(), payload["source"]["filter"])

    async def test_an_unset_optional_is_not_sent(self) -> None:
        """The override sends what it was given, so the platform's own defaults still apply."""
        payload = await self.override_payload(**self.inputs())
        self.assertNotIn("block_discretization", payload)

    async def test_preview_requires_explicit_opt_in(self) -> None:
        catalogue = kriging_catalogue(feature_flag="geostatistics-preview")
        with mock.patch.object(DiscoveryClient, "list_tasks", mock.AsyncMock(return_value=catalogue)):
            with _capture_submit("evo.compute.engine") as submit:
                with self.assertRaises(_SubmitCaptured):
                    await self.client.geostatistics.kriging.run(**self.inputs())
                self.assertFalse(submit.await_args.kwargs["preview"])

                with self.assertRaises(_SubmitCaptured):
                    await self.client.geostatistics.kriging.run(**self.inputs(), preview=True)
                self.assertTrue(submit.await_args.kwargs["preview"])
                self.assertNotIn("preview", submit.await_args.kwargs["parameters"])

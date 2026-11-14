"""Job lifecycle and shared execution/submission regression tests."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase, mock
from uuid import UUID

from evo.common import IContext
from evo.common.interfaces import IFeedback
from evo.common.test_tools import ORG as TEST_ORG
from evo.common.test_tools import MockResponse, TestWithConnector
from evo.common.utils import NoFeedback, Retry, get_header_metadata

from data import load_test_data
from evo.compute import ComputeClient, JobClient, JobStatusEnum, ParameterValidationError, TaskJob, TaskResult
from evo.compute.engine import _RUN_FEEDBACK_NOTE
from test_engine import SOURCE_URL, ComputeClientTestCase, FakeContext, _task

TEST_JOB_ID = UUID(int=4321)
RESULT_PAYLOAD = {
    "message": "done",
    "target": {"reference": "https://example.com/objects/1", "name": "grid"},
}
RESULT_SCHEMA = {
    "type": "object",
    "properties": {"target": {"type": "object", "output": "geoscience-object"}},
}


class TestTaskJob(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.backend = mock.MagicMock(spec=JobClient)
        self.backend.id = TEST_JOB_ID
        self.backend.url = "https://example.com/compute/jobs/1"
        self.backend.topic = "geostatistics"
        self.backend.task = "declustering"
        self.backend.wait_for_results.return_value = RESULT_PAYLOAD
        self.context = mock.MagicMock(spec=IContext)

        async def hydrate(payload: dict) -> TaskResult:
            return TaskResult(payload, RESULT_SCHEMA, self.context)

        self.hydrate = hydrate
        self.job = TaskJob(self.backend, self.hydrate)

    def test_identifiers_are_preserved(self) -> None:
        self.assertEqual(TEST_JOB_ID, self.job.id)
        self.assertEqual(self.backend.url, self.job.url)
        self.assertEqual("geostatistics", self.job.topic)
        self.assertEqual("declustering", self.job.task)
        self.assertIn(str(TEST_JOB_ID), repr(self.job))

    async def test_status_does_not_wait_for_results(self) -> None:
        self.backend.get_status.return_value = mock.sentinel.progress
        self.assertIs(mock.sentinel.progress, await self.job.status())
        self.backend.get_status.assert_awaited_once_with()
        self.backend.wait_for_results.assert_not_awaited()

    async def test_cancel_delegates_without_waiting(self) -> None:
        await self.job.cancel()
        self.backend.cancel.assert_awaited_once_with()
        self.backend.wait_for_results.assert_not_awaited()

    async def test_results_are_hydrated_with_async_loaders(self) -> None:
        result = await self.job.results()
        self.assertIsInstance(result, TaskResult)
        self.assertEqual(RESULT_PAYLOAD, result)
        self.assertEqual("grid", result.target.name)
        self.assertTrue(inspect.iscoroutinefunction(result.target.load))
        self.backend.wait_for_results.assert_awaited_once_with(polling_interval_seconds=5.0, retry=None, fb=NoFeedback)

    async def test_results_inherit_client_feedback(self) -> None:
        feedback = mock.MagicMock(spec=IFeedback)
        job = TaskJob(self.backend, self.hydrate, feedback)
        await job.results()
        self.assertIs(feedback, self.backend.wait_for_results.await_args.kwargs["fb"])

    async def test_results_forward_feedback_polling_and_retry_overrides(self) -> None:
        feedback = mock.MagicMock(spec=IFeedback)
        retry = mock.MagicMock(spec=Retry)
        await self.job.results(fb=feedback, polling_interval_seconds=0.0, retry=retry)
        self.backend.wait_for_results.assert_awaited_once_with(polling_interval_seconds=0.0, retry=retry, fb=feedback)

    async def test_map_awaits_native_result_conversion(self) -> None:
        convert = mock.AsyncMock(return_value=mock.sentinel.native_result)
        mapped = self.job.map(convert)
        self.backend.wait_for_results.assert_not_awaited()
        self.assertIs(mock.sentinel.native_result, await mapped.results())
        convert.assert_awaited_once()
        hydrated = convert.await_args.args[0]
        self.assertIsInstance(hydrated, TaskResult)
        self.assertEqual("grid", hydrated.target.name)
        self.assertEqual(self.job.url, mapped.url)
        await mapped.status()
        await mapped.cancel()
        self.backend.get_status.assert_awaited_once()
        self.backend.cancel.assert_awaited_once()

    async def test_map_preserves_feedback_and_wait_options(self) -> None:
        feedback = mock.MagicMock(spec=IFeedback)
        retry = mock.MagicMock(spec=Retry)
        self.job._fb = feedback
        mapped = self.job.map(mock.AsyncMock(return_value=123))
        self.assertEqual(123, await mapped.results(polling_interval_seconds=1.25, retry=retry))
        self.backend.wait_for_results.assert_awaited_once_with(polling_interval_seconds=1.25, retry=retry, fb=feedback)

    async def test_conversion_and_backend_errors_propagate(self) -> None:
        convert = mock.AsyncMock(side_effect=ValueError("invalid native result"))
        with self.assertRaisesRegex(ValueError, "invalid native result"):
            await self.job.map(convert).results()
        convert.reset_mock()
        self.backend.wait_for_results.side_effect = RuntimeError("job failed")
        with self.assertRaisesRegex(RuntimeError, "job failed"):
            await self.job.map(convert).results()
        convert.assert_not_awaited()

    async def test_maps_can_be_composed(self) -> None:
        first = mock.AsyncMock(return_value="native")
        second = mock.AsyncMock(return_value=6)
        self.assertEqual(6, await self.job.map(first).map(second).results())
        second.assert_awaited_once_with("native")


class RecordingFeedback(IFeedback):
    def __init__(self) -> None:
        self.reports: list[tuple[float, str | None]] = []

    def progress(self, progress: float, message: str | None = None) -> None:
        self.reports.append((progress, message))


class TestSubmitMatchesRun(ComputeClientTestCase):
    async def test_submission_does_not_wait(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            job = await self.client.geostatistics.declustering.submit(source=SOURCE_URL)
        self.assertIsInstance(job, TaskJob)
        submit.assert_awaited_once()
        submit.return_value.wait_for_results.assert_not_awaited()

    async def test_run_and_submit_have_identical_payloads(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.declustering.run(source=SOURCE_URL, power=2.0)
        ran = submit.await_args.kwargs
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.declustering.submit(source=SOURCE_URL, power=2.0)
        self.assertEqual(ran, submit.await_args.kwargs)

    async def test_both_paths_hydrate_the_same_results(self) -> None:
        with self.catalogue_response(), self.mock_job_client(results=RESULT_PAYLOAD):
            from_run = await self.client.geostatistics.declustering.run(source=SOURCE_URL)
        with self.catalogue_response(), self.mock_job_client(results=RESULT_PAYLOAD):
            job = await self.client.geostatistics.declustering.submit(source=SOURCE_URL)
            from_job = await job.results()
        self.assertEqual(from_run, from_job)
        self.assertIs(type(from_run), type(from_job))
        self.assertEqual("grid", from_job.target.name)

    async def test_invalid_parameters_do_not_submit(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaisesRegex(ParameterValidationError, "source"):
                await self.client.geostatistics.declustering.submit(sourse=SOURCE_URL)
        submit.assert_not_awaited()

    async def test_binding_errors_name_the_operation(self) -> None:
        for operation in ("run", "submit"):
            with self.subTest(operation=operation), self.catalogue_response(), self.mock_job_client():
                with self.assertRaises(ParameterValidationError) as caught:
                    await getattr(self.client.geostatistics.declustering, operation)(nonsense=1)
                self.assertIn(f"geostatistics.declustering.{operation}(): ", str(caught.exception))

    async def test_validation_and_resolution_errors_name_the_operation(self) -> None:
        schema = next(task for task in self.catalogue["results"] if task["name"] == "declustering")["parameters"]
        schema["additionalProperties"] = {"type": "integer"}
        schema["properties"]["settings"] = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "additionalProperties": False,
        }
        client = ComputeClient(self.context, deep_validation=True)
        task = client.geostatistics.declustering
        calls = {
            "run": ("run", lambda inputs: task.run(**inputs)),
            "submit": ("submit", lambda inputs: task.submit(**inputs)),
            "arun": ("run", lambda inputs: client.arun("geostatistics", "declustering", inputs)),
            "asubmit": ("submit", lambda inputs: client.asubmit("geostatistics", "declustering", inputs)),
        }
        problems = {
            "missing": ({"source": None}, "missing required parameter(s): source"),
            "unresolvable": ({"source": 12345}, "source: "),
            "closed": (
                {"source": SOURCE_URL, "settings": {"count": 1, "extra": True}},
                "settings: unexpected parameter(s): 'extra'",
            ),
            "typed extra": ({"source": SOURCE_URL, "future_parameter": "many"}, "future_parameter: "),
            "deep": ({"source": SOURCE_URL, "power": "high"}, "power: "),
        }
        for entry_point, (operation, call) in calls.items():
            for problem, (inputs, detail) in problems.items():
                with (
                    self.subTest(entry_point=entry_point, problem=problem),
                    self.catalogue_response(),
                    self.mock_job_client() as submit,
                ):
                    with self.assertRaises(ParameterValidationError) as caught:
                        await call(inputs)
                    message = str(caught.exception)
                    self.assertTrue(message.startswith(f"geostatistics.declustering.{operation}(): "), message)
                    self.assertIn(detail, message)
                    submit.assert_not_awaited()

    async def test_unspecified_extra_fields_warn_and_are_preserved_in_both_modes(self) -> None:
        spec = next(task for task in self.catalogue["results"] if task["name"] == "declustering")
        spec["parameters"].pop("additionalProperties", None)
        spec["parameters"]["properties"]["settings"] = {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1}},
            "required": ["count"],
        }
        parameters = {
            "source": SOURCE_URL,
            "settings": {"count": 2, "new_option": False, 0: None},
            "future-parameter": {"unset": None, "secret": "do-not-log-this-value"},
        }
        for validate, deep in ((True, False), (True, True), (False, False)):
            client = ComputeClient(self.context, validate=validate, deep_validation=deep)
            for operation in ("run", "submit"):
                with self.subTest(validate=validate, deep=deep, operation=operation):
                    with (
                        self.catalogue_response(),
                        self.mock_job_client() as submit,
                        self.assertLogs("compute.validation", level="WARNING") as logged,
                    ):
                        await getattr(client.geostatistics.declustering, operation)(**parameters)
                    self.assertEqual(parameters, submit.await_args.kwargs["parameters"])
                    self.assertEqual(2, len(logged.output))
                    messages = "\n".join(logged.output)
                    self.assertIn("future-parameter", messages)
                    self.assertIn("settings", messages)
                    self.assertIn("new_option", messages)
                    self.assertNotIn("do-not-log-this-value", messages)
                    submit.assert_awaited_once()

    async def test_closed_schemas_reject_extra_fields_for_run_and_submit(self) -> None:
        spec = next(task for task in self.catalogue["results"] if task["name"] == "declustering")
        spec["parameters"]["additionalProperties"] = False
        for validate in (False, True):
            client = ComputeClient(self.context, validate=validate)
            for operation in ("run", "submit"):
                with self.subTest(validate=validate, operation=operation):
                    with self.catalogue_response(), self.mock_job_client() as submit:
                        with self.assertRaisesRegex(
                            ParameterValidationError,
                            rf"^geostatistics\.declustering\.{operation}\(\): .*'future_parameter'",
                        ):
                            await getattr(client.geostatistics.declustering, operation)(
                                source=SOURCE_URL, future_parameter="do-not-log-this-value"
                            )
                    submit.assert_not_awaited()

    async def test_unknown_fields_do_not_hide_known_value_errors(self) -> None:
        client = ComputeClient(self.context, deep_validation=True)
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaisesRegex(ParameterValidationError, "power"):
                await client.asubmit(
                    "geostatistics",
                    "declustering",
                    {"source": SOURCE_URL, "power": "invalid", "future_parameter": True},
                )
        submit.assert_not_awaited()

    async def test_asubmit_preserves_the_callers_dictionary(self) -> None:
        parameters = {"source": SOURCE_URL, "preview": True}
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.asubmit("geostatistics", "declustering", parameters)
        self.assertEqual({"source": SOURCE_URL, "preview": True}, parameters)
        self.assertTrue(submit.await_args.kwargs["preview"])
        self.assertEqual({"source": SOURCE_URL}, submit.await_args.kwargs["parameters"])

    async def test_validation_options_reach_submission(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            with mock.patch("evo.compute.engine.validate_parameters") as validate:
                await self.client.asubmit(
                    "geostatistics",
                    "declustering",
                    {"source": SOURCE_URL},
                    validate=False,
                    deep_validation=True,
                    check_schemas=False,
                )
        validate.assert_not_called()
        submit.assert_awaited_once()

    async def test_submit_keeps_platform_topic_and_parameter_names(self) -> None:
        self.catalogue["results"].append(
            _task(
                "mesh-tools",
                "simplify",
                {"mesh": {"type": "string"}, "target-face-count": {"type": "integer"}},
                required=["mesh", "target-face-count"],
            )
        )
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.mesh_tools.simplify.submit(mesh=SOURCE_URL, target_face_count=1000)
        self.assertEqual("mesh-tools", submit.await_args.kwargs["topic"])
        self.assertEqual({"mesh": SOURCE_URL, "target-face-count": 1000}, submit.await_args.kwargs["parameters"])

    async def test_signatures_only_differ_in_return_type(self) -> None:
        with self.catalogue_response(), self.mock_job_client():
            await self.client.geostatistics.declustering.submit(source=SOURCE_URL)
        proxy = self.client.geostatistics.declustering
        run = inspect.signature(proxy.run)
        submit = inspect.signature(proxy.submit)
        self.assertEqual(run.parameters, submit.parameters)
        self.assertIs(TaskResult, run.return_annotation)
        self.assertIs(TaskJob, submit.return_annotation)

    def test_completion_lists_both_operations_without_discovery(self) -> None:
        proxy = self.client.geostatistics.declustering
        self.assertIn("run", dir(proxy))
        self.assertIn("submit", dir(proxy))
        self.transport.request.assert_not_called()

    async def test_run_and_submit_share_discovery(self) -> None:
        with self.catalogue_response(), self.mock_job_client():
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)
            await self.client.geostatistics.declustering.submit(source=SOURCE_URL)
        self.assertEqual(1, self.transport.request.call_count)

    async def test_default_feedback_is_silent(self) -> None:
        with self.catalogue_response(), self.mock_job_client() as submit:
            await self.client.geostatistics.declustering.run(source=SOURCE_URL)
        self.assertIs(NoFeedback, submit.return_value.wait_for_results.await_args.kwargs["fb"])

    async def test_client_feedback_reaches_run_and_separate_submission(self) -> None:
        feedback = RecordingFeedback()
        client = ComputeClient(self.context, fb=feedback)
        for operation in ("run", "submit"):
            with self.subTest(operation=operation), self.catalogue_response(), self.mock_job_client() as submit:
                result = await getattr(client.geostatistics.declustering, operation)(source=SOURCE_URL)
                if operation == "submit":
                    await result.results()
                self.assertIs(feedback, submit.return_value.wait_for_results.await_args.kwargs["fb"])

    async def test_wait_feedback_overrides_client_feedback(self) -> None:
        client = ComputeClient(self.context, fb=RecordingFeedback())
        feedback = RecordingFeedback()
        with self.catalogue_response(), self.mock_job_client() as submit:
            job = await client.geostatistics.declustering.submit(source=SOURCE_URL)
            await job.results(fb=feedback)
        self.assertIs(feedback, submit.return_value.wait_for_results.await_args.kwargs["fb"])

    async def test_run_says_where_feedback_goes(self) -> None:
        run = self.client.geostatistics.declustering.run
        self.assertEqual(_RUN_FEEDBACK_NOTE, inspect.getdoc(run))
        with self.catalogue_response(), self.mock_job_client():
            await run(source=SOURCE_URL)
        self.assertEqual(f"Compute declustering weights for a pointset.\n\n{_RUN_FEEDBACK_NOTE}", inspect.getdoc(run))
        self.assertNotIn("fb", inspect.signature(run).parameters)
        self.assertIn("job.results(fb=...)", inspect.getdoc(self.client.geostatistics.kriging.run))


class TestRestoreJob(ComputeClientTestCase):
    @property
    def job_url(self) -> str:
        return f"{self.connector.base_url}compute/orgs/{TEST_ORG.id}/geostatistics/declustering/{TEST_JOB_ID}/status"

    async def test_restored_job_hydrates_results_and_keeps_lifecycle(self) -> None:
        feedback = RecordingFeedback()
        with (
            self.catalogue_response(),
            mock.patch.object(JobClient, "wait_for_results", mock.AsyncMock(return_value=RESULT_PAYLOAD)) as wait,
            mock.patch.object(JobClient, "get_status", mock.AsyncMock()) as status,
            mock.patch.object(JobClient, "cancel", mock.AsyncMock()) as cancel,
        ):
            job = await TaskJob.from_url(self.context, self.job_url, fb=feedback)
            wait.assert_not_awaited()
            self.assertEqual(TEST_JOB_ID, job.id)
            self.assertEqual(self.job_url, job.url)
            result = await job.results()
            self.assertIsInstance(result, TaskResult)
            self.assertEqual("grid", result.target.name)
            self.assertTrue(inspect.iscoroutinefunction(result.target.load))
            await job.status()
            await job.cancel()
        wait.assert_awaited_once_with(polling_interval_seconds=5.0, retry=None, fb=feedback)
        status.assert_awaited_once()
        cancel.assert_awaited_once()
        self.assertEqual(1, self.transport.request.call_count)

    async def test_restoration_reuses_discovery(self) -> None:
        with self.catalogue_response():
            await TaskJob.from_url(self.context, self.job_url)
            await TaskJob.from_url(self.context, self.job_url)
        self.assertEqual(1, self.transport.request.call_count)

    async def test_preview_must_be_explicit_when_restoring(self) -> None:
        with self.catalogue_response():
            normal = await TaskJob.from_url(self.context, self.job_url)
            preview = await TaskJob.from_url(self.context, self.job_url, preview=True)
        self.assertEqual({}, normal._job._get_headers())
        self.assertEqual({"API-Preview": "opt-in"}, preview._job._get_headers())

    async def test_invalid_or_mismatched_urls_fail_before_discovery(self) -> None:
        for url in (
            self.job_url.replace(self.connector.base_url, "https://another-hub.test/"),
            self.job_url.replace(str(TEST_ORG.id), str(UUID(int=9876))),
            self.job_url.replace(str(TEST_JOB_ID), "not-a-uuid"),
            self.job_url.removesuffix("/status"),
            self.job_url + "/extra",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                await TaskJob.from_url(self.context, url)
        self.transport.request.assert_not_called()

    async def test_unknown_task_is_not_silently_restored_without_a_schema(self) -> None:
        with self.catalogue_response(), self.assertRaisesRegex(AttributeError, "missing-task"):
            await TaskJob.from_url(self.context, self.job_url.replace("declustering", "missing-task"))


class TestProgressArrives(TestWithConnector):
    def setUp(self) -> None:
        super().setUp()
        self.setup_universal_headers(get_header_metadata(JobClient.__module__))
        self.feedback = RecordingFeedback()

        async def hydrate(payload: dict) -> TaskResult:
            return TaskResult(payload, RESULT_SCHEMA, FakeContext(self.connector, TEST_ORG.id))

        self.job = TaskJob(
            JobClient(
                connector=self.connector,
                org_id=TEST_ORG.id,
                topic="test",
                task="job-handle",
                job_id=TEST_JOB_ID,
            ),
            convert_result=hydrate,
            fb=self.feedback,
        )

    @contextmanager
    def job_states(self, *data_files: str) -> Iterator[None]:
        responses = [
            MockResponse(
                status_code=202, headers={"Content-Type": "application/json"}, content=json.dumps(load_test_data(name))
            )
            for name in data_files
        ]
        responses.append(
            MockResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=json.dumps(load_test_data(data_files[-1])),
            )
        )
        previous = self.transport.request.side_effect
        self.transport.request.side_effect = responses
        try:
            yield
        finally:
            self.transport.request.side_effect = previous

    async def test_platform_progress_reaches_feedback(self) -> None:
        with self.job_states(
            "job-response-requested.json", "job-response-in-progress.json", "job-response-succeeded.json"
        ):
            await self.job.results(polling_interval_seconds=0.0)
        self.assertIn((0.5, "Job in progress"), self.feedback.reports)
        self.assertEqual((1.0, "Fetching results..."), self.feedback.reports[-1])

    async def test_no_feedback_still_collects_results(self) -> None:
        self.job._fb = NoFeedback
        with self.job_states(
            "job-response-requested.json", "job-response-in-progress.json", "job-response-succeeded.json"
        ):
            result = await self.job.results(polling_interval_seconds=0.0)
        self.assertIsInstance(result, TaskResult)
        self.assertEqual([], self.feedback.reports)

    async def test_status_does_not_start_polling(self) -> None:
        with self.job_states("job-response-in-progress.json"):
            progress = await self.job.status()
        self.assertEqual(JobStatusEnum.in_progress, progress.status)
        self.assertEqual(50, progress.progress)
        self.assertEqual(1, self.transport.request.call_count)

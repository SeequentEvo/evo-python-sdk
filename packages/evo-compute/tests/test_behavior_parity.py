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

"""Behavior parity between the hand-written task runners and the generic engine.

The companion suite in :mod:`test_payload_parity` shows the two paths agree on what they
send. This one shows they agree on what they refuse to send. Each case declares one set of
*invalid* inputs, hands that same set to a typed runner and to the discovery-driven engine,
and requires that neither reaches ``JobClient.submit`` and that both raise a comparable
error.

This suite states the contract it holds the engine to:

* Neither path submits a job. A job that the platform will reject is worse than a local
  error, so this is the part that actually matters and it is asserted on every case.
* Both raise a :class:`ValueError` -- pydantic's ``ValidationError`` on the runner side, the
  SDK's own :class:`~evo.compute.exceptions.ParameterValidationError` on the engine side.
* Both name the parameter at fault, so a caller can act on either message.

What is deliberately *not* promised is identical text or identical precision. A parameter
model knows the branch a caller meant and can fault the field inside it; a JSON Schema
``anyOf`` with no discriminator does not, so the engine reports the parameter and leaves the
branch open. ``test_an_unknown_kriging_method`` pins both messages side by side rather than
asserting a likeness that does not exist.

**Value checks are opt-in.** The engine only reproduces the runner's value-level refusals
under ``deep_validation=True``, which is what these tests use and what a caller who wants
this behavior has to ask for. Left off -- the default -- the engine still refuses a missing
parameter, an unknown one, and a reference it cannot resolve, but a value the schema
disallows is sent for the platform to reject. ``TestWhereTheTwoPathsDiffer`` pins that
boundary explicitly.

Every task class opens with a case proving its *valid* inputs are accepted by both paths.
Without it a broken fixture would refuse everything and the whole suite would pass for the
wrong reason.

The harness is shared with :mod:`test_payload_parity`: same mocked catalogue derived from
each runner's own parameter model, same fake context, so the two suites cannot drift into
testing different engines. As there, that makes this a test of the two code paths against
each other rather than of either against the live catalogue -- the constraints exercised here
are the ones the runners' own models publish. One of them publishes nothing:
``SearchNeighborhood`` hand-writes its own wire form, so the fixture can only advertise it as
an opaque object and the two paths go uncompared inside ``neighborhood``, the most-shared
nested parameter here. ``TestWhereTheTwoPathsDiffer`` asserts that gap rather than let the
silence read as agreement.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest import IsolatedAsyncioTestCase, mock

from pydantic import ValidationError

from evo.compute import ComputeClient, DiscoveryClient, ParameterValidationError
from evo.compute.tasks import CreateAttribute, Source, Target
from evo.compute.tasks.geostatistics.conditioned_simulator import ConSimRunner
from evo.compute.tasks.geostatistics.declustering import DeclusteringGrid, DeclusteringRunner, DeclusteringSource
from evo.compute.tasks.geostatistics.kriging import KrigingMethod, KrigingRunner
from evo.compute.tasks.geostatistics.location_wise import LocationWiseRunner, LocationWiseTarget
from test_payload_parity import (
    GRADE_ATTRIBUTE,
    GRID_URL,
    POINTSET_URL,
    SIMULATIONS_ATTRIBUTE,
    TARGET_URL,
    VARIOGRAM_URL,
    FakeContext,
    _capture_submit,
    _search,
    _SubmitCaptured,
    task_spec,
)

#: Where ``JobClient`` is looked up on each path, so submission can be captured there.
RUNNER_MODULE = "evo.compute.tasks.common.runner"
ENGINE_MODULE = "evo.compute.engine"


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class BehaviorParityTestCase(IsolatedAsyncioTestCase):
    """Runs one set of inputs through both paths and compares how each refuses it."""

    def setUp(self) -> None:
        super().setUp()
        self.context = FakeContext()
        # Deep validation is what makes the engine as strict about values as a parameter
        # model is. See TestWhereTheTwoPathsDiffer for what the default does without it.
        self.client = ComputeClient(self.context, deep_validation=True)

    # -- running one path -------------------------------------------------- #

    @contextmanager
    def engine(self, **options: Any) -> Iterator[None]:
        """Point the engine calls inside at a client configured differently from the fixture's."""
        previous, self.client = self.client, ComputeClient(self.context, **options)
        try:
            yield
        finally:
            self.client = previous

    async def _attempt(self, module: str, call) -> tuple[mock.AsyncMock, Exception | None]:
        """Run ``call`` with submission captured: what ``submit`` saw, and what was raised.

        Only the two error types the contract admits are caught, so a bug in this harness --
        an ``AttributeError`` out of a malformed fixture, say -- fails the test rather than
        passing for a refusal that never happened.
        """
        raised: Exception | None = None
        with _capture_submit(module) as submit:
            try:
                await call()
            except _SubmitCaptured:
                pass  # the payload is recorded; this path accepted the inputs
            except (ValueError, TypeError) as error:  # which of the two it is, is the question under test
                raised = error
        return submit, raised

    def _runner_call(self, runner_cls, inputs: dict[str, Any]):
        # Built inside the call so that a model the inputs cannot even construct is a refusal
        # by the runner, which is where a caller would meet it, rather than a test error.
        return lambda: runner_cls(self.context, runner_cls.params_type(**inputs))

    def _engine_call(self, runner_cls, inputs: dict[str, Any]):
        return lambda: self.client._run_generic("geostatistics", runner_cls.task, inputs)

    def _catalogue(self, runner_cls):
        """Serve the engine the task's schema, derived from the runner's own parameter model."""
        return mock.patch.object(DiscoveryClient, "list_tasks", mock.AsyncMock(return_value=[task_spec(runner_cls)]))

    async def runner_refusal(self, runner_cls, **inputs: Any) -> Exception:
        """The error the hand-written runner raises for ``inputs``, having submitted nothing."""
        submit, error = await self._attempt(RUNNER_MODULE, self._runner_call(runner_cls, inputs))
        return self._refused("runner", submit, error)

    async def engine_refusal(self, runner_cls, **inputs: Any) -> Exception:
        """The error the generic engine raises for the same inputs, having submitted nothing."""
        with self._catalogue(runner_cls):
            submit, error = await self._attempt(ENGINE_MODULE, self._engine_call(runner_cls, inputs))
        return self._refused("engine", submit, error)

    def _refused(self, path: str, submit: mock.AsyncMock, error: Exception | None) -> Exception:
        self.assertEqual(0, submit.await_count, f"the {path} submitted a job for parameters it should have refused")
        if error is None:
            self.fail(f"the {path} accepted parameters it should have refused")
        return error

    # -- comparing the two ------------------------------------------------- #

    async def assertBothRefuse(self, runner_cls, parameter: str, **inputs: Any) -> tuple[Exception, Exception]:
        """Require both paths to refuse ``inputs`` locally, blaming ``parameter``.

        Returns both errors so a case that wants to say more about them can.
        """
        runner_error = await self.runner_refusal(runner_cls, **inputs)
        engine_error = await self.engine_refusal(runner_cls, **inputs)
        self.assertIsInstance(runner_error, ValueError, f"the runner raised {type(runner_error).__name__}")
        self.assertIsInstance(engine_error, ParameterValidationError)
        for path, error in (("runner", runner_error), ("engine", engine_error)):
            self.assertBlames(path, error, parameter)
        return runner_error, engine_error

    def assertBlames(self, path: str, error: Exception, parameter: str) -> None:
        """Require ``error`` to name ``parameter`` as a whole word, not as part of another name.

        A plain substring would let a case pass on an error about a different parameter:
        ``source`` is inside ``source_filter``, ``variogram`` inside ``variogram_model``.
        """
        self.assertRegex(
            str(error), rf"\b{re.escape(parameter)}\b", f"the {path}'s message does not name {parameter!r}"
        )

    async def assertBothAccept(self, runner_cls, **inputs: Any) -> None:
        """Require both paths to get as far as submitting, which is what makes a refusal mean something."""
        await self.assertRunnerAccepts(runner_cls, **inputs)
        await self.assertEngineAccepts(runner_cls, **inputs)

    async def assertRunnerAccepts(self, runner_cls, **inputs: Any) -> dict[str, Any]:
        """Require the runner to submit, and return the payload it submitted."""
        submit, error = await self._attempt(RUNNER_MODULE, self._runner_call(runner_cls, inputs))
        self.assertEqual(1, submit.await_count, f"the runner refused parameters it should have accepted: {error}")
        return submit.await_args.kwargs["parameters"]

    async def assertEngineAccepts(self, runner_cls, **inputs: Any) -> dict[str, Any]:
        """Require the engine to submit, and return the payload it submitted."""
        with self._catalogue(runner_cls):
            submit, error = await self._attempt(ENGINE_MODULE, self._engine_call(runner_cls, inputs))
        self.assertEqual(1, submit.await_count, f"the engine refused parameters it should have accepted: {error}")
        return submit.await_args.kwargs["parameters"]


# --------------------------------------------------------------------------- #
# Valid baselines, one per task
# --------------------------------------------------------------------------- #


def declustering_inputs(**overrides: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = dict(
        source=DeclusteringSource(object=POINTSET_URL),
        grid=DeclusteringGrid(object=GRID_URL),
        target=Target(object=TARGET_URL, attribute=CreateAttribute(name="weights")),
        neighborhood=_search(),
        power=2.5,
    )
    inputs.update(overrides)
    return inputs


def kriging_inputs(**overrides: Any) -> dict[str, Any]:
    # Keyed by wire name: ``KrigingParameters`` populates by alias, so one set of inputs
    # serves both paths without restating anything.
    inputs: dict[str, Any] = dict(
        source=Source(object=POINTSET_URL, attribute=GRADE_ATTRIBUTE),
        target=Target(object=TARGET_URL, attribute=CreateAttribute(name="kriged_grade")),
        variogram=VARIOGRAM_URL,
        neighborhood=_search(),
        kriging_method=KrigingMethod.ORDINARY,
    )
    inputs.update(overrides)
    return inputs


def consim_inputs(**overrides: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = dict(
        source_object=POINTSET_URL,
        source_attribute=GRADE_ATTRIBUTE,
        target_object=GRID_URL,
        neighborhood=_search(),
        variogram_model=VARIOGRAM_URL,
    )
    inputs.update(overrides)
    return inputs


def location_wise_inputs(**overrides: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = dict(
        source=Source(object=POINTSET_URL, attribute=SIMULATIONS_ATTRIBUTE),
        target=LocationWiseTarget(object=GRID_URL),
        summary=True,
    )
    inputs.update(overrides)
    return inputs


# --------------------------------------------------------------------------- #
# Declustering
# --------------------------------------------------------------------------- #


class TestDeclusteringBehaviorParity(BehaviorParityTestCase):
    async def test_valid_parameters_are_accepted_by_both(self) -> None:
        """The guard for everything below: this baseline is a job both paths would run."""
        await self.assertBothAccept(DeclusteringRunner, **declustering_inputs())

    async def test_a_missing_required_parameter(self) -> None:
        """Caught as the call is bound to the schema's signature, so this refusal needs no validation."""
        inputs = declustering_inputs()
        del inputs["grid"]
        await self.assertBothRefuse(DeclusteringRunner, "grid", **inputs)

    async def test_a_power_below_the_allowed_range(self) -> None:
        """``power`` must be positive, and the bound is published in the schema as well as the model."""
        await self.assertBothRefuse(DeclusteringRunner, "power", **declustering_inputs(power=-1.0))

    async def test_a_power_that_is_not_a_number(self) -> None:
        """Neither path lets a word through where a number belongs."""
        await self.assertBothRefuse(DeclusteringRunner, "power", **declustering_inputs(power="strong"))

    async def test_a_create_target_without_a_name(self) -> None:
        """A nested field is as required as a top-level one, and both paths say which one it is."""
        target = {"object": TARGET_URL, "attribute": {"operation": "create"}}
        await self.assertBothRefuse(DeclusteringRunner, "target", **declustering_inputs(target=target))

    async def test_an_object_reference_that_is_not_a_url(self) -> None:
        """The reference is checked as it is converted, so this one does not need deep validation."""
        runner_error, engine_error = await self.assertBothRefuse(
            DeclusteringRunner, "source", **declustering_inputs(source="not-a-url")
        )
        for error in (runner_error, engine_error):
            self.assertIn("Reference must be a valid HTTPS URL", str(error))


# --------------------------------------------------------------------------- #
# Kriging
# --------------------------------------------------------------------------- #


class TestKrigingBehaviorParity(BehaviorParityTestCase):
    async def test_valid_parameters_are_accepted_by_both(self) -> None:
        await self.assertBothAccept(KrigingRunner, **kriging_inputs())

    async def test_a_missing_required_parameter(self) -> None:
        inputs = kriging_inputs()
        del inputs["variogram"]
        await self.assertBothRefuse(KrigingRunner, "variogram", **inputs)

    async def test_a_required_parameter_explicitly_set_to_none(self) -> None:
        """Supplying ``None`` is not the same as leaving a parameter out, and neither path accepts it."""
        await self.assertBothRefuse(KrigingRunner, "source", **kriging_inputs(source=None))

    async def test_an_unknown_kriging_method(self) -> None:
        """Both refuse an unknown method, but only the model can say which branch was meant.

        ``method`` is an untagged union in the model, so the schema derived from it is an
        ``anyOf`` with no discriminator. Pydantic reports every branch it tried; the engine
        has no way to pick one, so it reports the parameter and stops there. This is the
        precision the engine does not promise, written out rather than glossed over.
        """
        runner_error, engine_error = await self.assertBothRefuse(
            KrigingRunner, "kriging_method", **kriging_inputs(kriging_method={"type": "magic"})
        )
        self.assertIn("kriging_method.SimpleKriging.type", str(runner_error))
        self.assertIn(
            "kriging_method: {'type': 'magic'} is not valid under any of the given schemas", str(engine_error)
        )

    async def test_a_kriging_method_missing_its_own_field(self) -> None:
        """Simple kriging needs a mean; a well-formed tag is not a well-formed method."""
        await self.assertBothRefuse(
            KrigingRunner, "kriging_method", **kriging_inputs(kriging_method={"type": "simple"})
        )

    async def test_a_block_discretization_outside_its_range(self) -> None:
        """1 to 9 per axis, on both paths."""
        discretization = {"nx": 99, "ny": 3, "nz": 2}
        await self.assertBothRefuse(
            KrigingRunner, "block_discretization", **kriging_inputs(block_discretization=discretization)
        )

    async def test_an_attribute_that_is_not_a_name_or_expression(self) -> None:
        """Reference leaves are relaxed for deep validation, so resolution is the only gate -- and it holds."""
        source = {"object": POINTSET_URL, "attribute": 12345}
        _, engine_error = await self.assertBothRefuse(KrigingRunner, "source", **kriging_inputs(source=source))
        self.assertIn("source.attribute: expected an attribute name or expression, got int", str(engine_error))

    async def test_a_variogram_that_is_not_an_object_url(self) -> None:
        """A bare filename is not a geoscience object, however plausible it looks."""
        await self.assertBothRefuse(KrigingRunner, "variogram", **kriging_inputs(variogram="variogram.json"))


# --------------------------------------------------------------------------- #
# Conditional simulation (consim)
# --------------------------------------------------------------------------- #


class TestConSimBehaviorParity(BehaviorParityTestCase):
    async def test_valid_parameters_are_accepted_by_both(self) -> None:
        await self.assertBothAccept(ConSimRunner, **consim_inputs())

    async def test_a_missing_required_parameter(self) -> None:
        inputs = consim_inputs()
        del inputs["variogram_model"]
        await self.assertBothRefuse(ConSimRunner, "variogram_model", **inputs)

    async def test_more_simulations_than_the_task_allows(self) -> None:
        runner_error, engine_error = await self.assertBothRefuse(
            ConSimRunner, "number_of_simulations", **consim_inputs(number_of_simulations=500)
        )
        self.assertIn("less than or equal to 100", str(runner_error))
        self.assertIn("500 is greater than the maximum of 100", str(engine_error))

    async def test_fewer_lines_than_the_task_allows(self) -> None:
        await self.assertBothRefuse(ConSimRunner, "number_of_lines", **consim_inputs(number_of_lines=0))

    async def test_an_unknown_kriging_method(self) -> None:
        """A closed set of strings is the one union shape the engine can be precise about."""
        _, engine_error = await self.assertBothRefuse(
            ConSimRunner, "kriging_method", **consim_inputs(kriging_method="magic")
        )
        self.assertIn("kriging_method: must be one of ['simple', 'ordinary'], got 'magic'", str(engine_error))

    async def test_a_filter_with_an_unknown_operator(self) -> None:
        """A filter is a parameter like any other, and its contents are checked like any other."""
        source_filter = {
            "where": {"type": "condition", "attribute": GRADE_ATTRIBUTE, "operator": "roughly"},
        }
        await self.assertBothRefuse(ConSimRunner, "source_filter", **consim_inputs(source_filter=source_filter))

    async def test_a_discretization_outside_its_range(self) -> None:
        """A sub-object that is not optional is descended into, so the engine names the axis too."""
        discretization = {"nx": 0, "ny": 1, "nz": 1}
        _, engine_error = await self.assertBothRefuse(
            ConSimRunner, "block_discretization", **consim_inputs(block_discretization=discretization)
        )
        self.assertIn("block_discretization.nx: 0 is less than the minimum of 1", str(engine_error))


# --------------------------------------------------------------------------- #
# Location-wise
# --------------------------------------------------------------------------- #


class TestLocationWiseBehaviorParity(BehaviorParityTestCase):
    async def test_valid_parameters_are_accepted_by_both(self) -> None:
        await self.assertBothAccept(LocationWiseRunner, **location_wise_inputs())

    async def test_a_missing_required_parameter(self) -> None:
        inputs = location_wise_inputs()
        del inputs["target"]
        await self.assertBothRefuse(LocationWiseRunner, "target", **inputs)

    async def test_an_empty_list_of_cutoffs(self) -> None:
        """Asking for probabilities above no cutoffs at all is a mistake both paths catch."""
        await self.assertBothRefuse(
            LocationWiseRunner,
            "probability_above_cutoff",
            **location_wise_inputs(probability_above_cutoff={"cutoffs": []}),
        )

    async def test_quantiles_that_are_not_a_list(self) -> None:
        await self.assertBothRefuse(LocationWiseRunner, "quantiles", **location_wise_inputs(quantiles="all"))

    async def test_quantiles_that_are_not_numbers(self) -> None:
        """A list of the right shape but the wrong contents is still refused."""
        await self.assertBothRefuse(LocationWiseRunner, "quantiles", **location_wise_inputs(quantiles=["low", "high"]))

    async def test_a_source_without_an_attribute(self) -> None:
        await self.assertBothRefuse(
            LocationWiseRunner, "source", **location_wise_inputs(source={"object": POINTSET_URL})
        )


# --------------------------------------------------------------------------- #
# Where the two paths differ
# --------------------------------------------------------------------------- #


class TestWhereTheTwoPathsDiffer(BehaviorParityTestCase):
    """The cases the contract above does not cover, asserted as they are rather than left implied."""

    async def test_an_extra_parameter_warns_when_the_schema_leaves_its_policy_unspecified(self) -> None:
        """The legacy model ignores extra fields; the engine warns and preserves them."""
        misspelled = declustering_inputs(powr=2.5)
        del misspelled["power"]

        payload = await self.assertRunnerAccepts(DeclusteringRunner, **misspelled)
        self.assertNotIn("powr", payload)
        self.assertEqual(2.0, payload["power"], "the runner should have fallen back to its own default")

        with self.assertLogs("compute.validation", level="WARNING") as logs:
            payload = await self.assertEngineAccepts(DeclusteringRunner, **misspelled)
        self.assertEqual(2.5, payload["powr"])
        self.assertNotIn("power", payload)
        self.assertTrue(any("'powr'" in message for message in logs.output))

    async def test_a_value_the_model_coerces_is_refused_by_the_engine(self) -> None:
        """DIVERGENCE, the engine is stricter: pydantic reads ``"500"`` as ``500``; JSON Schema does not.

        The runner submits the number it parsed. The engine has no model to parse with, and
        forwarding the string as it stands would fail at the platform, so it refuses instead.
        """
        coerced = consim_inputs(number_of_lines="500")
        await self.assertRunnerAccepts(ConSimRunner, **coerced)
        error = await self.engine_refusal(ConSimRunner, **coerced)
        self.assertIn("number_of_lines: expected type integer, got str", str(error))

    async def test_an_unsupported_object_handle_is_refused_by_both_but_not_alike(self) -> None:
        """DIVERGENCE, in the error type: the runner raises ``TypeError``, outside the contract.

        ``_convert_object_reference`` raises ``TypeError`` for a value that is no kind of
        object handle, and pydantic re-wraps ``ValueError`` but not ``TypeError``, so it
        surfaces raw rather than as a ``ValidationError``. The engine meets it later and
        elsewhere: an int never looks like the object frame ``source`` declares, so it is
        deep validation rather than resolution that stops it. Both refuse and neither
        submits, so the part of the contract that matters holds; the ``except ValueError`` a
        caller might write around either path does not.
        """
        unsupported = declustering_inputs(source=12345)
        runner_error = await self.runner_refusal(DeclusteringRunner, **unsupported)
        engine_error = await self.engine_refusal(DeclusteringRunner, **unsupported)
        self.assertIsInstance(runner_error, TypeError)
        self.assertNotIsInstance(runner_error, ValidationError)
        self.assertIn("Cannot convert object reference from type int", str(runner_error))
        self.assertIsInstance(engine_error, ParameterValidationError)
        self.assertIn("source: expected type object, got int", str(engine_error))

    async def test_value_checks_are_opt_in_on_the_engine(self) -> None:
        """The clarified expectation: without ``deep_validation`` a bad value is the platform's to catch."""
        too_many = consim_inputs(number_of_simulations=500)
        with self.engine():  # the default
            await self.assertEngineAccepts(ConSimRunner, **too_many)
        with self.engine(deep_validation=True):
            await self.engine_refusal(ConSimRunner, **too_many)

    async def test_the_default_engine_still_refuses_the_mistakes_it_can_see(self) -> None:
        """Required parameters and resolvable references are always checked."""
        with self.engine():  # the default
            missing = consim_inputs()
            del missing["variogram_model"]
            self.assertBlames("engine", await self.engine_refusal(ConSimRunner, **missing), "variogram_model")

            unknown = consim_inputs(number_of_liness=500)
            with self.assertLogs("compute.validation", level="WARNING"):
                payload = await self.assertEngineAccepts(ConSimRunner, **unknown)
            self.assertEqual(500, payload["number_of_liness"])

            unresolvable = consim_inputs(variogram_model="variogram.json")
            self.assertBlames("engine", await self.engine_refusal(ConSimRunner, **unresolvable), "variogram_model")

    async def test_the_neighborhood_is_not_compared_field_by_field(self) -> None:
        """NOT A DIVERGENCE but a gap in the fixture, asserted so that it is not mistaken for parity.

        ``SearchNeighborhood`` hand-writes its own wire form, so its fields stop describing
        what it sends and ``_writes_its_own_wire_form`` has to advertise it as an opaque
        object. Deep validation then has nothing to check inside it, and three of the four
        tasks covered here take a neighborhood -- the most-shared nested parameter in the
        suite is the one the two paths are not compared on.

        The live catalogue publishes the real schema, so this is the fixture falling short of
        the engine rather than the engine falling short of the runner.
        """
        nonsense = {"ellipsoid": {"ranges": {"major": "wide"}}, "max_samples": 20}

        runner_error = await self.runner_refusal(KrigingRunner, **kriging_inputs(neighborhood=nonsense))
        self.assertBlames("runner", runner_error, "neighborhood")

        payload = await self.assertEngineAccepts(KrigingRunner, **kriging_inputs(neighborhood=nonsense))
        self.assertEqual(nonsense, payload["neighborhood"], "the engine saw something the fixture cannot show it")

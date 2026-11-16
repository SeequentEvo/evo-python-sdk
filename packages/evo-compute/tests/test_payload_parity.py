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

"""Payload parity between the hand-written task runners and the generic engine.

Each case declares one set of inputs, hands that same set to a typed runner and to the
discovery-driven engine, and asserts that ``JobClient.submit`` was given the same
``parameters`` payload by both -- equal in value *and* in type, so a ``1`` never passes
for a ``1.0``.

Feeding both paths the same inputs is the point. It is the difference between the engine
being a drop-in replacement for a runner and the engine merely being *capable* of the same
payload if the caller formats it by hand. The engine earns it through
:mod:`~evo.compute.resolution`, which turns objects and attributes into the references a
task's schema declares, reusing ``tasks/common/source_target.py`` so that both paths agree
by construction rather than by coincidence.

Where the two genuinely disagree, a test says so outright instead of hiding it by reshaping
the input. Two do: a runner materialises its own client-side field defaults where the engine
sends only what it was given, and a runner passes a bare attribute name straight through
where the engine expands it into the JMESPath the platform expects.

One case is deliberately not stated as a single input set. ``source_filter`` is a runner-only
argument -- ``exclude=True`` on the model, folded into ``source`` before anything is sent --
so the catalogue does not advertise it and the engine will not take it. That case compares
wire shapes instead, and the test beside it pins the rejection so the asymmetry stays visible
rather than reading as parity.

Discovery is mocked rather than recorded, so the suite needs no catalogue fixture and no
credentials. :func:`task_spec` derives each task's schema from the runner's own parameter
model: the wire names, which of them are required, and the reference annotations the
resolver reads. That makes this a test of the two code paths against each other, not of the
SDK against the live catalogue -- that is schema conformance, and is tested separately.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, get_args
from unittest import IsolatedAsyncioTestCase, mock

from evo.common.test_tools import ORG as TEST_ORG
from evo.objects import ObjectReference
from evo.objects.typed import Attribute, PendingAttribute
from evo.objects.typed.base import BaseObject
from pydantic import BaseModel

from evo.compute import ComputeClient, DiscoveryClient, ParameterValidationError, TaskResource
from evo.compute.tasks import CreateAttribute, SearchNeighborhood, Source, Target, UpdateAttribute
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges, Filter, FilterCondition, Rotation
from evo.compute.tasks.common.source_target import _convert_object_reference, _get_attribute_expression
from evo.compute.tasks.geostatistics.conditioned_simulator import ConSimRunner
from evo.compute.tasks.geostatistics.declustering import DeclusteringGrid, DeclusteringRunner, DeclusteringSource
from evo.compute.tasks.geostatistics.kriging import BlockDiscretization, KrigingMethod, KrigingRunner
from evo.compute.tasks.geostatistics.location_wise import (
    LocationWiseRunner,
    LocationWiseTarget,
    MeanAboveCutoff,
    ProbabilityAboveCutoff,
)

# --------------------------------------------------------------------------- #
# Shared inputs
# --------------------------------------------------------------------------- #


def _obj_url(suffix: str) -> str:
    """A structurally valid geoscience object URL, which the parameter models insist on."""
    return (
        "https://hub.test.evo.bentley.com/geoscience-object"
        "/orgs/00000000-0000-0000-0000-000000000001"
        "/workspaces/00000000-0000-0000-0000-000000000002"
        f"/objects/00000000-0000-0000-0000-0000000000{suffix}"
    )


POINTSET_URL = _obj_url("10")
GRID_URL = _obj_url("20")
TARGET_URL = _obj_url("30")
VARIOGRAM_URL = _obj_url("40")

GRADE_ATTRIBUTE = "locations.attributes[?name=='grade']"
SIMULATIONS_ATTRIBUTE = "locations.attributes[?name=='simulations']"


def _typed_object(url: str) -> mock.MagicMock:
    """Stands in for a typed geoscience object, which the models resolve down to its URL."""
    obj = mock.MagicMock(spec=BaseObject)
    obj.metadata.url = ObjectReference(url)
    return obj


def _existing_attribute(key: str, url: str, schema_path: str = "") -> mock.MagicMock:
    """Stands in for an attribute already on an object, which resolves to a key-based expression."""
    attribute = mock.MagicMock(spec=Attribute)
    attribute.key = key
    attribute.exists = True
    attribute._context = mock.MagicMock(schema_path=schema_path)
    attribute._obj = _typed_object(url)
    return attribute


def _pending_attribute(name: str, url: str) -> PendingAttribute:
    """An attribute that does not exist yet, which resolves to a create operation."""
    parent = mock.MagicMock()
    parent._obj = _typed_object(url)
    return PendingAttribute(parent, name)


def _search(min_samples: int | None = 4) -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(
            ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            rotation=Rotation(dip_azimuth=45.0, dip=10.0, pitch=5.0),
        ),
        max_samples=20,
        min_samples=min_samples,
    )


# --------------------------------------------------------------------------- #
# Type-strict comparison
# --------------------------------------------------------------------------- #


def payload_differences(expected: Any, actual: Any, path: str = "parameters") -> list[str]:
    """Compare two payloads by value *and* concrete type, reporting every divergence.

    ``==`` alone is too forgiving here: ``1 == 1.0`` and ``True == 1``, yet the platform
    reads those as different JSON types.
    """
    if type(expected) is not type(actual):
        return [f"{path}: expected {type(expected).__name__} {expected!r}, got {type(actual).__name__} {actual!r}"]

    if isinstance(expected, dict):
        differences = []
        for key in sorted(set(expected) - set(actual)):
            differences.append(f"{path}.{key}: missing, expected {expected[key]!r}")
        for key in sorted(set(actual) - set(expected)):
            differences.append(f"{path}.{key}: unexpected, got {actual[key]!r}")
        for key in expected.keys() & actual.keys():
            differences += payload_differences(expected[key], actual[key], f"{path}.{key}")
        return differences

    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: expected {len(expected)} item(s), got {len(actual)}"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences += payload_differences(left, right, f"{path}[{index}]")
        return differences

    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class FakeContext:
    """Minimal duck-typed IContext exposing just what the runners and the engine use."""

    def __init__(self) -> None:
        self._connector = mock.Mock()
        self._org_id = TEST_ORG.id

    def get_connector(self):
        return self._connector

    def get_org_id(self):
        return self._org_id

    def get_environment(self):
        return mock.Mock()


class _SubmitCaptured(Exception):
    """Raised in place of a real submission, once the payload has been recorded."""


@contextmanager
def _capture_submit(module: str) -> Iterator[mock.AsyncMock]:
    """Patch ``JobClient`` in ``module`` so submission stops as soon as the payload is known."""
    submit = mock.AsyncMock(side_effect=_SubmitCaptured)
    with mock.patch(f"{module}.JobClient") as job_client:
        job_client.submit = submit
        yield submit


_REFERENCE_ANNOTATIONS = {
    _convert_object_reference: {"reference_to": "geoscience-object"},
    _get_attribute_expression: {"reference_to": "attribute"},
}


def _annotation_for(field) -> dict[str, str] | None:
    """The catalogue annotation a model field corresponds to, if it carries a reference.

    The models already say this, in the validator each reference field is annotated with, so
    reading it back keeps the mocked schema honest without hand-writing any annotations.
    """
    for meta in field.metadata:
        if (annotation := _REFERENCE_ANNOTATIONS.get(getattr(meta, "func", None))) is not None:
            return annotation
    if {CreateAttribute, UpdateAttribute} & set(get_args(field.annotation)):
        return {"target": "attribute"}
    return None


def _models_reachable_from(model: type[BaseModel], seen: dict[str, type[BaseModel]]) -> dict[str, type[BaseModel]]:
    """Index every model in ``model``'s field graph by name, matching pydantic's ``$defs`` keys."""
    if model.__name__ in seen:
        return seen
    seen[model.__name__] = model
    for field in model.model_fields.values():
        for annotation in (field.annotation, *get_args(field.annotation)):
            for member in (annotation, *get_args(annotation)):
                if isinstance(member, type) and issubclass(member, BaseModel):
                    _models_reachable_from(member, seen)
    return seen


def _renest_folded_filters(schema: dict[str, Any], folded: dict[str, Any]) -> None:
    """Advertise a folded filter where the payload carries it, tagged as the catalogue tags it.

    ``source_filter`` is an input to the model that never reaches the wire: the serializer
    writes it to ``source.filter``. The catalogue describes the wire, so it advertises the
    filter under ``source`` and marks it ``composite: "filter"`` -- the annotation for a
    sub-object the platform composes from a group of fields.
    """
    for name, prop in folded.items():
        owner, _, leaf = name.rpartition("_")
        if leaf != "filter":
            continue
        reference = schema.get("properties", {}).get(owner, {}).get("$ref", "")
        if (block := schema.get("$defs", {}).get(reference.rpartition("/")[2])) is not None:
            block.setdefault("properties", {})[leaf] = dict(prop, composite="filter")


def _writes_its_own_wire_form(model: type[BaseModel]) -> bool:
    """Whether ``model`` hand-writes what it sends, rather than serialising its own fields.

    A plain ``@model_serializer`` replaces the payload outright, so the model's fields stop
    describing what reaches the wire -- ``SearchNeighborhood`` validates ``ellipsoid.ranges``
    and sends ``ellipsoid.ellipsoid_ranges``. A ``mode="wrap"`` serializer defers to the
    handler and only adds to what it returns, so those models still describe themselves.

    Advertising such a model's fields anyway would fault every payload built from it. Nothing
    in this module notices, because deep validation is off here; :mod:`test_behavior_parity`
    turns it on and is where that would surface. The cost is that nothing inside such a model
    is compared field by field in the derived-schema fixture.
    """
    decorators = model.__pydantic_decorators__.model_serializers.values()
    return any(decorator.info.mode == "plain" for decorator in decorators)


def task_spec(runner_cls) -> TaskResource:
    """The discovery spec the engine would fetch for ``runner_cls``, built from its own model.

    Departures from a plain ``model_json_schema`` keep this a fair stand-in for the catalogue,
    which describes the wire and nothing else. Fields the runner folds into another field are
    ``exclude=True`` -- inputs to the model that never reach the wire -- so they are advertised
    where the payload carries them rather than at the top level. Reference leaves carry the
    annotations the resolver reads, without which it would leave a caller's objects and
    attributes untouched. And a model that hand-writes its own wire form is advertised as the
    opaque object it is, because its fields are no longer a description of what it sends.
    """
    model = runner_cls.params_type
    schema = model.model_json_schema(by_alias=True, mode="validation")
    properties = schema.get("properties", {})
    folded = {field.alias or name for name, field in model.model_fields.items() if field.exclude}
    schema["properties"] = {name: prop for name, prop in properties.items() if name not in folded}
    schema["required"] = [name for name in schema.get("required", []) if name not in folded]
    _renest_folded_filters(schema, {name: properties[name] for name in folded if name in properties})

    models = _models_reachable_from(model, {})
    definitions = schema.get("$defs", {})
    for name, block in ((model.__name__, schema), *definitions.items()):
        if (owner := models.get(name)) is None:
            continue
        if name in definitions and _writes_its_own_wire_form(owner):
            definitions[name] = {"type": "object"}
            continue
        for field_name, field in owner.model_fields.items():
            key = field.alias or field_name
            if key in block.get("properties", {}) and (annotation := _annotation_for(field)) is not None:
                block["properties"][key].update(annotation)
    return TaskResource(topic=runner_cls.topic, name=runner_cls.task, parameters=schema)


class PayloadParityTestCase(IsolatedAsyncioTestCase):
    """Runs one job through both paths and compares what each hands to ``JobClient.submit``."""

    def setUp(self) -> None:
        super().setUp()
        self.context = FakeContext()
        self.client = ComputeClient(self.context)

    async def runner_payload(self, runner_cls, params) -> dict[str, Any]:
        """The payload the hand-written runner submits for ``params``."""
        with _capture_submit("evo.compute.tasks.common.runner") as submit:
            with self.assertRaises(_SubmitCaptured):
                await runner_cls(self.context, params)
        return submit.await_args.kwargs["parameters"]

    async def engine_payload(self, runner_cls, **parameters: Any) -> dict[str, Any]:
        """The payload the generic engine submits for the same job, described as plain JSON."""
        catalogue = mock.patch.object(
            DiscoveryClient, "list_tasks", mock.AsyncMock(return_value=[task_spec(runner_cls)])
        )
        with catalogue, _capture_submit("evo.compute.engine") as submit:
            with self.assertRaises(_SubmitCaptured):
                await self.client._run_generic("geostatistics", runner_cls.task, parameters)
        return submit.await_args.kwargs["parameters"]

    async def payload_difference(self, runner_cls, **inputs: Any) -> list[str]:
        """How the two payloads differ when both paths are given ``inputs``."""
        runner_payload = await self.runner_payload(runner_cls, runner_cls.params_type(**inputs))
        engine_payload = await self.engine_payload(runner_cls, **inputs)
        return payload_differences(runner_payload, engine_payload)

    async def assertSameInputsParity(self, runner_cls, **inputs: Any) -> None:
        """Require both paths to submit the same payload when handed the same inputs."""
        differences = await self.payload_difference(runner_cls, **inputs)
        if differences:
            self.fail("engine payload diverges from the runner payload:\n  " + "\n  ".join(differences))

    def assertPayloadParity(self, runner_payload: dict[str, Any], engine_payload: dict[str, Any]) -> None:
        differences = payload_differences(runner_payload, engine_payload)
        if differences:
            self.fail("engine payload diverges from the runner payload:\n  " + "\n  ".join(differences))


# --------------------------------------------------------------------------- #
# Declustering
# --------------------------------------------------------------------------- #


class TestDeclusteringPayloadParity(PayloadParityTestCase):
    def _inputs(self, **overrides: Any) -> dict[str, Any]:
        inputs: dict[str, Any] = dict(
            source=DeclusteringSource(object=POINTSET_URL),
            grid=DeclusteringGrid(object=GRID_URL),
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="weights")),
            neighborhood=_search(),
            power=2.5,
        )
        inputs.update(overrides)
        return inputs

    async def test_inverse_distance_weighting(self) -> None:
        """The everyday call: source, grid, target and a search, with an explicit power."""
        await self.assertSameInputsParity(DeclusteringRunner, **self._inputs())

    async def test_knn_mode_sends_a_null_power(self) -> None:
        """KNN mode is a literal ``null`` power, not an omitted one; both paths must forward it."""
        runner_payload = await self.runner_payload(
            DeclusteringRunner, DeclusteringRunner.params_type(**self._inputs(power=None))
        )
        self.assertIn("power", runner_payload)
        self.assertIsNone(runner_payload["power"])
        await self.assertSameInputsParity(DeclusteringRunner, **self._inputs(power=None))

    async def test_a_typed_attribute_target_becomes_a_create_operation(self) -> None:
        """The target is given as the attribute itself, and both paths expand it the same way."""
        await self.assertSameInputsParity(
            DeclusteringRunner, **self._inputs(target=_pending_attribute("weights", TARGET_URL))
        )

    async def test_search_without_min_samples_omits_the_key(self) -> None:
        """An unset ``min_samples`` is dropped by the neighborhood serializer, not sent as null."""
        runner_payload = await self.runner_payload(
            DeclusteringRunner, DeclusteringRunner.params_type(**self._inputs(neighborhood=_search(min_samples=None)))
        )
        self.assertNotIn("min_samples", runner_payload["neighborhood"])
        await self.assertSameInputsParity(DeclusteringRunner, **self._inputs(neighborhood=_search(min_samples=None)))

    async def test_integer_power_is_not_accepted_as_the_float_the_runner_sends(self) -> None:
        """Guards the comparison itself: the engine's ``2`` must not pass for the runner's ``2.0``."""
        runner_payload = await self.runner_payload(
            DeclusteringRunner, DeclusteringRunner.params_type(**self._inputs(power=2.0))
        )
        engine_payload = await self.engine_payload(DeclusteringRunner, **self._inputs(power=2))
        self.assertEqual(
            ["parameters.power: expected float 2.0, got int 2"],
            payload_differences(runner_payload, engine_payload),
        )

    async def test_bare_objects_fill_in_their_frames(self) -> None:
        """``source``/``grid`` are an object and nothing else, so the object goes straight in."""
        await self.assertSameInputsParity(
            DeclusteringRunner,
            **self._inputs(source=_typed_object(POINTSET_URL), grid=_typed_object(GRID_URL)),
        )


# --------------------------------------------------------------------------- #
# Kriging
# --------------------------------------------------------------------------- #


class TestKrigingPayloadParity(PayloadParityTestCase):
    def _inputs(self, **overrides: Any) -> dict[str, Any]:
        # Keyed by wire name throughout: ``KrigingParameters`` populates by alias, so one set
        # of inputs serves the runner and the engine without any restating.
        inputs: dict[str, Any] = dict(
            source=Source(object=POINTSET_URL, attribute=GRADE_ATTRIBUTE),
            target=Target(object=TARGET_URL, attribute=CreateAttribute(name="kriged_grade")),
            variogram=VARIOGRAM_URL,
            neighborhood=_search(),
            kriging_method=KrigingMethod.ORDINARY,
        )
        inputs.update(overrides)
        return inputs

    async def test_ordinary_kriging(self) -> None:
        """``neighborhood``/``kriging_method`` are the aliases of ``search``/``method``."""
        await self.assertSameInputsParity(KrigingRunner, **self._inputs())

    async def test_simple_kriging_with_block_discretization(self) -> None:
        """A tagged method object and an optional sub-block frame survive both paths intact."""
        await self.assertSameInputsParity(
            KrigingRunner,
            **self._inputs(
                kriging_method=KrigingMethod.simple(mean=12.5),
                block_discretization=BlockDiscretization(nx=3, ny=3, nz=2),
            ),
        )

    async def test_typed_attributes_become_urls_and_jmespath_expressions(self) -> None:
        """An attribute that already exists resolves to a key lookup, and to an *update* target."""
        runner_payload = await self.runner_payload(
            KrigingRunner,
            KrigingRunner.params_type(
                **self._inputs(
                    source=_existing_attribute("src-key-1", POINTSET_URL, schema_path="locations.attributes"),
                    target=_existing_attribute("tgt-key-9", TARGET_URL),
                )
            ),
        )
        self.assertEqual("locations.attributes[?key=='src-key-1']", runner_payload["source"]["attribute"])
        self.assertEqual(
            {"operation": "update", "reference": "attributes[?key=='tgt-key-9']"}, runner_payload["target"]["attribute"]
        )
        await self.assertSameInputsParity(
            KrigingRunner,
            **self._inputs(
                source=_existing_attribute("src-key-1", POINTSET_URL, schema_path="locations.attributes"),
                target=_existing_attribute("tgt-key-9", TARGET_URL),
            ),
        )

    async def test_a_folded_filter_reaches_the_wire_as_a_nested_one(self) -> None:
        """WIRE SHAPE, NOT SAME INPUTS: ``source_filter`` is an argument only the runner has.

        It is ``exclude=True`` on the model and folded into ``source`` before anything is sent,
        so the catalogue never advertises it and the engine rejects it -- the test below pins
        that. What is worth asserting is that the shape the runner folds a filter into is the
        shape an engine caller writes by hand, so this case compares payloads, not inputs.
        """
        source_filter = Filter(where=FilterCondition(attribute=GRADE_ATTRIBUTE, operator="greater_than", threshold=0.5))
        runner_payload = await self.runner_payload(
            KrigingRunner, KrigingRunner.params_type(**self._inputs(), source_filter=source_filter)
        )
        self.assertNotIn("source_filter", runner_payload)

        nested = {"object": POINTSET_URL, "attribute": GRADE_ATTRIBUTE, "filter": source_filter.model_dump()}
        engine_payload = await self.engine_payload(KrigingRunner, **self._inputs(source=nested))
        self.assertPayloadParity(runner_payload, engine_payload)

    async def test_the_folded_filter_fields_are_not_offered_as_engine_parameters(self) -> None:
        """Folding a filter is runner-side work, so ``source_filter`` is not a task parameter."""
        self.assertNotIn("source_filter", task_spec(KrigingRunner).parameters["properties"])
        with self.assertRaises(ParameterValidationError) as caught:
            await self.engine_payload(
                KrigingRunner,
                **self._inputs(),
                source_filter={"where": {"type": "condition", "attribute": GRADE_ATTRIBUTE, "operator": "in"}},
            )
        self.assertIn("source_filter", str(caught.exception))

    async def test_references_inside_a_composite_filter_are_resolved(self) -> None:
        """WIRE SHAPE, NOT SAME INPUTS: a ``composite`` node is walked like any other.

        The catalogue tags ``source.filter`` ``composite: "filter"``. That annotation only
        names the group of fields the platform composes, so it must not stop resolution at
        the node: the attribute the condition filters on carries ``reference_to: attribute``
        and has to reach the wire as an expression from both paths.

        The engine caller spells out ``values: None`` because the runner does. Its serializer
        dumps the filter model whole, and unlike the top-level payload that dump keeps the
        unset keys, so an engine caller matching it has to send them too.
        """
        attribute = _existing_attribute("filter-key-2", POINTSET_URL, schema_path="locations.attributes")
        source_filter = Filter(where=FilterCondition(attribute=attribute, operator="greater_than", threshold=0.5))
        runner_payload = await self.runner_payload(
            KrigingRunner, KrigingRunner.params_type(**self._inputs(), source_filter=source_filter)
        )

        nested = {
            "object": POINTSET_URL,
            "attribute": GRADE_ATTRIBUTE,
            "filter": {
                "where": {
                    "type": "condition",
                    "attribute": attribute,
                    "operator": "greater_than",
                    "threshold": 0.5,
                    "values": None,
                }
            },
        }
        engine_payload = await self.engine_payload(KrigingRunner, **self._inputs(source=nested))

        self.assertEqual(
            "locations.attributes[?key=='filter-key-2']", engine_payload["source"]["filter"]["where"]["attribute"]
        )
        self.assertPayloadParity(runner_payload, engine_payload)

    async def test_an_unset_method_is_defaulted_by_the_runner_only(self) -> None:
        """DIVERGENCE: the runner materialises its ``method`` default; the engine sends nothing.

        The engine forwards only what the caller passed, leaving the platform to apply its own
        default. Both reach the same job, but not by the same payload. This one is by design,
        unlike the resolution gaps this suite found, which were fixed rather than recorded.
        """
        inputs = self._inputs()
        del inputs["kriging_method"]
        self.assertEqual(
            ["parameters.kriging_method: missing, expected {'type': 'ordinary'}"],
            await self.payload_difference(KrigingRunner, **inputs),
        )

    async def test_a_bare_attribute_name_is_expanded_by_the_engine_only(self) -> None:
        """DIVERGENCE: the runner passes a bare attribute name through; the engine expands it.

        ``_get_attribute_expression`` treats any string as an expression already written, so a
        name reaches the platform as a name. The engine reads the leaf's ``reference_to:
        attribute`` and projects the name into the container that attribute's family keeps its
        attributes in -- ``attributes`` unless the schema names another through
        ``attribute_path``, which is the container the typed paths above resolve to as well.
        Only the engine's payload is one the platform can look the attribute up from.
        """
        self.assertEqual(
            ["parameters.source.attribute: expected 'grade', got \"attributes[?name=='grade']\""],
            await self.payload_difference(
                KrigingRunner, **self._inputs(source=Source(object=POINTSET_URL, attribute="grade"))
            ),
        )


# --------------------------------------------------------------------------- #
# Conditional simulation (consim)
# --------------------------------------------------------------------------- #


class TestConSimPayloadParity(PayloadParityTestCase):
    #: The runner materialises these from its own field defaults, so an engine caller has to
    #: pass them for the two payloads to agree. Sending them on both paths is what makes this
    #: one set of inputs rather than two.
    RUNNER_DEFAULTS: dict[str, Any] = {
        "block_discretization": {"nx": 1, "ny": 1, "nz": 1},
        "kriging_method": "simple",
        "number_of_lines": 500,
        "number_of_simulations": 1,
        "random_seed": 38239342,
        "number_of_simulations_to_save": 5,
        "perform_validation": False,
    }

    def _inputs(self, **overrides: Any) -> dict[str, Any]:
        inputs: dict[str, Any] = dict(
            source_object=POINTSET_URL,
            source_attribute=GRADE_ATTRIBUTE,
            target_object=GRID_URL,
            neighborhood=_search(),
            variogram_model=VARIOGRAM_URL,
            **self.RUNNER_DEFAULTS,
        )
        inputs.update(overrides)
        return inputs

    async def test_minimal_conditional_simulation(self) -> None:
        """The flat source shape: object and attribute as separate scalars, not a frame."""
        await self.assertSameInputsParity(ConSimRunner, **self._inputs())

    async def test_multiple_simulations_with_quantiles(self) -> None:
        """Counts stay ints and cutoffs stay floats across both paths."""
        await self.assertSameInputsParity(
            ConSimRunner,
            **self._inputs(
                number_of_simulations=25,
                number_of_simulations_to_save=3,
                random_seed=7,
                location_wise_quantiles=[0.1, 0.5, 0.9],
                probability_above_cutoff=[1.0, 2.5],
            ),
        )

    async def test_unset_defaults_are_supplied_by_the_runner_only(self) -> None:
        """DIVERGENCE: seven client-side defaults the engine leaves to the platform."""
        inputs = self._inputs()
        for name in self.RUNNER_DEFAULTS:
            del inputs[name]
        differences = await self.payload_difference(ConSimRunner, **inputs)
        self.assertEqual(
            sorted(f"parameters.{name}" for name in self.RUNNER_DEFAULTS),
            sorted(difference.split(":")[0] for difference in differences),
        )


# --------------------------------------------------------------------------- #
# Location-wise
# --------------------------------------------------------------------------- #


class TestLocationWisePayloadParity(PayloadParityTestCase):
    def _inputs(self, **overrides: Any) -> dict[str, Any]:
        inputs: dict[str, Any] = dict(
            source=Source(object=POINTSET_URL, attribute=SIMULATIONS_ATTRIBUTE),
            target=LocationWiseTarget(object=GRID_URL),
            summary=True,
        )
        inputs.update(overrides)
        return inputs

    async def test_summary_only(self) -> None:
        """The plain-model control: no aliases, no defaults, no folded fields."""
        await self.assertSameInputsParity(LocationWiseRunner, **self._inputs())

    async def test_all_statistics(self) -> None:
        """Quantiles and cutoff lists keep their float element types."""
        await self.assertSameInputsParity(
            LocationWiseRunner,
            **self._inputs(
                quantiles=[0.1, 0.5, 0.9],
                probability_above_cutoff=ProbabilityAboveCutoff(cutoffs=[1.0, 2.0]),
                mean_above_cutoff=MeanAboveCutoff(cutoffs=[1.5]),
            ),
        )

    async def test_unset_optionals_stay_off_the_wire(self) -> None:
        """Neither path invents a statistics block the caller never asked for."""
        runner_payload = await self.runner_payload(LocationWiseRunner, LocationWiseRunner.params_type(**self._inputs()))
        for unset in ("quantiles", "probability_above_cutoff", "mean_above_cutoff"):
            self.assertNotIn(unset, runner_payload)
        await self.assertSameInputsParity(LocationWiseRunner, **self._inputs())

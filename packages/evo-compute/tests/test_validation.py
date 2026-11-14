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

import json
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest import mock

from evo.common.test_tools import ORG as TEST_ORG
from evo.common.test_tools import TestWithConnector

from data import load_test_data
from evo.compute import ComputeClient, ParameterValidationError, TaskResource
from evo.compute.validation import _warn_unknown_fields, validate_parameters


def make_spec(parameters: dict[str, Any], *, topic: str = "demo", name: str = "widget") -> TaskResource:
    """Build a TaskResource carrying the given parameter schema."""
    return TaskResource.model_validate({"topic": topic, "name": name, "parameters": parameters})


class TestAdditionalPropertiesPolicy(unittest.TestCase):
    def test_closed_objects_reject_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ParameterValidationError, "new_option"):
            _warn_unknown_fields(
                {"properties": {}, "additionalProperties": False},
                {"new_option": True},
                ["config"],
                "demo.widget",
            )

    def test_additional_property_values_are_validated(self) -> None:
        with self.assertRaisesRegex(ParameterValidationError, "new_count"):
            _warn_unknown_fields(
                {"additionalProperties": {"type": "integer", "minimum": 1}},
                {"new_count": "invalid"},
                ["counts"],
                "demo.widget",
            )

    def test_explicitly_open_objects_do_not_warn(self) -> None:
        with self.assertNoLogs("compute.validation", level="WARNING"):
            _warn_unknown_fields(
                {"properties": {}, "additionalProperties": True},
                {"new_option": True},
                [],
                "demo.widget",
            )

    def test_unspecified_policy_warns_without_exposing_values(self) -> None:
        with self.assertLogs("compute.validation", level="WARNING") as logs:
            _warn_unknown_fields({"properties": {}}, {"new_option": "private value"}, [], "demo.widget")
        self.assertIn("to be submitted as is", logs.output[0])
        self.assertNotIn("private value", logs.output[0])
        self.assertNotIn("wire format", logs.output[0])

    def test_additional_value_schemas_can_reference_root_definitions(self) -> None:
        node = {"additionalProperties": {"$ref": "#/$defs/Count"}}
        root = {"$defs": {"Count": {"type": "integer", "minimum": 1}}}
        _warn_unknown_fields(node, {"new_count": 2}, ["counts"], "demo.widget", root=root)
        with self.assertRaisesRegex(ParameterValidationError, "counts.new_count"):
            _warn_unknown_fields(node, {"new_count": 0}, ["counts"], "demo.widget", root=root)

    def test_an_empty_additional_schema_accepts_values_without_warning(self) -> None:
        with self.assertNoLogs("compute.validation", level="WARNING"):
            _warn_unknown_fields(
                {"additionalProperties": {}},
                {"new_option": {"nested": [None, True]}},
                [],
                "demo.widget",
            )

    def test_pattern_properties_are_not_reported_as_extra_fields(self) -> None:
        with self.assertNoLogs("compute.validation", level="WARNING"):
            _warn_unknown_fields(
                {"patternProperties": {"^count_": {"type": "integer"}}, "additionalProperties": False},
                {"count_samples": 2},
                [],
                "demo.widget",
            )

    def test_unspecified_policy_warns_for_free_form_objects_too(self) -> None:
        with self.assertLogs("compute.validation", level="WARNING"):
            _warn_unknown_fields({"type": "object"}, {"new_option": True}, [], "demo.widget")


class TestValidateParameters(unittest.TestCase):
    """Unit tests for the schema-driven parameter validator (no network)."""

    # -- shallow ----------------------------------------------------------- #

    def test_shallow_passes_when_required_present(self) -> None:
        spec = make_spec({"type": "object", "properties": {"source": {"type": "object"}}, "required": ["source"]})
        validate_parameters(spec, {"source": "obj-1"})  # does not raise

    def test_shallow_rejects_missing_required(self) -> None:
        spec = make_spec({"type": "object", "properties": {"source": {"type": "object"}}, "required": ["source"]})
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {})
        self.assertIn("source", str(ctx.exception))
        self.assertEqual(["missing required parameter 'source'"], ctx.exception.errors)

    def test_shallow_rejects_required_resolved_to_none(self) -> None:
        """A required field dropped to None (so it would be omitted on the wire) is caught."""
        spec = make_spec({"type": "object", "properties": {"source": {"type": "object"}}, "required": ["source"]})
        with self.assertRaises(ParameterValidationError):
            validate_parameters(spec, {"source": None})

    def test_shallow_accepts_none_for_a_nullable_required_field(self) -> None:
        """``Optional[X]`` with no default is required *and* nullable, so None is a real value."""
        union = make_spec(
            {
                "type": "object",
                "properties": {"seed": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
                "required": ["seed"],
            }
        )
        validate_parameters(union, {"seed": None}, deep=True)  # does not raise

        type_list = make_spec(
            {"type": "object", "properties": {"seed": {"type": ["integer", "null"]}}, "required": ["seed"]}
        )
        validate_parameters(type_list, {"seed": None}, deep=True)  # does not raise

    def test_shallow_defers_to_the_schema_for_less_obvious_nullability(self) -> None:
        """Nullability is whatever Draft 2020-12 says it is, not just an explicit ``"null"`` type."""
        bare = make_spec({"type": "object", "properties": {"options": {"default": None}}, "required": ["options"]})
        validate_parameters(bare, {"options": None}, deep=True)  # unconstrained, so null is allowed

        behind_ref = make_spec(
            {
                "type": "object",
                "properties": {"seed": {"$ref": "#/$defs/MaybeSeed"}},
                "required": ["seed"],
                "$defs": {"MaybeSeed": {"type": ["integer", "null"]}},
            }
        )
        validate_parameters(behind_ref, {"seed": None}, deep=True)  # does not raise

        const_null = make_spec({"type": "object", "properties": {"unset": {"const": None}}, "required": ["unset"]})
        validate_parameters(const_null, {"unset": None}, deep=True)  # does not raise

    def test_shallow_still_rejects_an_omitted_nullable_required_field(self) -> None:
        spec = make_spec(
            {"type": "object", "properties": {"seed": {"type": ["integer", "null"]}}, "required": ["seed"]}
        )
        with self.assertRaises(ParameterValidationError):
            validate_parameters(spec, {})

    def test_shallow_ignores_type_mismatches(self) -> None:
        """Shallow validation only checks presence, not types."""
        spec = make_spec({"type": "object", "properties": {"count": {"type": "integer"}}})
        validate_parameters(spec, {"count": "not-an-int"})  # does not raise without deep

    # -- deep -------------------------------------------------------------- #

    def test_deep_rejects_bad_enum(self) -> None:
        spec = make_spec({"type": "object", "properties": {"mode": {"type": "string", "enum": ["fast", "accurate"]}}})
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"mode": "turbo"}, deep=True)
        message = str(ctx.exception)
        self.assertIn("mode", message)
        self.assertIn("fast", message)

    def test_deep_rejects_wrong_scalar_type(self) -> None:
        spec = make_spec({"type": "object", "properties": {"power": {"type": "number"}}})
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"power": "abc"}, deep=True)
        self.assertIn("power", str(ctx.exception))
        self.assertIn("number", str(ctx.exception))

    def test_deep_rejects_missing_nested_required(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {"seed": {"type": "integer"}},
                        "required": ["seed"],
                    }
                },
                "required": ["config"],
            }
        )
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"config": {}}, deep=True)
        self.assertIn("seed", str(ctx.exception))

    def test_deep_resolves_internal_ref(self) -> None:
        """``$ref``/``$defs`` are resolved so nested constraints are enforced."""
        spec = make_spec(
            {
                "type": "object",
                "properties": {"variogram": {"$ref": "#/$defs/Variogram"}},
                "required": ["variogram"],
                "$defs": {
                    "Variogram": {
                        "type": "object",
                        "properties": {"nugget": {"type": "number"}},
                        "required": ["nugget"],
                    }
                },
            }
        )
        validate_parameters(spec, {"variogram": {"nugget": 0.1}}, deep=True)  # does not raise
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"variogram": {}}, deep=True)
        self.assertIn("nugget", str(ctx.exception))

    def test_deep_accepts_valid_payload(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["fast", "accurate"]},
                    "iterations": {"type": "integer", "minimum": 1},
                },
                "required": ["mode"],
            }
        )
        validate_parameters(spec, {"mode": "fast", "iterations": 5}, deep=True)  # does not raise

    def test_deep_accepts_nullable_union(self) -> None:
        spec = make_spec({"type": "object", "properties": {"power": {"type": ["number", "null"]}}})
        validate_parameters(spec, {"power": None}, deep=True)  # does not raise

    def test_deep_accepts_anyof_union(self) -> None:
        spec = make_spec(
            {"type": "object", "properties": {"shape": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}
        )
        validate_parameters(spec, {"shape": "circle"}, deep=True)
        validate_parameters(spec, {"shape": 3}, deep=True)

    def test_closed_objects_reject_new_fields_without_weakening_known_constraints(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {"config": {"$ref": "#/$defs/Config"}},
                "required": ["config"],
                "additionalProperties": False,
                "$defs": {
                    "Config": {
                        "type": "object",
                        "properties": {"count": {"type": "integer", "minimum": 1}},
                        "required": ["count"],
                        "additionalProperties": False,
                    }
                },
            }
        )
        parameters = {"config": {"count": 2}}
        validate_parameters(spec, parameters, deep=True)
        self.assertEqual({"config": {"count": 2}}, parameters)

        for invalid in ({}, {"count": "2"}, {"count": 0}, {"count": 2, "new_option": None}):
            with self.subTest(invalid=invalid), self.assertRaises(ParameterValidationError):
                validate_parameters(spec, {"config": invalid}, deep=True)
        with self.assertRaisesRegex(ParameterValidationError, "new_parameter"):
            validate_parameters(spec, {**parameters, "new_parameter": False}, deep=True)

    def test_closed_discriminated_union_branches_reject_new_fields(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {
                    "method": {
                        "discriminator": {"propertyName": "kind"},
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {"kind": {"const": "ordinary"}},
                                "required": ["kind"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {"kind": {"const": "simple"}, "mean": {"type": "number"}},
                                "required": ["kind", "mean"],
                                "additionalProperties": False,
                            },
                        ],
                    }
                },
            }
        )
        validate_parameters(spec, {"method": {"kind": "simple", "mean": 2.0}}, deep=True)
        with self.assertRaisesRegex(ParameterValidationError, "new_option"):
            validate_parameters(spec, {"method": {"kind": "simple", "mean": 2.0, "new_option": True}}, deep=True)
        with self.assertRaisesRegex(ParameterValidationError, r"method\.mean"):
            validate_parameters(spec, {"method": {"kind": "simple", "mean": "invalid"}}, deep=True)

    def test_typed_maps_and_schema_literals_keep_their_constraints(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {
                    "counts": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}},
                    "literal": {"const": {"additionalProperties": False, "unevaluatedProperties": False}},
                },
                "unevaluatedProperties": False,
            }
        )
        original = json.dumps(spec.parameters, sort_keys=True)
        literal = {"additionalProperties": False, "unevaluatedProperties": False}
        validate_parameters(spec, {"counts": {"new_counter": 2}, "literal": literal}, deep=True)
        self.assertEqual(original, json.dumps(spec.parameters, sort_keys=True))
        with self.assertRaisesRegex(ParameterValidationError, "new_field"):
            validate_parameters(spec, {"new_field": None}, deep=True)
        for invalid in ("2", 0):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ParameterValidationError, r"counts\.new_counter"),
            ):
                validate_parameters(spec, {"counts": {"new_counter": invalid}}, deep=True)
        with self.assertRaisesRegex(ParameterValidationError, "literal"):
            validate_parameters(
                spec, {"literal": {"additionalProperties": True, "unevaluatedProperties": True}}, deep=True
            )

    # -- reference-leaf relaxation ----------------------------------------- #

    def test_deep_relaxes_a_reference_leaf_declared_as_its_resolved_form(self) -> None:
        """A leaf typed as the object it resolves to accepts the plain reference instead."""
        spec = make_spec(
            {
                "type": "object",
                "properties": {"source": {"type": "object", "reference_to": "geoscience-object"}},
                "required": ["source"],
            }
        )
        validate_parameters(spec, {"source": "https://example/objects/abc"}, deep=True)  # does not raise

    def test_deep_holds_a_scalar_reference_to_its_declared_type(self) -> None:
        """A reference transmitted as a string is still checked -- that is what the stub promises."""
        spec = make_spec(
            {
                "type": "object",
                "properties": {"source": {"type": "string", "format": "uri", "reference_to": "geoscience-object"}},
                "required": ["source"],
            }
        )
        validate_parameters(spec, {"source": "https://example/objects/abc"}, deep=True)  # does not raise
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"source": 12345}, deep=True)
        self.assertEqual(["source: expected type string, got int"], ctx.exception.errors)

    def test_deep_allows_null_for_a_nullable_scalar_reference(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {"source": {"type": ["string", "null"], "reference_to": "geoscience-object"}},
                "required": ["source"],
            }
        )
        validate_parameters(spec, {"source": None}, deep=True)  # does not raise
        with self.assertRaises(ParameterValidationError):
            validate_parameters(spec, {"source": 12345}, deep=True)

    def test_deep_keeps_the_transmitted_branch_of_a_mixed_declared_type(self) -> None:
        """Part scalar, part resolved form: the value still has to be one of the two."""
        spec = make_spec(
            {
                "type": "object",
                "properties": {"source": {"type": ["string", "object"], "reference_to": "geoscience-object"}},
                "required": ["source"],
            }
        )
        validate_parameters(spec, {"source": "https://example/objects/abc"}, deep=True)  # does not raise
        validate_parameters(spec, {"source": {"object": "abc"}}, deep=True)  # nor the resolved form
        with self.assertRaises(ParameterValidationError):
            validate_parameters(spec, {"source": 12345}, deep=True)

    def test_deep_relaxes_a_nullable_resolved_form(self) -> None:
        """``null`` says the leaf may be omitted, not that a reference travels as one.

        Counting it as a transmitted form would keep ``["object", "null"]`` declared, and
        resolution has already turned the reference into a URL by the time deep validation
        sees it -- so every non-null call would be rejected.
        """
        spec = make_spec(
            {
                "type": "object",
                "properties": {"source": {"type": ["object", "null"], "reference_to": "geoscience-object"}},
                "required": ["source"],
            }
        )
        validate_parameters(spec, {"source": "https://example/objects/abc"}, deep=True)  # does not raise
        validate_parameters(spec, {"source": None}, deep=True)  # nor the null the schema allows

    def test_deep_still_requires_reference_presence(self) -> None:
        """Relaxation loosens the value's shape but does not waive the required check."""
        spec = make_spec(
            {
                "type": "object",
                "properties": {"source": {"type": "object", "reference_to": "geoscience-object"}},
                "required": ["source"],
            }
        )
        with self.assertRaises(ParameterValidationError):
            validate_parameters(spec, {}, deep=True)

    def test_relaxation_leaves_literal_values_alone(self) -> None:
        """``reference_to`` inside a ``const`` literal is data, not a subschema to relax."""
        spec = make_spec(
            {"type": "object", "properties": {"meta": {"const": {"reference_to": "literal"}}}},
        )
        validate_parameters(spec, {"meta": {"reference_to": "literal"}}, deep=True)  # does not raise
        with self.assertRaises(ParameterValidationError):
            validate_parameters(spec, {"meta": {}}, deep=True)

    # -- friendly messages ------------------------------------------------- #

    def test_nested_required_message_names_its_location(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {
                    "config": {"type": "object", "properties": {"seed": {"type": "integer"}}, "required": ["seed"]}
                },
            }
        )
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"config": {}}, deep=True)
        self.assertEqual(["config: missing required parameter(s): 'seed'"], ctx.exception.errors)

    def test_top_level_required_message_is_unprefixed(self) -> None:
        """A root-level ``required`` the shallow pass can't see, reported without a location."""
        spec = make_spec(
            {"type": "object", "properties": {"mode": {"type": "string"}}, "allOf": [{"required": ["mode"]}]}
        )
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {}, deep=True)
        self.assertEqual(["missing required parameter(s): 'mode'"], ctx.exception.errors)

    def test_error_aggregates_multiple_failures(self) -> None:
        spec = make_spec(
            {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["fast"]},
                    "power": {"type": "number"},
                },
            }
        )
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"mode": "turbo", "power": "x"}, deep=True)
        self.assertEqual(2, len(ctx.exception.errors))

    def test_errors_at_mixed_type_keys_are_still_reported(self) -> None:
        """Sibling failures under int and str keys must not collide while being ordered."""
        spec = make_spec({"type": "object", "additionalProperties": {"type": "string"}})
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {0: 1, "name": 2}, deep=True)
        self.assertEqual(2, len(ctx.exception.errors))

    # -- discriminated unions ---------------------------------------------- #

    @staticmethod
    def _variogram_spec() -> TaskResource:
        """A Pydantic-style discriminated union whose branches are ``$ref``s."""
        return make_spec(
            {
                "type": "object",
                "properties": {
                    "variogram": {
                        "discriminator": {"propertyName": "kind"},
                        "oneOf": [{"$ref": "#/$defs/Spherical"}, {"$ref": "#/$defs/Exponential"}],
                    }
                },
                "required": ["variogram"],
                "$defs": {
                    "Spherical": {
                        "type": "object",
                        "properties": {"kind": {"const": "spherical"}, "range": {"type": "number"}},
                        "required": ["kind", "range"],
                    },
                    "Exponential": {
                        "type": "object",
                        "properties": {"kind": {"const": "exponential"}, "scale": {"type": "number"}},
                        "required": ["kind", "scale"],
                    },
                },
            }
        )

    def test_deep_accepts_a_valid_union_branch(self) -> None:
        validate_parameters(self._variogram_spec(), {"variogram": {"kind": "spherical", "range": 12.0}}, deep=True)

    def test_union_error_reports_the_selected_branch(self) -> None:
        """The discriminator picks the branch, so the message is about *that* branch."""
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(
                self._variogram_spec(), {"variogram": {"kind": "spherical", "range": "wide"}}, deep=True
            )
        self.assertEqual(["variogram.range: expected type number, got str"], ctx.exception.errors)

    def test_union_error_reports_a_missing_field_of_the_selected_branch(self) -> None:
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(self._variogram_spec(), {"variogram": {"kind": "spherical"}}, deep=True)
        self.assertEqual(["variogram: missing required parameter(s): 'range'"], ctx.exception.errors)

    def test_union_error_reports_allowed_discriminator_values(self) -> None:
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(self._variogram_spec(), {"variogram": {"kind": "gaussian"}}, deep=True)
        self.assertEqual(
            ["variogram.kind: must be one of ['spherical', 'exponential'], got 'gaussian'"], ctx.exception.errors
        )

    def test_undiscriminated_union_keeps_the_generic_message(self) -> None:
        """Without a discriminator no branch can be singled out, so no branch is guessed at."""
        spec = make_spec(
            {"type": "object", "properties": {"shape": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}
        )
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"shape": 1.5}, deep=True)
        self.assertIn("is not valid under any of the given schemas", ctx.exception.errors[0])

    def test_union_with_several_matching_branches_keeps_the_generic_message(self) -> None:
        """When more than one branch accepts the tag, no branch is guessed at either."""
        spec = make_spec(
            {
                "type": "object",
                "properties": {
                    "variogram": {
                        "discriminator": {"propertyName": "kind"},
                        "oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}],
                    }
                },
                "required": ["variogram"],
                "$defs": {
                    # Neither branch pins ``kind`` with a const, so both accept any kind.
                    "A": {
                        "type": "object",
                        "properties": {"kind": {"type": "string"}, "range": {"type": "number"}},
                        "required": ["kind", "range"],
                    },
                    "B": {
                        "type": "object",
                        "properties": {"kind": {"type": "string"}, "scale": {"type": "number"}},
                        "required": ["kind", "scale"],
                    },
                },
            }
        )
        with self.assertRaises(ParameterValidationError) as ctx:
            validate_parameters(spec, {"variogram": {"kind": "spherical", "range": "wide"}}, deep=True)
        self.assertIn("is not valid under any of the given schemas", ctx.exception.errors[0])


class TestEngineValidation(TestWithConnector):
    """Engine-level tests for the validation toggles (shallow default, deep opt-in)."""

    # Reference resolution runs between the two validation passes, so the engine-level
    # tests have to name an object the way a caller really would.
    SOURCE_URL = (
        "https://unittest.localhost/geoscience-object"
        "/orgs/00000000-0000-0000-0000-000000000001"
        "/workspaces/00000000-0000-0000-0000-000000000002"
        "/objects/00000000-0000-0000-0000-000000000010"
    )

    def setUp(self) -> None:
        super().setUp()
        self.context = FakeContext(self.connector, TEST_ORG.id)
        self.catalogue = load_test_data("discovery-validation.json")

    @contextmanager
    def catalogue_response(self) -> Iterator[None]:
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(self.catalogue),
            headers={"Content-Type": "application/json"},
        ):
            yield

    @contextmanager
    def mock_job_client(self) -> Iterator[mock.AsyncMock]:
        job = mock.Mock()
        job.wait_for_results = mock.AsyncMock(return_value={"ok": True})
        submit = mock.AsyncMock(return_value=job)
        with mock.patch("evo.compute.engine.JobClient") as mock_job_client:
            mock_job_client.submit = submit
            yield submit

    async def test_shallow_is_on_by_default_deep_is_off(self) -> None:
        """By default an enum violation slips through (deep off) and the job is submitted."""
        client = ComputeClient(self.context)
        with self.catalogue_response(), self.mock_job_client() as submit:
            await client.demo.widget.run(source=self.SOURCE_URL, mode="turbo")
        submit.assert_awaited_once()

    async def test_deep_validation_rejects_bad_enum(self) -> None:
        client = ComputeClient(self.context, deep_validation=True)
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaises(ParameterValidationError) as ctx:
                await client.demo.widget.run(source=self.SOURCE_URL, mode="turbo")
        self.assertIn("mode", str(ctx.exception))
        submit.assert_not_awaited()

    async def test_deep_validation_rejects_constraint_violation(self) -> None:
        client = ComputeClient(self.context, deep_validation=True)
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaises(ParameterValidationError):
                await client.demo.widget.run(source=self.SOURCE_URL, mode="fast", iterations=0)
        submit.assert_not_awaited()

    async def test_deep_validation_rejects_explicit_none_for_a_non_nullable_field(self) -> None:
        """An explicit ``None`` reaches validation rather than being silently dropped."""
        client = ComputeClient(self.context, deep_validation=True)
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaises(ParameterValidationError) as ctx:
                await client.demo.widget.run(source=self.SOURCE_URL, mode="fast", iterations=None)
        self.assertIn("iterations", str(ctx.exception))
        submit.assert_not_awaited()

    async def test_deep_validation_accepts_reference_string(self) -> None:
        """Deep validation passes when the required reference is supplied as a string."""
        client = ComputeClient(self.context, deep_validation=True)
        with self.catalogue_response(), self.mock_job_client() as submit:
            await client.demo.widget.run(source=self.SOURCE_URL, mode="fast")
        submit.assert_awaited_once()

    async def test_deep_validation_can_be_forced_per_call(self) -> None:
        client = ComputeClient(self.context)  # deep off at the client level
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaises(ParameterValidationError):
                await client.arun("demo", "widget", {"source": self.SOURCE_URL, "mode": "turbo"}, deep_validation=True)
        submit.assert_not_awaited()

    async def test_validation_can_be_disabled_per_call(self) -> None:
        """With validation off, a required field set to ``None`` goes out for the platform to reject."""
        client = ComputeClient(self.context)
        with self.catalogue_response(), self.mock_job_client() as submit:
            await client.arun("demo", "widget", {"source": self.SOURCE_URL, "mode": None}, validate=False)
        submit.assert_awaited_once()
        self.assertIsNone(submit.await_args.kwargs["parameters"]["mode"])

    async def test_validation_on_catches_required_set_to_none(self) -> None:
        client = ComputeClient(self.context)
        with self.catalogue_response(), self.mock_job_client() as submit:
            with self.assertRaises(ParameterValidationError):
                await client.arun("demo", "widget", {"source": self.SOURCE_URL, "mode": None})
        submit.assert_not_awaited()

    async def test_validate_off_at_the_client_disables_deep_validation_too(self) -> None:
        """``validate`` is the master switch, so ``deep_validation=True`` under it does nothing."""
        client = ComputeClient(self.context, validate=False, deep_validation=True)
        with self.catalogue_response(), self.mock_job_client() as submit:
            await client.demo.widget.run(source=self.SOURCE_URL, mode="turbo")
        submit.assert_awaited_once()

    async def test_validate_off_per_call_disables_deep_validation_too(self) -> None:
        client = ComputeClient(self.context, deep_validation=True)
        with self.catalogue_response(), self.mock_job_client() as submit:
            await client.arun("demo", "widget", {"source": self.SOURCE_URL, "mode": "turbo"}, validate=False)
        submit.assert_awaited_once()

    async def test_validate_off_per_call_overrides_deep_validation_on_per_call(self) -> None:
        client = ComputeClient(self.context)
        with self.catalogue_response(), self.mock_job_client() as submit:
            await client.arun(
                "demo",
                "widget",
                {"source": self.SOURCE_URL, "mode": "turbo"},
                validate=False,
                deep_validation=True,
            )
        submit.assert_awaited_once()


class FakeContext:
    """Minimal duck-typed IContext exposing just what ComputeClient uses."""

    def __init__(self, connector, org_id) -> None:
        self._connector = connector
        self._org_id = org_id

    def get_connector(self):
        return self._connector

    def get_org_id(self):
        return self._org_id

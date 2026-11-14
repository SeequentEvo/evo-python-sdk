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

"""Annotation-conformance tests for the generic engine's closed schema vocabulary.

The engine handles a task generically only because every task schema is built from
standard JSON Schema plus a small, closed set of custom annotations. These tests fail
if a schema uses an annotation the engine does not know about, flagging that the engine
needs updating before that task can be handled generically.

Two variants:

* Fixture-based (always runs): checks the bundled catalogue fixtures and the vocabulary
  constant.
* Live (opt-in): checks the organization's real, live catalogue. Enable by setting
  ``EVO_COMPUTE_LIVE_DISCOVERY=1`` along with ``EVO_ACCESS_TOKEN``, ``EVO_HUB_URL`` and
  ``EVO_ORG_ID``.
"""

from __future__ import annotations

import os
import unittest
from uuid import UUID

from data import load_test_data
from evo.compute import TaskResource
from evo.compute.validation import KNOWN_SCHEMA_ANNOTATIONS, unknown_annotation_keys

_EXPECTED_VOCABULARY = frozenset(
    {
        "reference_to",
        "output",
        "supported_schemas",
        "attribute_from",
        "attribute_path",
        "target",
        "discriminator",
        "composite",
    }
)


def _unknown_keys_for_task(task: TaskResource) -> set[str]:
    return unknown_annotation_keys(task.parameters) | unknown_annotation_keys(task.results)


class TestSchemaAnnotationConformance(unittest.TestCase):
    def test_vocabulary_matches_expected_set(self) -> None:
        """Guard the closed vocabulary against accidental drift."""
        self.assertEqual(_EXPECTED_VOCABULARY, KNOWN_SCHEMA_ANNOTATIONS)

    def test_bundled_catalogue_uses_only_known_annotations(self) -> None:
        offenders: dict[str, set[str]] = {}
        for fixture in ("discovery-tasks.json", "discovery-resolution.json"):
            for result in load_test_data(fixture)["results"]:
                task = TaskResource.model_validate(result)
                unknown = _unknown_keys_for_task(task)
                if unknown:
                    offenders[f"{fixture}:{task.name}"] = unknown
        self.assertEqual({}, offenders, f"unknown schema annotations found: {offenders}")

    def test_unknown_annotation_is_flagged(self) -> None:
        schema = {
            "type": "object",
            "properties": {"source": {"type": "object", "made_up_annotation": True}},
        }
        self.assertEqual({"made_up_annotation"}, unknown_annotation_keys(schema))

    def test_known_annotations_are_not_flagged(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "source": {
                    "type": "object",
                    "reference_to": "geoscience-object",
                    "supported_schemas": ["x"],
                },
                "attr": {"type": "string", "reference_to": "attribute", "attribute_from": "0/source"},
                "filter": {"type": "object", "composite": "filter"},
            },
            "$defs": {
                "Union": {
                    "discriminator": {"propertyName": "kind"},
                    "oneOf": [{"type": "object"}],
                }
            },
        }
        self.assertEqual(set(), unknown_annotation_keys(schema))

    def test_property_named_like_keyword_is_not_flagged(self) -> None:
        """A property whose *name* collides with a keyword must not be treated as one."""
        schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "reference_to": {"type": "string"},
                "output": {"type": "number"},
            },
        }
        self.assertEqual(set(), unknown_annotation_keys(schema))

    def test_data_values_are_not_walked(self) -> None:
        """Arbitrary data in enum/default/const must not be mistaken for schema keywords."""
        schema = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast"], "default": "fast"},
                "meta": {"const": {"made_up_annotation": 1}},
            },
        }
        self.assertEqual(set(), unknown_annotation_keys(schema))

    def test_output_annotation_in_results_is_known(self) -> None:
        results = {"type": "object", "properties": {"target": {"type": "object", "output": "geoscience-object"}}}
        self.assertEqual(set(), unknown_annotation_keys(results))


class TestLiveCatalogueConformance(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        os.getenv("EVO_COMPUTE_LIVE_DISCOVERY"),
        "set EVO_COMPUTE_LIVE_DISCOVERY=1 (+ EVO_ACCESS_TOKEN, EVO_HUB_URL, EVO_ORG_ID) to run the live check",
    )
    async def test_live_catalogue_uses_only_known_annotations(self) -> None:
        from evo.aio import AioTransport
        from evo.common import APIConnector
        from evo.oauth import AccessTokenAuthorizer

        from evo.compute import DiscoveryClient

        transport = AioTransport(user_agent="evo-compute-conformance-test")
        authorizer = AccessTokenAuthorizer(access_token=os.environ["EVO_ACCESS_TOKEN"])
        connector = APIConnector(base_url=os.environ["EVO_HUB_URL"], transport=transport, authorizer=authorizer)
        async with connector:
            client = DiscoveryClient(connector, UUID(os.environ["EVO_ORG_ID"]))
            tasks = await client.list_tasks()

        self.assertTrue(tasks, "live discovery returned no tasks")
        offenders = {task.name: unknown for task in tasks if (unknown := _unknown_keys_for_task(task))}
        self.assertEqual({}, offenders, f"unknown schema annotations in live catalogue: {offenders}")

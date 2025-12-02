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

import uuid
from unittest import TestCase
from unittest.mock import Mock

from evo.common import Environment, EvoContext
from evo.common.exceptions import ContextError

test_org_id = uuid.uuid4()
test_workspace_id = uuid.uuid4()


class TestEvoContext(TestCase):
    def test_from_parts_minimal(self):
        evo_context = EvoContext(
            transport=Mock(),
            authorizer=Mock(),
        )
        self.assertIsNone(evo_context.cache)

        with self.assertRaises(ContextError):
            evo_context.get_connector()

        with self.assertRaises(ContextError):
            evo_context.get_environment()

    def test_from_parts_only_org(self):
        cache = Mock()
        evo_context = EvoContext(
            transport=Mock(),
            authorizer=Mock(),
            cache=cache,
            hub_url="https://example.com",
        )
        self.assertIs(evo_context.cache, cache)
        connector = evo_context.get_connector()
        self.assertEqual(connector.base_url, "https://example.com/")

        with self.assertRaises(ContextError):
            evo_context.get_environment()

    def test_from_parts_full(self):
        cache = Mock()
        evo_context = EvoContext(
            transport=Mock(),
            authorizer=Mock(),
            cache=cache,
            hub_url="https://example.com",
            org_id=test_org_id,
            workspace_id=test_workspace_id,
        )
        self.assertIs(evo_context.cache, cache)
        connector = evo_context.get_connector()
        self.assertEqual(connector.base_url, "https://example.com/")

        environment = evo_context.get_environment()
        self.assertEqual(environment.hub_url, "https://example.com")
        self.assertEqual(environment.org_id, test_org_id)
        self.assertEqual(environment.workspace_id, test_workspace_id)

    def test_from_connector(self):
        connector = Mock()
        connector.base_url = "https://example.com/"
        cache = Mock()
        evo_context = EvoContext(
            connector=connector,
            cache=cache,
        )
        self.assertIs(evo_context.cache, cache)
        self.assertIs(evo_context.get_connector(), connector)

        with self.assertRaises(ContextError):
            evo_context.get_environment()

    def test_constructor_mixed(self):
        with self.assertRaises(ValueError):
            EvoContext(
                transport=Mock(),
                authorizer=Mock(),
                connector=Mock(),
            )

    def test_constructor_empty(self):
        with self.assertRaises(ValueError):
            EvoContext()

    def test_from_environment(self):
        environment = Environment(
            hub_url="https://example.com/",
            org_id=test_org_id,
            workspace_id=test_workspace_id,
        )
        connector = Mock()
        connector.base_url = "https://example.com/"
        cache = Mock()
        evo_context = EvoContext.from_environment(environment, connector, cache)
        self.assertIs(evo_context.cache, cache)
        self.assertIs(evo_context.get_connector(), connector)
        self.assertEqual(evo_context.get_environment(), environment)

    def test_from_environment_wrong_hub(self):
        environment = Environment(
            hub_url="https://hub.com/",
            org_id=test_org_id,
            workspace_id=test_workspace_id,
        )
        connector = Mock()
        connector.base_url = "https://example.com/"
        with self.assertRaises(ContextError):
            EvoContext.from_environment(environment, connector)

    def test_with_cache(self):
        evo_context = EvoContext(
            transport=Mock(),
            authorizer=Mock(),
            hub_url="https://example.com",
            org_id=test_org_id,
            workspace_id=test_workspace_id,
        )
        self.assertIsNone(evo_context.cache)

        cache = Mock()
        evo_context_with_cache = evo_context.with_cache(cache)
        self.assertIs(evo_context_with_cache.cache, cache)
        self.assertIsNot(evo_context, evo_context_with_cache)

    def test_with_cache_connector(self):
        connector = Mock()
        connector.base_url = "https://example.com/"
        evo_context = EvoContext(
            connector=Mock(),
        )
        cache = Mock()
        evo_context_with_cache = evo_context.with_cache(cache)
        self.assertIs(evo_context_with_cache.cache, cache)
        self.assertIsNot(evo_context, evo_context_with_cache)

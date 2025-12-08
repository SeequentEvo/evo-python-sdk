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

from unittest import TestCase
from unittest.mock import Mock

from evo.common import Environment, StaticContext
from evo.common.exceptions import ContextError
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID


class TestStaticContext(TestCase):
    def test_from_parts_minimal(self):
        evo_context = StaticContext(
            transport=Mock(),
            authorizer=Mock(),
        )
        self.assertIsNone(evo_context.get_cache())

        with self.assertRaises(ContextError):
            evo_context.get_connector()

        with self.assertRaises(ContextError):
            evo_context.get_environment()

    def test_from_parts_only_org(self):
        cache = Mock()
        evo_context = StaticContext(
            transport=Mock(),
            authorizer=Mock(),
            cache=cache,
            hub_url=BASE_URL,
        )
        self.assertIs(evo_context.get_cache(), cache)
        connector = evo_context.get_connector()
        self.assertEqual(connector.base_url, BASE_URL)
        with self.assertRaises(ContextError):
            evo_context.get_environment()

    def test_from_parts_full(self):
        cache = Mock()
        evo_context = StaticContext(
            transport=Mock(),
            authorizer=Mock(),
            cache=cache,
            hub_url=BASE_URL,
            org_id=ORG.id,
            workspace_id=WORKSPACE_ID,
        )
        self.assertIs(evo_context.get_cache(), cache)
        connector = evo_context.get_connector()
        self.assertEqual(connector.base_url, BASE_URL)

        environment = evo_context.get_environment()
        self.assertEqual(environment.hub_url, BASE_URL)
        self.assertEqual(environment.org_id, ORG.id)
        self.assertEqual(environment.workspace_id, WORKSPACE_ID)

    def test_from_connector(self):
        connector = Mock()
        connector.base_url = BASE_URL
        cache = Mock()
        evo_context = StaticContext(
            connector=connector,
            cache=cache,
        )
        self.assertIs(evo_context.get_cache(), cache)
        self.assertIs(evo_context.get_connector(), connector)

        with self.assertRaises(ContextError):
            evo_context.get_environment()

    def test_constructor_mixed(self):
        with self.assertRaises(ValueError):
            StaticContext(
                transport=Mock(),
                authorizer=Mock(),
                connector=Mock(),
            )

    def test_constructor_empty(self):
        with self.assertRaises(ValueError):
            StaticContext()

    def test_from_environment(self):
        environment = Environment(
            hub_url=BASE_URL,
            org_id=ORG.id,
            workspace_id=WORKSPACE_ID,
        )
        connector = Mock()
        connector.base_url = BASE_URL
        cache = Mock()
        evo_context = StaticContext.from_environment(environment, connector, cache)
        self.assertIs(evo_context.get_cache(), cache)
        self.assertIs(evo_context.get_connector(), connector)
        self.assertEqual(evo_context.get_environment(), environment)

    def test_from_environment_wrong_hub(self):
        environment = Environment(
            hub_url="https://wrong-hub.com/",
            org_id=ORG.id,
            workspace_id=WORKSPACE_ID,
        )
        connector = Mock()
        connector.base_url = BASE_URL
        with self.assertRaises(ContextError):
            StaticContext.from_environment(environment, connector)

    def test_copy_context(self):
        context = Mock()
        context.get_connector.return_value = Mock()
        context.get_connector.return_value.base_url = BASE_URL
        context.get_cache.return_value = Mock()
        context.get_environment.return_value = Environment(
            hub_url=BASE_URL,
            org_id=ORG.id,
            workspace_id=WORKSPACE_ID,
        )

        copied = StaticContext.create_copy(context)
        self.assertIs(copied.get_connector(), context.get_connector.return_value)
        self.assertIs(copied.get_cache(), context.get_cache.return_value)
        environment = copied.get_environment()
        self.assertEqual(environment.hub_url, BASE_URL)
        self.assertEqual(environment.org_id, ORG.id)
        self.assertEqual(environment.workspace_id, WORKSPACE_ID)
        self.assertEqual(copied.get_org_id(), ORG.id)

    def test_copy_context_no_workspace(self):
        context = Mock()
        context.get_connector.return_value = Mock()
        context.get_cache.return_value = Mock()
        context.get_environment.side_effect = ContextError()
        context.get_org_id.return_value = ORG.id

        copied = StaticContext.create_copy(context)
        self.assertIs(copied.get_connector(), context.get_connector.return_value)
        self.assertIs(copied.get_cache(), context.get_cache.return_value)
        with self.assertRaises(ContextError):
            copied.get_environment()
        self.assertEqual(copied.get_org_id(), ORG.id)

    def test_copy_context_no_org(self):
        context = Mock()
        context.get_connector.return_value = Mock()
        context.get_cache.return_value = Mock()
        context.get_environment.side_effect = ContextError()
        context.get_org_id.side_effect = ContextError()

        copied = StaticContext.create_copy(context)
        self.assertIs(copied.get_connector(), context.get_connector.return_value)
        self.assertIs(copied.get_cache(), context.get_cache.return_value)
        with self.assertRaises(ContextError):
            copied.get_environment()
        with self.assertRaises(ContextError):
            copied.get_org_id()

    def test_copy_context_no_connector(self):
        context = Mock()
        context.get_connector.side_effect = ContextError()

        with self.assertRaises(ContextError):
            StaticContext.create_copy(context)

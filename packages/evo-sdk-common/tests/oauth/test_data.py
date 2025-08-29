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

import unittest

from parameterized import parameterized

from evo.oauth import EvoScopes, OAuthScopes

EXPECTED_SCOPE_NAMES = {
    EvoScopes.offline_access: "offline_access",
    EvoScopes.evo_discovery: "evo.discovery",
    EvoScopes.evo_workspace: "evo.workspace",
    EvoScopes.evo_blocksync: "evo.blocksync",
    EvoScopes.evo_object: "evo.object",
    EvoScopes.evo_file: "evo.file",
    EvoScopes.evo_audit: "evo.audit",
}


class TestEvoScopes(unittest.TestCase):
    @parameterized.expand(
        [
            (EvoScopes.offline_access,),
            (EvoScopes.evo_discovery,),
            (EvoScopes.evo_workspace,),
            (EvoScopes.evo_blocksync,),
            (EvoScopes.evo_object,),
            (EvoScopes.evo_file,),
            (EvoScopes.evo_audit,),
        ],
        name_func=lambda func,
        index,
        params: f"{func.__name__}_{index}_{parameterized.to_safe_name(str(params[0][0]))}",
    )
    def test_single_scope_string(self, scope: EvoScopes) -> None:
        self.assertIn(scope, EXPECTED_SCOPE_NAMES)
        expected_string = EXPECTED_SCOPE_NAMES[scope]
        self.assertEqual(expected_string, str(scope))
        self.assertEqual(1, len(scope.members), "Expected a single scope")

    @classmethod
    def expand_scopes(cls, scopes: tuple[EvoScopes, ...]) -> list[EvoScopes]:
        """Expand a tuple of scopes into a list of individual scopes."""
        all_scopes = set()
        for scope in scopes:
            all_scopes.update(scope.members)

        return sorted(all_scopes)

    @parameterized.expand(
        [
            (
                EvoScopes.default,
                (
                    EvoScopes.evo_discovery,
                    EvoScopes.evo_workspace,
                ),
            ),
            (
                EvoScopes.all_evo,
                (
                    EvoScopes.default,
                    EvoScopes.evo_blocksync,
                    EvoScopes.evo_object,
                    EvoScopes.evo_file,
                ),
            ),
        ],
        name_func=lambda f, n, p: f"{f.__name__}_{n}_{parameterized.to_safe_name(p[0][0].name)}",
    )
    def test_multiple_scope_string(self, scopes: EvoScopes, expected_scopes: tuple[EvoScopes, ...]) -> None:
        expected_string = " ".join([EXPECTED_SCOPE_NAMES[scope] for scope in self.expand_scopes(expected_scopes)])
        self.assertEqual(expected_string, str(scopes))

        for scope in expected_scopes:
            self.assertIn(scope, scopes)

        self.assertEqual(len(self.expand_scopes(expected_scopes)), len(scopes.members))

    def test_oauth_scope_alias(self) -> None:
        """Test that the OAuthScopes alias points to EvoScopes for backwards compatibility."""
        self.assertIs(EvoScopes, OAuthScopes)

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

from evo.oauth import OAuthScopes
from evo.oauth.data import OIDCConfig
from evo.oauth.exceptions import OIDCError

EXPECTED_SCOPE_NAMES = {
    OAuthScopes.openid: "openid",
    OAuthScopes.profile: "profile",
    OAuthScopes.organization: "organization",
    OAuthScopes.email: "email",
    OAuthScopes.address: "address",
    OAuthScopes.phone: "phone",
    OAuthScopes.offline_access: "offline_access",
    OAuthScopes.evo_discovery: "evo.discovery",
    OAuthScopes.evo_workspace: "evo.workspace",
    OAuthScopes.evo_blocksync: "evo.blocksync",
    OAuthScopes.evo_object: "evo.object",
    OAuthScopes.evo_file: "evo.file",
}


class TestOAuthScopes(unittest.TestCase):
    @parameterized.expand(
        [
            (OAuthScopes.openid,),
            (OAuthScopes.profile,),
            (OAuthScopes.email,),
            (OAuthScopes.address,),
            (OAuthScopes.phone,),
            (OAuthScopes.offline_access,),
            (OAuthScopes.evo_discovery,),
            (OAuthScopes.evo_workspace,),
            (OAuthScopes.evo_blocksync,),
            (OAuthScopes.evo_object,),
            (OAuthScopes.evo_file,),
        ],
        name_func=lambda func,
        index,
        params: f"{func.__name__}_{index}_{parameterized.to_safe_name(str(params[0][0]))}",
    )
    def test_single_scope_string(self, scope: OAuthScopes) -> None:
        self.assertIn(scope, EXPECTED_SCOPE_NAMES)
        expected_string = EXPECTED_SCOPE_NAMES[scope]
        self.assertEqual(expected_string, str(scope))
        self.assertEqual(1, len(scope), "Expected a single scope")

    @classmethod
    def expand_scopes(cls, scopes: tuple[OAuthScopes, ...]) -> list[OAuthScopes]:
        """Expand a tuple of scopes into a list of individual scopes."""
        all_scopes = set()
        for scope in scopes:
            if len(scope) == 1:
                all_scopes.add(scope)
            else:
                for member in OAuthScopes:
                    if member in scope and len(member) == 1:
                        all_scopes.add(member)

        return sorted(all_scopes, key=lambda s: s.value)

    @parameterized.expand(
        [
            (
                OAuthScopes.openid | OAuthScopes.profile,
                (OAuthScopes.openid, OAuthScopes.profile),
            ),
            (
                OAuthScopes.default,
                (
                    OAuthScopes.openid,
                    OAuthScopes.profile,
                    OAuthScopes.email,
                    OAuthScopes.organization,
                    OAuthScopes.evo_discovery,
                    OAuthScopes.evo_workspace,
                ),
            ),
            (
                OAuthScopes.all_evo,
                (
                    OAuthScopes.default,
                    OAuthScopes.evo_blocksync,
                    OAuthScopes.evo_object,
                    OAuthScopes.evo_file,
                ),
            ),
        ],
        name_func=lambda f, n, p: f"{f.__name__}_{n}_{parameterized.to_safe_name(p[0][0].name)}",
    )
    def test_multiple_scope_string(self, scopes: OAuthScopes, expected_scopes: tuple[OAuthScopes, ...]) -> None:
        expected_string = " ".join([EXPECTED_SCOPE_NAMES[scope] for scope in self.expand_scopes(expected_scopes)])
        self.assertEqual(expected_string, str(scopes))

        for scope in expected_scopes:
            self.assertIn(scope, scopes)

        self.assertEqual(len(self.expand_scopes(expected_scopes)), len(scopes))


class TestOIDCConfig(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "invalid issuer for authorization_endpoint",
                "http://test.com",
                "http://wrong/auth",
                "http://test.com/token",
                "OIDC field authorization_endpoint must be a URL under the issuer",
            ),
            (
                "invalid issuer for token_endpoint",
                "http://test.com",
                "http://test.com/auth",
                "http://wrong/token",
                "OIDC field token_endpoint must be a URL under the issuer",
            ),
            (
                "missing issuer",
                None,
                "http://test.com/auth",
                "http://test.com/token",
                "Missing issuer url in OIDC config.",
            ),
        ]
    )
    def test_config_validation_errors(self, _, issuer, authorization_endpoint, token_endpoint, expected_msg):
        with self.assertRaises(OIDCError) as ex:
            OIDCConfig(
                issuer=issuer,
                authorization_endpoint=authorization_endpoint,
                token_endpoint=token_endpoint,
                response_types_supported=set(["code"]),
            )
        self.assertEqual(str(ex.exception), expected_msg)

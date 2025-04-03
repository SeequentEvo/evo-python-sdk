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

from __future__ import annotations

import json
from asyncio.log import logger
from base64 import urlsafe_b64decode
from datetime import datetime, timedelta, timezone
from enum import Flag, auto
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from evo.common.utils import BackoffLinear, Retry
from evo.oauth.exceptions import OAuthError, OIDCError
from urllib.parse import urlparse

__all__ = [
    "AccessToken",
    "DeviceFlowResponse",
    "OAuthScopes",
    "UserAccessToken",
]

_ALLOWABLE_CLOCK_DRIFT = timedelta(minutes=5)


def _utcnow() -> datetime:
    """Get the current UTC time.

    :return: The current UTC time.
    """
    return datetime.now(timezone.utc)


class OAuthScopes(Flag):
    """Public OAuth scopes for authenticating against Evo APIs.

    https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens#about-evo-access-tokens
    """

    # OpenID Scopes
    openid = auto()
    profile = auto()
    organization = auto()
    email = auto()
    address = auto()
    phone = auto()

    offline_access = auto()

    # Evo Scopes
    evo_discovery = auto()
    """Required for access to the Evo Discovery API, both discovery and token endpoints."""

    evo_workspace = auto()
    """Required for access to the Workspaces API."""

    evo_blocksync = auto()
    """Required for access to the BlockSync API."""

    evo_object = auto()
    """Required for access to the Geoscience Object API."""

    evo_file = auto()
    """Required for access to the File API."""

    # Useful combinations
    """Default scopes for Evo applications.

    Extra scopes to be added as needed include `evo_blocksync`, `evo_geoscienceobject`, or `evo_file`.
    """
    default = openid | profile | organization | email | evo_discovery | evo_workspace

    """All scopes for Evo applications."""
    all_evo = default | evo_blocksync | evo_object | evo_file

    def __str__(self) -> str:
        """Format a space-separated list of scopes for the OAuth request."""
        scopes = sorted([scope for scope in self.__class__ if scope in self and len(scope) == 1], key=lambda s: s.value)
        return " ".join([scopes_map[scope] for scope in scopes])

    def __len__(self) -> int:
        """Count the number of unique scopes that are set."""
        return len([scope for scope in self.__class__ if scope in self and (scope.value & (scope.value - 1)) == 0])


scopes_map = {
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


class AccessToken(BaseModel):
    """A base model for token responses for an OAuth server.
    https://www.rfc-editor.org/rfc/rfc6749#section-5.1
    """

    token_type: Literal["Bearer"]
    access_token: str
    """The access token issued by the authorization server."""

    expires_in: Optional[int] = None
    """The lifetime in seconds of the access token."""

    issued_at: datetime = Field(default_factory=_utcnow)
    """The time at which the token response was received."""

    scope: Optional[str] = None
    """The scope of the access token."""

    @property
    def expires_at(self) -> datetime | None:
        """The time at which the token expires, or None if the token lifetime is unknown.

        A small margin of error may occur due to the time taken to process the token response.
        """
        if self.expires_in is None:
            return None
        return self.issued_at + timedelta(seconds=self.expires_in)

    @property
    def ttl(self) -> int | None:
        """The time-to-live (TTL) of this access token in seconds, or None if the token lifetime is unknown.

        If the token is expired, the TTL will be 0.
        """
        if self.expires_at is None:
            return None
        ttl = self.expires_at - _utcnow()
        return max(round(ttl.total_seconds()), 0)

    @property
    def is_expired(self) -> bool:
        """Whether this access token has expired.

        True if the access token has expired, False otherwise.
        If the token lifetime is unknown, this method will always return False.
        """
        if (expiry := self.expires_at) is not None:
            return _utcnow() > expiry
        else:
            return False


class UserAccessToken(AccessToken):
    """A bearer token response from an OAuth server, for authorication_grant flows.

    https://www.rfc-editor.org/rfc/rfc6749#section-5.1
    """

    id_token: str
    """The ID token issued by the authorization server."""

    refresh_token: Optional[str] = None
    """The refresh token, which can be used to obtain new access tokens using the same authorization grant."""

    def validate_id_token(self, issuer: str, client_id: str) -> None:
        """Validate an ID token according to the OpenID Connect specification.

        https://openid.net/specs/openid-connect-basic-1_0.html#IDTokenValidation

        :param issuer: The issuer URL to validate against.
        :param client_id: The client ID to validate against.

        :raises OAuthError: If the ID token is invalid or cannot be decoded.
        """
        # The Client MUST validate the ID Token in the Token Response. To do this, the Client can split the ID Token at
        # the period (".") characters, take the second segment, and base64url decode it to obtain a JSON object containing
        # the ID Token Claims, which MUST be validated as follows:
        try:  # Broad exception handling to catch any errors during id_token validation.
            # Add surplus padding to make sure this can be decoded.
            # https://stackoverflow.com/a/49459036
            id_token = self.id_token.split(".")[1] + "=="
            claims: dict = json.loads(urlsafe_b64decode(id_token).decode("utf-8"))

            # 1. The Issuer Identifier for the OpenID Provider (which is typically obtained during Discovery) MUST exactly
            #    match the value of the iss (issuer) Claim.
            parsed_issuer = urlparse(issuer)
            if parsed_issuer.hostname and parsed_issuer.hostname.endswith(".seequent.com"):
                # Seequent ID is not fully compliant with OpenID Connect.
                logger.debug("Skipping issuer validation for Seequent ID.")
            elif claims.get("iss") != issuer:
                raise OAuthError("Issuer identifier does not match the value of the iss Claim.")

            # 2. The Client MUST validate that the aud (audience) Claim contains its client_id value registered at the Issuer
            #    identified by the iss (issuer) Claim as an audience. The ID Token MUST be rejected if the ID Token does not
            #    list the Client as a valid audience, or if it contains additional audiences not trusted by the Client.
            if client_id not in claims.get("aud"):  # `aud` may be a list of string, or a single string.
                raise OAuthError("Audience claim does not contain the client_id value.")

            # 3. The current time MUST be before the time represented by the exp Claim (possibly allowing for some small
            #    leeway to account for clock skew).
            exp_claim = datetime.fromtimestamp(claims.get("exp"), tz=timezone.utc) + _ALLOWABLE_CLOCK_DRIFT
            if self.issued_at > exp_claim:
                raise OAuthError("Token has expired.")

            # 4. The iat Claim can be used to reject tokens that were issued too far away from the current time, limiting the
            #    amount of time that nonces need to be stored to prevent attacks. The acceptable range is Client specific.
            min_iat = self.issued_at - _ALLOWABLE_CLOCK_DRIFT
            max_iat = self.issued_at + _ALLOWABLE_CLOCK_DRIFT
            iat_claim = datetime.fromtimestamp(claims.get("iat"), tz=timezone.utc)
            if not min_iat < iat_claim < max_iat:
                raise OAuthError("Token was issued too far away from the current time.")
        except OAuthError:
            raise  # Re-raise OAuthError exceptions.
        except Exception as e:
            raise OAuthError("Unable to decode ID token.") from e


class OIDCConfig(BaseModel):
    issuer: str
    """The issuer URL."""

    authorization_endpoint: str
    """The authorization endpoint, relative to the issuer URL."""

    token_endpoint: str
    """The token endpoint, relative to the issuer URL."""

    device_authorization_endpoint: str | None = None
    """The device authorization endpoint, relative to the issuer URL."""

    end_session_endpoint: str | None = None
    """The end session endpoint, relative to the issuer URL."""

    response_types_supported: set[str]
    """The supported response types for the authorization endpoint."""

    grant_types_supported: set[str] = set(["authorization_code", "implicit"])
    """The supported grant types for the token endpoint."""

    @field_validator(
        "authorization_endpoint",
        "token_endpoint",
        "device_authorization_endpoint",
        "end_session_endpoint",
        check_fields=True,
    )
    @classmethod
    def endpoints_exist_under_issuer(cls, v, info: ValidationInfo):
        if "issuer" in info.data:
            if not v.startswith(info.data["issuer"]):
                raise OIDCError(f"OIDC field {info.field_name} must be a URL under the issuer")
            validated_endpoint = v.removeprefix(info.data["issuer"])
            logger.info(f"Found OAuth {info.field_name} endpoint: {validated_endpoint}")
            return validated_endpoint
        else:
            raise OIDCError("Missing issuer url in OIDC config.")


class DeviceFlowResponse(BaseModel):
    """A response from the OAuth server to a device flow request.

    https://datatracker.ietf.org/doc/html/rfc8628#section-3.2
    """

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str] = None
    expires_in: int
    interval: int = 5

    def __str__(self) -> str:
        return f"Verification URL: {self.verification_uri}\nUser code: {self.user_code}"

    @property
    def _retry(self) -> Retry:
        max_attempts = self.expires_in // self.interval
        return Retry(logger=logger, max_attempts=max_attempts, backoff_method=BackoffLinear(self.interval))

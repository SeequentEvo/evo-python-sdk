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

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import deprecated

__all__ = [
    "AccessToken",
    "AnyScopes",
    "EvoScopes",
    "OAuthScopes",
]

_ALLOWABLE_CLOCK_DRIFT = timedelta(minutes=5)


def _utcnow() -> datetime:
    """Get the current UTC time.

    :return: The current UTC time.
    """
    return datetime.now(timezone.utc)


class Scopes(str):
    """A space-separated list of OAuth scopes."""

    def __new__(cls, *members: str) -> Scopes:
        if len(members) == 0:
            raise ValueError("At least one scope must be provided")

        # Get unique members.
        unique_members = set()
        for member in members:
            # Each member may itself be a space-separated list of scopes.
            unique_members.update(member.strip().split(" "))

        inst = str.__new__(cls, " ".join(sorted(unique_members)))
        return inst

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(sorted(self.split(" ")))

    def __repr__(self) -> str:
        return f"Scopes{self.members!r}"

    def __or__(self, other: str) -> Scopes:
        # Support Scopes | str
        if isinstance(other, str):
            return Scopes(self, other)
        return NotImplemented

    def __ror__(self, other: str) -> Scopes:
        # Support str | Scopes
        return self | other

    def __contains__(self, key: str) -> bool:
        members = set(self.members)
        others = set(Scopes(key).members)
        return others.issubset(members)


class EvoScopes(Scopes, Enum):
    """Public OAuth scopes for authenticating against Evo APIs.

    https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens#about-evo-access-tokens
    """

    offline_access = Scopes("offline_access")

    # Evo Scopes
    evo_discovery = Scopes("evo.discovery")
    """Required for access to the Evo Discovery API, both discovery and token endpoints."""

    evo_workspace = Scopes("evo.workspace")
    """Required for access to the Workspaces API."""

    evo_audit = Scopes("evo.audit")
    """Required for access to the Audit API."""

    evo_blocksync = Scopes("evo.blocksync")
    """Required for access to the BlockSync API."""

    evo_object = Scopes("evo.object")
    """Required for access to the Geoscience Object API."""

    evo_file = Scopes("evo.file")

    # Useful combinations
    """Default scopes for Evo applications.

    Extra scopes to be added as needed include `evo_blocksync`, `evo_object`, or `evo_file`, etc.
    """
    default = evo_discovery | evo_workspace

    """All scopes for Evo applications."""
    all_evo = default | evo_blocksync | evo_object | evo_file

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return Scopes.__repr__(self)


if TYPE_CHECKING:
    # Subclassing Enum is not supported at runtime, but it was the most elegant way to mark the old name as deprecated.
    # At runtime `OAuthScopes` is simply an alias for `EvoScopes`, and a deprecated warning will not be emitted.

    @deprecated("Use EvoScopes instead.")
    class OAuthScopes(EvoScopes): ...
else:
    OAuthScopes: TypeAlias = EvoScopes


AnyScopes: TypeAlias = str | Scopes | EvoScopes


class AccessToken(BaseModel):
    """An access token model, returned by the OAuth server.
    https://www.rfc-editor.org/rfc/rfc6749#section-5.1
    """

    token_type: Literal["Bearer"]

    access_token: str
    """The access token issued by the authorization server."""

    refresh_token: Optional[str] = None
    """The refresh token, which can be used to obtain new access tokens using the same authorization grant. Refresh tokens are only returned if you request the "offline access" scope."""

    expires_in: Optional[int] = None
    """The lifetime in seconds of the access token."""

    issued_at: datetime = Field(default_factory=_utcnow)
    """The time at which the token response was received."""

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

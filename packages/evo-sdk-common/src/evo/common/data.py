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

import copy
import enum
from abc import ABC, abstractmethod
from collections.abc import ItemsView, Iterator, KeysView, Mapping, MutableMapping, Sequence, ValuesView
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, TypeVar, overload
from uuid import UUID

from .exceptions import ServiceHealthCheckFailed

__all__ = [
    "DependencyStatus",
    "EmptyResponse",
    "Environment",
    "HTTPHeaderDict",
    "HTTPResponse",
    "HealthCheckType",
    "Page",
    "RequestMethod",
    "ResourceMetadata",
    "ServiceHealth",
    "ServiceStatus",
    "ServiceUser",
]


class RequestMethod(str, enum.Enum):
    """HTTP request method."""

    GET = "GET"
    """HTTP [`GET`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/GET)"""

    HEAD = "HEAD"
    """HTTP [`HEAD`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/HEAD)"""

    POST = "POST"
    """HTTP [`POST`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST)"""

    PUT = "PUT"
    """HTTP [`PUT`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/PUT)"""

    DELETE = "DELETE"
    """HTTP [`DELETE`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/DELETE)"""

    OPTIONS = "OPTIONS"
    """HTTP [`OPTIONS`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/OPTIONS)"""

    PATCH = "PATCH"
    """HTTP [`PATCH`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/PATCH)"""

    def __str__(self) -> str:
        return self.value


class HTTPHeaderDict(MutableMapping[str, str]):
    def __init__(self, seq: Mapping[str, str] | Sequence[tuple[str, str]] | None = None, **kwargs: str) -> None:
        self.__values: dict[str, str] = {}
        self.update(seq, **kwargs)

    def update(self, seq: Mapping[str, str] | Sequence[tuple[str, str]] | None = None, **kwargs: str) -> None:
        if isinstance(seq, Mapping):
            self.__update_from_mapping(seq)
        elif isinstance(seq, Sequence):
            self.__update_from_sequence(seq)

        self.__update_from_mapping(kwargs)

    def __update_from_mapping(self, mapping: Mapping[str, str]) -> None:
        for key, value in mapping.items():
            self.__setitem__(key, value)

    def __update_from_sequence(self, seq: Sequence[tuple[str, str]]) -> None:
        for key, value in seq:
            self.__setitem__(key, value)

    def __setitem__(self, key: str, value: str) -> None:
        lookup = key.title()
        if lookup in self.__values and lookup != "Set-Cookie":
            # RFC 7230 section 3.2.2: Field Order.
            # A recipient MAY combine multiple header fields with the same field
            # name into one "field-name: field-value" pair, without changing the
            # semantics of the message, by appending each subsequent field value to
            # the combined field value in order, separated by a comma.  The order
            # in which header fields with the same field name are received is
            # therefore significant to the interpretation of the combined field
            # value;
            # ...
            # In practice, the "Set-Cookie" header field ([RFC6265]) often
            # appears multiple times in a response message and does not use the
            # list syntax, violating the above requirements on multiple header
            # fields with the same name.  Since it cannot be combined into a
            # single field-value, recipients ought to handle "Set-Cookie" as a
            # special case while processing header fields.
            # https://www.rfc-editor.org/rfc/rfc7230#section-3.2.2
            self.__values[lookup] += "," + value
        else:
            self.__values[lookup] = value

    def __delitem__(self, key: str) -> None:
        del self.__values[key.title()]

    def __getitem__(self, key: str) -> str:
        return self.__values[key.title()]

    def __len__(self) -> int:
        return len(self.__values)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, item: str) -> bool:
        return item.title() in self.__values

    def __repr__(self) -> str:
        repr_data = {}
        for key, value in self.items():
            if key in ("Authorization", "Proxy-Authorization", "Cookie", "Set-Cookie"):
                # Do not expose sensitive information.
                value = "*****"
            repr_data[key] = value

        return f"{self.__class__.__name__}({repr_data!r})"

    def items(self) -> ItemsView[str, str]:
        return ItemsView(self)

    def keys(self) -> KeysView[str]:
        return KeysView(self.__values)

    def values(self) -> ValuesView[str]:
        return ValuesView(self.__values)

    def copy(self) -> HTTPHeaderDict:
        return copy.deepcopy(self)


@dataclass(frozen=True, kw_only=True)
class EmptyResponse:
    status: int
    reason: str | None = None
    headers: HTTPHeaderDict = field(default_factory=HTTPHeaderDict)

    def getheaders(self) -> HTTPHeaderDict:
        return self.headers.copy()

    def getheader(self, key: str, default: str | None = None) -> str:
        return self.headers.get(key, default)


@dataclass(frozen=True, kw_only=True)
class HTTPResponse(EmptyResponse):
    data: bytes


@dataclass(frozen=True, kw_only=True)
class Environment:
    """Service environment configuration.

    This class is used by service clients to determine the organisation and workspace context for requests.
    """

    hub_url: str
    """The base URL of the Evo service hub."""

    org_id: UUID
    """The organisation ID for the current environment."""

    workspace_id: UUID
    """The workspace ID for the current environment."""


class _UserModel(Protocol):
    id: UUID
    name: str | None
    email: str | None


@dataclass(frozen=True, kw_only=True)
class ServiceUser:
    """Metadata about a user."""

    id: UUID
    """The user UUID."""

    name: str | None
    """The user display name."""

    email: str | None
    """The user email address."""

    @classmethod
    def from_model(cls, model: _UserModel) -> ServiceUser:
        """Create a new instance from an instance of a generated model."""
        return cls(id=model.id, name=model.name, email=model.email)


@dataclass(frozen=True, kw_only=True)
class ResourceMetadata(ABC):
    """Metadata about a resource stored in an Evo service."""

    environment: Environment
    """The environment in which the resource is stored."""

    id: UUID
    """The resource UUID."""

    name: str
    """The resource name."""

    created_at: datetime
    """The resource creation timestamp."""

    created_by: ServiceUser | None = None
    """The user who created the resource."""

    @property
    @abstractmethod
    def url(self) -> str:
        """The URL of the resource.

        Resource URLs should point to the appropriate API endpoint to get the resource metadata.
        Prefer URLs that reference resources by UUID and include the resource version.
        """
        pass


_Metadata = TypeVar("_Metadata", bound=ResourceMetadata)


class Page(Sequence[_Metadata]):
    """A page of resource metadata from a paginated response.

    This type exposes paginated metadata from an API response, including the offset, limit, and total number of items.
    The contained items are the actual metadata from the response.
    """

    def __init__(self, *, offset: int, limit: int, total: int, items: Sequence[_Metadata]) -> None:
        self._offset = offset
        self._limit = limit
        self._total = total
        self._items = items

    @property
    def offset(self) -> int:
        """The offset of the first item in the page."""
        return self._offset

    @property
    def limit(self) -> int:
        """The maximum number of items per page."""
        return self._limit

    @property
    def size(self) -> int:
        """The number of items in the page."""
        return len(self._items)

    def __len__(self) -> int:
        """The number of items in the page."""
        return self.size

    @property
    def total(self) -> int:
        """The total number of items that are available, as reported by the API."""
        return self._total

    def items(self) -> list[_Metadata]:
        """Get the items that are in the page.

        Items are copied to prevent modification of the original items.

        :returns: A list of items in the page.
        """
        return [copy.deepcopy(item) for item in self._items]

    @overload
    def __getitem__(self, key: int) -> _Metadata: ...

    @overload
    def __getitem__(self, key: slice) -> list[_Metadata]: ...

    def __getitem__(self, key: int | slice) -> _Metadata | list[_Metadata]:
        """Get an item or items from the page.

        If the key is an integer, the item at that index is returned. If the key is a slice, a list of items in the
        slice are returned.

        Items are copied to prevent modification of the original items.

        :param key: The index of the item to get, or a slice of items to get.

        :returns: The item or items from the page.
        """
        if isinstance(key, int):
            return copy.deepcopy(self._items[key])
        elif isinstance(key, slice):
            return [copy.deepcopy(item) for item in self._items[key]]
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    @property
    def next_offset(self) -> int:
        """The offset of the next page."""
        return self.offset + self.size

    @property
    def is_last(self) -> bool:
        """Whether the page is the last page."""
        return self.next_offset >= self.total


class HealthCheckType(enum.Enum):
    """The type of health check to be performed.

    This enum determines which parameters are provided in a service health check request, which in turn determines the
    type of health check that is performed by the service.
    """

    BASIC = enum.auto()
    """Perform a basic health check, ignoring service dependencies."""

    FULL = enum.auto()
    """Perform a full health check, including service dependencies.
    A failed dependency may not necessarily result in a failed health check. If a dependency is unhealthy but the
    service can still handle requests, the service may report report a degraded status.
    """

    STRICT = enum.auto()
    """Perform a strict health check, including service dependencies.
    A strict health check will always fail if any dependency is unhealthy.
    """


class ServiceStatus(str, enum.Enum):
    """Service status enumeration."""

    HEALTHY = "pass"
    """The service is healthy and able to handle requests."""

    DEGRADED = "degraded"
    """The service is running but some features may not work, or performance may be reduced."""

    UNHEALTHY = "fail"
    """The service is unable to handle requests."""


class DependencyStatus(str, enum.Enum):
    """Dependency status enumeration."""

    HEALTHY = "pass"
    """The dependency is healthy and able to handle requests."""

    UNHEALTHY = "fail"
    """The dependency is unable to handle requests."""


@dataclass(frozen=True, kw_only=True)
class ServiceHealth:
    service: str
    """The name of the service."""

    status_code: int
    """The status code of the health check response."""

    status: ServiceStatus
    """The current status of the service."""

    version: str
    """An arbitrary string representing the service version."""

    dependencies: dict[str, DependencyStatus] | None
    """A dictionary of service dependencies and their status."""

    def raise_for_status(self) -> None:
        """Raise an exception if the service is unhealthy.

        :raises ServiceHealthCheckFailed: If `status_code` is not 200.
        """
        if self.status_code != 200:
            msg = f"Health check for {self.service} service failed with status {self.status} ({self.status_code})"
            if self.dependencies is not None:
                for dependency, status in self.dependencies.items():
                    msg += f"\n  {dependency}: {status}"
            raise ServiceHealthCheckFailed(msg)


class OrderByOperatorEnum(str, enum.Enum):
    """Enumeration for order_by operators for listing endpoints."""

    asc = "asc"
    desc = "desc"

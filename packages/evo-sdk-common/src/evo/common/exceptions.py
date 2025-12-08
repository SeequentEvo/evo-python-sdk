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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

if TYPE_CHECKING:
    from typing import Self

    from .data import HTTPHeaderDict


class EvoClientException(Exception):
    """The base exception class for all Evo client exceptions."""


_Condition: TypeAlias = type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], bool]


def _get_filter_func(condition: _Condition) -> Callable[[Exception], bool]:
    if not isinstance(condition, type) and callable(condition):
        return condition
    else:
        return lambda e: isinstance(e, condition)


class EvoExceptionGroup(EvoClientException):
    """A custom exception group type that closely resembles `ExceptionGroup` which was introduced in python 3.11

    This implementation is based off PEP 654 and the reference implementation linked within.
    https://github.com/iritkatriel/cpython/pull/10/files#diff-96c4ad902cdfac48712fef8fff0d0c78a71eb135fb48264f3ebb67db1013fc94

    EvoExceptionGroup wraps the exceptions in the sequence excs. The msg parameter must be a string.
    """

    def __new__(cls, msg: str, excs: Sequence[Exception]):
        grp = super().__new__(cls)
        grp._msg = msg
        grp._excs = tuple(excs)
        return grp

    @property
    def message(self) -> str:
        """The msg argument to the constructor. This is a read-only attribute."""
        return self._msg

    @property
    def exceptions(self) -> tuple[Exception, ...]:
        """A tuple of the exceptions in the excs sequence given to the constructor. This is a read-only attribute."""
        return copy.copy(self._excs)

    def derive(self, excs: Sequence[Exception]) -> Self:
        """Create a new instance of the current exception group type with the same message, but wrapping the exceptions
        in excs.

        This method is used by subgroup() and split(). A subclass needs to override it in order to make subgroup() and
        split() return instances of the subclass rather than ExceptionGroup.

        subgroup() and split() copy the __traceback__, __cause__, and __context__ fields from the original
        exception group to the one returned by derive(), so these fields do not need to be updated by derive().

        :param excs: Exceptions to wrap in the derived exception group.

        :return: An exception group with the same message, but which wraps the exceptions in excs.
        """
        return EvoExceptionGroup(self.message, excs)

    def _derive_or_none(self, excs: Sequence[Exception]) -> Self | None:
        """Derive a new exception group with the given exceptions, or return None.

        :param excs: Exceptions to include in the derived exception group.

        :return: A new instance of the current type, with __traceback__, __cause__, and __context__ fields copied. If
            excs is empty, None is returned instead.
        """
        if len(excs) == 0:
            return None
        else:
            derived = self.derive(excs)
            # Copy the original context, cause, and traceback.
            derived.__context__ = copy.copy(self.__context__)
            derived.__cause__ = copy.copy(self.__cause__)
            derived.__traceback__ = copy.copy(self.__traceback__)
            return derived

    def split(self, condition: _Condition) -> tuple[Self | None, Self | None]:
        """Like subgroup(), but returns the pair (match, rest) where match is subgroup(condition) and rest is the
        remaining non-matching part.

        :param condition: Either a function that accepts an exception and returns true for those that should be in
            the subgroup, or it can be an exception type or a tuple of exception types, which is used to check for a
            match using the same check that is used in an except clause.

        :return: (match, rest) where match is subgroup(condition) and rest is the remaining non-matching part.
        """
        filter_func = _get_filter_func(condition)
        if filter_func(self):
            return self._derive_or_none(self.exceptions), None
        matched = []
        unmatched = []
        for exc in self.exceptions:
            if filter_func(exc):
                # Satisfies condition.
                matched.append(exc)
            elif isinstance(exc, EvoExceptionGroup):
                # Recursive splitting.
                matched_subgroup, unmatched_subgroup = exc.split(filter_func)
                if matched_subgroup is not None:
                    matched.append(matched_subgroup)
                if unmatched_subgroup is not None:
                    unmatched.append(unmatched_subgroup)
            else:
                # Does not satisfy condition.
                unmatched.append(exc)

        return self._derive_or_none(matched), self._derive_or_none(unmatched)

    def subgroup(self, condition: _Condition) -> Self | None:
        """Returns an exception group that contains only the exceptions from the current group that match condition, or
        None if the result is empty.

        The nesting structure of the current exception is preserved in the result, as are the values of its message,
        __traceback__, __cause__, and __context__ fields. Empty nested groups are omitted from the result.

        The condition is checked for all exceptions in the nested exception group, including the top-level and any
        nested exception groups. If the condition is true for such an exception group, it is included in the result in
        full.

        :param condition: Either a function that accepts an exception and returns true for those that should be in
            the subgroup, or it can be an exception type or a tuple of exception types, which is used to check for a
            match using the same check that is used in an except clause.

        :return: An exception group that contains only the exceptions from the current group that match condition, or
            None if the result is empty.
        """
        return self.split(condition)[0]

    def __str__(self) -> str:
        excs = self.exceptions
        n_sub_excs = len(excs)
        tb_lines = [
            f"{self.__class__.__name__}: {self.message} ({n_sub_excs} sub-exception{'' if n_sub_excs == 1 else 's'})"
        ]
        for i, exc in enumerate(excs):
            tb_lines.append(f"+---------------- {i + 1} ----------------")
            tb_lines.append(f"| {type(exc).__name__}:")
            for exc_line in str(exc).split("\n"):
                tb_lines.append(f"| {exc_line}")
        return "\n".join(tb_lines)


class RetryError(EvoExceptionGroup):
    """Custom ExceptionGroup for wrapping exceptions from multiple retry attempts."""


class StorageFileNotFoundError(FileNotFoundError, EvoClientException):
    """Raised when a file or directory is requested but doesn't exist."""


class StorageFileExistsError(FileExistsError, EvoClientException):
    """Raised when trying to create a file or directory which already exists."""


class FileNameTooLongError(EvoClientException):
    """Raised when a user tries to download a file with a name that is too long to be saved in the filesystem"""


class ServiceHealthCheckFailed(EvoClientException):
    """Raised when a service health check fails."""


class _WrappedError(EvoClientException):
    """Wrapper for standard exceptions that occur while parsing service responses."""

    def __init__(self, msg: str, caused_by: Exception | None = None):
        """
        :param msg: The exception message.
        :param caused_by: The original error that caused deserialization to fail.
        """
        self.caused_by = caused_by
        full_msg = msg
        if caused_by:
            full_msg = f"{msg}: {str(caused_by)}"
        super(_WrappedError, self).__init__(full_msg)


class TransportError(_WrappedError):
    """Wraps errors raised by the underlying HTTP transport."""


class ClientTypeError(_WrappedError, TypeError):
    """Raised when an operation or function is applied to an object of inappropriate type.
    The associated value is a string giving details about the type mismatch.
    """

    def __init__(
        self,
        msg: str,
        caused_by: Exception | None = None,
        valid_classes: tuple[type, ...] | None = None,
        key_type: bool | None = None,
    ):
        """
        :param msg: The exception message.
        :param caused_by: The original error that caused deserialization to fail.
        :param valid_classes: The primitive classes that current item should be an instance of.
        :param key_type: False if our value is a value in a dict. True if it is a key in a dict. False if our item is an
            item in a list.
        """
        super(ClientTypeError, self).__init__(msg, caused_by)
        self.valid_classes = valid_classes
        self.key_type = key_type


class ClientValueError(_WrappedError, ValueError):
    """Raised when an operation or function receives an argument that has the right type but an inappropriate value,
    and the situation is not described by a more precise exception such as IndexError.
    """


class EvoAPIException(EvoClientException):
    """Base class for all service errors."""

    def __init__(self, status: int, reason: str | None, content: object | None, headers: HTTPHeaderDict | None):
        """
        :param status: HTTP status code.
        :param reason: Reason.
        :param content: Deserialized content from the response.
        :param headers: Response headers
        """
        self.status = status
        self.reason = reason
        self.content = content
        self.headers = headers

    def __str__(self) -> str:
        error_message = f"({self.status})"
        if reason := self.reason:
            error_message += f" {reason}"
        if content := self.content:
            error_message += f"\n{content}"
        return error_message


class UnknownResponseError(EvoAPIException):
    """The service sent an unknown response."""


class BaseTypedError(EvoAPIException):
    """Base class for service errors that have a defined type.

    Concrete error types must subclass BaseTypedError and define the class attribute `TYPE_ID`, which will be used
    to map service error responses to the corresponding implementation.

    The fallback type `DefaultTypedError` will be used if an error response has a title and no other concrete type
    matches the type ID in the response. It is worth noting that the type may be missing from the response,
    in which case `DefaultTypedError` will be used, and the instance `type_` will be 'about:blank' (as is specified).
    """

    status: int
    """The HTTP status code generated by the origin server for this occurrence of the problem."""

    content: dict[str, Any]
    """Deserialized content from the response."""

    @property
    def type_(self) -> str:
        """A URI reference that identifies the problem type."""
        return str(self.content["type"])

    @property
    def title(self) -> str:
        """A short, human-readable summary of the problem type.

        This is advisory only.
        """
        return self.content["title"]

    @property
    def detail(self) -> str | None:
        """A human-readable explanation specific to this occurrence of the problem."""
        return self.content.get("detail")

    def __str__(self) -> str:
        error_message = f"Error: ({self.status})"
        if reason := self.reason:
            error_message += f" {reason}"
        error_message += f"\nType: {self.type_}"
        error_message += f"\nTitle: {self.title}"
        if detail := self.detail:
            error_message += f"\nDetail: {detail}"
        return error_message


class CustomTypedError(BaseTypedError):
    """Base class for service errors that have a type.

    Concrete error types must subclass CustomTypedError and define the class attribute `TYPE_ID`, which will be used
    to map service error responses to the corresponding implementation.

    The fallback type `DefaultTypedError` will be used if an error response has a title and no other concrete type
    matches the type ID in the response. It is worth noting that the type may be missing from the response,
    in which case `DefaultTypedError` will be used, and the instance `type_` will be 'about:blank' (as is specified).
    """

    __CONCRETE_TYPES: dict[str, type[CustomTypedError]] = {}

    TYPE_ID: ClassVar[str]
    """The type ID used to match concrete error types.

    Subclasses should omit the URL portion of the type ID, in case the URL is changed in the future. The full type ID,
    as provided in the response, will be used by the `type_` property.
    """

    @staticmethod
    def from_type_id(type_id: str | None) -> type[CustomTypedError]:
        """Get a concrete error type, based on the type ID.

        :param type_id: The type ID of the received error.

        :return: The concrete implementation of the requested type.
        """
        if type_id is None:
            return DefaultTypedError

        for defined_type_id, error_type in CustomTypedError.__CONCRETE_TYPES.items():
            if type_id.endswith(defined_type_id):
                return error_type
        return DefaultTypedError

    @staticmethod
    def provided_by(content: object) -> bool:
        """Determine whether the content of an error response has a title

        :param content: The deserialized content of a 4xx or 5xx error response.

        :return: True if the content satisfies has a title.
        """
        return isinstance(content, dict) and "title" in content

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            type_id = cls.TYPE_ID
        except AttributeError:
            raise ValueError(f"{cls} must define TYPE_ID.")
        if type_id is None:
            return  # abstract class
        if not isinstance(type_id, str):
            raise ValueError(f"{cls} TYPE_ID must be a string.")
        if existing_cls := CustomTypedError.__CONCRETE_TYPES.get(type_id):
            raise ValueError(f"Duplicated TYPE_ID between {cls} and {existing_cls}")
        CustomTypedError.__CONCRETE_TYPES[type_id] = cls


class DefaultTypedError(CustomTypedError):
    """Fallback Typed Error implementation that meets the minimum specification."""

    TYPE_ID = "about:blank"

    @property
    def type_(self) -> str:
        """A URI reference that identifies the problem type.

        When this member is not present, its value is assumed to be `about:blank`. This is the primary identifier for
        the problem type.
        """
        return self.content.get("type", self.TYPE_ID)


class GeneralizedTypedError(BaseTypedError):
    """Base class for Typed errors that should be generalized based on status code.

    Generalized error types must subclass GeneralizedTypedError and define the class attribute `STATUS_CODE`, which will
    be used to map service error codes to the corresponding generalization.

    This mechanism allows specific error codes to be treated consistently, regardless of service implementation.
    """

    __GENERALIZED_TYPES: dict[int, type[GeneralizedTypedError]] = {}

    TYPE_ID = "about:blank"
    STATUS_CODE: int

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        status_code = getattr(cls, "STATUS_CODE")
        GeneralizedTypedError.__GENERALIZED_TYPES[status_code] = cls

    @staticmethod
    def from_status_code(status_code: int) -> type[GeneralizedTypedError] | None:
        """Get a generalized error type, based on the status code.

        :param status_code: The status code of the error response.

        :return: The generalized implementation for the requested status code.
        """
        return GeneralizedTypedError.__GENERALIZED_TYPES.get(status_code, None)

    @property
    def type_(self) -> str:
        """A URI reference that identifies the problem type.

        When this member is not present, its value is assumed to be `about:blank`. This is the primary identifier for
        the problem type.
        """
        if isinstance(self.content, dict) and "type" in self.content:
            return self.content["type"]
        else:
            return self.TYPE_ID

    @property
    def title(self) -> str:
        """A short, human-readable summary of the problem type.

        This is advisory only.
        """
        if isinstance(self.content, dict) and "title" in self.content:
            return str(self.content["title"])
        else:
            return self.reason

    @property
    def detail(self) -> str | None:
        """A human-readable explanation specific to this occurrence of the problem."""
        if isinstance(self.content, dict) and "detail" in self.content:
            return str(self.content["detail"])
        else:
            return None


class BadRequestException(GeneralizedTypedError):
    """The service cannot process the request due to a client error (400 - Bad Request)."""

    STATUS_CODE = 400


class UnauthorizedException(GeneralizedTypedError):
    """The client must authenticate to get a response (401 - Unauthorized)."""

    STATUS_CODE = 401


class ForbiddenException(GeneralizedTypedError):
    """The client does not have access rights to the content (403- Forbidden)."""

    STATUS_CODE = 403


class NotFoundException(GeneralizedTypedError):
    """The Service API could not find the requested resource (404 - Not Found)."""

    STATUS_CODE = 404


class GoneException(GeneralizedTypedError):
    """The requested resource is deleted (410 - Gone)."""

    STATUS_CODE = 410


class ContextError(EvoClientException):
    """Raised when the context lacks required information."""


class SelectionError(ContextError):
    """Raised when a selection error occurs."""

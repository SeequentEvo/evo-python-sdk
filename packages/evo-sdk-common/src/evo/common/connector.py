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

import datetime
import functools
import json
import re
from collections.abc import Mapping
from enum import Enum
from inspect import isclass
from types import GenericAlias, NoneType, TracebackType
from typing import Any, ParamSpec, TypeVar
from urllib.parse import quote, urlencode
from uuid import UUID

from dateutil.parser import parse
from pydantic import BaseModel

from evo import logging

from .data import EmptyResponse, HTTPHeaderDict, HTTPResponse, RequestMethod
from .exceptions import (
    ClientTypeError,
    ClientValueError,
    CustomTypedError,
    EvoAPIException,
    EvoClientException,
    GeneralizedTypedError,
    UnauthorizedException,
    UnknownResponseError,
)
from .interfaces import IAuthorizer, ITransport

logger = logging.getLogger("connector")

__all__ = [
    "APIConnector",
    "NoAuth",
]

T = TypeVar("T")
P = ParamSpec("P")


def retry_on_auth_error(func):  # No type annotation to prevent hiding the signature of the decorated function.
    @functools.wraps(func)
    async def wrapper(self: APIConnector, *args: P.args, **kwargs: P.kwargs) -> T:
        # Always use the connector in a context manager to ensure the transport is opened and closed correctly.
        # ITransport implementations are required to be re-entrant, so this is safe. It may still be useful to open
        # and close the transport elsewhere in client code to prevent unnecessarily destroying and recreating any
        # underlying sessions.
        async with self:
            try:
                return await func(self, *args, **kwargs)
            except UnauthorizedException:
                logger.debug("Unauthorized exception caught. Attempting to refresh the access token", exc_info=True)
                if not await self._authorizer.refresh_token():
                    logger.debug("Failed to refresh the access token.", exc_info=True)
                    raise
                else:
                    logger.debug("Access token refreshed. Retrying the request.")
                    return await func(self, *args, **kwargs)
            except:
                logger.debug("An error occurred while calling the API.", exc_info=True)
                raise

    return wrapper


class _NoAuth(IAuthorizer):
    """An authorizer that does not provide any authentication."""

    async def get_default_headers(self) -> HTTPHeaderDict:
        """Return an empty header dictionary."""
        return HTTPHeaderDict()

    async def refresh_token(self) -> bool:
        """Return False, as there is no token to refresh."""
        return False


NoAuth = _NoAuth()
"""An authorizer that does not provide any authentication."""


class APIConnector:
    """Generic client for facilitating API requests."""

    def __init__(
        self,
        base_url: str,
        transport: ITransport,
        authorizer: IAuthorizer = NoAuth,
        additional_headers: Mapping[str, Any] | None = None,
    ) -> None:
        """
        :param base_url: The host URL of the API.
        :param transport: The transport to use for sending requests.
        :param authorizer: The authorizer to use for authenticating requests.
        :param additional_headers: Additional headers to include in each request.
        """
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._authorizer = authorizer
        self._additional_headers = additional_headers

    @property
    def base_url(self) -> str:
        """The base_url of the connected API."""
        return self._base_url + "/"

    @property
    def transport(self) -> ITransport:
        """The transport used to send requests."""
        return self._transport

    async def open(self) -> None:
        """Open the HTTP transport."""
        await self._transport.open()

    async def close(self) -> None:
        """Close the HTTP transport."""
        await self._transport.close()

    async def __aenter__(self) -> APIConnector:
        await self.open()
        return self

    async def __aexit__(
        self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: TracebackType | None
    ) -> None:
        await self.close()

    @retry_on_auth_error
    async def call_api(
        self,
        method: RequestMethod,
        resource_path: str,
        path_params: Mapping[str, Any] | None = None,
        query_params: Mapping[str, Any] | None = None,
        header_params: Mapping[str, Any] | None = None,
        post_params: Mapping[str, Any] | None = None,
        body: object | str | bytes | None = None,
        collection_formats: Mapping[str, str] | None = None,
        response_types_map: Mapping[str, type[T]] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> T:
        """Call the API with the given parameters and deserialize the response.

        Errors raised by `ITransport.request` are not handled by this method.

        :param method: HTTP request method.
        :param resource_path: Path to the API endpoint.
        :param path_params: Path parameters to embed in the url.
        :param query_params: Query parameters to embed in the url.
        :param header_params: Header parameters to be placed in the request header.
        :param post_params: Post request form parameters.
        :param body: Body to send with the request.
        :param collection_formats: Dict of collection formats for path, query, header, and post parameters, where the
            key is the parameter name, and the value is one of 'multi', 'ssv', 'pipes', or 'csv'.
            'csv' is the default collection format.
        :param response_types_map: Mapping of response status codes to response data types. The response will
            be deserialized to the corresponding type.
        :param request_timeout: Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: The deserialized response, in the format determined by the response types map.

        :raise BadRequestException: If the server responds with HTTP status 400.
        :raise UnauthorizedException: If the server responds with HTTP status 401.
        :raise ForbiddenException: If the server responds with HTTP status 403.
        :raise NotFoundException: If the server responds with HTTP status 404.
        :raise BaseTypedError: If the server responds with any other HTTP status between 400 and 599, and the body
            of the response has a type.
        :raise EvoAPIException: If the server responds with any other HTTP status between 400 and 599, and the body of
            the response does not have a type.
        :raise UnknownResponseError: For other HTTP status codes with no corresponding response type in
            `response_types_map`.
        """
        collection_formats = collection_formats or {}

        # Process URL parameters.
        resource_url = self._encode_url_parameters(resource_path, path_params, query_params, collection_formats)

        # Process request headers.
        headers = await self._authorizer.get_default_headers()
        headers.update(self._additional_headers)
        if header_params is not None:
            specified_headers = set()  # So we make sure each header is only overridden once.
            for key, value in self._parameters_to_tuples(header_params, collection_formats):
                if key in specified_headers or isinstance(value, list):
                    raise RuntimeError(f"Multiple values not supported in header '{key}'")
                else:
                    headers[key] = value
                    specified_headers.add(key)

        # Process post parameters.
        post_params = self._parameters_to_tuples(post_params, collection_formats) if post_params else None

        # Sanitize body
        body = self._sanitize_for_serialization(body) if body else None

        # Perform request.
        logger.debug(f"Making {method} request to {resource_url}")
        response = await self._transport.request(
            method=method,
            url=resource_url,
            headers=headers,
            post_params=post_params,
            body=body,
            request_timeout=request_timeout,
        )

        # Prepare response type.
        if (default_response_type := GeneralizedTypedError.from_status_code(response.status)) is not None:
            pass  # Use the response type returned above.
        elif 400 <= response.status <= 599:  # Error status code.
            default_response_type = EvoAPIException
        else:
            default_response_type = UnknownResponseError

        if response_types_map is not None:
            # Always use the type from response_types_map if it is available.
            response_type = response_types_map.get(str(response.status), default_response_type)
        else:
            response_type = default_response_type

        # Decode response.
        response_object = self._deserialize(response, response_type)

        return response_object

    def _encode_url_parameters(
        self,
        resource_path: str,
        path_params: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        collection_formats: dict[str, str] | None = None,
    ) -> str:
        """Encode path and query parameters within the resource path.

        :param resource_path: The resource path.
        :param path_params: Parameters that are used to replace template parameters in the resource path.
        :param query_params: Query parameters that are to be encoded at the end of the URL.
        :param collection_formats: Dict of collection formats for path and query parameters.

        :return: A URL with all path and query parameters encoded.
        """
        resource_url = self._base_url + "/" + resource_path.lstrip("/")

        if path_params:
            for key, value in self._parameters_to_tuples(path_params, collection_formats):
                resource_url = resource_url.replace(f"{{{key}}}", quote(str(value)))

        if query_params:
            resource_url += "?" + self._parameters_to_url_query(query_params, collection_formats)

        return resource_url

    @classmethod
    def _sanitize_for_serialization(cls, obj: Any | None) -> Any | None:
        """Builds a JSON object for serialization.

        If obj is None return None.
        If obj is an Enum sanitize the value.
        If obj is a primitive return directly.
        If obj is a date or datetime convert to string in iso8601 format.
        If obj is a UUID convert to string.
        If obj is a list or tuple, sanitize each element.
        If obj is a dict, sanitize the dict.
        If obj is an API model, convert to dict.

        :param obj: The data to serialize.

        :return: The serialized form of data.
        """
        if obj is None:
            # If obj is None return None.
            return None
        if isinstance(obj, Enum):
            # If obj is an Enum sanitize the value.
            return cls._sanitize_for_serialization(obj.value)
        elif isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            # If obj is a primitive return directly.
            return obj
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            # If obj is a date or datetime convert to string in iso8601 format.
            return obj.isoformat()
        elif isinstance(obj, UUID):
            # If obj is a UUID convert to string.
            return str(obj)
        elif isinstance(obj, list):
            # If obj is a list sanitize each element in the list.
            return [cls._sanitize_for_serialization(sub_obj) for sub_obj in obj]
        elif isinstance(obj, tuple):
            # If obj is a tuple sanitize each element in the tuple.
            return tuple(cls._sanitize_for_serialization(sub_obj) for sub_obj in obj)

        if isinstance(obj, Mapping):
            # If obj is a dict, sanitize the dict.
            obj_dict = obj
        elif isinstance(obj, BaseModel):
            # If obj is an API model, convert to dict.
            obj_dict = obj.model_dump(mode="json", by_alias=True, exclude_unset=True)
        else:
            raise ClientTypeError(
                msg=f"{type(obj)} could not be serialized.",
                valid_classes=(
                    NoneType,
                    str,
                    int,
                    float,
                    bool,
                    bytes,
                    list,
                    tuple,
                    dict,
                    BaseModel,
                ),
            )

        return {str(key): cls._sanitize_for_serialization(val) for key, val in obj_dict.items()}

    @classmethod
    def _parameters_to_tuples(
        cls, params: Mapping | list[tuple[str, Any]], collection_formats: Mapping[str, str]
    ) -> list[tuple[str, Any]]:
        """Get parameters as list of tuples, formatting collections.

        :param params: Parameters as dict or list of two-tuples.
        :param collection_formats: Parameter collection formats. One of 'multi', 'ssv', 'pipes', or 'csv'.
            'csv' is the default collection format.

        :return: Parameters as list of tuples, collections formatted
        """
        params = cls._sanitize_for_serialization(params)
        if not isinstance(collection_formats, dict):
            raise ClientTypeError("collection_formats must be a dict.", valid_classes=(dict,))
        new_params = []
        for key, value in list(params.items() if isinstance(params, dict) else params):
            if isinstance(value, (list, tuple)):
                # Implements OpenAPI multi-value parameter styles. Defaults to simple (csv).
                # https://swagger.io/specification/#style-values
                collection_format = collection_formats.get(key, "csv")
                if collection_format == "multi":
                    new_params.extend((key, value) for value in value)
                else:
                    if collection_format == "ssv":
                        delimiter = " "
                    elif collection_format == "pipes":
                        delimiter = "|"
                    else:  # csv is the default.
                        delimiter = ","
                    new_params.append((key, delimiter.join(value)))
            else:
                new_params.append((key, value))
        return new_params

    @classmethod
    def _parameters_to_url_query(cls, params: dict | list[tuple[str, Any]], collection_formats: dict[str, str]) -> str:
        """Get parameters as list of tuples, formatting collections.

        :param params: Parameters as dict or list of two-tuples.
        :param collection_formats: Parameter collection formats. One of 'multi', 'ssv', tsv', 'pipes', or 'csv'.
            'csv' is the default collection format.

        :return: URL query string (e.g. a=Hello%20World&b=123)
        """
        query_params = cls._parameters_to_tuples(params, collection_formats)
        return urlencode(query_params)

    @classmethod
    def _deserialize(cls, response: HTTPResponse, response_type: type[T] | None) -> T:
        """Deserializes dict, list, or str into an object.

        :param response: Value as dict, list, or str.
        :param response_type: Target type to deserialize data to. Can be class literal, list[T], or dict[str, T].

        :return: The deserialized object.
        """
        if response_type is not None and isclass(response_type) and issubclass(response_type, HTTPResponse):
            # Return the response object directly.
            return response

        if response_type is not bytes and response_type is not bytearray:
            match = None
            content_type = response.getheader("content-type")
            if content_type is not None:
                match = re.search(r"charset=([a-zA-Z\-\d]+)[\s;]?", content_type)
            encoding = match.group(1) if match else "utf-8"
            response_data = response.data.decode(encoding)
        else:
            response_data = response.data

        try:
            response_data = json.loads(response_data)
        except ValueError:
            pass  # data must not be JSON formatted.

        if response_type is None:  # Return decoded data for a known response without a schema.
            return response_data
        elif response_type is EmptyResponse and response_data == "":
            return EmptyResponse(status=response.status, reason=response.reason, headers=response.getheaders())
        elif response_type is EmptyResponse and response_data != "":
            raise ClientValueError(msg=f"Unexpected content with '{response.status}' status code")
        elif isclass(response_type) and issubclass(response_type, EvoAPIException):  # Raise if API exception.
            if response_type is EvoAPIException and CustomTypedError.provided_by(response_data):
                # Use Typed Error types, but don't override generalized error types.
                response_type = CustomTypedError.from_type_id(response_data.get("type"))
            raise response_type(
                status=response.status, reason=response.reason, content=response_data, headers=response.headers
            )
        else:
            try:
                return cls.__deserialize(response_data, response_type)
            except EvoClientException:
                raise
            except Exception as e:
                raise ClientValueError(msg="Could not deserialize result", caused_by=e)

    @classmethod
    def __deserialize(cls, data: dict | list | str, response_type: type[T]) -> T:
        """Deserializes dict, list, or str into an object.

        :param data: Value as dict, list, or str.
        :param response_type: Target type to deserialize data to. Can be class literal, list[T], or dict[str, T].

        :return: The deserialized object.
        """
        if data is None:
            return None

        if isinstance(response_type, GenericAlias):  # list[T], dict[str, T].
            return cls.__deserialize_generic(data, response_type)
        elif response_type in {str, int, float, bool, bytes, bytearray, dict}:
            return cls.__deserialize_primitive(data, response_type)
        elif response_type is datetime.datetime:
            return cls.__deserialize_datetime(data)
        elif response_type is datetime.date:
            return cls.__deserialize_datetime(data).date()
        elif response_type is datetime.time:
            return cls.__deserialize_time(data)
        elif issubclass(response_type, BaseModel):  # API Models.
            return response_type.model_validate(data)
        else:
            raise ClientValueError("Could not parse content.")

    @classmethod
    def __deserialize_generic(cls, data: list | dict, klass: GenericAlias) -> list | dict[str, Any]:
        """Deserializes list or dict into a list or dict of objects.

        :param data: Value as list or dict.
        :param klass: Target type to deserialize data to. Can be list[T] or dict[str, T].

        :return: The deserialized object.

        :raises ClientTypeError: If the data could not be deserialized.
        """
        if klass.__origin__ is list and isinstance(data, list):
            # Deserialize list.
            (inner_klass,) = klass.__args__
            return [cls.__deserialize(sub_data, inner_klass) for sub_data in data]

        elif klass.__origin__ is dict and klass.__args__[0] is str and isinstance(data, dict):
            # Deserialize dict.
            _, value_klass = klass.__args__
            return {str(key): cls.__deserialize(value, value_klass) for key, value in data.items()}

        else:
            raise ClientTypeError(msg=f"Could not deserialize '{type(data)}' as '{klass}'.")

    @staticmethod
    def __deserialize_primitive(data: str, klass: type):
        """Deserializes string to primitive type.

        :param data: str.
        :param klass: class literal.

        :return: str, int, float, bool.

        :raises ClientTypeError: If the data could not be deserialized.
        """
        try:
            return klass(data)
        except TypeError as e:
            raise ClientTypeError(msg="Could not deserialize primitive", caused_by=e)

    @staticmethod
    def __deserialize_datetime(string: str) -> datetime.datetime:
        """Deserializes string to datetime.

        :param string: Date string.

        :return: datetime object.

        :raises ClientValueError: If the string could not be parsed as datetime.
        """
        try:
            return parse(string)
        except ValueError as e:
            raise ClientTypeError(msg="Could not deserialize datetime", caused_by=e)

    @classmethod
    def __deserialize_time(cls, string: str) -> datetime.time:
        """Deserializes string to time.

        :param string: time string.

        :return: time object.

        :raises ClientValueError: If the string could not be parsed as datetime.
        """
        datetime_value = cls.__deserialize_datetime(string)
        return datetime_value.time().replace(tzinfo=datetime_value.tzinfo)

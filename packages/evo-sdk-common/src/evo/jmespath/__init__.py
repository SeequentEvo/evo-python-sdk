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
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Generic, TypeVar, overload
from uuid import UUID

try:
    import jmespath
    from jmespath.exceptions import JMESPathError
    from jmespath.parser import ParsedResult as UpstreamParsedResult
    from jmespath.visitor import Options
except ImportError:
    raise ImportError(
        "The 'jmespath' package is required for evo.json.jmespath. "
        "Please install 'evo-sdk-common' with the 'jmespath' extra."
    ) from None

__all__ = [
    "JMESPathArrayProxy",
    "JMESPathError",
    "JMESPathObjectProxy",
    "Options",
    "ParsedResult",
    "compile",
    "proxy",
    "search",
]

T = TypeVar("T")


class _JMESPathViewEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, MappingProxyType):
            return dict(obj)

        if isinstance(obj, UUID):
            return str(obj)

        return super().default(obj)


class _JMESPathProxyMixin(Generic[T]):
    def __init__(self, data: T) -> None:
        self._data = data

    @property
    def raw(self) -> T:
        return self._data

    def search(self, expression: str) -> Any:
        """Search the proxied data with a JMESPath expression.

        :param expression: The JMESPath expression to compile and search.

        :return: The result of the search, as JMESArrayProxy, JMESObjectProxy, or a primitive type.

        :raises JMESPathError: If the expression is invalid.
        """
        return search(expression, self._data)

    def json_dumps(
        self,
        /,
        skipkeys: bool = False,
        ensure_ascii: bool = False,
        check_circular: bool = True,
        allow_nan: bool = True,
        cls: type[json.JSONEncoder] | None = None,
        indent: int | str | None = None,
        separators: tuple[str, str] | None = None,
        default: Callable[[Any], Any] | None = None,
        sort_keys: bool = False,
        **kwargs: Any,
    ) -> str:
        """Serialize the contained data to a JSON string.

        This method is the same as json.dumps, but uses a custom encoder by default that
        handles MappingProxyType and UUID.

        All parameters are the same as for json.dumps, except that cls defaults to
        our custom encoder.
        """
        if cls is None:
            # Use our own encoder that handles MappingProxyType and UUID.
            cls = _JMESPathViewEncoder

        return json.dumps(
            self._data,
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            cls=cls,
            indent=indent,
            separators=separators,
            default=default,
            sort_keys=sort_keys,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.json_dumps(indent=2)})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _JMESPathProxyMixin):
            return self.raw == other.raw
        else:
            return self.raw == other


class JMESPathArrayProxy(Generic[T], _JMESPathProxyMixin[Sequence[T]], Sequence[T]):
    def __getitem__(self, index: int | str) -> Any:
        if isinstance(index, int):
            return proxy(self.raw[index])
        else:
            return self.search(index)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        for item in self.raw:
            yield proxy(item)


class JMESPathObjectProxy(Generic[T], _JMESPathProxyMixin[Mapping[str, T]], Mapping[str, T]):
    def __getitem__(self, key: str) -> Any:
        return self.search(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


@overload
def proxy(value: Mapping[str, T]) -> JMESPathObjectProxy[T]: ...


@overload
def proxy(value: Sequence[T]) -> JMESPathArrayProxy[T]: ...


@overload
def proxy(value: T) -> T: ...


def proxy(value: Any) -> Any:
    """Convert a JSON-like value into a JMESPath proxy type if applicable."""
    if isinstance(value, Mapping):
        return JMESPathObjectProxy(value)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        return JMESPathArrayProxy(value)
    else:
        return value


class ParsedResult(UpstreamParsedResult):
    def search(self, value: Any, options: Options | None = None) -> Any:
        return proxy(super().search(value, options))


def compile(expression: str) -> ParsedResult:
    """Thin wrapper around jmespath.compile to return our own version of ParsedResult.

    :param expression: The JMESPath expression to compile.

    :return: A ParsedResult instance.

    :raises JMESPathError: If the expression is invalid.
    """
    result = jmespath.compile(expression)
    return ParsedResult(result.expression, result.parsed)


def search(expression: str, data: Any, options: Options | None = None) -> Any:
    """Reimplementation of jmespath.search that returns our proxy types for JSON results.

    :param expression: The JMESPath expression to compile.
    :param data: The data to search.
    :param options: Optional jmespath Options.

    :return: The result of the search, as JMESArrayProxy, JMESObjectProxy, or a primitive type.

    :raises JMESPathError: If the expression is invalid.
    """
    return compile(expression).search(data, options=options)

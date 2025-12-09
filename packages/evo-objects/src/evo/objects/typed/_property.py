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

from typing import Any, Callable, Generic, Protocol, TypeVar, overload

from pydantic import TypeAdapter

from evo import jmespath

from ._utils import (
    assign_jmespath_value,
    delete_jmespath_value,
)

_T = TypeVar("_T")


class WithDocument(Protocol):
    """Protocol for objects that have an underlying document."""

    _document: dict[str, Any]

    def search(self, jmespath_expr: str) -> Any:
        """Search the underlying document using a JMESPath expression.

        :param jmespath_expr: The JMESPath expression to evaluate.
        :return: The result of the JMESPath evaluation.
        """


class SchemaProperty(Generic[_T]):
    """Descriptor for data within a Geoscience Object schema.

    This can be used on either typed objects classes or dataset classes.
    """

    def __init__(
        self, jmespath_expr: str, typed_adapter: TypeAdapter[_T], default_factory: Callable[[], _T] | None = None
    ) -> None:
        self._jmespath_expr = jmespath_expr
        self._typed_adapter = typed_adapter
        self._default_factory = default_factory

    @overload
    def __get__(self, instance: None, owner: type[WithDocument]) -> SchemaProperty[_T]: ...

    @overload
    def __get__(self, instance: WithDocument, owner: type[WithDocument]) -> _T: ...

    def __get__(self, instance: WithDocument | None, owner: type[WithDocument]) -> Any:
        if instance is None:
            return self

        value = instance.search(self._jmespath_expr)
        if value is None and self._default_factory is not None:
            return self._default_factory()
        if isinstance(value, (jmespath.JMESPathArrayProxy, jmespath.JMESPathObjectProxy)):
            value = value.raw
        return self._typed_adapter.validate_python(value)

    def __set__(self, instance: WithDocument, value: Any) -> None:
        self.apply_to(instance._document, value)

    def apply_to(self, document: dict[str, Any], value: _T) -> None:
        dumped_value = self._typed_adapter.dump_python(value)

        if dumped_value is None:
            # Remove the property from the document if the value is None
            delete_jmespath_value(document, self._jmespath_expr)
        else:
            # Update the document with the new value
            assign_jmespath_value(document, self._jmespath_expr, dumped_value)

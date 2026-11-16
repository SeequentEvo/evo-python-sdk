#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Output hydration: the references a task returns, back as typed objects.

A task's results are references too -- ``{"reference": "<url>", "name": ..., "schema_id":
...}`` for an object it wrote, a JMESPath expression for the attribute it filled in. The
task's published ``results`` schema marks each of those with an ``output`` annotation, which
is all :class:`TaskResult` needs to hand the caller a loadable object instead of a URL::

    result = await client.geostatistics.kriging.run(...)
    grid = await result.target.load()
    frame = await result.target.to_dataframe()
    estimates = await result.target.attribute.to_dataframe()

An ``output: attribute`` node names an attribute rather than an object, so it carries the
same two annotations the input side uses to place one: ``attribute_from``, a relative JSON
pointer to the object the attribute lives on, and ``attribute_path``, the container each
object family keeps attributes in. A task publishes one expression for every family it
supports, so the container it names can be the wrong one for the object actually written --
:meth:`ResultNode.load` repairs it from ``attribute_path`` rather than reporting a miss.

:class:`TaskResult` and :class:`ResultNode` subclass :class:`dict`, so the raw payload is
still available exactly as the platform sent it -- ``result["target"]["reference"]`` and
``result == {...}`` behave as they did before hydration existed. Attribute access and the
loaders are additions on top, driven entirely by the schema; nothing here is task-specific.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from evo.common import IContext
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference

# ``attribute_path`` means the same thing on the way in and on the way out, so both sides
# read it through the same helpers rather than agreeing by convention.
from .resolution import _container_for, _container_of, _pointer_value
from .tasks.common.source_target import AnyTypedAttribute

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "ResultNode",
    "TaskResult",
]


class ResultNode(dict):
    """One object in a task's result payload, viewed through its ``results`` schema.

    Nested objects and arrays of objects are hydrated into further nodes, so the whole
    result is reachable by attribute access. Nodes annotated ``output: geoscience-object``
    or ``output: attribute`` additionally offer :meth:`load` and :meth:`to_dataframe`.
    """

    def __init__(
        self,
        payload: dict,
        schema: dict[str, Any] | None,
        context: IContext,
        *,
        root: dict | None = None,
        path: tuple[str, ...] = (),
    ) -> None:
        """
        :param payload: The raw payload for this node.
        :param schema: The ``results`` subschema describing it, if the task published one.
        :param context: An authenticated Evo context, used to load referenced objects.
        :param root: The whole result payload, which ``attribute_from`` pointers index into.
        :param path: Where this node sits in that payload, which they are relative to.
        """
        self._schema = schema if isinstance(schema, dict) else {}
        self._context = context
        self._root: dict = self if root is None else root
        self._path = path
        properties = self._schema.get("properties", {}) or {}
        super().__init__(
            {
                name: _hydrate(value, properties.get(name), context, self._root, (*path, name))
                for name, value in payload.items()
            }
        )

    def __getattr__(self, name: str) -> Any:
        # Result fields are whatever the payload carried, so they cannot be declared up
        # front. Only fires for names that are not real attributes of the dict itself.
        if name.startswith("_") or name not in self:
            raise AttributeError(name)
        return self[name]

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | {key for key in self if isinstance(key, str) and key.isidentifier()})

    async def load(self) -> BaseObject | AnyTypedAttribute:
        """Load whatever this node refers to, as its typed class.

        :return: The typed object for an ``output: geoscience-object`` node, or the typed
            attribute for an ``output: attribute`` one.

        :raises TypeError: If this node is not annotated as either.
        :raises ValueError: If an attribute's owning object cannot be identified, or the
            attribute is not on it.
        """
        match self._schema.get("output"):
            case "geoscience-object":
                return await object_from_reference(self._context, self["reference"])
            case "attribute":
                return await self._load_attribute()
        raise TypeError("this result is not a geoscience object or attribute reference")

    async def to_dataframe(self, *keys: str) -> pd.DataFrame:
        """Load what this node refers to and read it into a DataFrame.

        :param keys: Attributes to include, if the object's type accepts a selection. An
            ``output: attribute`` node is a single column and takes none.

        :raises TypeError: If this node is not a geoscience object or attribute reference.
        """
        loaded = await self.load()
        if isinstance(loaded, BaseObject):
            return await loaded.to_dataframe(*keys)
        if keys:
            raise TypeError("an attribute result is a single column and takes no keys")
        return await loaded.to_dataframe()

    async def _load_attribute(self) -> AnyTypedAttribute:
        """The typed attribute this node names, on the object ``attribute_from`` points to."""
        leaf = (self._schema.get("properties") or {}).get("reference") or {}
        pointer = leaf.get("attribute_from")
        reference = _pointer_value(pointer, [*self._path, "reference"], self._root) if pointer else None
        if not isinstance(reference, str):
            raise ValueError(f"cannot tell which object this attribute belongs to (attribute_from={pointer!r})")

        owner = await object_from_reference(self._context, reference)
        expression = self["reference"]
        if (match := _search(owner, expression)) is None:
            expression = _healed(expression, leaf.get("attribute_path"), owner.metadata.schema_id)
            match = _search(owner, expression)
        if match is None:
            raise ValueError(f"{expression!r} matches no attribute on {reference}")

        # Every family that can carry a compute attribute exposes its ``attribute_path``
        # container as ``attributes`` -- ``locations.attributes`` on a pointset,
        # ``cell_attributes`` on a grid -- so the typed lookup needs no per-family code.
        container = getattr(owner, "attributes", None)
        found = None if container is None else container[match.get("key") or match["name"]]
        if found is None or not found.exists:
            raise ValueError(f"{expression!r} is not an attribute {type(owner).__name__} exposes")
        return found


class TaskResult(ResultNode):
    """What a task returned, hydrated against the ``results`` schema it published.

    A :class:`dict` of the raw payload, plus attribute access and typed loaders for the
    parts the schema marks as outputs. A task that publishes no ``results`` schema simply
    yields the payload unhydrated.
    """


def _hydrate(value: Any, schema: Any, context: IContext, root: dict, path: tuple[str, ...]) -> Any:
    """Wrap the objects inside ``value``, leaving scalars and nulls as they came."""
    if isinstance(value, dict):
        return ResultNode(value, schema, context, root=root, path=path)
    if isinstance(value, list):
        items = schema.get("items") if isinstance(schema, dict) else None
        return [_hydrate(item, items, context, root, (*path, str(index))) for index, item in enumerate(value)]
    return value


def _search(owner: BaseObject, expression: str) -> Mapping[str, Any] | None:
    """The first attribute a JMESPath expression selects on an object, if it selects any."""
    found = owner.search(expression)
    if isinstance(found, Sequence) and not isinstance(found, (str, bytes)):
        found = found[0] if len(found) else None
    return found if isinstance(found, Mapping) else None


def _healed(expression: str, containers: Any, schema: ObjectSchema) -> str:
    """Point an attribute expression at the container this object's family actually uses.

    A task supporting several object families publishes one expression for all of them, so a
    result can name ``locations.attributes`` for an object that keeps its attributes under
    ``cell_attributes``. ``attribute_path`` says which is right; the predicate is left alone.
    """
    if (container := _container_for(containers, schema)) is None:
        return expression
    return container + expression[len(_container_of(expression)) :]

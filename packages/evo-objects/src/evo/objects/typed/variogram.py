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

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import TypeAdapter

from evo.objects import SchemaVersion

from ._property import SchemaProperty
from .base import BaseObject, BaseObjectData, ConstructableObject

__all__ = [
    "Variogram",
    "VariogramData",
]


@dataclass(kw_only=True, frozen=True)
class VariogramData(BaseObjectData):
    """Data for creating a Variogram.

    A variogram is a geostatistical model describing spatial correlation structure.
    The variogram model is defined by the nugget and multiple structures using the
    leapfrog-convention rotation.
    """

    sill: float
    """The variance of the variogram. Must be within a very small tolerance of the nugget
    plus the sum of all structure's contributions."""

    is_rotation_fixed: bool
    """Boolean value specifying whether all structure's rotations are the same."""

    structures: list[dict[str, Any]]
    """A list of at least one mathematical model, which are parameterised to represent
    the spatial structure of the variogram model. Can include exponential, gaussian,
    generalisedcauchy, spherical, spheroidal, linear, or cubic structures."""

    nugget: float = 0.0
    """The variance between two samples separated by near-zero lag distance, representing
    the randomness present. When plotted, this value is the y-intercept."""

    data_variance: float | None = None
    """The variance of the data, if different from the sill value, this is used for
    normalising or rescaling the variogram."""

    modelling_space: Literal["data", "normalscore"] | None = None
    """The modelling space the variogram model was fitted in - either 'data' for original
    units or 'normalscore' for gaussian space."""

    domain: str | None = None
    """The domain the variogram is modelled for."""

    attribute: str | None = None
    """The attribute the variogram is modelled for."""


class Variogram(ConstructableObject[VariogramData]):
    """A GeoscienceObject representing a variogram.

    The variogram describes the spatial correlation structure of a variable,
    used in geostatistical modeling and kriging interpolation.
    """

    _data_class = VariogramData

    sub_classification = "variogram"
    creation_schema_version = SchemaVersion(major=1, minor=2, patch=0)

    sill: float = SchemaProperty("sill", TypeAdapter(float))
    """The variance of the variogram."""

    is_rotation_fixed: bool = SchemaProperty("is_rotation_fixed", TypeAdapter(bool))
    """Boolean value specifying whether all structure's rotations are the same."""

    structures: list[dict[str, Any]] = SchemaProperty("structures", TypeAdapter(list[dict[str, Any]]))
    """List of variogram structures (exponential, gaussian, spherical, etc.)."""

    nugget: float = SchemaProperty("nugget", TypeAdapter(float), default_factory=lambda: 0.0)
    """The variance between two samples separated by near-zero lag distance."""

    data_variance: float | None = SchemaProperty("data_variance", TypeAdapter(float | None))
    """The variance of the data, used for normalising or rescaling the variogram."""

    modelling_space: Literal["data", "normalscore"] | None = SchemaProperty(
        "modelling_space", TypeAdapter(Literal["data", "normalscore"] | None)
    )
    """The modelling space the variogram model was fitted in."""

    domain: str | None = SchemaProperty("domain", TypeAdapter(str | None))
    """The domain the variogram is modelled for."""

    attribute: str | None = SchemaProperty("attribute", TypeAdapter(str | None))
    """The attribute the variogram is modelled for."""

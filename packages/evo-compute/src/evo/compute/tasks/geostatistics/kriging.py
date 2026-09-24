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

"""
Kriging compute task client.

This module provides typed dataclass models and convenience functions for running
the Kriging task (geostatistics/kriging).

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.geostatistics.kriging import KrigingParameters
    >>>
    >>> params = KrigingParameters(
    ...     source=pointset.attributes["grade"],
    ...     target=Target.new_attribute(block_model, "kriged_grade"),
    ...     variogram=variogram,
    ...     search=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ... )
    >>> result = await run(manager, params, preview=True)
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Protocol, runtime_checkable
from uuid import UUID

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

# Import shared components
from ..common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    CreateAttribute,
    Filter,
    GeoscienceObjectReference,
    SearchNeighborhood,
    UpdateAttribute,
    attribute_spec,
)
from ..common.results import TaskAttribute, TaskTarget
from ..common.runner import TaskRunner
from ..common.source_target import AnyTypedAttribute

__all__ = [
    # Kriging-specific (users import from evo.compute.tasks.kriging)
    "RECOMMENDED_DIAGNOSTIC_NAMES",
    "BlockDiscretisation",
    "Filter",
    "KrigingDiagnostics",
    "KrigingDiagnosticsResult",
    "KrigingMethod",
    "KrigingParameters",
    "KrigingResult",
    "KrigingResultModel",
    "KrigingRunner",
    "KrigingTargetResult",
    "OrdinaryKriging",
    "SimpleKriging",
]


# =============================================================================
# Kriging Method Types
# =============================================================================


class SimpleKriging(BaseModel):
    """Simple kriging method with a known constant mean.

    Use when the mean of the variable is known and constant across the domain.

    Example:
        >>> method = SimpleKriging(mean=100.0)
    """

    type: Literal["simple"] = "simple"
    """The method type discriminator."""

    mean: float
    """The mean value, assumed to be constant across the domain."""


class OrdinaryKriging(BaseModel):
    """Ordinary kriging method with unknown local mean.

    The most common kriging method. Estimates the local mean from nearby samples.
    This is the default kriging method if none is specified.
    """

    type: Literal["ordinary"] = "ordinary"
    """The method type discriminator."""


class KrigingMethod:
    """Factory for kriging methods.

    Provides convenient access to kriging method types.

    Example:
        >>> # Use ordinary kriging (most common)
        >>> method = KrigingMethod.ORDINARY
        >>>
        >>> # Use simple kriging with known mean
        >>> method = KrigingMethod.simple(mean=100.0)
    """

    ORDINARY: OrdinaryKriging = OrdinaryKriging()
    """Ordinary kriging - estimates local mean from nearby samples."""

    @staticmethod
    def simple(mean: float) -> SimpleKriging:
        """Create a simple kriging method with the given mean.

        Args:
            mean: The known constant mean value across the domain.

        Returns:
            SimpleKriging instance configured with the given mean.
        """
        return SimpleKriging(mean=mean)


# =============================================================================
# Block Discretisation
# =============================================================================


class BlockDiscretisation(BaseModel):
    """Sub-block discretisation for block kriging.

    When provided, each target block is subdivided into ``nx * ny * nz``
    sub-cells and the kriged value is averaged across these sub-cells.
    When omitted (``None``), point kriging is performed.

    Only applicable when the target is a 3D grid or block model.

    Each dimension must be an integer between 1 and 9 (inclusive).
    The default value of 1 in every direction is equivalent to point kriging.

    Example:
        >>> discretisation = BlockDiscretisation(nx=3, ny=3, nz=2)
    """

    nx: int = Field(1, ge=1, le=9)
    """Number of subdivisions in the x direction (1-9)."""

    ny: int = Field(1, ge=1, le=9)
    """Number of subdivisions in the y direction (1-9)."""

    nz: int = Field(1, ge=1, le=9)
    """Number of subdivisions in the z direction (1-9)."""


# =============================================================================
# Diagnostic Outputs
# =============================================================================

RECOMMENDED_DIAGNOSTIC_NAMES: dict[str, str] = {
    "valid": "valid",
    "kriging_variance": "KV",
    "slope_of_regression": "SoR",
    "kriging_efficiency": "KE",
    "kriging_mean": "KM",
    "num_samples": "NS",
    "num_drillholes": "NDh",
    "num_duplicates": "ND",
    "num_equidistant": "NeD",
    "sum_weights": "Sum",
    "sum_positive_weights": "SumP",
    "sum_negative_weights": "SumN",
    "min_distance": "MinD",
    "mean_distance": "AvgD",
    "aniso_min_distance": "MinAD",
    "aniso_mean_distance": "AvgAD",
}
"""Recommended attribute name for each diagnostic, matching Leapfrog's naming convention."""

# The attribute types that can hold each kind of diagnostic value, for geoscience objects and block models.
_ATTRIBUTE_TYPES_BY_KIND: dict[str, frozenset[str]] = {
    "boolean": frozenset({"bool", "Boolean"}),
    "integer": frozenset({"integer", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}),
    "floating-point": frozenset({"scalar", "Float64"}),
}

_DIAGNOSTIC_KINDS: dict[str, str] = {
    "valid": "boolean",
    **dict.fromkeys(("num_samples", "num_drillholes", "num_duplicates", "num_equidistant"), "integer"),
}
"""The kind of value each diagnostic writes, when it is not floating-point."""


def _object_identity(reference: str) -> str:
    """The object a reference URL points at, ignoring any version query string."""
    return reference.split("?", 1)[0]


def _claim_attribute(claimed: dict[str, str], attribute: str, output: str) -> None:
    """Record that ``output`` writes to ``attribute``, raising if another output already does."""
    if attribute in claimed:
        raise ValueError(
            f"{claimed[attribute]} and {output} both write to the attribute {attribute!r}. "
            "Every kriging output needs its own attribute."
        )
    claimed[attribute] = output


class KrigingDiagnostics(BaseModel):
    """Optional per-location diagnostics written alongside the kriging estimate.

    Each field selects one diagnostic and says where to write it on the target object.
    Only the diagnostics you set are computed; the rest are left out entirely.

    Every field accepts:

    - ``True`` — create an attribute using the Leapfrog-aligned recommended name from
      :data:`RECOMMENDED_DIAGNOSTIC_NAMES` (for example ``KV`` for ``kriging_variance``).
    - a ``str`` — create an attribute with that name.
    - a typed attribute from the target object — update it if it already exists,
      otherwise create it.
    - a :class:`~evo.compute.tasks.common.CreateAttribute` or
      :class:`~evo.compute.tasks.common.UpdateAttribute` for full control.

    Every output of a kriging task writes to its own attribute, so each diagnostic name
    must differ from the estimate's attribute name and from every other diagnostic.

    An existing attribute must be able to hold the diagnostic's values: boolean for ``valid``,
    integer for the ``num_*`` counts, and floating-point for the rest.

    Example:
        >>> diagnostics = KrigingDiagnostics(
        ...     kriging_variance=True,  # creates "KV"
        ...     num_samples="sample_count",  # creates "sample_count"
        ...     slope_of_regression=block_model.attributes["SoR"],  # updates if it exists
        ... )
    """

    model_config = {"extra": "forbid"}

    _source_objects: dict[str, str] = PrivateAttr(default_factory=dict)
    """The object each typed attribute came from, checked against the kriging target."""

    _attribute_names: dict[str, str] = PrivateAttr(default_factory=dict)
    """The name of each typed attribute, which an update reference does not carry."""

    valid: CreateAttribute | UpdateAttribute | None = None
    """Whether each target location's neighbourhood satisfied the search constraints."""

    kriging_variance: CreateAttribute | UpdateAttribute | None = None
    """The kriging variance (estimation uncertainty)."""

    slope_of_regression: CreateAttribute | UpdateAttribute | None = None
    """The slope of regression, a diagnostic for conditional bias."""

    kriging_efficiency: CreateAttribute | UpdateAttribute | None = None
    """The proportion of point or intra-block variance explained by the kriging weights."""

    kriging_mean: CreateAttribute | UpdateAttribute | None = None
    """The GLS mean for ordinary kriging, or the supplied constant mean for simple kriging."""

    num_samples: CreateAttribute | UpdateAttribute | None = None
    """The number of data samples used in the estimate."""

    num_drillholes: CreateAttribute | UpdateAttribute | None = None
    """The number of distinct drillholes contributing to the estimate.

    Requires a downhole intervals source object.
    """

    num_duplicates: CreateAttribute | UpdateAttribute | None = None
    """The number of duplicate sample locations used in the estimate."""

    num_equidistant: CreateAttribute | UpdateAttribute | None = None
    """A hint of how many other samples could have replaced the last returned sample
    because they were the same distance from the target location. Counted after searching
    and before clipping to the maximum number of samples."""

    sum_weights: CreateAttribute | UpdateAttribute | None = None
    """The sum of kriging weights applied to data samples."""

    sum_positive_weights: CreateAttribute | UpdateAttribute | None = None
    """The sum of positive kriging weights applied to data samples."""

    sum_negative_weights: CreateAttribute | UpdateAttribute | None = None
    """The sum of negative kriging weights applied to data samples (zero when
    negative-weight removal is enabled)."""

    min_distance: CreateAttribute | UpdateAttribute | None = None
    """The minimum isotropic Euclidean distance from the target location to any neighbour."""

    mean_distance: CreateAttribute | UpdateAttribute | None = None
    """The mean isotropic Euclidean distance from the target location to all neighbours."""

    aniso_min_distance: CreateAttribute | UpdateAttribute | None = None
    """The minimum anisotropic (ellipsoid-space) distance from the target location to any neighbour."""

    aniso_mean_distance: CreateAttribute | UpdateAttribute | None = None
    """The mean anisotropic (ellipsoid-space) distance from the target location to all neighbours."""

    @field_validator("*", mode="before")
    @classmethod
    def _resolve_attribute(cls, value: Any, info: ValidationInfo) -> Any:
        """Expand the shorthands each diagnostic accepts into an attribute specification."""
        if value is True:
            return CreateAttribute(name=RECOMMENDED_DIAGNOSTIC_NAMES[info.field_name])
        if value is False:
            return None
        if isinstance(value, str):
            return CreateAttribute(name=value)
        if isinstance(value, AnyTypedAttribute) and value.exists:
            kind = _DIAGNOSTIC_KINDS.get(info.field_name, "floating-point")
            if value.attribute_type not in _ATTRIBUTE_TYPES_BY_KIND[kind]:
                raise ValueError(
                    f"Diagnostic {info.field_name!r} writes {kind} values, so it cannot update {value.name!r}, "
                    f"which is a {value.attribute_type} attribute. Pass a name to create a new attribute instead."
                )
        return attribute_spec(value)

    @model_validator(mode="wrap")
    @classmethod
    def _remember_typed_attributes(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> KrigingDiagnostics:
        """Note the name and parent object of each typed attribute, before it is reduced to a specification."""
        names, sources = {}, {}
        if isinstance(data, dict):
            for field, value in data.items():
                if isinstance(value, AnyTypedAttribute):
                    names[field] = value.name
                    if value._obj is not None:
                        sources[field] = str(value._obj.metadata.url)
        diagnostics = handler(data)
        diagnostics._attribute_names.update(names)
        diagnostics._source_objects.update(sources)
        return diagnostics


# =============================================================================
# Kriging Parameters
# =============================================================================


class KrigingParameters(BaseModel):
    """Parameters for the kriging task.

    Defines all inputs needed to run a kriging interpolation task.

    Example:
        >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
        >>> from evo.compute.tasks.geostatistics.kriging import KrigingParameters
        >>> from evo.compute.tasks.common import Filter, FilterCondition
        >>>
        >>> params = KrigingParameters(
        ...     source=pointset.attributes["grade"],  # Source attribute
        ...     target=block_model.attributes["kriged_grade"],  # Target attribute (creates if doesn't exist)
        ...     variogram=variogram,  # Variogram model
        ...     search=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ...     # method defaults to ordinary kriging
        ... )
        >>>
        >>> # With a target filter to restrict kriging to specific categories on the target:
        >>> params_filtered = KrigingParameters(
        ...     source=pointset.attributes["grade"],
        ...     target=block_model.attributes["kriged_grade"],
        ...     variogram=variogram,
        ...     search=SearchNeighborhood(...),
        ...     target_filter=Filter(
        ...         where=FilterCondition(
        ...             attribute=block_model.attributes["domain"],
        ...             operator="in",
        ...             values=["LMS1", "LMS2"],
        ...         ),
        ...     ),
        ... )
        >>>
        >>> # With diagnostics written alongside the estimate:
        >>> params_with_diagnostics = KrigingParameters(
        ...     source=pointset.attributes["grade"],
        ...     target=block_model.attributes["kriged_grade"],
        ...     variogram=variogram,
        ...     search=SearchNeighborhood(...),
        ...     diagnostics=KrigingDiagnostics(kriging_variance=True, num_samples=True),
        ... )
    """

    model_config = {"populate_by_name": True}

    source: AnySourceAttribute
    """The source object and attribute containing known values."""

    target: AnyTargetAttribute
    """The target object and attribute to create or update with kriging results."""

    variogram: GeoscienceObjectReference
    """Model of the covariance within the domain (Variogram object or reference)."""

    search: SearchNeighborhood = Field(alias="neighborhood")
    """Search neighborhood parameters."""

    method: SimpleKriging | OrdinaryKriging = Field(default_factory=OrdinaryKriging, alias="kriging_method")
    """The kriging method to use. Defaults to ordinary kriging if not specified."""

    source_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict kriging to a subset of the source data."""

    target_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict kriging to a subset of the target object."""

    block_discretisation: BlockDiscretisation | None = None
    """Optional sub-block discretisation for block kriging.

    When provided, each target block is subdivided into nx × ny × nz sub-cells
    and the kriged value is averaged across these sub-cells. When omitted,
    point kriging is performed. Only applicable when the target is a 3D grid
    or block model.
    """

    diagnostics: KrigingDiagnostics | None = Field(None, exclude=True)
    """Optional diagnostics to write onto the target object alongside the estimate.

    Only the diagnostics you select are computed. See :class:`KrigingDiagnostics`
    for the available outputs and the shorthands each of them accepts.
    """

    @model_validator(mode="wrap")
    @classmethod
    def _validate_outputs(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> KrigingParameters:
        """Check that every output writes to its own attribute on the target object."""
        raw_target = data.get("target") if isinstance(data, dict) else None
        params = handler(data)
        diagnostics = params.diagnostics
        if diagnostics is None:
            return params

        target_object = _object_identity(params.target.object)
        for field, source in diagnostics._source_objects.items():
            if _object_identity(source) != target_object:
                raise ValueError(
                    f"Diagnostic {field!r} references an attribute of a different object than the kriging "
                    f"target. Diagnostics are written onto the target object, {target_object}."
                )

        # An update reference does not carry the attribute's name, so it is only known for typed attributes.
        estimate_name = raw_target.name if isinstance(raw_target, AnyTypedAttribute) else None
        outputs: list[tuple[str, CreateAttribute | UpdateAttribute, str | None]] = [
            ("the estimate", params.target.attribute, estimate_name)
        ]
        outputs += [
            (f"diagnostic {field!r}", spec, diagnostics._attribute_names.get(field))
            for field in KrigingDiagnostics.model_fields
            if (spec := getattr(diagnostics, field)) is not None
        ]

        claimed_names: dict[str, str] = {}
        claimed_references: dict[str, str] = {}
        for label, spec, known_name in outputs:
            name = spec.name if isinstance(spec, CreateAttribute) else known_name
            if name is not None:
                _claim_attribute(claimed_names, name, label)
            if isinstance(spec, UpdateAttribute):
                _claim_attribute(claimed_references, spec.reference, label)
        return params

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.source_filter is not None:
            result["source"]["filter"] = self.source_filter.model_dump()
        if self.target_filter is not None:
            result["target"]["filter"] = self.target_filter.model_dump()
        if self.diagnostics is not None:
            result["target"]["diagnostics"] = self.diagnostics.model_dump(mode="json", exclude_none=True)
        return result


# =============================================================================
# Kriging Result Types
# =============================================================================

# TODO: tidy up `to_dataframe` implementations for better consistency. in _theory_ and spatial object does implement
# `to_dataframe()` (and should!!), but `BaseSPatialObject` does not declare this, and the `BlockModel` object has a different
# signature.


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    """Protocol for objects that can convert themselves to a DataFrame."""

    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


@runtime_checkable
class _BlockModelToDataFrameProtocol(Protocol):
    """Protocol for block models that can convert themselves to a DataFrame."""

    async def to_dataframe(
        self,
        columns: list[str] | None = None,
        version_uuid: UUID | Literal["latest"] | None = None,
        fb: IFeedback = ...,
    ) -> pd.DataFrame: ...


class KrigingDiagnosticsResult(BaseModel):
    """The diagnostic attributes that were written alongside the kriging estimate.

    Diagnostics that were not requested come back as ``None``.
    """

    valid: TaskAttribute | None = None
    kriging_variance: TaskAttribute | None = None
    slope_of_regression: TaskAttribute | None = None
    kriging_efficiency: TaskAttribute | None = None
    kriging_mean: TaskAttribute | None = None
    num_samples: TaskAttribute | None = None
    num_drillholes: TaskAttribute | None = None
    num_duplicates: TaskAttribute | None = None
    num_equidistant: TaskAttribute | None = None
    sum_weights: TaskAttribute | None = None
    sum_positive_weights: TaskAttribute | None = None
    sum_negative_weights: TaskAttribute | None = None
    min_distance: TaskAttribute | None = None
    mean_distance: TaskAttribute | None = None
    aniso_min_distance: TaskAttribute | None = None
    aniso_mean_distance: TaskAttribute | None = None


class KrigingTargetResult(TaskTarget):
    """Target information from a kriging task result."""

    diagnostics: KrigingDiagnosticsResult | None = None
    """The diagnostic attributes that were written, when any were requested."""


class KrigingResultModel(BaseModel):
    """Base class for compute task results.

    Provides common functionality for all task results including:
    - Pretty-printing in Jupyter notebooks
    - Portal URL extraction
    - Access to target object and data
    """

    message: str
    """A message describing what happened in the task."""

    target: KrigingTargetResult
    """Target information from the task result."""


class KrigingResult:
    TASK_DISPLAY_NAME: ClassVar[str] = "Kriging"

    def __init__(self, context: IContext, model: KrigingResultModel) -> None:
        self._target = model.target
        self._message = model.message
        self._context = context

    @property
    def message(self) -> str:
        """A message describing what happened in the task."""
        return self._message

    @property
    def target_name(self) -> str:
        """The name of the target object."""
        return self._target.name

    @property
    def target_reference(self) -> str:
        """Reference URL to the target object."""
        return self._target.reference

    @property
    def attribute_name(self) -> str:
        """The name of the attribute that was created/updated."""
        return self._target.attribute.name

    @property
    def diagnostics(self) -> dict[str, TaskAttribute]:
        """The diagnostic attributes that were written, keyed by diagnostic name.

        Only the diagnostics requested through
        :attr:`KrigingParameters.diagnostics` are present.

        Example:
            >>> result = await run(manager, params, preview=True)
            >>> result.diagnostics["kriging_variance"].name
            'KV'
        """
        diagnostics = self._target.diagnostics
        if diagnostics is None:
            return {}
        return {
            name: attribute
            for name in type(diagnostics).model_fields
            if (attribute := getattr(diagnostics, name)) is not None
        }

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the target object (e.g., 'regular-masked-3d-grid').

        Uses ``ObjectSchema.from_id`` to parse the schema ID. Falls back to the
        raw ``schema_id`` string when it cannot be parsed.
        """
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        """Load and return the target geoscience object.

        Args:
            context: Optional context to use. If not provided, uses the context
                    from when the task was run.

        Returns:
            The typed geoscience object (e.g., Regular3DGrid, RegularMasked3DGrid, BlockModel)

        Example:
            >>> result = await run(manager, params)
            >>> target = await result.get_target_object()
            >>> target  # Pretty-prints with Portal/Viewer links
        """
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Get the task results as a DataFrame.

        This is the simplest way to access the task output data. It loads
        the target object and returns its data as a pandas DataFrame.

        Args:
            context: Optional context to use. If not provided, uses the context
                    from when the task was run.
            columns: Optional list of column names to include. If None, includes
                    all columns. Use ["*"] to explicitly request all columns.

        Returns:
            A pandas DataFrame containing the task results.

        Example:
            >>> result = await run(manager, params)
            >>> df = await result.to_dataframe()
            >>> df.head()
        """
        target_obj = await self.get_target_object()

        if isinstance(target_obj, _ObjToDataframeProtocol):
            return await target_obj.to_dataframe(*columns)
        elif isinstance(target_obj, _BlockModelToDataFrameProtocol):
            return target_obj.to_dataframe(columns=list(columns) if columns else None)
        else:
            raise TypeError(
                f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
                "Use get_target_object() and access the data manually."
            )

    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message:   {self.message}",
            f"  Target:    {self.target_name}",
            f"  Attribute: {self.attribute_name}",
        ]
        if diagnostics := self.diagnostics:
            lines.append(f"  Diagnostics: {', '.join(a.name for a in diagnostics.values())}")
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class KrigingRunner(
    TaskRunner[KrigingParameters, KrigingResultModel, KrigingResult],
    topic="geostatistics",
    task="kriging",
):
    """Runner for kriging compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await KrigingRunner(context, params, preview=True)
    """

    async def _get_result(self, raw_result: KrigingResultModel) -> KrigingResult:
        return KrigingResult(self._context, raw_result)

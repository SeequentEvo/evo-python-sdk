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

Example::

    >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
    >>> from evo.compute.tasks.geostatistics.kriging import KrigingParameters
    >>>
    >>> params = KrigingParameters(
    ...     source=pointset.attributes["grade"],
    ...     target=Target.new_attribute(block_model, "kriged_grade"),
    ...     variogram=variogram,
    ...     neighborhood=SearchNeighborhood(
    ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...         max_samples=20,
    ...     ),
    ... )
    >>> result = await run(manager, params, preview=True)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal

from evo.common import IContext
from evo.objects.typed import BaseSpatialObject, DownholeIntervals, Variogram
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

# Import shared components
from ..common import (
    AnySourceAttribute,
    AnyTargetAttribute,
    BlockDiscretization,
    CreateAttribute,
    Filter,
    GeoscienceObjectReference,
    OutputTarget,
    SearchNeighborhood,
    SingleTargetTaskResult,
    SourceTargetTaskParameters,
    TaskInputModel,
    UpdateAttribute,
    attribute_spec,
    canonicalize_attribute_selector,
    extract_typed_object,
    object_identity,
    resolve_output_target,
)
from ..common.results import TaskAttribute, TaskTarget
from ..common.runner import TaskRunner
from ..common.source_target import AnyTypedAttribute

__all__ = [
    # Kriging-specific (users import from evo.compute.tasks.kriging)
    "RECOMMENDED_DIAGNOSTIC_NAMES",
    "BlockDiscretization",
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


class SimpleKriging(TaskInputModel):
    """Simple kriging method with a known constant mean.

    Use when the mean of the variable is known and constant across the domain.

    Example::

        >>> method = SimpleKriging(mean=100.0)
    """

    type: Literal["simple"] = "simple"
    """The method type discriminator."""

    mean: float
    """The mean value, assumed to be constant across the domain."""


class OrdinaryKriging(TaskInputModel):
    """Ordinary kriging method with unknown local mean.

    The most common kriging method. Estimates the local mean from nearby samples.
    This is the default kriging method if none is specified.
    """

    type: Literal["ordinary"] = "ordinary"
    """The method type discriminator."""


class KrigingMethod:
    """Factory for kriging methods.

    Provides convenient access to kriging method types.

    Example::

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


class KrigingDiagnostics(TaskInputModel):
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

    Example::

        >>> diagnostics = KrigingDiagnostics(
        ...     kriging_variance=True,  # creates "KV"
        ...     num_samples="sample_count",  # creates "sample_count"
        ...     slope_of_regression=block_model.attributes["SoR"],  # updates if it exists
        ... )
    """

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
        names: dict[str, str] = {}
        sources: dict[str, str] = {}
        if isinstance(data, KrigingDiagnostics):
            names.update(data._attribute_names)
            sources.update(data._source_objects)
        elif isinstance(data, Mapping):
            for field, value in data.items():
                if isinstance(value, AnyTypedAttribute):
                    names[field] = value.name
                    if value._obj is not None:
                        sources[field] = str(value._obj.metadata.url)
        diagnostics = handler(data)
        diagnostics._attribute_names.update(names)
        diagnostics._source_objects.update(sources)
        return diagnostics


_KRIGING_METHOD_ADAPTER: TypeAdapter[SimpleKriging | OrdinaryKriging] = TypeAdapter(SimpleKriging | OrdinaryKriging)


# =============================================================================
# Kriging Parameters
# =============================================================================


class KrigingParameters(SourceTargetTaskParameters):
    """Parameters for the kriging task.

    Defines all inputs needed to run a kriging interpolation task.

    Example::

        >>> from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
        >>> from evo.compute.tasks.geostatistics.kriging import KrigingParameters
        >>> from evo.compute.tasks.common import Filter, FilterCondition
        >>>
        >>> params = KrigingParameters(
        ...     source=pointset.attributes["grade"],  # Source attribute
        ...     target=block_model.attributes["kriged_grade"],  # Target attribute (creates if doesn't exist)
        ...     variogram=variogram,  # Variogram model
        ...     neighborhood=SearchNeighborhood(
        ...         ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        ...         max_samples=20,
        ...     ),
        ...     # kriging_method defaults to ordinary kriging
        ... )
        >>>
        >>> # With a target filter to restrict kriging to specific categories on the target:
        >>> params_filtered = KrigingParameters(
        ...     source=pointset.attributes["grade"],
        ...     target=block_model.attributes["kriged_grade"],
        ...     variogram=variogram,
        ...     neighborhood=SearchNeighborhood(...),
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
        ...     neighborhood=SearchNeighborhood(...),
        ...     diagnostics=KrigingDiagnostics(kriging_variance=True, num_samples=True),
        ... )
    """

    _alias_pairs: ClassVar[tuple[tuple[str, str], ...]] = (
        ("neighborhood", "search"),
        ("kriging_method", "method"),
    )
    _alias_validators: ClassVar[Mapping[str, Callable[[Any], Any]]] = {
        "neighborhood": SearchNeighborhood.model_validate,
        "kriging_method": _KRIGING_METHOD_ADAPTER.validate_python,
    }
    _nested_target_diagnostics_type: ClassVar[type[BaseModel] | None] = KrigingDiagnostics
    _object_roles: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "source": (BaseSpatialObject,),
        "target": (BaseSpatialObject,),
        "variogram": (Variogram,),
    }
    _drillhole_source_field: ClassVar[str | None] = "source"

    source: AnySourceAttribute
    """The source object and attribute containing known values."""

    target: AnyTargetAttribute
    """The target object and attribute to create or update with kriging results."""

    variogram: GeoscienceObjectReference
    """Model of the covariance within the domain (Variogram object or reference)."""

    neighborhood: SearchNeighborhood = Field(validation_alias=AliasChoices("neighborhood", "search"))
    """Search neighborhood parameters."""

    kriging_method: SimpleKriging | OrdinaryKriging = Field(
        default_factory=OrdinaryKriging,
        validation_alias=AliasChoices("kriging_method", "method"),
    )
    """The kriging method to use. Defaults to ordinary kriging if not specified."""

    @property
    def search(self) -> SearchNeighborhood:
        """Backwards-compatible alias for :attr:`neighborhood`."""
        return self.neighborhood

    @property
    def method(self) -> SimpleKriging | OrdinaryKriging:
        """Backwards-compatible alias for :attr:`kriging_method`."""
        return self.kriging_method

    source_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict kriging to a subset of the source data."""

    target_filter: Filter | None = Field(None, exclude=True)
    """Optional filter to restrict kriging to a subset of the target object."""

    block_discretization: BlockDiscretization | None = None
    """Optional sub-block discretization for block kriging.

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

    @model_validator(mode="after")
    def _validate_outputs(self) -> KrigingParameters:
        """Check that every output writes to its own attribute on the target object."""
        diagnostics = self.diagnostics
        if diagnostics is None:
            return self

        if diagnostics.num_drillholes is not None:
            source_obj = extract_typed_object(self.source)
            if source_obj is not None and not isinstance(source_obj, DownholeIntervals):
                raise ValueError(
                    f"Diagnostic 'num_drillholes' requires a DownholeIntervals source object, "
                    f"got {type(source_obj).__name__}."
                )

        target_object = _object_identity(self.target.object)
        for field, source in diagnostics._source_objects.items():
            if _object_identity(source) != target_object:
                raise ValueError(
                    f"Diagnostic {field!r} references an attribute of a different object than the kriging "
                    f"target. Diagnostics are written onto the target object, {target_object}."
                )

        # An update reference does not carry the attribute's name unless it came from a typed attribute
        # or can be resolved through the target BaseObject.
        target_obj = extract_typed_object(self.target) or self._typed_objects.get(target_object)
        outputs: list[tuple[str, CreateAttribute | UpdateAttribute, str | None]] = [
            ("the estimate", self.target.attribute, self.target._attribute_name)
        ]
        outputs += [
            (f"diagnostic {field!r}", spec, diagnostics._attribute_names.get(field))
            for field in KrigingDiagnostics.model_fields
            if (spec := getattr(diagnostics, field)) is not None
        ]

        claimed_names: dict[str, str] = {}
        claimed_references: dict[str, str] = {}
        claimed_canonical_refs: dict[str, tuple[str, str]] = {}
        for label, spec, known_name in outputs:
            name = spec.name if isinstance(spec, CreateAttribute) else known_name
            if name is None and isinstance(spec, UpdateAttribute):
                canonical_sel = canonicalize_attribute_selector(spec.reference)
                if canonical_sel.startswith("primary:name:"):
                    name = canonical_sel.removeprefix("primary:name:")
                elif target_obj is not None:
                    resolved = resolve_output_target(target_object, spec, target_obj=target_obj)
                    for alias in resolved.attribute_aliases or ():
                        if alias.startswith("primary:name:"):
                            name = alias.removeprefix("primary:name:")
                            break
            if isinstance(spec, UpdateAttribute):
                _claim_attribute(claimed_references, spec.reference, label)
                canonical_sel = canonicalize_attribute_selector(spec.reference)
                if canonical_sel in claimed_canonical_refs:
                    prev_label, prev_ref = claimed_canonical_refs[canonical_sel]
                    raise ValueError(
                        f"{prev_label} and {label} both write to the attribute {prev_ref!r}. "
                        "Every kriging output needs its own attribute."
                    )
                claimed_canonical_refs[canonical_sel] = (label, spec.reference)
            if name is not None:
                _claim_attribute(claimed_names, name, label)
        return self

    def output_targets(self) -> list[OutputTarget]:
        targets = super().output_targets()
        if self.diagnostics is None:
            return targets
        target_obj = extract_typed_object(self.target) or self._typed_objects.get(object_identity(self.target.object))
        for field in KrigingDiagnostics.model_fields:
            spec = getattr(self.diagnostics, field)
            if spec is not None:
                targets.append(
                    resolve_output_target(
                        self.target.object,
                        spec,
                        target_obj=target_obj,
                        known_name=self.diagnostics._attribute_names.get(field),
                    )
                )
        return targets


# =============================================================================
# Kriging Result Types
# =============================================================================


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


class KrigingResult(SingleTargetTaskResult[KrigingTargetResult]):
    TASK_DISPLAY_NAME: ClassVar[str] = "Kriging"

    def __init__(self, context: IContext, model: KrigingResultModel) -> None:
        super().__init__(context, model)

    @property
    def diagnostics(self) -> dict[str, TaskAttribute]:
        """The diagnostic attributes that were written, keyed by diagnostic name.

        Only the diagnostics requested through
        :attr:`KrigingParameters.diagnostics` are present.

        Example::

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

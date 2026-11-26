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

"""Cross-validation compute task client.

This module provides typed models for running the cross-validation task
(``geostatistics/cross-validation``). Cross-validation holds samples out of the
source data, re-estimates them at their own locations from the remaining
samples, writes the estimates (and optional signed errors and kriging
diagnostics) back onto the source object, and returns summary statistics.

Two tagged unions select how each run behaves without exposing conditionally
valid fields:

- ``validation`` (:data:`ValidationStrategy`) — what is held out:

  - :class:`LeaveOneSampleOut` holds out each sample and any coincident samples.
  - :class:`LeaveOneDrillholeOut` holds out every sample sharing one drillhole
    ID and requires a :class:`~evo.objects.typed.DownholeIntervals` source.

- ``estimator`` (:data:`CrossValidationEstimator`) — how held-out locations are
  re-estimated:

  - :class:`KrigingEstimator` — simple or ordinary kriging with a variogram and
    optional :class:`~evo.compute.tasks.geostatistics.kriging.KrigingDiagnostics`.
  - :class:`KNNEstimator` — k-nearest-neighbour arithmetic mean.
  - :class:`IDWEstimator` — inverse-distance weighting with a positive ``power``.

Example — Leave-one-sample-out cross-validation on a pointset::

    >>> from evo.compute.tasks import Ellipsoid, EllipsoidRanges, KrigingDiagnostics, SearchNeighborhood, run
    >>> from evo.compute.tasks.geostatistics.cross_validation import (
    ...     CrossValidationOutputs,
    ...     CrossValidationParameters,
    ...     KrigingEstimator,
    ...     LeaveOneSampleOut,
    ... )
    >>>
    >>> params = CrossValidationParameters(
    ...     source=pointset.attributes["grade"],
    ...     validation=LeaveOneSampleOut(),
    ...     estimator=KrigingEstimator(
    ...         variogram=variogram,
    ...         neighborhood=SearchNeighborhood(
    ...             ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...             max_samples=20,
    ...         ),
    ...         diagnostics=KrigingDiagnostics(kriging_variance=True),
    ...     ),
    ...     outputs=CrossValidationOutputs(estimate="cv_estimate", error="cv_error"),
    ... )
    >>> result = await run(manager, params, preview=True)
    >>> result.statistics.root_mean_squared_error

Example — Leave-one-drillhole-out cross-validation on downhole intervals::

    >>> from evo.compute.tasks.geostatistics.cross_validation import (
    ...     CrossValidationOutputs,
    ...     CrossValidationParameters,
    ...     IDWEstimator,
    ...     LeaveOneDrillholeOut,
    ... )
    >>>
    >>> params = CrossValidationParameters(
    ...     source=downhole_intervals.attributes["grade"],
    ...     validation=LeaveOneDrillholeOut(),
    ...     estimator=IDWEstimator(
    ...         neighborhood=SearchNeighborhood(
    ...             ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
    ...             max_samples=24,
    ...             max_samples_per_drillhole=4,
    ...         ),
    ...         power=2.0,
    ...     ),
    ...     outputs=CrossValidationOutputs(estimate="idw_cv_estimate", error="idw_cv_error"),
    ... )
    >>> result = await run(manager, params, preview=True)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, Protocol, TypeAlias, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, BlockModel, object_from_reference
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ..common import (
    CreateAttribute,
    Filter,
    GeoscienceObjectReference,
    SearchNeighborhood,
    Source,
    UpdateAttribute,
    attribute_spec,
)
from ..common.results import TaskAttribute
from ..common.runner import TaskRunner
from ..common.source_target import AnyTypedAttribute, _source_from_attribute
from .kriging import (
    _ATTRIBUTE_TYPES_BY_KIND,
    KrigingDiagnostics,
    KrigingDiagnosticsResult,
    OrdinaryKriging,
    SimpleKriging,
    _object_identity,
)

__all__ = [
    "CrossValidationEstimator",
    "CrossValidationOutputs",
    "CrossValidationParameters",
    "CrossValidationResult",
    "CrossValidationResultModel",
    "CrossValidationRunner",
    "CrossValidationSource",
    "CrossValidationSourceResult",
    "CrossValidationStatistics",
    "IDWEstimator",
    "KNNEstimator",
    "KrigingCrossValidationStatistics",
    "KrigingEstimator",
    "LeaveOneDrillholeOut",
    "LeaveOneSampleOut",
    "ValidationStrategy",
]


# =============================================================================
# Validation Strategy Tagged Union
# =============================================================================


class LeaveOneSampleOut(BaseModel):
    """Hold out each sample and any samples coincident with its location.

    Estimates every selected sample in one pass, excluding the sample's own
    location (including coincident samples strictly closer than ``1e-8`` world
    units) before applying neighbour quotas.

    Example::

        >>> validation = LeaveOneSampleOut()
    """

    model_config = {"extra": "forbid"}

    type: Literal["sample"] = "sample"
    """Discriminator selecting leave-one-sample-out validation."""


class LeaveOneDrillholeOut(BaseModel):
    """Hold out one drillhole at a time, estimating its samples from the remaining holes.

    Requires a :class:`~evo.objects.typed.DownholeIntervals` source object to
    supply drillhole IDs. Each query excludes every sample sharing its drillhole
    ID before neighbour quotas are applied, while keeping samples from other
    holes eligible.

    Example::

        >>> validation = LeaveOneDrillholeOut()
    """

    model_config = {"extra": "forbid"}

    type: Literal["drillhole"] = "drillhole"
    """Discriminator selecting leave-one-drillhole-out validation."""


ValidationStrategy: TypeAlias = Annotated[
    LeaveOneSampleOut | LeaveOneDrillholeOut,
    Field(discriminator="type"),
]
"""Tagged union of supported cross-validation hold-out strategies."""


# =============================================================================
# Estimator Tagged Union
# =============================================================================

_ALLOWED_NEIGHBORHOOD_KEYS: frozenset[str] = frozenset(
    {
        *SearchNeighborhood.model_fields,
        "max_drillholes_per_estimate",
        "min_drillholes_per_estimate",
    }
)


def _validate_cross_validation_neighborhood(value: Any) -> Any:
    """Reject unknown fields on raw neighborhood mappings before delegating to :class:`SearchNeighborhood`."""
    if isinstance(value, Mapping):
        unknown = set(value) - _ALLOWED_NEIGHBORHOOD_KEYS
        if unknown:
            raise ValueError(f"Unknown neighborhood field(s): {', '.join(sorted(map(str, unknown)))}.")
    return value


_CrossValidationNeighborhood: TypeAlias = Annotated[
    SearchNeighborhood,
    BeforeValidator(_validate_cross_validation_neighborhood),
]


class KrigingEstimator(BaseModel):
    """Re-estimate held-out locations using simple or ordinary kriging.

    Carries the variogram, kriging method, search neighborhood, and optional
    per-sample :class:`~evo.compute.tasks.geostatistics.kriging.KrigingDiagnostics`
    written onto the source object alongside the cross-validated estimate.

    Example::

        >>> from evo.compute.tasks import KrigingDiagnostics, SearchNeighborhood
        >>> from evo.compute.tasks.geostatistics.kriging import KrigingMethod
        >>>
        >>> estimator = KrigingEstimator(
        ...     variogram=variogram,
        ...     neighborhood=SearchNeighborhood(ellipsoid=ellipsoid, max_samples=20),
        ...     kriging_method=KrigingMethod.ORDINARY,
        ...     diagnostics=KrigingDiagnostics(kriging_variance=True, num_samples=True),
        ... )
    """

    model_config = {"populate_by_name": True, "extra": "forbid"}

    type: Literal["kriging"] = "kriging"
    """Discriminator selecting the kriging estimator."""

    variogram: GeoscienceObjectReference
    """Model of the covariance within the domain (Variogram object or reference URL)."""

    neighborhood: _CrossValidationNeighborhood = Field(validation_alias=AliasChoices("neighborhood", "search"))
    """Search neighborhood deciding which remaining samples estimate each held-out location."""

    kriging_method: Annotated[SimpleKriging | OrdinaryKriging, Field(discriminator="type")] = Field(
        default_factory=OrdinaryKriging,
        validation_alias=AliasChoices("kriging_method", "method"),
    )
    """The kriging method to use. Defaults to ordinary kriging."""

    diagnostics: KrigingDiagnostics | None = None
    """Optional per-sample kriging diagnostic attributes to write onto the source object."""

    @property
    def search(self) -> SearchNeighborhood:
        """Backwards-compatible alias for :attr:`neighborhood`."""
        return self.neighborhood

    @property
    def method(self) -> SimpleKriging | OrdinaryKriging:
        """Backwards-compatible alias for :attr:`kriging_method`."""
        return self.kriging_method

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.diagnostics is None:
            result.pop("diagnostics", None)
        else:
            result["diagnostics"] = self.diagnostics.model_dump(mode="json", exclude_none=True)
        return result


class KNNEstimator(BaseModel):
    """Re-estimate held-out locations with the arithmetic mean of their nearest neighbours.

    Example::

        >>> estimator = KNNEstimator(
        ...     neighborhood=SearchNeighborhood(ellipsoid=ellipsoid, max_samples=12),
        ... )
    """

    model_config = {"populate_by_name": True, "extra": "forbid"}

    type: Literal["knn"] = "knn"
    """Discriminator selecting the k-nearest-neighbour estimator."""

    neighborhood: _CrossValidationNeighborhood = Field(validation_alias=AliasChoices("neighborhood", "search"))
    """Search neighborhood deciding which remaining samples estimate each held-out location."""

    @property
    def search(self) -> SearchNeighborhood:
        """Backwards-compatible alias for :attr:`neighborhood`."""
        return self.neighborhood


class IDWEstimator(BaseModel):
    """Re-estimate held-out locations using inverse-distance weighting.

    Example::

        >>> estimator = IDWEstimator(
        ...     neighborhood=SearchNeighborhood(ellipsoid=ellipsoid, max_samples=20),
        ...     power=2.0,
        ... )
    """

    model_config = {"populate_by_name": True, "extra": "forbid"}

    type: Literal["idw"] = "idw"
    """Discriminator selecting the inverse-distance-weighting estimator."""

    neighborhood: _CrossValidationNeighborhood = Field(validation_alias=AliasChoices("neighborhood", "search"))
    """Search neighborhood deciding which remaining samples estimate each held-out location."""

    power: float = Field(gt=0.0, allow_inf_nan=False)
    """Distance-weighting exponent (must be strictly positive, e.g. ``1.0`` or ``2.0``)."""

    @property
    def search(self) -> SearchNeighborhood:
        """Backwards-compatible alias for :attr:`neighborhood`."""
        return self.neighborhood

    @field_validator("power", mode="before")
    @classmethod
    def _reject_boolean_power(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("power must be a positive number, not a boolean.")
        return value


CrossValidationEstimator: TypeAlias = Annotated[
    KrigingEstimator | KNNEstimator | IDWEstimator,
    Field(discriminator="type"),
]
"""Tagged union of supported cross-validation estimators."""


# =============================================================================
# Source & Output Specifications
# =============================================================================


def _claim_cross_validation_attribute(claimed: dict[str, str], attribute: str, output: str) -> None:
    """Record that ``output`` writes to ``attribute``, raising if another output already does."""
    if attribute in claimed:
        raise ValueError(
            f"{claimed[attribute]} and {output} both write to the attribute {attribute!r}. "
            "Every cross-validation output needs its own attribute."
        )
    claimed[attribute] = output


def _validate_attribute_spec_mapping(value: Any) -> Any:
    """Reject extra keys on raw attribute-operation dicts so invalid combinations fail fast."""
    if isinstance(value, dict):
        operation = value.get("operation")
        allowed = {"operation", "name"} if operation == "create" else {"operation", "reference"}
        if operation in ("create", "update"):
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"Operation {operation!r} does not accept field(s): {', '.join(sorted(unknown))}.")
    return value


class CrossValidationSource(Source):
    """The samples to cross-validate and the attribute holding their known values.

    Can be constructed explicitly or converted automatically from a typed
    attribute such as ``pointset.attributes["grade"]`` or
    ``downhole_intervals.attributes["grade"]``.
    """

    model_config = {"extra": "forbid"}

    filter: Filter | None = None
    """Optional filter restricting which source locations are cross-validated and used as neighbours."""

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.filter is None:
            result.pop("filter", None)
        else:
            result["filter"] = self.filter.model_dump(mode="json", exclude_none=True)
        return result


def _convert_cross_validation_source(value: Any) -> Any:
    """Convert a typed attribute or :class:`Source` into a :class:`CrossValidationSource`."""
    if isinstance(value, CrossValidationSource):
        return value
    converted = _source_from_attribute(value)
    if isinstance(converted, Source) and not isinstance(converted, CrossValidationSource):
        return CrossValidationSource(object=converted.object, attribute=converted.attribute)
    return converted


class CrossValidationOutputs(BaseModel):
    """Per-sample attributes written back onto the source object.

    Both ``estimate`` and ``error`` accept:

    - a ``str`` — create a new attribute with that name.
    - a typed attribute from the source object — update it if it already exists
      (must be a floating-point attribute), otherwise create it.
    - a :class:`~evo.compute.tasks.common.CreateAttribute` or
      :class:`~evo.compute.tasks.common.UpdateAttribute` for explicit control.

    Samples that produced no valid estimate carry ``NaN`` in the written
    ``estimate`` and ``error`` attributes. The ``error`` attribute stores the
    signed error ``estimate - true``, so a positive value is an overestimate.

    Example::

        >>> outputs = CrossValidationOutputs(
        ...     estimate="cv_estimate",
        ...     error="cv_error",
        ... )
    """

    model_config = {"extra": "forbid"}

    _source_objects: dict[str, str] = PrivateAttr(default_factory=dict)
    """The object each typed attribute came from, checked against the cross-validation source."""

    _attribute_names: dict[str, str] = PrivateAttr(default_factory=dict)
    """The name of each typed attribute, which an update reference does not carry."""

    estimate: Annotated[CreateAttribute | UpdateAttribute, Field(discriminator="operation")]
    """Attribute to create or update on the source object with each sample's cross-validated estimate."""

    error: Annotated[CreateAttribute | UpdateAttribute, Field(discriminator="operation")] | None = None
    """Optional attribute to create or update with the signed error (``estimate - true``)."""

    @field_validator("estimate", "error", mode="before")
    @classmethod
    def _resolve_attribute(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str):
            return CreateAttribute(name=value)
        if isinstance(value, AnyTypedAttribute):
            if value.exists and value.attribute_type not in _ATTRIBUTE_TYPES_BY_KIND["floating-point"]:
                raise ValueError(
                    f"Output {info.field_name!r} writes floating-point values, so it cannot update {value.name!r}, "
                    f"which is a {value.attribute_type} attribute. Pass a name to create a new attribute instead."
                )
            return attribute_spec(value)
        return _validate_attribute_spec_mapping(value)

    @model_validator(mode="wrap")
    @classmethod
    def _remember_typed_attributes_and_check_duplicates(
        cls, data: Any, handler: ValidatorFunctionWrapHandler
    ) -> CrossValidationOutputs:
        names: dict[str, str] = {}
        sources: dict[str, str] = {}
        if isinstance(data, dict):
            for field, value in data.items():
                if isinstance(value, AnyTypedAttribute):
                    names[field] = value.name
                    if value._obj is not None:
                        sources[field] = str(value._obj.metadata.url)

        outputs = handler(data)
        outputs._attribute_names.update(names)
        outputs._source_objects.update(sources)

        if outputs.error is not None:
            claimed_names: dict[str, str] = {}
            claimed_references: dict[str, str] = {}
            for field in ("estimate", "error"):
                spec = getattr(outputs, field)
                label = f"output {field!r}"
                name = spec.name if isinstance(spec, CreateAttribute) else outputs._attribute_names.get(field)
                if name is not None:
                    _claim_cross_validation_attribute(claimed_names, name, label)
                if isinstance(spec, UpdateAttribute):
                    _claim_cross_validation_attribute(claimed_references, spec.reference, label)

        return outputs

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.error is None:
            result.pop("error", None)
        return result


# =============================================================================
# Cross-Validation Parameters
# =============================================================================


class CrossValidationParameters(BaseModel):
    """Parameters for the ``geostatistics/cross-validation`` compute task.

    Example::

        >>> from evo.compute.tasks import Filter, FilterCondition, KrigingDiagnostics, SearchNeighborhood, run
        >>> from evo.compute.tasks.geostatistics.cross_validation import (
        ...     CrossValidationOutputs,
        ...     CrossValidationParameters,
        ...     KrigingEstimator,
        ...     LeaveOneSampleOut,
        ... )
        >>>
        >>> params = CrossValidationParameters(
        ...     source=pointset.attributes["Cu_pct"],
        ...     source_filter=Filter(
        ...         where=FilterCondition(
        ...             attribute=pointset.attributes["domain"],
        ...             operator="in",
        ...             values=["Early Diorite"],
        ...         ),
        ...     ),
        ...     validation=LeaveOneSampleOut(),
        ...     estimator=KrigingEstimator(
        ...         variogram=variogram,
        ...         neighborhood=SearchNeighborhood(ellipsoid=ellipsoid, max_samples=24),
        ...         diagnostics=KrigingDiagnostics(kriging_variance=True, slope_of_regression=True),
        ...     ),
        ...     outputs=CrossValidationOutputs(estimate="Cu_cv_est", error="Cu_cv_err"),
        ... )
        >>> result = await run(manager, params, preview=True)
    """

    model_config = {"populate_by_name": True, "extra": "forbid"}

    source: Annotated[CrossValidationSource, BeforeValidator(_convert_cross_validation_source)]
    """Source object and attribute containing the known sample values to cross-validate."""

    validation: ValidationStrategy
    """Hold-out strategy (:class:`LeaveOneSampleOut` or :class:`LeaveOneDrillholeOut`)."""

    estimator: CrossValidationEstimator
    """Estimator configuration (:class:`KrigingEstimator`, :class:`KNNEstimator`, or :class:`IDWEstimator`)."""

    outputs: CrossValidationOutputs
    """Per-sample output attributes to write back onto the source object."""

    source_filter: Filter | None = Field(None, exclude=True)
    """Optional filter restricting cross-validation to a subset of the source samples.

    Excluded from top-level serialization and injected into ``source["filter"]``.
    """

    @model_validator(mode="after")
    def _validate_cross_validation_outputs(self) -> CrossValidationParameters:
        if self.source.filter is not None and self.source_filter is not None:
            raise ValueError("Specify a source filter either on 'source.filter' or via 'source_filter', not both.")

        source_object = _object_identity(self.source.object)
        for field, attr_source in self.outputs._source_objects.items():
            if _object_identity(attr_source) != source_object:
                raise ValueError(
                    f"Output {field!r} references an attribute of a different object than the cross-validation "
                    f"source. Outputs are written onto the source object, {source_object}."
                )

        diagnostics = self.estimator.diagnostics if isinstance(self.estimator, KrigingEstimator) else None
        if diagnostics is not None:
            for field, attr_source in diagnostics._source_objects.items():
                if _object_identity(attr_source) != source_object:
                    raise ValueError(
                        f"Diagnostic {field!r} references an attribute of a different object than the "
                        f"cross-validation source. Diagnostics are written onto the source object, {source_object}."
                    )

        outputs_to_check: list[tuple[str, CreateAttribute | UpdateAttribute, str | None]] = [
            ("output 'estimate'", self.outputs.estimate, self.outputs._attribute_names.get("estimate")),
        ]
        if self.outputs.error is not None:
            outputs_to_check.append(
                ("output 'error'", self.outputs.error, self.outputs._attribute_names.get("error")),
            )
        if diagnostics is not None:
            outputs_to_check.extend(
                (f"diagnostic {field!r}", spec, diagnostics._attribute_names.get(field))
                for field in KrigingDiagnostics.model_fields
                if (spec := getattr(diagnostics, field)) is not None
            )

        claimed_names: dict[str, str] = {}
        claimed_references: dict[str, str] = {}
        for label, spec, known_name in outputs_to_check:
            name = spec.name if isinstance(spec, CreateAttribute) else known_name
            if name is not None:
                _claim_cross_validation_attribute(claimed_names, name, label)
            if isinstance(spec, UpdateAttribute):
                _claim_cross_validation_attribute(claimed_references, spec.reference, label)

        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result = handler(self)
        if self.source_filter is not None:
            result["source"]["filter"] = self.source_filter.model_dump(mode="json", exclude_none=True)
        return result


# =============================================================================
# Cross-Validation Result Types
# =============================================================================


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class CrossValidationStatistics(BaseModel):
    """Summary statistics comparing held-out true values with their estimates.

    Statistics are computed over samples that produced a valid estimate
    (``valid_count``). Samples without a valid estimate are counted in
    ``invalid_count``, carry ``NaN`` in the written attributes, and are excluded
    from the summary statistics. Any statistic that is undefined for the
    available valid samples (such as ``correlation`` when fewer than two samples
    are valid or variance is zero) is ``None``.
    """

    model_config = {"extra": "forbid"}

    evaluated_count: int
    """Number of samples cross-validated after applying any source filter."""

    valid_count: int
    """Number of cross-validated samples that produced a finite estimate."""

    invalid_count: int
    """Number of cross-validated samples that produced no estimate."""

    mean_true: float | None
    """Mean of the held-out true values over the valid samples."""

    mean_estimate: float | None
    """Mean of the cross-validated estimates over the valid samples."""

    variance_true: float | None
    """Population variance of the held-out true values over the valid samples."""

    variance_estimate: float | None
    """Population variance of the cross-validated estimates over the valid samples."""

    mean_error: float | None
    """Mean of ``estimate - true``. Positive values indicate overestimation bias."""

    mean_absolute_error: float | None
    """Mean of ``|estimate - true|``."""

    mean_squared_error: float | None
    """Mean of ``(estimate - true) ** 2``."""

    root_mean_squared_error: float | None
    """Square root of ``mean_squared_error``, in the units of the estimated attribute."""

    error_variance: float | None
    """Population variance of ``estimate - true``."""

    correlation: float | None
    """Pearson correlation coefficient between true values and estimates, or ``None`` when undefined."""

    regression_slope: float | None
    """Slope of true values regressed on estimates, or ``None`` when undefined."""

    regression_intercept: float | None
    """Intercept of true values regressed on estimates, or ``None`` when ``regression_slope`` is ``None``."""


class KrigingCrossValidationStatistics(BaseModel):
    """Standardized error summary statistics returned for kriging cross-validation only.

    Computed over valid samples that also have a positive kriging variance
    (``standardized_count``). Absent (``None`` on the result) for KNN and IDW.
    """

    model_config = {"extra": "forbid"}

    standardized_count: int
    """Number of valid samples with a positive kriging variance."""

    mean_kriging_variance: float | None
    """Mean kriging variance over the standardized samples, or ``None`` when ``standardized_count == 0``."""

    mean_standardized_error: float | None
    """Mean of ``(estimate - true) / sqrt(kriging_variance)``, or ``None`` when ``standardized_count == 0``."""

    standardized_error_variance: float | None
    """Population variance of the standardized error, or ``None`` when ``standardized_count == 0``."""


class CrossValidationSourceResult(BaseModel):
    """The source object updated with the per-sample cross-validation attributes."""

    model_config = {"extra": "forbid"}

    reference: str
    """Reference URL to the updated source geoscience object."""

    name: str
    """Name of the source geoscience object."""

    description: str | None = None
    """Optional description of the source geoscience object."""

    schema_id: str
    """Schema identifier of the source geoscience object."""

    estimate: TaskAttribute
    """Attribute written with each sample's cross-validated estimate."""

    error: TaskAttribute | None = None
    """Attribute written with each sample's signed error, when requested."""

    diagnostics: KrigingDiagnosticsResult | None = None
    """Kriging diagnostic attributes written alongside the estimate, when requested."""


class CrossValidationResultModel(BaseModel):
    """Pydantic model for the raw cross-validation task result payload."""

    model_config = {"extra": "forbid"}

    message: str
    """A message describing what happened in the task."""

    source: CrossValidationSourceResult
    """The source object updated with the cross-validation output attributes."""

    statistics: CrossValidationStatistics
    """Summary statistics comparing held-out true values with their estimates."""

    kriging_statistics: KrigingCrossValidationStatistics | None = None
    """Standardized kriging statistics (present for kriging only; ``None`` for KNN and IDW)."""


class CrossValidationResult:
    """Rich result object for the cross-validation task.

    Provides typed access to the updated source object, output attributes,
    kriging diagnostics, common summary statistics, and kriging-only
    standardized error statistics.
    """

    TASK_DISPLAY_NAME: ClassVar[str] = "Cross-Validation"

    def __init__(self, context: IContext, model: CrossValidationResultModel) -> None:
        self._context = context
        self._message = model.message
        self._source = model.source
        self._statistics = model.statistics
        self._kriging_statistics = model.kriging_statistics

    @property
    def message(self) -> str:
        """A message describing what happened in the task."""
        return self._message

    @property
    def source(self) -> CrossValidationSourceResult:
        """Metadata and written attribute references for the updated source object."""
        return self._source

    @property
    def source_name(self) -> str:
        """The name of the source object updated with cross-validation attributes."""
        return self._source.name

    @property
    def source_reference(self) -> str:
        """Reference URL to the updated source object."""
        return self._source.reference

    @property
    def estimate_attribute(self) -> TaskAttribute:
        """The attribute containing each sample's cross-validated estimate."""
        return self._source.estimate

    @property
    def estimate_name(self) -> str:
        """The name of the attribute containing each sample's cross-validated estimate."""
        return self._source.estimate.name

    @property
    def error_attribute(self) -> TaskAttribute | None:
        """The attribute containing each sample's signed error, or ``None`` if not requested."""
        return self._source.error

    @property
    def error_name(self) -> str | None:
        """The name of the signed-error attribute, or ``None`` if not requested."""
        if self._source.error is None:
            return None
        return self._source.error.name

    @property
    def diagnostics(self) -> dict[str, TaskAttribute]:
        """The kriging diagnostic attributes that were written, keyed by diagnostic name.

        Empty when no diagnostics were requested or when the estimator is KNN or IDW.
        """
        diagnostics = self._source.diagnostics
        if diagnostics is None:
            return {}
        return {
            name: attribute
            for name in type(diagnostics).model_fields
            if (attribute := getattr(diagnostics, name)) is not None
        }

    @property
    def statistics(self) -> CrossValidationStatistics:
        """Summary statistics comparing held-out true values with their estimates."""
        return self._statistics

    @property
    def kriging_statistics(self) -> KrigingCrossValidationStatistics | None:
        """Kriging standardized error statistics, or ``None`` for KNN and IDW."""
        return self._kriging_statistics

    @property
    def schema(self) -> ObjectSchema:
        """The schema type of the source object (e.g., ``'pointset'`` or ``'downhole-intervals'``)."""
        return ObjectSchema.from_id(self._source.schema_id)

    async def get_source_object(self) -> BaseObject:
        """Load and return the updated source geoscience object."""
        return await object_from_reference(self._context, self._source.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        """Load the updated source object and return its data as a DataFrame.

        Args:
            *columns: Optional column names to select. If omitted, returns all columns.

        Returns:
            A pandas DataFrame containing the source data and written cross-validation attributes.

        Raises:
            TypeError: If the source object type does not support DataFrame export.
        """
        source_obj = await self.get_source_object()
        if isinstance(source_obj, BlockModel):
            return await source_obj.to_dataframe(columns=list(columns) if columns else None)
        if isinstance(source_obj, _ObjToDataframeProtocol):
            return await source_obj.to_dataframe(*columns)
        raise TypeError(
            f"Don't know how to get DataFrame from {type(source_obj).__name__}. "
            "Use get_source_object() and access the data manually."
        )

    def __str__(self) -> str:
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message:   {self.message}",
            f"  Source:    {self.source_name}",
            f"  Estimate:  {self.estimate_name}",
        ]
        if self.error_name is not None:
            lines.append(f"  Error:     {self.error_name}")
        if diagnostics := self.diagnostics:
            lines.append(f"  Diagnostics: {', '.join(a.name for a in diagnostics.values())}")
        stats = self._statistics
        lines.append(f"  Samples:   {stats.valid_count}/{stats.evaluated_count} valid ({stats.invalid_count} invalid)")
        if stats.root_mean_squared_error is not None:
            lines.append(f"  RMSE:      {stats.root_mean_squared_error:.4f}")
        return "\n".join(lines)


# =============================================================================
# Task Runner
# =============================================================================


class CrossValidationRunner(
    TaskRunner[CrossValidationParameters, CrossValidationResultModel, CrossValidationResult],
    topic="geostatistics",
    task="cross-validation",
):
    """Runner for cross-validation compute tasks.

    Automatically registered — used by :func:`~evo.compute.tasks.run` for
    dispatch, or directly::

        result = await CrossValidationRunner(context, params, preview=True)
    """

    async def _get_result(self, raw_result: CrossValidationResultModel) -> CrossValidationResult:
        return CrossValidationResult(self._context, raw_result)

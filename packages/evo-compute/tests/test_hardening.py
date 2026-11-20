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

"""Comprehensive validation, round-trip, typed-contract, and batch-collision tests for evo-compute."""

from __future__ import annotations

import math
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from evo.common import APIConnector, Environment
from evo.objects import DownloadedObject, ObjectMetadata, ObjectReference, ObjectSchema
from evo.objects.typed import (
    BlockModel,
    DownholeIntervals,
    PointSet,
    Regular3DGrid,
    RegularMasked3DGrid,
    Variogram,
)
from pydantic import BaseModel, ValidationError

from evo.compute.tasks import (
    AllOfFilter,
    AnyOfFilter,
    BlockDiscretization,
    CreateAttribute,
    Ellipsoid,
    EllipsoidRanges,
    Filter,
    FilterCondition,
    Rotation,
    SearchNeighborhood,
    Source,
    Target,
    TaskResultList,
    UpdateAttribute,
    run,
)
from evo.compute.tasks.common import (
    ExistingAttributeExpression,
    SingleTargetTaskResult,
    TaskAttribute,
    TaskInputModel,
    TaskParameters,
    TaskRegistry,
    TaskRunner,
    TaskTarget,
    canonicalize_attribute_selector,
    run_tasks,
)
from evo.compute.tasks.geostatistics.break_ties import (
    BreakTiesParameters,
    BreakTiesResult,
    BreakTiesResultModel,
)
from evo.compute.tasks.geostatistics.conditional_turning_bands import (
    ConditionalTurningBandsParameters,
    ConditionalTurningBandsResult,
    ConditionalTurningBandsResultModel,
    ConditionalTurningBandsTargetResult,
)
from evo.compute.tasks.geostatistics.conditioned_simulator import (
    ConSimCutoffAttribute,
    ConSimLinks,
    ConSimLossCalculation,
    ConSimMaterialCategory,
    ConSimParameters,
    ConSimQuantileAttribute,
    ConSimResult,
    ConSimResultModel,
    ConSimSummaryAttributes,
    ConSimTargetResult,
    ConSimValidationReport,
    ConSimValidationSummary,
    DistributionParams,
    LowerTailParams,
    ReportContext,
    ReportMeanThresholds,
    TailExtrapolationParams,
    UpperTailParams,
    ValidationReportContextItem,
)
from evo.compute.tasks.geostatistics.continuous_distribution import (
    ContinuousDistributionParameters,
    ContinuousDistributionResult,
    ContinuousDistributionResultModel,
    DistributionOutput,
    DistributionTarget,
    LowerTail,
    TailExtrapolation,
    UpperTail,
)
from evo.compute.tasks.geostatistics.declustering import (
    DeclusteringGrid,
    DeclusteringParameters,
    DeclusteringResult,
    DeclusteringResultModel,
    DeclusteringSource,
)
from evo.compute.tasks.geostatistics.idw import IDWParameters, IDWResult, IDWResultModel
from evo.compute.tasks.geostatistics.knn import KNNParameters, KNNResult, KNNResultModel
from evo.compute.tasks.geostatistics.kriging import (
    KrigingDiagnostics,
    KrigingMethod,
    KrigingParameters,
    KrigingResult,
    KrigingResultModel,
    KrigingTargetResult,
    OrdinaryKriging,
    SimpleKriging,
)
from evo.compute.tasks.geostatistics.location_wise import (
    LocationWiseParameters,
    LocationWiseResult,
    LocationWiseResultModel,
    LocationWiseTarget,
    LocationWiseTargetResult,
    MeanAboveCutoff,
    ProbabilityAboveCutoff,
)
from evo.compute.tasks.geostatistics.loss_calculation import (
    LossCalculationParameters,
    LossCalculationResult,
    LossCalculationResultModel,
    MaterialCategory,
)
from evo.compute.tasks.geostatistics.normal_score import (
    NormalScoreParameters,
    NormalScoreResult,
    NormalScoreResultModel,
)
from evo.compute.tasks.geostatistics.profit_calculation import (
    ProfitCalculationParameters,
    ProfitCalculationResult,
    ProfitCalculationResultModel,
)
from evo.compute.tasks.geostatistics.simulation_report import (
    SimReportContext,
    SimReportContextItem,
    SimReportLinks,
    SimReportThresholds,
    SimReportValidationReport,
    SimReportValidationSummary,
    SimulationReportBlockDiscretization,
    SimulationReportDistribution,
    SimulationReportParameters,
    SimulationReportResult,
    SimulationReportResultModel,
)

_BASE = "https://hub.test.evo.bentley.com"
_ORG = uuid.UUID("00000000-0000-0000-0000-000000000001")
_WS = uuid.UUID("00000000-0000-0000-0000-000000000002")
_ENV = Environment(hub_url=f"{_BASE}/", org_id=_ORG, workspace_id=_WS)


def _obj_url(suffix: int, version: str | None = None) -> str:
    obj_id = uuid.UUID(f"00000000-0000-0000-0000-{suffix:012d}")
    return str(ObjectReference.new(_ENV, object_id=obj_id, version_id=version))


POINTSET_URL = _obj_url(10)
DOWNHOLE_URL = _obj_url(11)
GRID_URL = _obj_url(20)
GRID_B_URL = _obj_url(21)
MASKED_GRID_URL = _obj_url(22)
VARIOGRAM_URL = _obj_url(30)
BLOCKMODEL_URL = _obj_url(40)
DIST_URL = _obj_url(50)
DIST_B_URL = _obj_url(51)


# ---------------------------------------------------------------------------
# Real typed GeoscienceObject builders (backed by in-memory documents)
# ---------------------------------------------------------------------------


class _StubDownloadedObject(DownloadedObject):
    """Lightweight in-memory DownloadedObject for constructing real typed BaseObject instances."""

    def __init__(self, document: dict, version_id: str = "1", parent_path: str = "/geology") -> None:
        self._doc = document
        self._meta = Mock(spec=ObjectMetadata)
        self._meta.schema_id = ObjectSchema.from_id(document["schema"])
        self._meta.id = uuid.UUID(document["uuid"])
        self._meta.version_id = version_id
        self._meta.environment = _ENV
        self._meta.name = f"{document['name']}.json"
        self._meta.parent = parent_path
        self._meta.path = f"{parent_path}/{self._meta.name}".lstrip(".")
        self._meta.url = ObjectReference.new(_ENV, object_id=self._meta.id, version_id=version_id)
        self._connector = Mock(spec=APIConnector)
        self._connector.base_url = _ENV.hub_url

    @property
    def metadata(self) -> ObjectMetadata:
        return self._meta

    def get_environment(self) -> Environment:
        return _ENV

    def get_org_id(self) -> uuid.UUID:
        return _ORG

    def get_connector(self) -> APIConnector:
        return self._connector

    def get_cache(self):
        return None

    def as_dict(self) -> dict:
        return self._doc


def _attr_doc(name: str, key: str, length: int = 4, attribute_type: str = "scalar") -> dict:
    return {
        "name": name,
        "key": key,
        "attribute_type": attribute_type,
        "values": {"data": f"sha-{key}", "length": length, "width": 1, "data_type": "float64"},
    }


def _make_pointset(suffix: int = 10, name: str = "AssayPoints") -> PointSet:
    obj_uuid = f"00000000-0000-0000-0000-{suffix:012d}"
    doc = {
        "name": name,
        "uuid": obj_uuid,
        "schema": "/objects/pointset/1.2.0/pointset.schema.json",
        "bounding_box": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10},
        "coordinate_reference_system": "unspecified",
        "locations": {
            "coordinates": {"data": "coords", "length": 4, "width": 3, "data_type": "float64"},
            "attributes": [
                _attr_doc("grade", "key-grade", length=4),
                _attr_doc("domain", "key-domain", length=4, attribute_type="category"),
                _attr_doc("weights", "key-weights", length=4),
            ],
        },
    }
    return PointSet(_StubDownloadedObject(doc))


def _make_downhole_intervals(suffix: int = 11) -> DownholeIntervals:
    obj_uuid = f"00000000-0000-0000-0000-{suffix:012d}"
    tbl3 = {"data": "c", "length": 4, "width": 3, "data_type": "float64"}
    doc = {
        "name": "Composites",
        "uuid": obj_uuid,
        "schema": "/objects/downhole-intervals/1.3.0/downhole-intervals.schema.json",
        "bounding_box": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10},
        "coordinate_reference_system": "unspecified",
        "is_composited": True,
        "start": {"coordinates": tbl3},
        "end": {"coordinates": tbl3},
        "mid_points": {"coordinates": tbl3},
        "from_to": {"intervals": {"start_and_end": {"data": "ft", "length": 4, "width": 2, "data_type": "float64"}}},
        "hole_id": {
            "values": {"data": "hv", "length": 4, "width": 1, "data_type": "int32"},
            "table": {"data": "ht", "length": 2, "keys_data_type": "int32", "values_data_type": "string"},
        },
        "attributes": [
            _attr_doc("grade", "dh-grade", length=4),
            _attr_doc("lithology", "dh-lith", length=4, attribute_type="category"),
        ],
    }
    return DownholeIntervals(_StubDownloadedObject(doc))


def _make_regular_grid(suffix: int = 20) -> Regular3DGrid:
    obj_uuid = f"00000000-0000-0000-0000-{suffix:012d}"
    doc = {
        "name": "SimGrid",
        "uuid": obj_uuid,
        "schema": "/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
        "bounding_box": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10},
        "coordinate_reference_system": "unspecified",
        "origin": [0.0, 0.0, 0.0],
        "size": [2, 2, 2],
        "cell_size": [5.0, 5.0, 5.0],
        "cell_attributes": [
            _attr_doc("kriged_grade", "grid-key-kriged", length=8),
            _attr_doc("domain", "grid-key-domain", length=8, attribute_type="category"),
        ],
        "vertex_attributes": [],
    }
    return Regular3DGrid(_StubDownloadedObject(doc))


def _make_masked_grid(suffix: int = 22) -> RegularMasked3DGrid:
    obj_uuid = f"00000000-0000-0000-0000-{suffix:012d}"
    doc = {
        "name": "MaskedGrid",
        "uuid": obj_uuid,
        "schema": "/objects/regular-masked-3d-grid/1.3.0/regular-masked-3d-grid.schema.json",
        "bounding_box": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10},
        "coordinate_reference_system": "unspecified",
        "origin": [0.0, 0.0, 0.0],
        "size": [2, 2, 2],
        "cell_size": [5.0, 5.0, 5.0],
        "number_of_active_cells": 4,
        "mask": {
            "name": "mask",
            "key": "mask-key",
            "attribute_type": "bool",
            "values": {"data": "m", "length": 8, "width": 1, "data_type": "bool"},
        },
        "cell_attributes": [
            _attr_doc("domain", "mgrid-domain", length=4, attribute_type="category"),
        ],
    }
    return RegularMasked3DGrid(_StubDownloadedObject(doc))


def _make_block_model(suffix: int = 40) -> BlockModel:
    obj_uuid = f"00000000-0000-0000-0000-{suffix:012d}"
    doc = {
        "name": "RegBM",
        "uuid": obj_uuid,
        "schema": "/objects/block-model/1.0.0/block-model.schema.json",
        "bounding_box": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10},
        "coordinate_reference_system": "unspecified",
        "block_model_uuid": "11111111-1111-1111-1111-111111111111",
        "geometry": {
            "model_type": "regular",
            "origin": [0.0, 0.0, 0.0],
            "n_blocks": [2, 2, 2],
            "block_size": [5.0, 5.0, 5.0],
        },
        "attributes": [
            {"name": "Cu_OK", "attribute_type": "Float64"},
            {"name": "Zone", "attribute_type": "Category"},
        ],
    }
    return BlockModel(_StubDownloadedObject(doc))


def _make_variogram(suffix: int = 30) -> Variogram:
    obj_uuid = f"00000000-0000-0000-0000-{suffix:012d}"
    doc = {
        "name": "CuVariogram",
        "uuid": obj_uuid,
        "schema": "/objects/variogram/1.1.0/variogram.schema.json",
        "sill": 1.0,
        "nugget": 0.1,
        "is_rotation_fixed": True,
        "structures": [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 100.0, "semi_major": 80.0, "minor": 40.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            }
        ],
    }
    return Variogram(_StubDownloadedObject(doc))


def _search(max_samples: int = 20, **kwargs) -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(
            ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            rotation=Rotation(dip_azimuth=45.0, dip=15.0, pitch=0.0),
        ),
        max_samples=max_samples,
        **kwargs,
    )


def _all_valid_task_parameter_instances() -> list[TaskParameters]:
    """Construct a representative valid parameter instance for every one of the 13 geostatistics tasks."""
    src = Source(object=POINTSET_URL, attribute="locations.attributes[?name=='grade']")
    tgt = Target.new_attribute(GRID_URL, "out_attr")
    src_f = Filter(where=FilterCondition(attribute="locations.attributes[?name=='domain']", operator="in", values=[1]))
    tgt_f = Filter(where=FilterCondition(attribute="cell_attributes[?name=='domain']", operator="in", values=["A"]))
    cats = [
        MaterialCategory(cutoff_grade=0.0, metal_price=0.0, label="waste"),
        MaterialCategory(cutoff_grade=0.5, metal_price=200.0, label="ore"),
    ]
    return [
        KrigingParameters(
            source=src,
            target=tgt,
            variogram=VARIOGRAM_URL,
            neighborhood=_search(min_samples=2, min_drillholes=2, max_drillholes=6),
            kriging_method=KrigingMethod.simple(mean=1.25),
            source_filter=src_f,
            target_filter=tgt_f,
            block_discretization=BlockDiscretization(nx=2, ny=2, nz=2),
            diagnostics=KrigingDiagnostics(kriging_variance=True, num_samples="ns_col"),
        ),
        IDWParameters(
            source=src,
            target=tgt,
            neighborhood=_search(max_drillholes=4),
            power=2.5,
            source_filter=src_f,
            target_filter=tgt_f,
        ),
        KNNParameters(
            source=src,
            target=tgt,
            neighborhood=_search(max_samples_per_octant=3),
            source_filter=src_f,
            target_filter=tgt_f,
        ),
        DeclusteringParameters(
            source=POINTSET_URL,
            grid=GRID_URL,
            target=Target.new_attribute(POINTSET_URL, "declus_wt"),
            neighborhood=_search(),
            power=None,
        ),
        BreakTiesParameters(
            source=src,
            target=Target.new_attribute(POINTSET_URL, "grade_bt"),
            neighborhood=_search(),
            seed=42,
        ),
        ConSimParameters(
            source_object=POINTSET_URL,
            source_attribute="locations.attributes[?name=='grade']",
            target_object=GRID_URL,
            filter=tgt_f,
            source_filter=src_f,
            neighborhood=_search(),
            variogram_model=VARIOGRAM_URL,
            block_discretization=BlockDiscretization(nx=2, ny=2, nz=2),
            distribution=DistributionParams(
                weights="locations.attributes[?name=='weights']",
                tail_extrapolation=TailExtrapolationParams(
                    upper=UpperTailParams(power=0.5, max=100.0),
                    lower=LowerTailParams(power=0.4, min=0.0),
                ),
            ),
            number_of_lines=250,
            number_of_simulations=10,
            number_of_simulations_to_save=4,
            loss_calculation=ConSimLossCalculation(
                material_categories=[
                    ConSimMaterialCategory(cutoff_grade=0.0, metal_price=0.0, label="waste"),
                    ConSimMaterialCategory(cutoff_grade=0.5, metal_price=150.0, label="ore"),
                ],
                processing_cost=15.0,
                mining_waste_cost=4.0,
                mining_ore_cost=8.0,
                metal_recovery_fraction=0.9,
                target_attribute=CreateAttribute(name="loss_cat"),
            ),
            location_wise_quantiles=[0.1, 0.5, 0.9],
            probability_above_cutoff=[0.5, 1.0],
            mean_above_cutoff=[0.5],
            perform_validation=True,
            report_context=ReportContext(
                title="QC",
                details=[ValidationReportContextItem(label="Domain", value="Zone1")],
            ),
            report_mean_thresholds=ReportMeanThresholds(acceptable=0.02, marginal=0.05),
        ),
        ConditionalTurningBandsParameters(
            source=POINTSET_URL,
            source_attribute="locations.attributes[?name=='grade']",
            target=GRID_URL,
            filter=tgt_f,
            source_filter=src_f,
            neighborhood=_search(),
            distribution=DIST_URL,
            variogram_model=VARIOGRAM_URL,
            block_discretization=BlockDiscretization(nx=3, ny=3, nz=2),
            number_of_lines=300,
            realizations=8,
        ),
        ContinuousDistributionParameters(
            source=src,
            weights=Source(object=POINTSET_URL, attribute="locations.attributes[?name=='weights']"),
            tail_extrapolation=TailExtrapolation(
                upper=UpperTail(power=0.5, max=50.0),
                lower=LowerTail(power=0.5, min=0.0),
            ),
            target=DistributionTarget(reference=DIST_URL, overwrite=True, description="dist", tags={"k": "v"}),
        ),
        LocationWiseParameters(
            source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
            target=LocationWiseTarget(object=GRID_URL),
            summary=True,
            quantiles=[0.1, 0.5, 0.9],
            probability_above_cutoff=ProbabilityAboveCutoff(cutoffs=[0.5, 1.0]),
            mean_above_cutoff=MeanAboveCutoff(cutoffs=[0.5]),
        ),
        LossCalculationParameters(
            source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
            target=tgt,
            material_categories=cats,
            processing_cost=20.0,
            mining_waste_cost=5.0,
            mining_ore_cost=10.0,
            metal_recovery_fraction=0.85,
        ),
        ProfitCalculationParameters(
            source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
            target=tgt,
            material_categories=cats,
            processing_cost=20.0,
            mining_waste_cost=5.0,
            mining_ore_cost=10.0,
            metal_recovery_fraction=0.85,
        ),
        NormalScoreParameters(
            method="forward",
            source=src,
            target=Target.new_attribute(POINTSET_URL, "grade_ns"),
            distribution=DIST_URL,
        ),
        SimulationReportParameters(
            simulation_source=POINTSET_URL,
            source_attribute="locations.attributes[0]",
            simulation_target=GRID_URL,
            point_simulations="cell_attributes[0]",
            point_simulations_normal_score="cell_attributes[1]",
            block_simulations="cell_attributes[2]",
            block_simulations_normal_score="cell_attributes[3]",
            variogram_model=VARIOGRAM_URL,
            neighborhood=_search(),
            block_discretization=SimulationReportBlockDiscretization(nx=2, ny=2, nz=2),
            distribution=SimulationReportDistribution(
                weights="locations.attributes[1]",
                tail_extrapolation=TailExtrapolationParams(upper=UpperTailParams(power=0.5, max=50.0)),
            ),
            number_of_simulations=20,
            number_of_lines=400,
            report_context=SimReportContext(
                title="Report",
                details=[SimReportContextItem(label="Elem", value="Cu")],
            ),
            report_mean_thresholds=SimReportThresholds(acceptable=0.03, marginal=0.08),
        ),
    ]


# ===========================================================================
# 1. Strict Extra-Field Rejection, Alias Deduplication, Finite Floats & Schemas
# ===========================================================================


class TestStrictValidationAndSchemas(unittest.TestCase):
    def test_extra_forbid_on_all_task_parameters(self):
        for params in _all_valid_task_parameter_instances():
            cls = type(params)
            payload = params.model_dump(mode="json")
            payload["unexpected_extra_field"] = 123
            with self.subTest(model=cls.__name__), self.assertRaises(ValidationError):
                cls.model_validate(payload)

    def test_extra_forbid_on_supporting_input_models(self):
        supporting_cases = [
            (BlockDiscretization, {"nx": 1, "ny": 1, "nz": 1}),
            (Source, {"object": POINTSET_URL, "attribute": "grade"}),
            (CreateAttribute, {"name": "a"}),
            (UpdateAttribute, {"reference": "attributes[0]"}),
            (Target, {"object": GRID_URL, "attribute": {"operation": "create", "name": "a"}}),
            (FilterCondition, {"attribute": "grade", "operator": "equal", "threshold": 1.0}),
            (
                AllOfFilter,
                {"filters": [{"type": "condition", "attribute": "grade", "operator": "equal", "threshold": 1.0}]},
            ),
            (
                AnyOfFilter,
                {"filters": [{"type": "condition", "attribute": "grade", "operator": "equal", "threshold": 1.0}]},
            ),
            (
                Filter,
                {"where": {"type": "condition", "attribute": "grade", "operator": "equal", "threshold": 1.0}},
            ),
            (SearchNeighborhood, {"ellipsoid": {"ranges": [100, 50, 25]}, "max_samples": 10}),
            (SimpleKriging, {"mean": 1.0}),
            (OrdinaryKriging, {}),
            (KrigingDiagnostics, {"kriging_variance": True}),
            (UpperTailParams, {"power": 0.5, "max": 10.0}),
            (LowerTailParams, {"power": 0.5, "min": 0.0}),
            (TailExtrapolationParams, {"upper": {"power": 0.5, "max": 10.0}}),
            (DistributionParams, {"weights": "w"}),
            (ConSimMaterialCategory, {"cutoff_grade": 0.0, "metal_price": 0.0, "label": "waste"}),
            (
                ConSimLossCalculation,
                {
                    "material_categories": [{"cutoff_grade": 0.0, "metal_price": 0.0, "label": "waste"}],
                    "processing_cost": 1.0,
                    "mining_waste_cost": 1.0,
                    "mining_ore_cost": 1.0,
                    "metal_recovery_fraction": 0.8,
                    "target_attribute": {"operation": "create", "name": "loss"},
                },
            ),
            (ValidationReportContextItem, {"label": "k", "value": "v"}),
            (ReportContext, {"title": "t", "details": [{"label": "k", "value": "v"}]}),
            (ReportMeanThresholds, {"acceptable": 0.01, "marginal": 0.05}),
            (DeclusteringSource, {"object": POINTSET_URL}),
            (DeclusteringGrid, {"object": GRID_URL}),
            (LocationWiseTarget, {"object": GRID_URL}),
            (ProbabilityAboveCutoff, {"cutoffs": [1.0]}),
            (MeanAboveCutoff, {"cutoffs": [1.0]}),
            (MaterialCategory, {"cutoff_grade": 0.0, "metal_price": 0.0, "label": "waste"}),
            (UpperTail, {"power": 0.5, "max": 10.0}),
            (LowerTail, {"power": 0.5, "min": 0.0}),
            (TailExtrapolation, {"upper": {"power": 0.5, "max": 10.0}}),
            (DistributionTarget, {"reference": DIST_URL}),
            (SimulationReportBlockDiscretization, {"nx": 1, "ny": 1, "nz": 1}),
            (SimulationReportDistribution, {"weights": "w"}),
            (SimReportContextItem, {"label": "k", "value": "v"}),
            (SimReportContext, {"title": "t"}),
            (SimReportThresholds, {"acceptable": 0.01, "marginal": 0.05}),
        ]
        for model_cls, valid_payload in supporting_cases:
            with self.subTest(model=model_cls.__name__):
                self.assertTrue(issubclass(model_cls, TaskInputModel))
                model_cls.model_validate(valid_payload)
                with self.assertRaises(ValidationError):
                    model_cls.model_validate({**valid_payload, "unknown_key": "boom"})

    def test_result_models_remain_tolerant_of_new_server_fields(self):
        target_payload = {
            "reference": GRID_URL,
            "name": "Grid",
            "schema_id": "/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
            "attribute": {"reference": "cell_attributes[0]", "name": "est", "server_extra": 1},
            "server_target_extra": True,
        }
        for model_cls in (
            KrigingResultModel,
            IDWResultModel,
            KNNResultModel,
            DeclusteringResultModel,
            BreakTiesResultModel,
            NormalScoreResultModel,
            LossCalculationResultModel,
            ProfitCalculationResultModel,
        ):
            with self.subTest(result_model=model_cls.__name__):
                parsed = model_cls.model_validate(
                    {"message": "ok", "target": target_payload, "future_server_field": "ignored"}
                )
                self.assertEqual(parsed.target.name, "Grid")

        ConSimResultModel.model_validate(
            {
                "target": {
                    "reference": GRID_URL,
                    "name": "Grid",
                    "schema_id": "/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                    "summary_attributes": {
                        k: {"reference": f"r_{k}", "name": k, "extra": 1} for k in ("mean", "variance", "min", "max")
                    },
                    "extra_target": 1,
                },
                "links": {"dashboard": "https://example.com", "extra_link": "x"},
                "extra_root": 1,
            }
        )
        ConditionalTurningBandsResultModel.model_validate(
            {
                "target": {
                    "reference": GRID_URL,
                    "name": "Grid",
                    "schema_id": "/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                    "extra": 1,
                },
                "extra": 2,
            }
        )
        ContinuousDistributionResultModel.model_validate(
            {
                "message": "ok",
                "distribution": {
                    "reference": DIST_URL,
                    "name": "d",
                    "schema_id": "/objects/continuous-distribution/1.0.0/continuous-distribution.schema.json",
                    "extra": 1,
                },
                "extra": 2,
            }
        )
        LocationWiseResultModel.model_validate(
            {
                "message": "ok",
                "target": {
                    "reference": GRID_URL,
                    "name": "Grid",
                    "schema_id": "/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                    "attributes": [{"reference": "r", "name": "n", "extra": 1}],
                    "extra": 2,
                },
                "extra": 3,
            }
        )
        SimulationReportResultModel.model_validate({"extra_server_field": "ok"})

    def test_alias_deduplication_and_conflict_rejection(self):
        search = _search(max_drillholes=6, min_drillholes=2)

        # SearchNeighborhood equal aliases deduplicate under extra="forbid"
        sn = SearchNeighborhood.model_validate(
            {
                "ellipsoid": {"ranges": [200, 150, 100]},
                "max_samples": 20,
                "max_drillholes": 6,
                "max_drillholes_per_estimate": 6,
                "min_drillholes": 2,
                "min_drillholes_per_estimate": 2,
            }
        )
        self.assertEqual(sn.max_drillholes, 6)
        self.assertEqual(sn.min_drillholes, 2)

        # KrigingParameters equal aliases (instances and equivalent dicts) deduplicate
        kp_same = KrigingParameters.model_validate(
            {
                "source": {"object": POINTSET_URL, "attribute": "grade"},
                "target": {"object": GRID_URL, "attribute": {"operation": "create", "name": "est"}},
                "variogram": VARIOGRAM_URL,
                "neighborhood": search,
                "search": search.model_dump(mode="json"),
                "kriging_method": SimpleKriging(mean=3.5),
                "method": {"type": "simple", "mean": 3.5},
            }
        )
        self.assertEqual(kp_same.neighborhood, search)
        self.assertEqual(kp_same.kriging_method, SimpleKriging(mean=3.5))

        # Conflicting KrigingParameters aliases raise ValueError
        for conflict in (
            {"neighborhood": search, "search": _search(max_samples=10)},
            {"kriging_method": OrdinaryKriging(), "method": SimpleKriging(mean=3.5)},
            {"kriging_method": SimpleKriging(mean=1.0), "method": SimpleKriging(mean=2.0)},
        ):
            with self.subTest(conflict=list(conflict)), self.assertRaisesRegex(ValidationError, "Conflicting values"):
                KrigingParameters.model_validate(
                    {
                        "source": {"object": POINTSET_URL, "attribute": "grade"},
                        "target": {"object": GRID_URL, "attribute": {"operation": "create", "name": "est"}},
                        "variogram": VARIOGRAM_URL,
                        "neighborhood": search,
                        **conflict,
                    }
                )

    def test_nan_and_inf_rejected_across_float_fields_and_spatial_types(self):
        for non_finite in (math.nan, math.inf, -math.inf):
            with self.subTest(non_finite=non_finite):
                with self.assertRaises(ValidationError):
                    IDWParameters(
                        source=Source(object=POINTSET_URL, attribute="grade"),
                        target=Target.new_attribute(GRID_URL, "idw"),
                        neighborhood=_search(),
                        power=non_finite,
                    )
                with self.assertRaises(ValidationError):
                    DeclusteringParameters(
                        source=POINTSET_URL,
                        grid=GRID_URL,
                        target=Target.new_attribute(POINTSET_URL, "w"),
                        neighborhood=_search(),
                        power=non_finite,
                    )
                with self.assertRaises(ValidationError):
                    SimpleKriging(mean=non_finite)
                with self.assertRaises(ValidationError):
                    FilterCondition(attribute="grade", operator="greater_than", threshold=non_finite)
                with self.assertRaises(ValidationError):
                    UpperTailParams(power=0.5, max=non_finite)
                with self.assertRaises(ValidationError):
                    LowerTail(power=0.5, min=non_finite)
                with self.assertRaises(ValidationError):
                    ProbabilityAboveCutoff(cutoffs=[non_finite])
                with self.assertRaises(ValidationError):
                    MeanAboveCutoff(cutoffs=[0.5, non_finite])
                with self.assertRaises(ValidationError):
                    MaterialCategory(cutoff_grade=non_finite)
                with self.assertRaises(ValidationError):
                    SimReportThresholds(acceptable=non_finite, marginal=1.0)
                with self.assertRaises(ValueError):
                    EllipsoidRanges(non_finite, 10.0, 10.0)
                with self.assertRaises(ValueError):
                    Rotation(non_finite, 0.0, 0.0)

        # Even a mutated EllipsoidRanges on an existing Ellipsoid is caught during SearchNeighborhood validation
        ell = Ellipsoid(ranges=EllipsoidRanges(100.0, 50.0, 25.0))
        ell.ranges.major = math.nan
        with self.assertRaises(ValidationError):
            SearchNeighborhood(ellipsoid=ell, max_samples=10)

        for bad_rot in ((-1.0, 0.0, 0.0), (361.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 181.0, 0.0), (0.0, 0.0, 361.0)):
            with self.subTest(bad_rot=bad_rot), self.assertRaises(ValidationError):
                SearchNeighborhood.model_validate(
                    {
                        "ellipsoid": {
                            "ranges": {"major": 100.0, "semi_major": 50.0, "minor": 25.0},
                            "rotation": {"dip_azimuth": bad_rot[0], "dip": bad_rot[1], "pitch": bad_rot[2]},
                        },
                        "max_samples": 10,
                    }
                )


# ===========================================================================
# 2. Lossless model_dump() -> model_validate() Round-Tripping
# ===========================================================================


class TestLosslessRoundTrip(unittest.TestCase):
    def test_every_task_round_trips_in_default_and_wire_modes(self):
        instances = _all_valid_task_parameter_instances()
        # Also include DeclusteringParameters with non-None power (IDW mode)
        instances.append(
            DeclusteringParameters(
                source=POINTSET_URL,
                grid=GRID_URL,
                target=Target.new_attribute(POINTSET_URL, "declus_idw"),
                neighborhood=_search(),
                power=2.5,
            )
        )
        for params in instances:
            cls = type(params)
            with self.subTest(model=cls.__name__, power=getattr(params, "power", "n/a")):
                dumped_full = params.model_dump(mode="json")
                reloaded_full = cls.model_validate(dumped_full)
                self.assertEqual(reloaded_full, params)
                self.assertEqual(reloaded_full.model_dump(mode="json"), dumped_full)

                dumped_wire = params.model_dump(mode="json", by_alias=True, exclude_none=True)
                reloaded_wire = cls.model_validate(dumped_wire)
                self.assertEqual(reloaded_wire, params)
                self.assertEqual(
                    reloaded_wire.model_dump(mode="json", by_alias=True, exclude_none=True),
                    dumped_wire,
                )

    def test_nested_and_top_level_filter_and_diagnostics_conflicts_rejected(self):
        f_a = Filter(where=FilterCondition(attribute="a", operator="in", values=["X"]))
        f_b = Filter(where=FilterCondition(attribute="b", operator="in", values=["Y"]))

        for model_cls, extra_kwargs in (
            (KrigingParameters, {"variogram": VARIOGRAM_URL}),
            (IDWParameters, {"power": 2.0}),
            (KNNParameters, {}),
        ):
            with self.subTest(model=model_cls.__name__):
                # Identical top-level and nested filters deduplicate
                accepted = model_cls.model_validate(
                    {
                        "source": {"object": POINTSET_URL, "attribute": "g", "filter": f_a.model_dump(mode="json")},
                        "source_filter": f_a,
                        "target": {
                            "object": GRID_URL,
                            "attribute": {"operation": "create", "name": "out"},
                            "filter": f_b.model_dump(mode="json"),
                        },
                        "target_filter": f_b,
                        "neighborhood": _search(),
                        **extra_kwargs,
                    }
                )
                self.assertEqual(accepted.source_filter, f_a)
                self.assertEqual(accepted.target_filter, f_b)

                # Conflicting source_filter vs source.filter raises
                with self.assertRaisesRegex(ValidationError, "Conflicting values for 'source_filter'"):
                    model_cls.model_validate(
                        {
                            "source": {"object": POINTSET_URL, "attribute": "g", "filter": f_a.model_dump(mode="json")},
                            "source_filter": f_b,
                            "target": {"object": GRID_URL, "attribute": {"operation": "create", "name": "out"}},
                            "neighborhood": _search(),
                            **extra_kwargs,
                        }
                    )

                # Conflicting target_filter vs target.filter raises
                with self.assertRaisesRegex(ValidationError, "Conflicting values for 'target_filter'"):
                    model_cls.model_validate(
                        {
                            "source": {"object": POINTSET_URL, "attribute": "g"},
                            "target": {
                                "object": GRID_URL,
                                "attribute": {"operation": "create", "name": "out"},
                                "filter": f_a.model_dump(mode="json"),
                            },
                            "target_filter": f_b,
                            "neighborhood": _search(),
                            **extra_kwargs,
                        }
                    )

        # Conflicting diagnostics vs target.diagnostics on KrigingParameters raises
        with self.assertRaisesRegex(ValidationError, "Conflicting values for 'diagnostics'"):
            KrigingParameters.model_validate(
                {
                    "source": {"object": POINTSET_URL, "attribute": "g"},
                    "target": {
                        "object": GRID_URL,
                        "attribute": {"operation": "create", "name": "out"},
                        "diagnostics": {"kriging_variance": {"operation": "create", "name": "KV"}},
                    },
                    "diagnostics": KrigingDiagnostics(kriging_variance="OTHER_KV"),
                    "variogram": VARIOGRAM_URL,
                    "neighborhood": _search(),
                }
            )


# ===========================================================================
# 3. Domain Parameter Invariants Across Tasks
# ===========================================================================


class TestDomainParameterInvariants(unittest.TestCase):
    def _consim(self, **kwargs) -> ConSimParameters:
        base = dict(
            source_object=POINTSET_URL,
            source_attribute="locations.attributes[?name=='grade']",
            target_object=GRID_URL,
            neighborhood=_search(),
            variogram_model=VARIOGRAM_URL,
            number_of_simulations=10,
            number_of_simulations_to_save=5,
        )
        base.update(kwargs)
        return ConSimParameters(**base)

    def test_consim_invariants(self):
        with self.assertRaisesRegex(ValidationError, "cannot exceed number_of_simulations"):
            self._consim(number_of_simulations=4, number_of_simulations_to_save=5)

        with self.assertRaisesRegex(ValidationError, "requires number_of_simulations_to_save >= 1"):
            self._consim(perform_validation=True, number_of_simulations_to_save=0)

        with self.assertRaisesRegex(ValidationError, "report_context requires perform_validation=True"):
            self._consim(
                perform_validation=False,
                report_context=ReportContext(title="T", details=[ValidationReportContextItem(label="a", value="b")]),
            )

        with self.assertRaisesRegex(ValidationError, "report_mean_thresholds requires perform_validation=True"):
            self._consim(
                perform_validation=False,
                report_mean_thresholds=ReportMeanThresholds(acceptable=0.01, marginal=0.05),
            )

        for bad_quantiles in ([], [-0.1], [1.1], [math.nan], [math.inf]):
            with self.subTest(quantiles=bad_quantiles), self.assertRaises(ValidationError):
                self._consim(location_wise_quantiles=bad_quantiles)

        for bad_cutoffs in ([], [math.nan], [math.inf], [-math.inf]):
            with self.subTest(cutoffs=bad_cutoffs):
                with self.assertRaises(ValidationError):
                    self._consim(probability_above_cutoff=bad_cutoffs)
                with self.assertRaises(ValidationError):
                    self._consim(mean_above_cutoff=bad_cutoffs)

        with self.assertRaisesRegex(ValidationError, "marginal threshold"):
            ReportMeanThresholds(acceptable=0.10, marginal=0.05)

        with self.assertRaisesRegex(ValidationError, "strictly less than upper.max"):
            TailExtrapolationParams(
                upper=UpperTailParams(power=0.5, max=10.0),
                lower=LowerTailParams(power=0.5, min=10.0),
            )

    def test_location_wise_invariants(self):
        for summary_val in (None, False):
            with self.subTest(summary=summary_val), self.assertRaisesRegex(ValidationError, "At least one"):
                LocationWiseParameters(
                    source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                    target=LocationWiseTarget(object=GRID_URL),
                    summary=summary_val,
                )

        for bad_quantiles in ([], [-0.01], [1.01], [math.nan]):
            with self.subTest(quantiles=bad_quantiles), self.assertRaises(ValidationError):
                LocationWiseParameters(
                    source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                    target=LocationWiseTarget(object=GRID_URL),
                    quantiles=bad_quantiles,
                )

    def test_loss_and_profit_calculation_invariants(self):
        for calc_cls in (LossCalculationParameters, ProfitCalculationParameters):
            with self.subTest(cls=calc_cls.__name__):
                # Empty categories rejected
                with self.assertRaises(ValidationError):
                    calc_cls(
                        source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                        target=Target.new_attribute(GRID_URL, "cat"),
                        material_categories=[],
                        processing_cost=10.0,
                        mining_waste_cost=2.0,
                        mining_ore_cost=4.0,
                        metal_recovery_fraction=0.8,
                    )

                # Non-increasing cutoff_grade rejected
                for cutoffs in ((0.5, 0.5), (1.0, 0.2)):
                    with self.assertRaisesRegex(ValidationError, "strictly increasing cutoff_grade"):
                        calc_cls(
                            source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                            target=Target.new_attribute(GRID_URL, "cat"),
                            material_categories=[
                                MaterialCategory(cutoff_grade=cutoffs[0], metal_price=0.0, label="c1"),
                                MaterialCategory(cutoff_grade=cutoffs[1], metal_price=100.0, label="c2"),
                            ],
                            processing_cost=10.0,
                            mining_waste_cost=2.0,
                            mining_ore_cost=4.0,
                            metal_recovery_fraction=0.8,
                        )

                # Duplicate category labels rejected
                with self.assertRaisesRegex(ValidationError, "Duplicate material category label"):
                    calc_cls(
                        source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                        target=Target.new_attribute(GRID_URL, "cat"),
                        material_categories=[
                            MaterialCategory(cutoff_grade=0.0, metal_price=0.0, label="ore"),
                            MaterialCategory(cutoff_grade=0.5, metal_price=100.0, label="ore"),
                        ],
                        processing_cost=10.0,
                        mining_waste_cost=2.0,
                        mining_ore_cost=4.0,
                        metal_recovery_fraction=0.8,
                    )

    def test_simulation_report_upper_bounds_and_thresholds(self):
        base = dict(
            simulation_source=POINTSET_URL,
            source_attribute="locations.attributes[0]",
            simulation_target=GRID_URL,
            point_simulations="cell_attributes[0]",
            point_simulations_normal_score="cell_attributes[1]",
            block_simulations="cell_attributes[2]",
            block_simulations_normal_score="cell_attributes[3]",
            variogram_model=VARIOGRAM_URL,
            neighborhood=_search(),
            number_of_simulations=10,
        )
        with self.assertRaises(ValidationError):
            SimulationReportParameters(**{**base, "number_of_simulations": 101})
        with self.assertRaises(ValidationError):
            SimulationReportParameters(**{**base, "number_of_lines": 1001})
        with self.assertRaisesRegex(ValidationError, "marginal threshold"):
            SimReportThresholds(acceptable=0.2, marginal=0.1)


# ===========================================================================
# 4 & 5. Typed Object Roles, Drillhole Checks, ExistingAttributeExpression & Filter.resolve()
# ===========================================================================


class TestTypedObjectRolesAndFilterBinding(unittest.TestCase):
    def setUp(self):
        self.pointset = _make_pointset()
        self.downhole = _make_downhole_intervals()
        self.grid = _make_regular_grid()
        self.masked_grid = _make_masked_grid()
        self.block_model = _make_block_model()
        self.variogram = _make_variogram()

    def test_drillhole_limits_require_downhole_intervals_source(self):
        dh_search = _search(max_samples_per_drillhole=3, max_drillholes=5)

        # DownholeIntervals accepted across all 4 drillhole-capable tasks
        KrigingParameters(
            source=self.downhole.attributes["grade"],
            target=Target.new_attribute(self.grid, "est"),
            variogram=self.variogram,
            neighborhood=_search(
                max_samples_per_drillhole=3,
                max_drillholes=5,
                min_drillholes=2,
                drillhole_limit_scope="octant",
            ),
            diagnostics=KrigingDiagnostics(num_drillholes=True),
        )
        IDWParameters(
            source=self.downhole.attributes["grade"],
            target=Target.new_attribute(self.grid, "est"),
            neighborhood=dh_search,
            power=2.0,
        )
        KNNParameters(
            source=self.downhole.attributes["grade"],
            target=Target.new_attribute(self.grid, "est"),
            neighborhood=dh_search,
        )
        DeclusteringParameters(
            source=self.downhole,
            grid=self.grid,
            target=Target.new_attribute(self.downhole, "wt"),
            neighborhood=dh_search,
        )

        # PointSet rejected when drillhole limits are active
        for factory in (
            lambda: KrigingParameters(
                source=self.pointset.attributes["grade"],
                target=Target.new_attribute(self.grid, "est"),
                variogram=self.variogram,
                neighborhood=dh_search,
            ),
            lambda: KrigingParameters(
                source=Source(object=self.pointset, attribute="locations.attributes[0]"),
                target=Target.new_attribute(self.grid, "est"),
                variogram=self.variogram,
                neighborhood=_search(min_drillholes=2),
            ),
            lambda: KrigingParameters(
                source=self.pointset.attributes["grade"],
                target=Target.new_attribute(self.grid, "est"),
                variogram=self.variogram,
                neighborhood=_search(),
                diagnostics=KrigingDiagnostics(num_drillholes=True),
            ),
            lambda: IDWParameters(
                source=self.pointset.attributes["grade"],
                target=Target.new_attribute(self.grid, "est"),
                neighborhood=dh_search,
                power=2.0,
            ),
            lambda: KNNParameters(
                source=self.pointset.attributes["grade"],
                target=Target.new_attribute(self.grid, "est"),
                neighborhood=dh_search,
            ),
            lambda: DeclusteringParameters(
                source=self.pointset,
                grid=self.grid,
                target=Target.new_attribute(self.pointset, "wt"),
                neighborhood=dh_search,
            ),
        ):
            with self.assertRaisesRegex(ValidationError, "DownholeIntervals.*PointSet"):
                factory()

    def test_consim_and_turning_bands_typed_object_roles(self):
        # Valid with PointSet + Regular3DGrid or RegularMasked3DGrid + Variogram
        for target_grid in (self.grid, self.masked_grid):
            ConSimParameters(
                source_object=self.pointset,
                source_attribute=self.pointset.attributes["grade"],
                target_object=target_grid,
                variogram_model=self.variogram,
                neighborhood=_search(),
            )
            ConditionalTurningBandsParameters(
                source=self.pointset,
                source_attribute=self.pointset.attributes["grade"],
                target=target_grid,
                distribution=DIST_URL,
                variogram_model=self.variogram,
                neighborhood=_search(),
            )

        # Wrong source role (DownholeIntervals instead of PointSet)
        with self.assertRaisesRegex(
            ValidationError, "Expected 'source_object' to be a PointSet, got DownholeIntervals"
        ):
            ConSimParameters(
                source_object=self.downhole,
                source_attribute="attributes[0]",
                target_object=self.grid,
                variogram_model=self.variogram,
                neighborhood=_search(),
            )

        # Wrong target role (BlockModel instead of Regular3DGrid | RegularMasked3DGrid)
        with self.assertRaisesRegex(
            ValidationError, "Expected 'target_object' to be a Regular3DGrid or RegularMasked3DGrid, got BlockModel"
        ):
            ConSimParameters(
                source_object=self.pointset,
                source_attribute=self.pointset.attributes["grade"],
                target_object=self.block_model,
                variogram_model=self.variogram,
                neighborhood=_search(),
            )

        # Wrong variogram role
        with self.assertRaisesRegex(ValidationError, "Expected 'variogram_model' to be a Variogram, got PointSet"):
            ConSimParameters(
                source_object=self.pointset,
                source_attribute=self.pointset.attributes["grade"],
                target_object=self.grid,
                variogram_model=self.pointset,
                neighborhood=_search(),
            )

        # Wrong distribution role on ConditionalTurningBandsParameters and NormalScoreParameters
        with self.assertRaisesRegex(ValidationError, "continuous distribution object, got PointSet"):
            ConditionalTurningBandsParameters(
                source=self.pointset,
                source_attribute=self.pointset.attributes["grade"],
                target=self.grid,
                distribution=self.pointset,
                variogram_model=self.variogram,
                neighborhood=_search(),
            )
        with self.assertRaisesRegex(ValidationError, "continuous distribution object, got Variogram"):
            NormalScoreParameters(
                method="forward",
                source=self.pointset.attributes["grade"],
                target=Target.new_attribute(self.pointset, "ns"),
                distribution=self.variogram,
            )

    def test_existing_attribute_expression_rejects_pending_attributes_with_available_names(self):
        from pydantic import TypeAdapter

        ta = TypeAdapter(ExistingAttributeExpression)
        # Existing attribute passes
        self.assertEqual(
            ta.validate_python(self.pointset.attributes["grade"]), "locations.attributes[?key=='key-grade']"
        )

        # Pending attribute on PointSet lists available attributes
        with self.assertRaises(ValidationError) as ctx:
            ta.validate_python(self.pointset.attributes["non_existent"])
        msg = str(ctx.exception)
        self.assertIn("non_existent", msg)
        self.assertIn("grade", msg)
        self.assertIn("domain", msg)

        # Pending attribute on BlockModel lists available attributes
        with self.assertRaises(ValidationError) as ctx_bm:
            FilterCondition(attribute=self.block_model.attributes["missing_col"], operator="in", values=["A"])
        msg_bm = str(ctx_bm.exception)
        self.assertIn("missing_col", msg_bm)
        self.assertIn("Cu_OK", msg_bm)
        self.assertIn("Zone", msg_bm)

        # Weights and simulation_report fields also reject PendingAttribute
        with self.assertRaises(ValidationError):
            DistributionParams(weights=self.pointset.attributes["missing_wt"])
        with self.assertRaises(ValidationError):
            SimulationReportDistribution(weights=self.pointset.attributes["missing_wt"])
        with self.assertRaises(ValidationError):
            SimulationReportParameters(
                simulation_source=self.pointset,
                source_attribute=self.pointset.attributes["grade"],
                simulation_target=self.grid,
                point_simulations=self.grid.attributes["missing_sim"],
                point_simulations_normal_score="cell_attributes[1]",
                block_simulations="cell_attributes[2]",
                block_simulations_normal_score="cell_attributes[3]",
                variogram_model=self.variogram,
                neighborhood=_search(),
                number_of_simulations=5,
            )

    def test_filter_resolve_and_automatic_binding_on_tasks(self):
        compound_source_filter = Filter(
            where=AllOfFilter(
                filters=[
                    FilterCondition(attribute="domain", operator="in", values=["D1"]),
                    AnyOfFilter(
                        filters=[
                            FilterCondition(attribute="grade", operator="greater_than", threshold=0.2),
                            FilterCondition(
                                attribute="locations.attributes[?key=='already-jmes']",
                                operator="less_than",
                                threshold=10.0,
                            ),
                        ]
                    ),
                ]
            )
        )
        target_grid_filter = Filter(where=FilterCondition(attribute="domain", operator="in", values=["A"]))
        target_bm_filter = Filter(where=FilterCondition(attribute="Zone", operator="in", values=["Ore"]))

        # Standalone Filter.resolve()
        resolved = compound_source_filter.resolve(self.pointset)
        self.assertEqual(
            resolved.where.filters[0].attribute,
            "locations.attributes[?key=='key-domain']",
        )
        self.assertEqual(
            resolved.where.filters[1].filters[0].attribute,
            "locations.attributes[?key=='key-grade']",
        )
        self.assertEqual(
            resolved.where.filters[1].filters[1].attribute,
            "locations.attributes[?key=='already-jmes']",
        )

        # Automatic binding on KrigingParameters, IDWParameters, KNNParameters, ConSimParameters, ConditionalTurningBandsParameters
        krig = KrigingParameters(
            source=self.pointset.attributes["grade"],
            source_filter=compound_source_filter,
            target=Target.new_attribute(self.block_model, "Cu_est"),
            target_filter=target_bm_filter,
            variogram=self.variogram,
            neighborhood=_search(),
        )
        self.assertEqual(krig.source_filter.where.filters[0].attribute, "locations.attributes[?key=='key-domain']")
        self.assertEqual(krig.target_filter.where.attribute, "attributes[?name=='Zone']")

        idw_p = IDWParameters(
            source=Source(object=self.pointset, attribute="locations.attributes[?key=='key-grade']"),
            source_filter=Filter(where=FilterCondition(attribute="domain", operator="in", values=["D1"])),
            target=Target.new_attribute(self.grid, "idw_est"),
            target_filter=target_grid_filter,
            neighborhood=_search(),
            power=2.0,
        )
        self.assertEqual(idw_p.source_filter.where.attribute, "locations.attributes[?key=='key-domain']")
        self.assertEqual(idw_p.target_filter.where.attribute, "cell_attributes[?key=='grid-key-domain']")

        consim_p = ConSimParameters(
            source_object=self.pointset,
            source_attribute=self.pointset.attributes["grade"],
            source_filter=Filter(where=FilterCondition(attribute="domain", operator="in", values=["D1"])),
            target_object=self.grid,
            filter=target_grid_filter,
            variogram_model=self.variogram,
            neighborhood=_search(),
        )
        self.assertEqual(consim_p.source_filter.where.attribute, "locations.attributes[?key=='key-domain']")
        self.assertEqual(consim_p.filter.where.attribute, "cell_attributes[?key=='grid-key-domain']")

        ctb_p = ConditionalTurningBandsParameters(
            source=self.pointset,
            source_attribute=self.pointset.attributes["grade"],
            source_filter=Filter(where=FilterCondition(attribute="domain", operator="in", values=["D1"])),
            target=self.grid,
            filter=target_grid_filter,
            distribution=DIST_URL,
            variogram_model=self.variogram,
            neighborhood=_search(),
        )
        self.assertEqual(ctb_p.source_filter.where.attribute, "locations.attributes[?key=='key-domain']")
        self.assertEqual(ctb_p.filter.where.attribute, "cell_attributes[?key=='grid-key-domain']")

        # Unknown bare attribute name in Filter raises ValueError listing available attributes
        bad_filter = Filter(where=FilterCondition(attribute="typo_domain", operator="in", values=["A"]))
        with self.assertRaises(ValidationError) as ctx:
            KNNParameters(
                source=self.pointset.attributes["grade"],
                target=Target.new_attribute(self.grid, "knn_est"),
                target_filter=bad_filter,
                neighborhood=_search(),
            )
        err = str(ctx.exception)
        self.assertIn("typo_domain", err)
        self.assertIn("kriged_grade", err)
        self.assertIn("domain", err)

    def test_mapping_form_wrappers_enforce_object_and_drillhole_roles(self):
        # Mapping-form Source, Target, DeclusteringSource/Grid, LocationWiseTarget, and weights reject Variogram
        for factory in (
            lambda: KrigingParameters(
                source={"object": self.variogram, "attribute": "attributes[0]"},
                target=Target.new_attribute(self.grid, "est"),
                variogram=self.variogram,
                neighborhood=_search(),
            ),
            lambda: KrigingParameters(
                source=self.pointset.attributes["grade"],
                target={"object": self.variogram, "attribute": {"operation": "create", "name": "est"}},
                variogram=self.variogram,
                neighborhood=_search(),
            ),
            lambda: IDWParameters(
                source={"object": self.variogram, "attribute": "attributes[0]"},
                target=Target.new_attribute(self.grid, "est"),
                neighborhood=_search(),
                power=2.0,
            ),
            lambda: KNNParameters(
                source={"object": self.variogram, "attribute": "attributes[0]"},
                target=Target.new_attribute(self.grid, "est"),
                neighborhood=_search(),
            ),
            lambda: DeclusteringParameters(
                source={"object": self.variogram},
                grid=self.grid,
                target=Target.new_attribute(self.pointset, "wt"),
                neighborhood=_search(),
            ),
            lambda: DeclusteringParameters(
                source=self.pointset,
                grid={"object": self.variogram},
                target=Target.new_attribute(self.pointset, "wt"),
                neighborhood=_search(),
            ),
            lambda: LocationWiseParameters(
                source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                target={"object": self.variogram},
                summary=True,
            ),
            lambda: ContinuousDistributionParameters(
                source=self.pointset.attributes["grade"],
                weights={"object": self.variogram, "attribute": "attributes[0]"},
                target=DistributionTarget(reference=DIST_URL),
            ),
        ):
            with self.assertRaisesRegex(ValidationError, "BaseSpatialObject.*Variogram"):
                factory()

        # Mapping-form PointSet source also enforces DownholeIntervals requirement when drillhole limits are active
        with self.assertRaisesRegex(ValidationError, "DownholeIntervals.*PointSet"):
            KrigingParameters(
                source={"object": self.pointset, "attribute": "locations.attributes[0]"},
                target=Target.new_attribute(self.grid, "est"),
                variogram=self.variogram,
                neighborhood=_search(max_drillholes=4),
            )

    def test_source_rejects_mismatched_attribute_parent_and_supports_path_uuid_equivalence(self):
        pointset_b = _make_pointset(suffix=19, name="OtherPoints")

        # Direct Source(object=A, attribute=B.attributes["grade"]) rejected
        with self.assertRaisesRegex(ValidationError, "belongs to a different object"):
            Source(object=self.pointset, attribute=pointset_b.attributes["grade"])

        # Mapping-form source={"object": A, "attribute": B.attributes["grade"]} rejected
        with self.assertRaisesRegex(ValidationError, "belongs to a different object"):
            KrigingParameters(
                source={"object": self.pointset, "attribute": pointset_b.attributes["grade"]},
                target=Target.new_attribute(self.grid, "est"),
                variogram=self.variogram,
                neighborhood=_search(),
            )

        # URL-only object=A_URL with B.attributes["grade"] rejected (never leaks B into source_filter resolution)
        with self.assertRaisesRegex(ValidationError, "belongs to a different object"):
            KrigingParameters(
                source={"object": POINTSET_URL, "attribute": pointset_b.attributes["grade"]},
                source_filter=Filter(where=FilterCondition(attribute="domain", operator="in", values=["D1"])),
                target=Target.new_attribute(self.grid, "est"),
                variogram=self.variogram,
                neighborhood=_search(),
            )

        # Equivalent path-based ObjectReference URL for self.pointset matches its UUID-backed typed attributes
        matching_path_url = str(ObjectReference.new(_ENV, object_path="geology/AssayPoints.json"))
        encoded_path_url = matching_path_url.replace("AssayPoints.json", "Assay%50oints.json")
        other_path_url = str(ObjectReference.new(_ENV, object_path="geology/OtherPoints.json"))

        src_equiv = Source(object=matching_path_url, attribute=self.pointset.attributes["grade"])
        self.assertEqual(src_equiv.object, matching_path_url)

        krig_equiv = KrigingParameters(
            source={"object": encoded_path_url, "attribute": self.pointset.attributes["grade"]},
            source_filter=Filter(where=FilterCondition(attribute="domain", operator="in", values=["D1"])),
            target=Target.new_attribute(self.grid, "est"),
            variogram=self.variogram,
            neighborhood=_search(),
        )
        self.assertEqual(krig_equiv.source_filter.where.attribute, "locations.attributes[?key=='key-domain']")

        consim_equiv = ConSimParameters(
            source_object=matching_path_url,
            source_attribute=self.pointset.attributes["grade"],
            source_filter=Filter(where=FilterCondition(attribute="domain", operator="in", values=["D1"])),
            distribution=DistributionParams(weights=self.pointset.attributes["weights"]),
            target_object=self.grid,
            variogram_model=self.variogram,
            neighborhood=_search(),
        )
        self.assertEqual(consim_equiv.source_filter.where.attribute, "locations.attributes[?key=='key-domain']")

        # Non-matching path reference is rejected
        with self.assertRaisesRegex(ValidationError, "belongs to a different object"):
            Source(object=other_path_url, attribute=self.pointset.attributes["grade"])
        with self.assertRaisesRegex(ValidationError, "belongs to a different object"):
            ConSimParameters(
                source_object=other_path_url,
                source_attribute=self.pointset.attributes["grade"],
                target_object=self.grid,
                variogram_model=self.variogram,
                neighborhood=_search(),
            )


# ===========================================================================
# 6. Batch Output-Collision Detection in run_tasks() & Future Task Inheritance
# ===========================================================================


class TestBatchOutputCollisions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.grid = _make_regular_grid(suffix=20)
        self.grid_b = _make_regular_grid(suffix=21)
        self.pointset = _make_pointset(suffix=10)
        self.variogram = _make_variogram(suffix=30)
        self.ctx = MagicMock()
        self.ctx.get_connector.return_value = MagicMock()
        self.ctx.get_org_id.return_value = _ORG

    def test_jmespath_selector_canonicalization_variants(self):
        key_variants = [
            "cell_attributes[?key=='grid-key-kriged']",
            "cell_attributes[? 'grid-key-kriged' == key ]",
            'cell_attributes[? @."key" == `"grid-key-kriged"` ]',
            'locations.attributes[?key==`"grid-key-kriged"`]',
            "attributes[?\"key\"=='grid-key-kriged']",
            "attributes[?key=='grid-key-kriged']",
        ]
        canonical_keys = {canonicalize_attribute_selector(v) for v in key_variants}
        self.assertEqual(canonical_keys, {"primary:key:grid-key-kriged"})

        name_variants = [
            "cell_attributes[?name=='kriged_grade']",
            "attributes[?\"name\"=='kriged_grade']",
            'attributes[? `"kriged_grade"` == @."name" ]',
            "locations.attributes[?@.name == 'kriged_grade']",
        ]
        self.assertEqual({canonicalize_attribute_selector(v) for v in name_variants}, {"primary:name:kriged_grade"})

        # Double-quoted right-hand side without backticks compares two fields (name == grade), NOT literal 'grade'
        field_cmp = canonicalize_attribute_selector('attributes[?name=="kriged_grade"]')
        self.assertNotEqual(field_cmp, "primary:name:kriged_grade")
        self.assertEqual(field_cmp, canonicalize_attribute_selector('cell_attributes[?"kriged_grade" == @.name]'))

        self.assertEqual(
            canonicalize_attribute_selector("cell_attributes[ 0 ]"),
            canonicalize_attribute_selector('@."locations"."attributes"[0]'),
        )

    async def test_single_attribute_collisions_blocked_before_submit(self):
        src = Source(object=POINTSET_URL, attribute="locations.attributes[0]")
        collision_batches = [
            # Same attribute name across Kriging + IDW with versioned vs unversioned target URL
            [
                KrigingParameters(
                    source=src,
                    target=Target.new_attribute(_obj_url(20, version="1"), "cu_est"),
                    variogram=VARIOGRAM_URL,
                    neighborhood=_search(),
                ),
                IDWParameters(
                    source=src,
                    target=Target.new_attribute(_obj_url(20, version="2"), "cu_est"),
                    neighborhood=_search(),
                    power=2.0,
                ),
            ],
            # Kriging diagnostic "KV" colliding with KNN target "KV"
            [
                KrigingParameters(
                    source=src,
                    target=Target.new_attribute(GRID_URL, "cu_ok"),
                    variogram=VARIOGRAM_URL,
                    neighborhood=_search(),
                    diagnostics=KrigingDiagnostics(kriging_variance=True),
                ),
                KNNParameters(
                    source=src,
                    target=Target.new_attribute(GRID_URL, "KV"),
                    neighborhood=_search(),
                ),
            ],
            # Equivalent JMESPath UpdateAttribute selectors (quoted identifier vs unquoted, without typed target)
            [
                IDWParameters(
                    source=src,
                    target=Target(
                        object=GRID_URL,
                        attribute=UpdateAttribute(reference="attributes[?name=='grade']"),
                    ),
                    neighborhood=_search(),
                    power=2.0,
                ),
                KNNParameters(
                    source=src,
                    target=Target(
                        object=GRID_URL,
                        attribute=UpdateAttribute(reference="attributes[?\"name\"=='grade']"),
                    ),
                    neighborhood=_search(),
                ),
            ],
            # Equivalent JMESPath UpdateAttribute key selectors (backtick JSON literal vs single-quoted literal)
            [
                IDWParameters(
                    source=src,
                    target=Target(
                        object=GRID_URL,
                        attribute=UpdateAttribute(reference="cell_attributes[?key=='abc']"),
                    ),
                    neighborhood=_search(),
                    power=2.0,
                ),
                KNNParameters(
                    source=src,
                    target=Target(
                        object=GRID_URL,
                        attribute=UpdateAttribute(reference='cell_attributes[? `"abc"` == @."key" ]'),
                    ),
                    neighborhood=_search(),
                ),
            ],
            # CreateAttribute(name="kriged_grade") vs UpdateAttribute by index [0] resolved via typed grid
            [
                IDWParameters(
                    source=self.pointset.attributes["grade"],
                    target=Target.new_attribute(self.grid, "kriged_grade"),
                    neighborhood=_search(),
                    power=2.0,
                ),
                KNNParameters(
                    source=src,
                    target=Target(
                        object=GRID_URL,
                        attribute=UpdateAttribute(reference="cell_attributes[0]"),
                    ),
                    neighborhood=_search(),
                ),
            ],
            # Typed existing Attribute (key=='grid-key-kriged') vs CreateAttribute(name='kriged_grade') on plain URL
            [
                BreakTiesParameters(
                    source=src,
                    target=self.grid.attributes["kriged_grade"],
                    neighborhood=_search(),
                ),
                NormalScoreParameters(
                    method="forward",
                    source=src,
                    target=Target.new_attribute(GRID_URL, "kriged_grade"),
                    distribution=DIST_URL,
                ),
            ],
            # LossCalculation vs ProfitCalculation writing to same target attribute
            [
                LossCalculationParameters(
                    source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                    target=Target.new_attribute(GRID_URL, "econ_cat"),
                    material_categories=[MaterialCategory(cutoff_grade=0.0, metal_price=10.0, label="ore")],
                    processing_cost=1.0,
                    mining_waste_cost=1.0,
                    mining_ore_cost=1.0,
                    metal_recovery_fraction=0.9,
                ),
                ProfitCalculationParameters(
                    source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
                    target=Target(
                        object=GRID_URL,
                        attribute=UpdateAttribute(reference="cell_attributes[?name=='econ_cat']"),
                    ),
                    material_categories=[MaterialCategory(cutoff_grade=0.0, metal_price=10.0, label="ore")],
                    processing_cost=1.0,
                    mining_waste_cost=1.0,
                    mining_ore_cost=1.0,
                    metal_recovery_fraction=0.9,
                ),
            ],
        ]

        with patch("evo.compute.tasks.common.runner.JobClient.submit", new_callable=AsyncMock) as mock_submit:
            for idx, batch in enumerate(collision_batches):
                with self.subTest(batch=idx), self.assertRaisesRegex(ValueError, "Output collision in batch"):
                    await run(self.ctx, batch, preview=True)
            mock_submit.assert_not_called()

        # Local KrigingParameters check also catches quoted-identifier equivalence and avoids false field-cmp collisions
        with self.assertRaisesRegex(ValidationError, "both write to the attribute"):
            KrigingParameters(
                source=src,
                target=Target(object=GRID_URL, attribute=UpdateAttribute(reference="attributes[?name=='KV']")),
                variogram=VARIOGRAM_URL,
                neighborhood=_search(),
                diagnostics=KrigingDiagnostics(
                    kriging_variance=UpdateAttribute(reference="attributes[?\"name\"=='KV']"),
                ),
            )
        with self.assertRaisesRegex(ValidationError, "both write to the attribute 'KV'"):
            KrigingParameters(
                source=src,
                target=Target.new_attribute(GRID_URL, "KV"),
                variogram=VARIOGRAM_URL,
                neighborhood=_search(),
                diagnostics=KrigingDiagnostics(
                    kriging_variance=UpdateAttribute(reference="attributes[?\"name\"=='KV']"),
                ),
            )

        # Field-to-field JMESPath comparison `attributes[?name=="grade"]` is NOT a literal 'grade' selector
        KrigingParameters(
            source=src,
            target=Target.new_attribute(GRID_URL, "grade"),
            variogram=VARIOGRAM_URL,
            neighborhood=_search(),
            diagnostics=KrigingDiagnostics(
                kriging_variance=UpdateAttribute(reference='attributes[?name=="grade"]'),
            ),
        )
        fake_idw_job = AsyncMock()
        fake_idw_job.wait_for_results.return_value = IDWResultModel(
            message="ok",
            target=TaskTarget(
                reference=GRID_URL,
                name="Grid",
                schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                attribute=TaskAttribute(reference="attributes[0]", name="grade"),
            ),
        )
        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=fake_idw_job,
        ) as mock_submit:
            non_colliding_results = await run_tasks(
                self.ctx,
                [
                    IDWParameters(
                        source=src,
                        target=Target(
                            object=GRID_URL,
                            attribute=UpdateAttribute(reference="attributes[?name=='grade']"),
                        ),
                        neighborhood=_search(),
                        power=2.0,
                    ),
                    IDWParameters(
                        source=src,
                        target=Target(
                            object=GRID_URL,
                            attribute=UpdateAttribute(reference='attributes[?name=="grade"]'),
                        ),
                        neighborhood=_search(),
                        power=2.0,
                    ),
                ],
            )
        self.assertEqual(len(non_colliding_results), 2)
        self.assertEqual(mock_submit.call_count, 2)

    async def test_whole_grid_and_distribution_collisions_blocked_and_distinct_targets_allowed(self):
        consim_a1 = ConSimParameters(
            source_object=POINTSET_URL,
            source_attribute="locations.attributes[0]",
            target_object=_obj_url(20, version="1"),
            variogram_model=VARIOGRAM_URL,
            neighborhood=_search(),
        )
        consim_a2 = ConSimParameters(
            source_object=POINTSET_URL,
            source_attribute="locations.attributes[0]",
            target_object=_obj_url(20, version="2"),
            variogram_model=VARIOGRAM_URL,
            neighborhood=_search(),
        )
        lw_a = LocationWiseParameters(
            source=Source(object=GRID_URL, attribute="cell_attributes[0]"),
            target=LocationWiseTarget(object=GRID_URL),
            summary=True,
        )
        ctb_a = ConditionalTurningBandsParameters(
            source=POINTSET_URL,
            source_attribute="locations.attributes[0]",
            target=GRID_URL,
            distribution=DIST_URL,
            variogram_model=VARIOGRAM_URL,
            neighborhood=_search(),
        )
        dist_a1 = ContinuousDistributionParameters(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[0]"),
            target=DistributionTarget(reference=DIST_URL),
        )
        dist_a2 = ContinuousDistributionParameters(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[1]"),
            target=DistributionTarget(reference=DIST_URL, overwrite=True),
        )

        with patch("evo.compute.tasks.common.runner.JobClient.submit", new_callable=AsyncMock) as mock_submit:
            for pair in ((consim_a1, consim_a2), (consim_a1, lw_a), (consim_a1, ctb_a), (dist_a1, dist_a2)):
                with self.assertRaisesRegex(ValueError, "Output collision in batch"):
                    await run_tasks(self.ctx, list(pair))
            mock_submit.assert_not_called()

        # SimulationReport writes no target geoscience object, so two reports on the same grid do not collide
        rep = SimulationReportParameters(
            simulation_source=POINTSET_URL,
            source_attribute="locations.attributes[0]",
            simulation_target=GRID_URL,
            point_simulations="cell_attributes[0]",
            point_simulations_normal_score="cell_attributes[1]",
            block_simulations="cell_attributes[2]",
            block_simulations_normal_score="cell_attributes[3]",
            variogram_model=VARIOGRAM_URL,
            neighborhood=_search(),
            number_of_simulations=5,
        )
        fake_job = AsyncMock()
        fake_job.wait_for_results.return_value = SimulationReportResultModel()
        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=fake_job,
        ) as mock_submit:
            results = await run_tasks(self.ctx, [rep, rep])
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_submit.call_count, 2)

    async def test_future_task_subclass_inherits_all_common_guarantees(self):
        registry = TaskRegistry()

        class FutureTaskParameters(TaskParameters):
            _filter_bindings = (("source_filter", "source"),)
            source: Source
            target: Target
            smoothing: float
            source_filter: Filter | None = None

        class FutureResultModel(BaseModel):
            message: str
            target: TaskTarget

        class FutureResult(SingleTargetTaskResult[TaskTarget]):
            TASK_DISPLAY_NAME = "Future Task"

        class FutureRunner(
            TaskRunner[FutureTaskParameters, FutureResultModel, FutureResult],
            topic="geostatistics",
            task="future-task",
        ):
            async def _get_result(self, raw_result: FutureResultModel) -> FutureResult:
                return FutureResult(self._context, raw_result)

        try:
            # 1. Extra fields & NaN rejected automatically
            with self.assertRaises(ValidationError):
                FutureTaskParameters(
                    source=Source(object=POINTSET_URL, attribute="g"),
                    target=Target.new_attribute(GRID_URL, "out"),
                    smoothing=math.nan,
                )
            with self.assertRaises(ValidationError):
                FutureTaskParameters(
                    source=Source(object=POINTSET_URL, attribute="g"),
                    target=Target.new_attribute(GRID_URL, "out"),
                    smoothing=1.0,
                    extra_bad=1,  # type: ignore[call-arg]
                )

            # 2. Automatic Filter bare-attribute binding
            p1 = FutureTaskParameters(
                source=Source(object=self.pointset, attribute="locations.attributes[0]"),
                target=Target.new_attribute(self.grid, "out_1"),
                smoothing=0.5,
                source_filter=Filter(where=FilterCondition(attribute="domain", operator="in", values=["D1"])),
            )
            self.assertEqual(p1.source_filter.where.attribute, "locations.attributes[?key=='key-domain']")

            # 3. Automatic batch output-collision detection in run_tasks()
            p2 = FutureTaskParameters(
                source=Source(object=POINTSET_URL, attribute="locations.attributes[0]"),
                target=Target.new_attribute(GRID_URL, "out_1"),
                smoothing=0.8,
            )
            with self.assertRaisesRegex(ValueError, "Output collision in batch"):
                await run_tasks(self.ctx, [p1, p2])
        finally:
            registry._runners.pop(FutureTaskParameters, None)


# ===========================================================================
# 7. Public Result Properties Across All 13 Task Result Classes
# ===========================================================================


class TestAllTaskResultPublicProperties(unittest.TestCase):
    def test_consim_result_public_properties(self):
        attr = TaskAttribute(reference="cell_attributes[0]", name="mean_attr")
        model = ConSimResultModel(
            target=ConSimTargetResult(
                reference=GRID_URL,
                name="SimGrid",
                description="Grid desc",
                schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                summary_attributes=ConSimSummaryAttributes(mean=attr, variance=attr, min=attr, max=attr),
                quantile_attributes=[ConSimQuantileAttribute(reference="q", name="p50", quantile=0.5)],
                simulations=TaskAttribute(reference="s", name="block_sims"),
                simulations_normal_score=TaskAttribute(reference="sns", name="block_sims_ns"),
                point_simulations=TaskAttribute(reference="ps", name="pt_sims"),
                point_simulations_normal_score=TaskAttribute(reference="psns", name="pt_sims_ns"),
                loss_calculation_attribute=TaskAttribute(reference="lc", name="loss_cat"),
                probability_above_cutoff_attributes=[ConSimCutoffAttribute(reference="pc", name="prob", cutoff=0.5)],
                mean_above_cutoff_attributes=[ConSimCutoffAttribute(reference="mc", name="mean_c", cutoff=0.5)],
            ),
            validation_summary=ConSimValidationSummary(reference_mean=1.2, mean=1.25),
            validation_report=ConSimValidationReport(reference="https://example.com/report.html", name="QC Report"),
            links=ConSimLinks(dashboard="https://example.com/dashboard"),
        )
        res = ConSimResult(MagicMock(), model)
        self.assertIs(res.target, model.target)
        self.assertEqual(res.target_name, "SimGrid")
        self.assertEqual(res.target_reference, GRID_URL)
        self.assertEqual(res.target_description, "Grid desc")
        self.assertEqual(res.simulations_attribute.name, "block_sims")
        self.assertEqual(res.simulations_normal_score_attribute.name, "block_sims_ns")
        self.assertEqual(res.point_simulations_attribute.name, "pt_sims")
        self.assertEqual(res.point_simulations_normal_score_attribute.name, "pt_sims_ns")
        self.assertIs(res.validation_report, model.validation_report)
        self.assertEqual(res.validation_report.name, "QC Report")
        self.assertEqual(res.report_reference, "https://example.com/report.html")
        self.assertIs(res.links, model.links)
        self.assertEqual(res.dashboard_url, "https://example.com/dashboard")

    def test_public_properties_on_remaining_result_classes(self):
        target = TaskTarget(
            reference=GRID_URL,
            name="Grid",
            description="Target desc",
            schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
            attribute=TaskAttribute(reference="cell_attributes[0]", name="est"),
        )
        for res_cls, model in (
            (KrigingResult, KrigingResultModel(message="ok", target=KrigingTargetResult(**target.model_dump()))),
            (IDWResult, IDWResultModel(message="ok", target=target)),
            (KNNResult, KNNResultModel(message="ok", target=target)),
            (DeclusteringResult, DeclusteringResultModel(message="ok", target=target)),
            (BreakTiesResult, BreakTiesResultModel(message="ok", target=target)),
            (NormalScoreResult, NormalScoreResultModel(message="ok", target=target)),
            (LossCalculationResult, LossCalculationResultModel(message="ok", target=target)),
            (ProfitCalculationResult, ProfitCalculationResultModel(message="ok", target=target)),
        ):
            with self.subTest(result_cls=res_cls.__name__):
                r = res_cls(MagicMock(), model)
                self.assertEqual(r.message, "ok")
                self.assertEqual(r.target.name, "Grid")
                self.assertEqual(r.target_name, "Grid")
                self.assertEqual(r.target_reference, GRID_URL)
                self.assertEqual(r.target_description, "Target desc")
                self.assertEqual(r.attribute.name, "est")
                self.assertEqual(r.attribute_name, "est")
                self.assertEqual(r.attribute_reference, "cell_attributes[0]")

        ctb = ConditionalTurningBandsResult(
            MagicMock(),
            ConditionalTurningBandsResultModel(
                target=ConditionalTurningBandsTargetResult(
                    reference=GRID_URL,
                    name="Grid",
                    description="CTB desc",
                    schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                    simulations=TaskAttribute(reference="cell_attributes[0]", name="sims"),
                )
            ),
        )
        self.assertEqual(ctb.target.name, "Grid")
        self.assertEqual(ctb.target_description, "CTB desc")

        cd = ContinuousDistributionResult(
            MagicMock(),
            ContinuousDistributionResultModel(
                message="done",
                distribution=DistributionOutput(
                    reference=DIST_URL,
                    name="Dist",
                    description="Dist desc",
                    schema_id="/objects/continuous-distribution/1.0.0/continuous-distribution.schema.json",
                ),
            ),
        )
        self.assertEqual(cd.distribution.name, "Dist")
        self.assertEqual(cd.distribution_description, "Dist desc")

        lw = LocationWiseResult(
            MagicMock(),
            LocationWiseResultModel(
                message="done",
                target=LocationWiseTargetResult(
                    reference=GRID_URL,
                    name="Grid",
                    description="LW desc",
                    schema_id="/objects/regular-3d-grid/1.3.0/regular-3d-grid.schema.json",
                    attributes=[TaskAttribute(reference="cell_attributes[0]", name="mean")],
                ),
            ),
        )
        self.assertEqual(lw.target.name, "Grid")
        self.assertEqual(lw.target_description, "LW desc")
        self.assertEqual(lw.attributes[0].name, "mean")

        sr = SimulationReportResult(
            MagicMock(),
            SimulationReportResultModel(
                validation_summary=SimReportValidationSummary(reference_mean=1.0, mean=1.1),
                validation_report=SimReportValidationReport(reference="https://example.com/r"),
                links=SimReportLinks(dashboard="https://example.com/d"),
            ),
        )
        self.assertEqual(sr.validation_report.reference, "https://example.com/r")
        self.assertEqual(sr.links.dashboard, "https://example.com/d")

        # TaskResultList.__str__ handles non-single-attribute results gracefully too
        self.assertIn("Dist", str(TaskResultList([cd])))
        self.assertIn("https://example.com/r", str(TaskResultList([sr])))
        self.assertIn("mean", str(TaskResultList([lw])))


if __name__ == "__main__":
    unittest.main()

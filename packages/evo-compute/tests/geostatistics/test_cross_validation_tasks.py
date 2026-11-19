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

"""Contract and unit tests for the cross-validation compute task SDK client."""

from __future__ import annotations

import ast
import math
import re
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import nbformat
import numpy as np
import pandas as pd
from evo.objects import ObjectReference
from evo.objects.typed import BlockModel
from evo.objects.typed.attributes import (
    Attribute,
    BlockModelAttribute,
    BlockModelPendingAttribute,
    PendingAttribute,
)
from pydantic import ValidationError

from evo.compute.tasks import (
    AllOfFilter,
    AnyOfFilter,
    BlockDiscretization,
    CreateAttribute,
    Ellipsoid,
    EllipsoidRanges,
    Filter,
    FilterCondition,
    KrigingDiagnostics,
    Rotation,
    SearchNeighborhood,
    Source,
    Target,
    TaskResultList,
    UpdateAttribute,
    run,
)
from evo.compute.tasks import (
    CrossValidationResult as TopLevelCrossValidationResult,
)
from evo.compute.tasks.common.results import TaskAttribute
from evo.compute.tasks.common.runner import TaskRegistry
from evo.compute.tasks.geostatistics import CrossValidationResult as TopicCrossValidationResult
from evo.compute.tasks.geostatistics.cross_validation import (
    CrossValidationOutputs,
    CrossValidationParameters,
    CrossValidationResult,
    CrossValidationResultModel,
    CrossValidationRunner,
    CrossValidationSource,
    CrossValidationSourceResult,
    CrossValidationStatistics,
    IDWEstimator,
    KNNEstimator,
    KrigingCrossValidationStatistics,
    KrigingEstimator,
    LeaveOneDrillholeOut,
    LeaveOneSampleOut,
)
from evo.compute.tasks.geostatistics.kriging import (
    KrigingDiagnosticsResult,
    KrigingMethod,
    SimpleKriging,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_BASE = "https://hub.test.evo.bentley.com"
_ORG = "00000000-0000-0000-0000-000000000001"
_WS = "00000000-0000-0000-0000-000000000002"


def _obj_url(obj_id: str = "00000000-0000-0000-0000-000000000003") -> str:
    return f"{_BASE}/geoscience-object/orgs/{_ORG}/workspaces/{_WS}/objects/{obj_id}"


POINTSET_URL = _obj_url("00000000-0000-0000-0000-000000000010")
DOWNHOLE_URL = _obj_url("00000000-0000-0000-0000-000000000020")
VARIOGRAM_URL = _obj_url("00000000-0000-0000-0000-000000000030")
OTHER_URL = _obj_url("00000000-0000-0000-0000-000000000040")


def _search(**kwargs) -> SearchNeighborhood:
    defaults = dict(
        ellipsoid=Ellipsoid(
            ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            rotation=Rotation(dip_azimuth=15.0, dip=70.0, pitch=0.0),
        ),
        max_samples=20,
    )
    defaults.update(kwargs)
    return SearchNeighborhood(**defaults)


def _create_mock_attribute(
    name: str,
    key: str,
    object_url: str = POINTSET_URL,
    schema_path: str = "locations.attributes",
    attribute_type: str = "scalar",
) -> MagicMock:
    attr = MagicMock(spec=Attribute)
    attr.name = name
    attr.key = key
    attr.exists = True
    attr.attribute_type = attribute_type

    mock_context = MagicMock()
    mock_context.schema_path = schema_path
    attr._context = mock_context

    mock_obj = MagicMock()
    mock_obj.metadata.url = ObjectReference(object_url)
    attr._obj = mock_obj
    return attr


def _create_pending_attribute(name: str, object_url: str | None = POINTSET_URL) -> PendingAttribute:
    mock_parent = MagicMock()
    if object_url is not None:
        mock_obj = MagicMock()
        mock_obj.metadata.url = ObjectReference(object_url)
        mock_parent._obj = mock_obj
    else:
        mock_parent._obj = None
    return PendingAttribute(mock_parent, name)


def _dump(params: CrossValidationParameters) -> dict:
    return params.model_dump(mode="json", by_alias=True, exclude_none=True)


def _make_statistics(**overrides) -> CrossValidationStatistics:
    values = dict(
        evaluated_count=50,
        valid_count=48,
        invalid_count=2,
        mean_true=1.25,
        mean_estimate=1.22,
        variance_true=0.64,
        variance_estimate=0.49,
        mean_error=-0.03,
        mean_absolute_error=0.18,
        mean_squared_error=0.0625,
        root_mean_squared_error=0.25,
        error_variance=0.0616,
        correlation=0.88,
        regression_slope=0.96,
        regression_intercept=0.0788,
    )
    values.update(overrides)
    return CrossValidationStatistics(**values)


def _make_kriging_statistics(**overrides) -> KrigingCrossValidationStatistics:
    values = dict(
        standardized_count=48,
        mean_kriging_variance=0.065,
        mean_standardized_error=-0.04,
        standardized_error_variance=0.97,
    )
    values.update(overrides)
    return KrigingCrossValidationStatistics(**values)


def _make_result_model(
    *,
    with_error: bool = True,
    with_diagnostics: bool = False,
    with_kriging_stats: bool = False,
    schema_id: str = "/objects/pointset/1.2.0/pointset.schema.json",
    source_url: str = POINTSET_URL,
    source_name: str = "WP comps",
    statistics: CrossValidationStatistics | None = None,
    kriging_statistics: KrigingCrossValidationStatistics | None = None,
) -> CrossValidationResultModel:
    diagnostics = (
        KrigingDiagnosticsResult(
            kriging_variance=TaskAttribute(reference="locations.attributes[?key=='kv']", name="KV"),
            num_samples=TaskAttribute(reference="locations.attributes[?key=='ns']", name="NS"),
        )
        if with_diagnostics
        else None
    )
    return CrossValidationResultModel(
        message="Cross-validation completed: 48 of 50 samples produced an estimate.",
        source=CrossValidationSourceResult(
            reference=source_url,
            name=source_name,
            description="Composited samples",
            schema_id=schema_id,
            estimate=TaskAttribute(reference="locations.attributes[?key=='est']", name="cv_estimate"),
            error=(
                TaskAttribute(reference="locations.attributes[?key=='err']", name="cv_error") if with_error else None
            ),
            diagnostics=diagnostics,
        ),
        statistics=statistics or _make_statistics(),
        kriging_statistics=kriging_statistics
        if kriging_statistics is not None
        else (_make_kriging_statistics() if with_kriging_stats else None),
    )


# ---------------------------------------------------------------------------
# Registration & Public Surface
# ---------------------------------------------------------------------------


class TestCrossValidationRegistration(unittest.TestCase):
    def test_runner_registered(self):
        self.assertIs(TaskRegistry().get_runner(CrossValidationParameters), CrossValidationRunner)

    def test_topic_and_task(self):
        self.assertEqual(CrossValidationRunner.topic, "geostatistics")
        self.assertEqual(CrossValidationRunner.task, "cross-validation")

    def test_runner_generic_types(self):
        self.assertIs(CrossValidationRunner.params_type, CrossValidationParameters)
        self.assertIs(CrossValidationRunner.result_model_type, CrossValidationResultModel)
        self.assertIs(CrossValidationRunner.result_type, CrossValidationResult)

    def test_result_reexported_at_package_levels(self):
        self.assertIs(TopLevelCrossValidationResult, CrossValidationResult)
        self.assertIs(TopicCrossValidationResult, CrossValidationResult)


# ---------------------------------------------------------------------------
# Tagged Union Contract Tests: Validation & Estimator Branches
# ---------------------------------------------------------------------------


class TestCrossValidationTaggedUnions(unittest.TestCase):
    """Verify all six (validation × estimator) branches and reject invalid combinations."""

    def test_all_six_validation_and_estimator_branches_serialize_to_openapi_contract(self):
        validations = [
            ("sample", LeaveOneSampleOut(), POINTSET_URL, "locations.attributes[?key=='cu']"),
            ("drillhole", LeaveOneDrillholeOut(), DOWNHOLE_URL, "attributes[?key=='cu']"),
        ]
        estimators = [
            (
                "kriging",
                KrigingEstimator(
                    variogram=VARIOGRAM_URL,
                    neighborhood=_search(min_samples=3),
                    kriging_method=KrigingMethod.ORDINARY,
                    diagnostics=KrigingDiagnostics(kriging_variance=True),
                ),
            ),
            (
                "knn",
                KNNEstimator(neighborhood=_search()),
            ),
            (
                "idw",
                IDWEstimator(neighborhood=_search(), power=2.0),
            ),
        ]

        for expected_val_type, validation, source_url, attr_expr in validations:
            for expected_est_type, estimator in estimators:
                with self.subTest(validation=expected_val_type, estimator=expected_est_type):
                    params = CrossValidationParameters(
                        source=Source(object=source_url, attribute=attr_expr),
                        validation=validation,
                        estimator=estimator,
                        outputs=CrossValidationOutputs(estimate="cv_est", error="cv_err"),
                    )
                    payload = _dump(params)

                    self.assertEqual(set(payload), {"source", "validation", "estimator", "outputs"})
                    self.assertEqual(payload["source"], {"object": source_url, "attribute": attr_expr})
                    self.assertEqual(payload["validation"], {"type": expected_val_type})
                    self.assertEqual(payload["estimator"]["type"], expected_est_type)
                    self.assertEqual(
                        payload["outputs"],
                        {
                            "estimate": {"operation": "create", "name": "cv_est"},
                            "error": {"operation": "create", "name": "cv_err"},
                        },
                    )

                    if expected_est_type == "kriging":
                        self.assertEqual(
                            set(payload["estimator"]),
                            {"type", "variogram", "neighborhood", "kriging_method", "diagnostics"},
                        )
                        self.assertEqual(payload["estimator"]["kriging_method"], {"type": "ordinary"})
                        self.assertEqual(
                            payload["estimator"]["diagnostics"],
                            {"kriging_variance": {"operation": "create", "name": "KV"}},
                        )
                    elif expected_est_type == "knn":
                        self.assertEqual(set(payload["estimator"]), {"type", "neighborhood"})
                    else:
                        self.assertEqual(set(payload["estimator"]), {"type", "neighborhood", "power"})
                        self.assertEqual(payload["estimator"]["power"], 2.0)

                    # Round-trip wire payload back through CrossValidationParameters
                    round_tripped = CrossValidationParameters.model_validate(payload)
                    self.assertEqual(_dump(round_tripped), payload)

    def test_simple_kriging_branch_and_method_search_aliases(self):
        search = _search()
        params = CrossValidationParameters(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[?key=='cu']"),
            validation={"type": "sample"},
            estimator=KrigingEstimator(
                variogram=VARIOGRAM_URL,
                search=search,
                method=KrigingMethod.simple(mean=1.45),
            ),
            outputs=CrossValidationOutputs(estimate="cv_est"),
        )
        self.assertIsInstance(params.validation, LeaveOneSampleOut)
        self.assertIsInstance(params.estimator, KrigingEstimator)
        self.assertIsInstance(params.estimator.kriging_method, SimpleKriging)
        self.assertIs(params.estimator.search, params.estimator.neighborhood)
        self.assertIs(params.estimator.method, params.estimator.kriging_method)
        self.assertIs(KNNEstimator(search=search).search, search)
        self.assertIs(IDWEstimator(search=search, power=2.0).search, search)

        payload = _dump(params)
        self.assertEqual(payload["estimator"]["kriging_method"], {"type": "simple", "mean": 1.45})
        self.assertNotIn("diagnostics", payload["estimator"])
        self.assertNotIn("error", payload["outputs"])
        # Plain model_dump without exclude_none also omits None diagnostics and error
        plain = params.model_dump(mode="json")
        self.assertNotIn("diagnostics", plain["estimator"])
        self.assertNotIn("error", plain["outputs"])
        self.assertNotIn("filter", plain["source"])

    def test_neighborhood_drillhole_and_sector_limits_serialize_to_openapi_names(self):
        search = _search(
            min_samples=4,
            max_empty_octants=3,
            max_samples_per_octant=3,
            max_empty_quadrants=1,
            max_samples_per_quadrant=5,
            max_samples_per_drillhole=4,
            max_drillholes=6,
            min_drillholes=2,
            drillhole_limit_scope="octant",
        )
        for estimator in (
            KrigingEstimator(variogram=VARIOGRAM_URL, neighborhood=search),
            KNNEstimator(neighborhood=search),
            IDWEstimator(neighborhood=search, power=1.5),
        ):
            with self.subTest(estimator=estimator.type):
                params = CrossValidationParameters(
                    source=Source(object=DOWNHOLE_URL, attribute="attributes[?key=='cu']"),
                    validation=LeaveOneDrillholeOut(),
                    estimator=estimator,
                    outputs=CrossValidationOutputs(estimate="cv_est"),
                )
                payload = _dump(params)
                nbr = payload["estimator"]["neighborhood"]
                self.assertEqual(nbr["min_samples"], 4)
                self.assertEqual(nbr["max_empty_octants"], 3)
                self.assertEqual(nbr["max_samples_per_octant"], 3)
                self.assertEqual(nbr["max_empty_quadrants"], 1)
                self.assertEqual(nbr["max_samples_per_quadrant"], 5)
                self.assertEqual(nbr["max_samples_per_drillhole"], 4)
                self.assertEqual(nbr["max_drillholes"], 6)
                self.assertEqual(nbr["min_drillholes"], 2)
                self.assertEqual(nbr["drillhole_limit_scope"], "octant")
                self.assertNotIn("max_drillholes_per_estimate", nbr)
                self.assertNotIn("min_drillholes_per_estimate", nbr)

                # Round-trip wire payload carrying canonical drillhole properties
                reloaded = CrossValidationParameters.model_validate(payload)
                self.assertEqual(reloaded.estimator.neighborhood.max_drillholes, 6)
                self.assertEqual(reloaded.estimator.neighborhood.max_drillholes_per_estimate, 6)
                self.assertEqual(reloaded.estimator.neighborhood.min_drillholes, 2)
                self.assertEqual(reloaded.estimator.neighborhood.drillhole_limit_scope, "octant")
                self.assertEqual(_dump(reloaded), payload)

        # Backwards-compatible legacy drillhole aliases also serialize to canonical OpenAPI names
        legacy_search = _search(
            max_drillholes_per_estimate=5, min_drillholes_per_estimate=0, drillhole_limit_scope="global"
        )
        legacy_payload = _dump(
            CrossValidationParameters(
                source=Source(object=DOWNHOLE_URL, attribute="attributes[?key=='cu']"),
                validation=LeaveOneDrillholeOut(),
                estimator=KNNEstimator(neighborhood=legacy_search),
                outputs=CrossValidationOutputs(estimate="cv_est"),
            )
        )
        legacy_nbr = legacy_payload["estimator"]["neighborhood"]
        self.assertEqual(legacy_nbr["max_drillholes"], 5)
        self.assertNotIn("max_drillholes_per_estimate", legacy_nbr)
        self.assertNotIn("min_drillholes", legacy_nbr)
        self.assertNotIn("drillhole_limit_scope", legacy_nbr)

    def test_validation_rejects_invalid_strategies_and_arbitrary_groups(self):
        base = dict(
            source={"object": POINTSET_URL, "attribute": "locations.attributes[?key=='cu']"},
            estimator={"type": "knn", "neighborhood": _search().model_dump()},
            outputs={"estimate": {"operation": "create", "name": "cv_est"}},
        )
        invalid_validations = [
            ("missing-type", {}),
            ("unknown-k-fold", {"type": "k_fold", "folds": 5}),
            ("arbitrary-group", {"type": "group", "samples": [0, 1, 2]}),
            ("sample-subset", {"type": "sample", "samples": [0, 5]}),
            ("single-selected-hole", {"type": "drillhole", "drillhole_id": "DH001"}),
            ("multiple-holes", {"type": "drillhole", "drillhole_ids": ["DH001", "DH002"]}),
        ]
        for label, invalid_validation in invalid_validations:
            with self.subTest(case=label), self.assertRaises(ValidationError):
                CrossValidationParameters.model_validate({**base, "validation": invalid_validation})

    def test_estimator_rejects_cross_branch_and_invalid_combinations(self):
        nbr = _search().model_dump()
        base = dict(
            source={"object": POINTSET_URL, "attribute": "locations.attributes[?key=='cu']"},
            validation={"type": "sample"},
            outputs={"estimate": {"operation": "create", "name": "cv_est"}},
        )
        invalid_estimators = [
            ("missing-type", {"neighborhood": nbr}),
            ("unknown-type", {"type": "rbf", "neighborhood": nbr}),
            ("knn-with-power", {"type": "knn", "neighborhood": nbr, "power": 2.0}),
            ("knn-with-variogram", {"type": "knn", "neighborhood": nbr, "variogram": VARIOGRAM_URL}),
            ("knn-with-diagnostics", {"type": "knn", "neighborhood": nbr, "diagnostics": {}}),
            ("knn-with-kriging-method", {"type": "knn", "neighborhood": nbr, "kriging_method": {"type": "ordinary"}}),
            ("idw-without-power", {"type": "idw", "neighborhood": nbr}),
            ("idw-zero-power", {"type": "idw", "neighborhood": nbr, "power": 0.0}),
            ("idw-negative-power", {"type": "idw", "neighborhood": nbr, "power": -2.0}),
            ("idw-nan-power", {"type": "idw", "neighborhood": nbr, "power": math.nan}),
            ("idw-inf-power", {"type": "idw", "neighborhood": nbr, "power": math.inf}),
            ("idw-bool-power", {"type": "idw", "neighborhood": nbr, "power": True}),
            ("idw-with-variogram", {"type": "idw", "neighborhood": nbr, "power": 2.0, "variogram": VARIOGRAM_URL}),
            ("idw-with-diagnostics", {"type": "idw", "neighborhood": nbr, "power": 2.0, "diagnostics": {}}),
            (
                "idw-with-kriging-method",
                {"type": "idw", "neighborhood": nbr, "power": 2.0, "kriging_method": {"type": "ordinary"}},
            ),
            ("kriging-without-variogram", {"type": "kriging", "neighborhood": nbr}),
            (
                "kriging-with-power",
                {"type": "kriging", "variogram": VARIOGRAM_URL, "neighborhood": nbr, "power": 2.0},
            ),
            (
                "ordinary-kriging-with-mean",
                {
                    "type": "kriging",
                    "variogram": VARIOGRAM_URL,
                    "neighborhood": nbr,
                    "kriging_method": {"type": "ordinary", "mean": 1.0},
                },
            ),
            (
                "simple-kriging-without-mean",
                {
                    "type": "kriging",
                    "variogram": VARIOGRAM_URL,
                    "neighborhood": nbr,
                    "kriging_method": {"type": "simple"},
                },
            ),
            (
                "unknown-kriging-method",
                {
                    "type": "kriging",
                    "variogram": VARIOGRAM_URL,
                    "neighborhood": nbr,
                    "kriging_method": {"type": "universal"},
                },
            ),
        ]
        for label, invalid_estimator in invalid_estimators:
            with self.subTest(case=label), self.assertRaises(ValidationError):
                CrossValidationParameters.model_validate({**base, "estimator": invalid_estimator})

    def test_top_level_rejects_flattened_estimator_or_target_fields(self):
        base = dict(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[?key=='cu']"),
            validation=LeaveOneSampleOut(),
            estimator=KNNEstimator(neighborhood=_search()),
            outputs=CrossValidationOutputs(estimate="cv_est"),
        )
        flattened_extras = [
            ("target", Target.new_attribute(POINTSET_URL, "cv_est")),
            ("target_filter", Filter(where=FilterCondition(attribute="dom", operator="in", values=["A"]))),
            ("block_discretization", BlockDiscretization(nx=2, ny=2, nz=2)),
            ("variogram", VARIOGRAM_URL),
            ("power", 2.0),
            ("diagnostics", KrigingDiagnostics(kriging_variance=True)),
        ]
        for key, value in flattened_extras:
            with self.subTest(extra_field=key), self.assertRaises(ValidationError):
                CrossValidationParameters(**base, **{key: value})

    def test_neighborhood_mapping_validation_rejects_unknown_or_conflicting_fields(self):
        valid_ellipsoid = _search().ellipsoid.to_dict()
        bad_neighborhoods = [
            ("unknown-key", {"ellipsoid": valid_ellipsoid, "max_samples": 10, "k": 5}),
            (
                "conflicting-max-drillholes",
                {
                    "ellipsoid": valid_ellipsoid,
                    "max_samples": 10,
                    "max_drillholes": 3,
                    "max_drillholes_per_estimate": 4,
                },
            ),
            (
                "conflicting-min-drillholes",
                {
                    "ellipsoid": valid_ellipsoid,
                    "max_samples": 10,
                    "min_drillholes": 2,
                    "min_drillholes_per_estimate": 3,
                },
            ),
            ("zero-max-drillholes", {"ellipsoid": valid_ellipsoid, "max_samples": 10, "max_drillholes": 0}),
            ("unknown-ellipsoid-key", {"ellipsoid": {**valid_ellipsoid, "extra": 1}, "max_samples": 10}),
            ("missing-ellipsoid-ranges", {"ellipsoid": {"rotation": valid_ellipsoid["rotation"]}, "max_samples": 10}),
            ("incomplete-ranges", {"ellipsoid": {"ellipsoid_ranges": {"major": 100}}, "max_samples": 10}),
            (
                "unknown-rotation-key",
                {
                    "ellipsoid": {
                        "ellipsoid_ranges": valid_ellipsoid["ellipsoid_ranges"],
                        "rotation": {"dip": 10, "roll": 5},
                    },
                    "max_samples": 10,
                },
            ),
            (
                "out-of-bounds-rotation",
                {
                    "ellipsoid": {
                        "ellipsoid_ranges": valid_ellipsoid["ellipsoid_ranges"],
                        "rotation": {"dip": 200},
                    },
                    "max_samples": 10,
                },
            ),
        ]
        for label, bad_nbr in bad_neighborhoods:
            with self.subTest(case=label), self.assertRaises(ValidationError):
                KNNEstimator.model_validate({"type": "knn", "neighborhood": bad_nbr})

    def test_ellipsoid_mapping_without_or_partial_rotation_defaults_cleanly(self):
        est = KNNEstimator.model_validate(
            {
                "type": "knn",
                "neighborhood": {
                    "ellipsoid": {"ellipsoid_ranges": {"major": 100.0, "semi_major": 80.0, "minor": 50.0}},
                    "max_samples": 10,
                },
            }
        )
        self.assertEqual(est.neighborhood.ellipsoid.rotation, Rotation(0.0, 0.0, 0.0))

        partial_rot = KNNEstimator.model_validate(
            {
                "type": "knn",
                "neighborhood": {
                    "ellipsoid": {
                        "ellipsoid_ranges": {"major": 100.0, "semi_major": 80.0, "minor": 50.0},
                        "rotation": {"dip": 25.0},
                    },
                    "max_samples": 10,
                },
            }
        )
        self.assertEqual(partial_rot.neighborhood.ellipsoid.rotation, Rotation(0.0, 25.0, 0.0))


# ---------------------------------------------------------------------------
# Source, Filter, Outputs, & Diagnostics Validation
# ---------------------------------------------------------------------------


class TestCrossValidationSourceAndOutputs(unittest.TestCase):
    def test_typed_pointset_and_downhole_attributes_convert_to_source(self):
        ps_attr = _create_mock_attribute("Cu_pct", "cu-key", POINTSET_URL, schema_path="locations.attributes")
        dh_attr = _create_mock_attribute("Cu_pct", "dh-cu-key", DOWNHOLE_URL, schema_path="attributes")

        ps_params = CrossValidationParameters(
            source=ps_attr,
            validation=LeaveOneSampleOut(),
            estimator=KNNEstimator(neighborhood=_search()),
            outputs=CrossValidationOutputs(estimate="cv_est"),
        )
        dh_params = CrossValidationParameters(
            source=dh_attr,
            validation=LeaveOneDrillholeOut(),
            estimator=KNNEstimator(neighborhood=_search()),
            outputs=CrossValidationOutputs(estimate="cv_est"),
        )

        self.assertEqual(
            _dump(ps_params)["source"],
            {"object": POINTSET_URL, "attribute": "locations.attributes[?key=='cu-key']"},
        )
        self.assertEqual(
            _dump(dh_params)["source"],
            {"object": DOWNHOLE_URL, "attribute": "attributes[?key=='dh-cu-key']"},
        )

    def test_pending_source_attribute_is_rejected(self):
        pending = _create_pending_attribute("missing_grade")
        with self.assertRaisesRegex(ValidationError, "does not exist"):
            CrossValidationParameters(
                source=pending,
                validation=LeaveOneSampleOut(),
                estimator=KNNEstimator(neighborhood=_search()),
                outputs=CrossValidationOutputs(estimate="cv_est"),
            )

    def test_source_rejects_unknown_extra_fields(self):
        with self.assertRaises(ValidationError):
            CrossValidationParameters.model_validate(
                {
                    "source": {"object": POINTSET_URL, "attribute": "grade", "unexpected": True},
                    "validation": {"type": "sample"},
                    "estimator": {"type": "knn", "neighborhood": _search().model_dump()},
                    "outputs": {"estimate": {"operation": "create", "name": "cv_est"}},
                }
            )

    def test_source_filter_injected_into_source_and_composite_filters_supported(self):
        domain_attr = _create_mock_attribute("domain", "dom-key", POINTSET_URL, attribute_type="category")
        grade_attr = _create_mock_attribute("grade", "grd-key", POINTSET_URL, attribute_type="scalar")

        source_filter = Filter(
            where=AllOfFilter(
                filters=[
                    FilterCondition(attribute=domain_attr, operator="in", values=["LMS1", "LMS2"]),
                    AnyOfFilter(
                        filters=[
                            FilterCondition(attribute=grade_attr, operator="greater_than", threshold=0.2),
                            FilterCondition(attribute=grade_attr, operator="equal", threshold=0.0),
                        ]
                    ),
                ]
            )
        )
        params = CrossValidationParameters(
            source=grade_attr,
            source_filter=source_filter,
            validation=LeaveOneSampleOut(),
            estimator=KNNEstimator(neighborhood=_search()),
            outputs=CrossValidationOutputs(estimate="cv_est"),
        )
        payload = _dump(params)
        self.assertNotIn("source_filter", payload)
        self.assertEqual(payload["source"]["filter"]["where"]["type"], "all_of")
        self.assertNotIn("threshold", payload["source"]["filter"]["where"]["filters"][0])

    def test_both_source_filter_and_nested_source_filter_rejected(self):
        f = Filter(where=FilterCondition(attribute="domain", operator="in", values=["A"]))
        with self.assertRaisesRegex(ValidationError, "not both"):
            CrossValidationParameters(
                source=CrossValidationSource(object=POINTSET_URL, attribute="grade", filter=f),
                source_filter=f,
                validation=LeaveOneSampleOut(),
                estimator=KNNEstimator(neighborhood=_search()),
                outputs=CrossValidationOutputs(estimate="cv_est"),
            )

    def test_outputs_accept_strings_typed_attributes_and_explicit_specs(self):
        existing_est = _create_mock_attribute("cv_est", "est-key", POINTSET_URL, attribute_type="scalar")
        pending_err = _create_pending_attribute("cv_err", POINTSET_URL)

        outputs = CrossValidationOutputs(estimate=existing_est, error=pending_err)
        self.assertEqual(outputs.estimate, UpdateAttribute(reference="locations.attributes[?key=='est-key']"))
        self.assertEqual(outputs.error, CreateAttribute(name="cv_err"))

        mock_bm = MagicMock()
        mock_bm.metadata.url = ObjectReference(POINTSET_URL)
        bm_est = BlockModelAttribute(name="bm_est", attribute_type="Float64", obj=mock_bm)
        bm_err = BlockModelPendingAttribute(obj=mock_bm, name="bm_err")
        bm_outputs = CrossValidationOutputs(estimate=bm_est, error=bm_err)
        self.assertEqual(bm_outputs.estimate, UpdateAttribute(reference="attributes[?name=='bm_est']"))
        self.assertEqual(bm_outputs.error, CreateAttribute(name="bm_err"))

    def test_outputs_reject_non_floating_point_existing_attributes(self):
        for field in ("estimate", "error"):
            for bad_type in ("bool", "integer", "category"):
                with self.subTest(field=field, attribute_type=bad_type):
                    bad_attr = _create_mock_attribute("flag", "flag-key", POINTSET_URL, attribute_type=bad_type)
                    kwargs = {"estimate": "cv_est", field: bad_attr}
                    with self.assertRaisesRegex(ValidationError, f"Output {field!r} writes floating-point values"):
                        CrossValidationOutputs(**kwargs)

    def test_outputs_reject_invalid_attribute_operation_mappings(self):
        invalid_specs = [
            {"operation": "create", "name": "cv_est", "reference": "locations.attributes[0]"},
            {"operation": "update", "reference": "locations.attributes[0]", "name": "cv_est"},
            {"operation": "delete", "name": "cv_est"},
        ]
        for spec in invalid_specs:
            with self.subTest(spec=spec), self.assertRaises(ValidationError):
                CrossValidationOutputs.model_validate({"estimate": spec})

    def test_outputs_and_diagnostics_must_belong_to_source_object(self):
        wrong_attr = _create_mock_attribute("other_col", "other-key", OTHER_URL)

        with self.assertRaisesRegex(ValidationError, "Output 'estimate' references an attribute of a different object"):
            CrossValidationParameters(
                source=Source(object=POINTSET_URL, attribute="grade"),
                validation=LeaveOneSampleOut(),
                estimator=KNNEstimator(neighborhood=_search()),
                outputs=CrossValidationOutputs(estimate=wrong_attr),
            )

        with self.assertRaisesRegex(ValidationError, "Output 'error' references an attribute of a different object"):
            CrossValidationParameters(
                source=Source(object=POINTSET_URL, attribute="grade"),
                validation=LeaveOneSampleOut(),
                estimator=KNNEstimator(neighborhood=_search()),
                outputs=CrossValidationOutputs(estimate="cv_est", error=wrong_attr),
            )

        with self.assertRaisesRegex(
            ValidationError, "Diagnostic 'kriging_variance' references an attribute of a different object"
        ):
            CrossValidationParameters(
                source=Source(object=POINTSET_URL, attribute="grade"),
                validation=LeaveOneSampleOut(),
                estimator=KrigingEstimator(
                    variogram=VARIOGRAM_URL,
                    neighborhood=_search(),
                    diagnostics=KrigingDiagnostics(kriging_variance=wrong_attr),
                ),
                outputs=CrossValidationOutputs(estimate="cv_est"),
            )

    def test_version_query_string_ignored_when_matching_source_object(self):
        attr_without_version = _create_mock_attribute("cv_est", "est-key", POINTSET_URL)
        params = CrossValidationParameters(
            source=Source(object=f"{POINTSET_URL}?version=123", attribute="grade"),
            validation=LeaveOneSampleOut(),
            estimator=KNNEstimator(neighborhood=_search()),
            outputs=CrossValidationOutputs(estimate=attr_without_version),
        )
        self.assertEqual(
            _dump(params)["outputs"]["estimate"],
            {"operation": "update", "reference": "locations.attributes[?key=='est-key']"},
        )

    def test_duplicate_output_and_diagnostic_attributes_rejected(self):
        with self.assertRaisesRegex(ValidationError, "output 'estimate' and output 'error'"):
            CrossValidationOutputs(estimate="same_col", error="same_col")

        ref = "locations.attributes[?key=='same']"
        with self.assertRaisesRegex(ValidationError, "output 'estimate' and output 'error'"):
            CrossValidationOutputs(
                estimate=UpdateAttribute(reference=ref),
                error=UpdateAttribute(reference=ref),
            )

        with self.assertRaisesRegex(ValidationError, "output 'estimate' and diagnostic 'kriging_variance'"):
            CrossValidationParameters(
                source=Source(object=POINTSET_URL, attribute="grade"),
                validation=LeaveOneSampleOut(),
                estimator=KrigingEstimator(
                    variogram=VARIOGRAM_URL,
                    neighborhood=_search(),
                    diagnostics=KrigingDiagnostics(kriging_variance=True),
                ),
                outputs=CrossValidationOutputs(estimate="KV"),
            )

        with self.assertRaisesRegex(ValidationError, "output 'error' and diagnostic 'num_samples'"):
            CrossValidationParameters(
                source=Source(object=POINTSET_URL, attribute="grade"),
                validation=LeaveOneSampleOut(),
                estimator=KrigingEstimator(
                    variogram=VARIOGRAM_URL,
                    neighborhood=_search(),
                    diagnostics=KrigingDiagnostics(num_samples=True),
                ),
                outputs=CrossValidationOutputs(estimate="cv_est", error="NS"),
            )

        typed_kv = _create_mock_attribute("KV", "kv-key", POINTSET_URL)
        with self.assertRaisesRegex(ValidationError, "output 'estimate' and diagnostic 'kriging_variance'"):
            CrossValidationParameters(
                source=Source(object=POINTSET_URL, attribute="grade"),
                validation=LeaveOneSampleOut(),
                estimator=KrigingEstimator(
                    variogram=VARIOGRAM_URL,
                    neighborhood=_search(),
                    diagnostics=KrigingDiagnostics(
                        kriging_variance=UpdateAttribute(reference="locations.attributes[?key=='kv-key']")
                    ),
                ),
                outputs=CrossValidationOutputs(estimate=typed_kv),
            )


# ---------------------------------------------------------------------------
# Result Contract, Null/Invalid Behavior, & DataFrame Loading
# ---------------------------------------------------------------------------


class TestCrossValidationResults(unittest.IsolatedAsyncioTestCase):
    def test_common_and_kriging_results_expose_all_contract_fields(self):
        model = _make_result_model(with_error=True, with_diagnostics=True, with_kriging_stats=True)
        result = CrossValidationResult(MagicMock(), model)

        self.assertEqual(result.message, "Cross-validation completed: 48 of 50 samples produced an estimate.")
        self.assertIs(result.source, model.source)
        self.assertEqual(result.source_name, "WP comps")
        self.assertEqual(result.source_reference, POINTSET_URL)
        self.assertEqual(result.estimate_name, "cv_estimate")
        self.assertEqual(result.estimate_attribute.name, "cv_estimate")
        self.assertEqual(result.error_name, "cv_error")
        self.assertIsNotNone(result.error_attribute)
        self.assertEqual(
            {k: v.name for k, v in result.diagnostics.items()}, {"kriging_variance": "KV", "num_samples": "NS"}
        )
        self.assertEqual(result.schema.sub_classification, "pointset")
        self.assertEqual(result.statistics.evaluated_count, 50)
        self.assertEqual(result.statistics.valid_count, 48)
        self.assertEqual(result.statistics.invalid_count, 2)
        self.assertAlmostEqual(result.statistics.root_mean_squared_error, 0.25)
        self.assertIsNotNone(result.kriging_statistics)
        self.assertEqual(result.kriging_statistics.standardized_count, 48)
        self.assertAlmostEqual(result.kriging_statistics.standardized_error_variance, 0.97)

        rendered = str(result)
        self.assertIn("Cross-Validation Result", rendered)
        self.assertIn("WP comps", rendered)
        self.assertIn("cv_estimate", rendered)
        self.assertIn("cv_error", rendered)
        self.assertIn("KV, NS", rendered)
        self.assertIn("48/50 valid (2 invalid)", rendered)
        self.assertIn("0.2500", rendered)

    def test_knn_and_idw_results_omit_optional_error_diagnostics_and_kriging_stats(self):
        model = _make_result_model(
            with_error=False,
            with_diagnostics=False,
            with_kriging_stats=False,
            schema_id="/objects/downhole-intervals/1.3.0/downhole-intervals.schema.json",
            source_url=DOWNHOLE_URL,
            source_name="Drillholes",
        )
        result = CrossValidationResult(MagicMock(), model)

        self.assertIsNone(result.error_attribute)
        self.assertIsNone(result.error_name)
        self.assertEqual(result.diagnostics, {})
        self.assertIsNone(result.kriging_statistics)
        self.assertEqual(result.schema.sub_classification, "downhole-intervals")

        rendered = str(result)
        self.assertNotIn("Error:", rendered)
        self.assertNotIn("Diagnostics:", rendered)

    def test_null_statistics_and_zero_standardized_samples_preserved(self):
        """Single valid sample or zero-variance estimates produce null correlation/regression and null kriging stats."""
        stats = _make_statistics(
            evaluated_count=10,
            valid_count=1,
            invalid_count=9,
            variance_true=0.0,
            variance_estimate=0.0,
            correlation=None,
            regression_slope=None,
            regression_intercept=None,
            root_mean_squared_error=None,
        )
        kriging_stats = KrigingCrossValidationStatistics(
            standardized_count=0,
            mean_kriging_variance=None,
            mean_standardized_error=None,
            standardized_error_variance=None,
        )
        payload = _make_result_model(
            with_error=False,
            statistics=stats,
            kriging_statistics=kriging_stats,
        ).model_dump(mode="json")

        parsed = CrossValidationResultModel.model_validate(payload)
        result = CrossValidationResult(MagicMock(), parsed)

        self.assertEqual(result.statistics.evaluated_count, 10)
        self.assertEqual(result.statistics.valid_count, 1)
        self.assertEqual(result.statistics.invalid_count, 9)
        self.assertIsNone(result.statistics.correlation)
        self.assertIsNone(result.statistics.regression_slope)
        self.assertIsNone(result.statistics.regression_intercept)
        self.assertIsNone(result.statistics.root_mean_squared_error)
        self.assertEqual(result.kriging_statistics.standardized_count, 0)
        self.assertIsNone(result.kriging_statistics.mean_kriging_variance)
        self.assertIsNone(result.kriging_statistics.mean_standardized_error)
        self.assertIsNone(result.kriging_statistics.standardized_error_variance)
        self.assertNotIn("RMSE:", str(result))

    def test_result_models_reject_unexpected_or_missing_fields(self):
        valid_payload = _make_result_model().model_dump(mode="json")
        with self.assertRaises(ValidationError):
            CrossValidationResultModel.model_validate({**valid_payload, "extra_field": 1})

        missing_stat = dict(valid_payload["statistics"])
        missing_stat.pop("correlation")
        with self.assertRaises(ValidationError):
            CrossValidationResultModel.model_validate({**valid_payload, "statistics": missing_stat})

    def test_task_result_list_renders_cross_validation_source_and_estimate(self):
        r1 = CrossValidationResult(MagicMock(), _make_result_model())
        result_list = TaskResultList([r1])
        self.assertIn("WP comps → cv_estimate", str(result_list))

    async def test_to_dataframe_preserves_nan_estimates_for_spatial_and_block_model_sources(self):
        ctx = MagicMock()
        result = CrossValidationResult(ctx, _make_result_model())
        expected_df = pd.DataFrame({"grade": [1.2, 0.8], "cv_estimate": [1.1, np.nan], "cv_error": [-0.1, np.nan]})

        spatial_obj = MagicMock()
        spatial_obj.to_dataframe = AsyncMock(return_value=expected_df)
        with patch(
            "evo.compute.tasks.geostatistics.cross_validation.object_from_reference",
            new_callable=AsyncMock,
            return_value=spatial_obj,
        ) as mock_load:
            df = await result.to_dataframe("grade", "cv_estimate", "cv_error")

        mock_load.assert_awaited_once_with(ctx, POINTSET_URL)
        spatial_obj.to_dataframe.assert_awaited_once_with("grade", "cv_estimate", "cv_error")
        self.assertTrue(math.isnan(df.loc[1, "cv_estimate"]))

        bm_obj = MagicMock(spec=BlockModel)
        bm_obj.to_dataframe = AsyncMock(return_value=expected_df)
        with patch(
            "evo.compute.tasks.geostatistics.cross_validation.object_from_reference",
            new_callable=AsyncMock,
            return_value=bm_obj,
        ):
            await result.to_dataframe("grade", "cv_estimate")
        bm_obj.to_dataframe.assert_awaited_once_with(columns=["grade", "cv_estimate"])

        with patch(
            "evo.compute.tasks.geostatistics.cross_validation.object_from_reference",
            new_callable=AsyncMock,
            return_value=object(),
        ):
            with self.assertRaisesRegex(TypeError, "Don't know how to get DataFrame"):
                await result.to_dataframe()


# ---------------------------------------------------------------------------
# Async Runner & tasks.run Integration
# ---------------------------------------------------------------------------


class TestCrossValidationRunnerAsync(unittest.IsolatedAsyncioTestCase):
    def _make_context(self) -> MagicMock:
        ctx = MagicMock()
        ctx.get_connector.return_value = MagicMock()
        ctx.get_org_id.return_value = _ORG
        return ctx

    async def test_runner_and_unified_run_submit_expected_payload(self):
        ctx = self._make_context()
        params = CrossValidationParameters(
            source=Source(object=POINTSET_URL, attribute="locations.attributes[?key=='cu']"),
            validation=LeaveOneSampleOut(),
            estimator=IDWEstimator(neighborhood=_search(), power=2.0),
            outputs=CrossValidationOutputs(estimate="cv_estimate", error="cv_error"),
        )
        job = AsyncMock()
        job.wait_for_results.return_value = _make_result_model()

        with patch(
            "evo.compute.tasks.common.runner.JobClient.submit",
            new_callable=AsyncMock,
            return_value=job,
        ) as mock_submit:
            result = await run(ctx, params, preview=True)

        mock_submit.assert_awaited_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs["topic"], "geostatistics")
        self.assertEqual(kwargs["task"], "cross-validation")
        self.assertIs(kwargs["result_type"], CrossValidationResultModel)
        self.assertTrue(kwargs["preview"])
        self.assertEqual(kwargs["parameters"], _dump(params))
        self.assertIsInstance(result, CrossValidationResult)


# ---------------------------------------------------------------------------
# Example Notebook Validation
# ---------------------------------------------------------------------------


class TestCrossValidationExampleNotebook(unittest.TestCase):
    NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "docs" / "examples" / "cross_validation.ipynb"

    def test_notebook_valid_syntax_clean_outputs_and_no_client_ids(self):
        nb = nbformat.read(str(self.NOTEBOOK_PATH), as_version=4)
        nbformat.validate(nb)

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        self.assertGreaterEqual(len(code_cells), 8)

        full_source = "\n".join(c.source for c in code_cells)
        self.assertIn("LeaveOneSampleOut", full_source)
        self.assertIn("LeaveOneDrillholeOut", full_source)
        self.assertIn("KrigingEstimator", full_source)
        self.assertIn("IDWEstimator", full_source)
        self.assertIn("KNNEstimator", full_source)
        self.assertIsNone(re.search(r"native-[A-Za-z0-9]{12,}", full_source))

        for cell in code_cells:
            self.assertEqual(cell.outputs, [])
            self.assertIsNone(cell.execution_count)
            filtered_lines = [line for line in cell.source.splitlines() if not line.strip().startswith("%")]
            ast.parse("\n".join(filtered_lines), mode="exec", feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()

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

import contextlib
import dataclasses
import uuid
from unittest.mock import patch

from parameterized import parameterized

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import (
    CubicStructure,
    Ellipsoid,
    EllipsoidRanges,
    ExponentialStructure,
    GaussianStructure,
    GeneralisedCauchyStructure,
    LinearStructure,
    Rotation,
    SphericalStructure,
    SpheroidalStructure,
    Variogram,
    VariogramData,
)
from evo.objects.typed.base import BaseObject

from .helpers import MockClient


class TestVariogram(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
        )

    @contextlib.contextmanager
    def _mock_geoscience_objects(self):
        mock_client = MockClient(self.environment)
        with (
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.DownloadedObject.from_context", mock_client.from_reference),
        ):
            yield mock_client

    example_variogram = VariogramData(
        name="Test Variogram",
        sill=1.5,
        is_rotation_fixed=True,
        structures=[
            SphericalStructure(
                contribution=0.8,
                anisotropy=Ellipsoid(
                    ranges=EllipsoidRanges(major=100.0, semi_major=50.0, minor=25.0),
                    rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
                ),
            ),
            ExponentialStructure(
                contribution=0.5,
                anisotropy=Ellipsoid(
                    ranges=EllipsoidRanges(major=200.0, semi_major=100.0, minor=50.0),
                    rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
                ),
            ),
        ],
        nugget=0.2,
        data_variance=1.5,
        modelling_space="data",
        domain="ore_zone",
        attribute="gold_grade",
    )

    @parameterized.expand([BaseObject, Variogram])
    async def test_create(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=self.example_variogram)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Test Variogram")

    @parameterized.expand([BaseObject, Variogram])
    async def test_replace(self, class_to_call):
        data = dataclasses.replace(self.example_variogram, name="Replaced Variogram")
        with self._mock_geoscience_objects():
            result = await class_to_call.replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Replaced Variogram")

    @parameterized.expand([BaseObject, Variogram])
    async def test_create_or_replace(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create_or_replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=self.example_variogram,
            )
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Test Variogram")

    async def test_from_reference(self):
        with self._mock_geoscience_objects():
            original = await Variogram.create(context=self.context, data=self.example_variogram)

            result = await Variogram.from_reference(context=self.context, reference=original.metadata.url)
            self.assertEqual(result.name, "Test Variogram")

    async def test_update(self):
        with self._mock_geoscience_objects():
            obj = await Variogram.create(context=self.context, data=self.example_variogram)

            self.assertEqual(obj.metadata.version_id, "1")
            obj.name = "Updated Variogram"
            obj.description = "An updated variogram model"

            await obj.update()

            self.assertEqual(obj.name, "Updated Variogram")
            self.assertEqual(obj.description, "An updated variogram model")
            self.assertEqual(obj.metadata.version_id, "2")

    async def test_minimal_variogram(self):
        """Test creating a variogram with only required fields using typed classes."""
        minimal_data = VariogramData(
            name="Minimal Variogram",
            sill=1.0,
            is_rotation_fixed=False,
            structures=[
                SphericalStructure(
                    contribution=1.0,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=100.0, semi_major=100.0, minor=100.0),
                    ),
                )
            ],
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=minimal_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Minimal Variogram")

    async def test_variogram_with_normalscore(self):
        """Test creating a variogram with normalscore modelling space."""
        normalscore_data = VariogramData(
            name="Normalscore Variogram",
            sill=1.0,
            is_rotation_fixed=True,
            structures=[
                GaussianStructure(
                    contribution=0.9,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=150.0, semi_major=75.0, minor=30.0),
                        rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
                    ),
                )
            ],
            nugget=0.1,
            modelling_space="normalscore",
            attribute="copper_grade",
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=normalscore_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Normalscore Variogram")

    async def test_variogram_multiple_structures(self):
        """Test creating a variogram with multiple structure types using typed classes."""
        multi_structure_data = VariogramData(
            name="Multi-Structure Variogram",
            sill=2.0,
            is_rotation_fixed=False,
            structures=[
                SphericalStructure(
                    contribution=0.5,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=100.0, semi_major=50.0, minor=25.0),
                        rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
                    ),
                ),
                ExponentialStructure(
                    contribution=0.8,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=200.0, semi_major=100.0, minor=50.0),
                        rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
                    ),
                ),
                GaussianStructure(
                    contribution=0.5,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=300.0, semi_major=150.0, minor=75.0),
                        rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
                    ),
                ),
            ],
            nugget=0.2,
            data_variance=2.0,
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=multi_structure_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Multi-Structure Variogram")

    async def test_cubic_structure(self):
        """Test creating a variogram with cubic structure."""
        cubic_data = VariogramData(
            name="Cubic Variogram",
            sill=1.0,
            is_rotation_fixed=True,
            structures=[
                CubicStructure(
                    contribution=1.0,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=100.0, semi_major=100.0, minor=100.0),
                    ),
                )
            ],
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=cubic_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Cubic Variogram")

    def test_structure_to_dict(self):
        """Test that typed structures correctly convert to dictionaries."""
        structure = SphericalStructure(
            contribution=0.8,
            anisotropy=Ellipsoid(
                ranges=EllipsoidRanges(major=100.0, semi_major=50.0, minor=25.0),
                rotation=Rotation(dip_azimuth=45.0, dip=30.0, pitch=15.0),
            ),
        )
        result = structure.to_dict()

        self.assertEqual(result["variogram_type"], "spherical")
        self.assertEqual(result["contribution"], 0.8)
        self.assertEqual(result["anisotropy"]["ellipsoid_ranges"]["major"], 100.0)
        self.assertEqual(result["anisotropy"]["ellipsoid_ranges"]["semi_major"], 50.0)
        self.assertEqual(result["anisotropy"]["ellipsoid_ranges"]["minor"], 25.0)
        self.assertEqual(result["anisotropy"]["rotation"]["dip_azimuth"], 45.0)
        self.assertEqual(result["anisotropy"]["rotation"]["dip"], 30.0)
        self.assertEqual(result["anisotropy"]["rotation"]["pitch"], 15.0)

    def test_ellipsoid_ranges_to_dict(self):
        """Test EllipsoidRanges to_dict method."""
        ranges = EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0)
        result = ranges.to_dict()

        self.assertEqual(result, {"major": 200.0, "semi_major": 150.0, "minor": 100.0})

    def test_ellipsoid_default_rotation(self):
        """Test Ellipsoid with default rotation."""
        ellipsoid = Ellipsoid(
            ranges=EllipsoidRanges(major=100.0, semi_major=50.0, minor=25.0),
        )
        result = ellipsoid.to_dict()

        self.assertEqual(result["ellipsoid_ranges"]["major"], 100.0)
        self.assertEqual(result["rotation"]["dip_azimuth"], 0.0)
        self.assertEqual(result["rotation"]["dip"], 0.0)
        self.assertEqual(result["rotation"]["pitch"], 0.0)

    def test_variogram_data_get_structures_as_dicts(self):
        """Test that VariogramData correctly converts mixed typed and dict structures."""
        data = VariogramData(
            name="Mixed Structures",
            sill=1.0,
            is_rotation_fixed=True,
            structures=[
                SphericalStructure(
                    contribution=0.5,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=100.0, semi_major=50.0, minor=25.0),
                    ),
                ),
                {
                    "variogram_type": "exponential",
                    "contribution": 0.5,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                        "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                    },
                },
            ],
        )
        result = data.get_structures_as_dicts()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["variogram_type"], "spherical")
        self.assertEqual(result[1]["variogram_type"], "exponential")

    def test_all_structure_types(self):
        """Test that all structure types have correct variogram_type."""
        self.assertEqual(
            SphericalStructure(
                contribution=1.0,
                anisotropy=Ellipsoid(ranges=EllipsoidRanges(major=1, semi_major=1, minor=1)),
            ).variogram_type,
            "spherical",
        )

        self.assertEqual(
            ExponentialStructure(
                contribution=1.0,
                anisotropy=Ellipsoid(ranges=EllipsoidRanges(major=1, semi_major=1, minor=1)),
            ).variogram_type,
            "exponential",
        )

        self.assertEqual(
            GaussianStructure(
                contribution=1.0,
                anisotropy=Ellipsoid(ranges=EllipsoidRanges(major=1, semi_major=1, minor=1)),
            ).variogram_type,
            "gaussian",
        )

        self.assertEqual(
            CubicStructure(
                contribution=1.0,
                anisotropy=Ellipsoid(ranges=EllipsoidRanges(major=1, semi_major=1, minor=1)),
            ).variogram_type,
            "cubic",
        )

        self.assertEqual(
            LinearStructure(
                contribution=1.0,
                anisotropy=Ellipsoid(ranges=EllipsoidRanges(major=1, semi_major=1, minor=1)),
            ).variogram_type,
            "linear",
        )

        self.assertEqual(
            SpheroidalStructure(
                contribution=1.0,
                alpha=5,
                anisotropy=Ellipsoid(ranges=EllipsoidRanges(major=1, semi_major=1, minor=1)),
            ).variogram_type,
            "spheroidal",
        )

        self.assertEqual(
            GeneralisedCauchyStructure(
                contribution=1.0,
                alpha=7,
                anisotropy=Ellipsoid(ranges=EllipsoidRanges(major=1, semi_major=1, minor=1)),
            ).variogram_type,
            "generalisedcauchy",
        )

    async def test_linear_structure(self):
        """Test creating a variogram with linear structure."""
        linear_data = VariogramData(
            name="Linear Variogram",
            sill=1.0,
            is_rotation_fixed=True,
            structures=[
                LinearStructure(
                    contribution=1.0,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=100.0, semi_major=100.0, minor=100.0),
                    ),
                )
            ],
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=linear_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Linear Variogram")

    async def test_spheroidal_structure(self):
        """Test creating a variogram with spheroidal structure."""
        spheroidal_data = VariogramData(
            name="Spheroidal Variogram",
            sill=1.0,
            is_rotation_fixed=True,
            structures=[
                SpheroidalStructure(
                    contribution=1.0,
                    alpha=5,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=100.0, semi_major=75.0, minor=50.0),
                        rotation=Rotation(dip_azimuth=45.0, dip=30.0, pitch=15.0),
                    ),
                )
            ],
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=spheroidal_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Spheroidal Variogram")

    async def test_generalisedcauchy_structure(self):
        """Test creating a variogram with generalised cauchy structure."""
        cauchy_data = VariogramData(
            name="Generalised Cauchy Variogram",
            sill=1.0,
            is_rotation_fixed=True,
            structures=[
                GeneralisedCauchyStructure(
                    contribution=1.0,
                    alpha=7,
                    anisotropy=Ellipsoid(
                        ranges=EllipsoidRanges(major=150.0, semi_major=100.0, minor=75.0),
                    ),
                )
            ],
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=cauchy_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Generalised Cauchy Variogram")

    def test_spheroidal_structure_to_dict(self):
        """Test that spheroidal structure correctly includes alpha in dict."""
        structure = SpheroidalStructure(
            contribution=0.8,
            alpha=5,
            anisotropy=Ellipsoid(
                ranges=EllipsoidRanges(major=100.0, semi_major=50.0, minor=25.0),
            ),
        )
        result = structure.to_dict()

        self.assertEqual(result["variogram_type"], "spheroidal")
        self.assertEqual(result["contribution"], 0.8)
        self.assertEqual(result["alpha"], 5)
        self.assertEqual(result["anisotropy"]["ellipsoid_ranges"]["major"], 100.0)

    def test_generalisedcauchy_structure_to_dict(self):
        """Test that generalised cauchy structure correctly includes alpha in dict."""
        structure = GeneralisedCauchyStructure(
            contribution=0.7,
            alpha=9,
            anisotropy=Ellipsoid(
                ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
                rotation=Rotation(dip_azimuth=90.0, dip=45.0, pitch=0.0),
            ),
        )
        result = structure.to_dict()

        self.assertEqual(result["variogram_type"], "generalisedcauchy")
        self.assertEqual(result["contribution"], 0.7)
        self.assertEqual(result["alpha"], 9)
        self.assertEqual(result["anisotropy"]["rotation"]["dip_azimuth"], 90.0)

    def test_linear_structure_to_dict(self):
        """Test that linear structure correctly converts to dict."""
        structure = LinearStructure(
            contribution=0.5,
            anisotropy=Ellipsoid(
                ranges=EllipsoidRanges(major=100.0, semi_major=100.0, minor=100.0),
            ),
        )
        result = structure.to_dict()

        self.assertEqual(result["variogram_type"], "linear")
        self.assertEqual(result["contribution"], 0.5)
        self.assertNotIn("alpha", result)  # Linear doesn't have alpha

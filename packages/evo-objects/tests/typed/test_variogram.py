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
from evo.objects.typed import Variogram, VariogramData
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
            patch("evo.objects.typed.dataset.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.typed.base.download_geoscience_object", mock_client.from_reference),
        ):
            yield mock_client

    example_variogram = VariogramData(
        name="Test Variogram",
        sill=1.5,
        is_rotation_fixed=True,
        structures=[
            {
                "type": "spherical",
                "contribution": 0.8,
                "range": {"major": 100.0, "minor": 50.0, "vertical": 25.0},
            },
            {
                "type": "exponential",
                "contribution": 0.5,
                "range": {"major": 200.0, "minor": 100.0, "vertical": 50.0},
            },
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
        """Test creating a variogram with only required fields."""
        minimal_data = VariogramData(
            name="Minimal Variogram",
            sill=1.0,
            is_rotation_fixed=False,
            structures=[
                {
                    "type": "spherical",
                    "contribution": 1.0,
                    "range": {"major": 100.0, "minor": 100.0, "vertical": 100.0},
                }
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
                {
                    "type": "gaussian",
                    "contribution": 0.9,
                    "range": {"major": 150.0, "minor": 75.0, "vertical": 30.0},
                }
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
        """Test creating a variogram with multiple structure types."""
        multi_structure_data = VariogramData(
            name="Multi-Structure Variogram",
            sill=2.0,
            is_rotation_fixed=False,
            structures=[
                {
                    "type": "spherical",
                    "contribution": 0.5,
                    "range": {"major": 100.0, "minor": 50.0, "vertical": 25.0},
                },
                {
                    "type": "exponential",
                    "contribution": 0.8,
                    "range": {"major": 200.0, "minor": 100.0, "vertical": 50.0},
                },
                {
                    "type": "gaussian",
                    "contribution": 0.5,
                    "range": {"major": 300.0, "minor": 150.0, "vertical": 75.0},
                },
            ],
            nugget=0.2,
            data_variance=2.0,
        )
        with self._mock_geoscience_objects():
            result = await Variogram.create(context=self.context, data=multi_structure_data)
        self.assertIsInstance(result, Variogram)
        self.assertEqual(result.name, "Multi-Structure Variogram")

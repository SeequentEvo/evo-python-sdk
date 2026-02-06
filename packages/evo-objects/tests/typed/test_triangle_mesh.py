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
import uuid
from unittest.mock import patch

import pandas as pd
from parameterized import parameterized

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import BoundingBox, TriangleMesh, TriangleMeshData
from evo.objects.typed.base import BaseObject
from evo.objects.typed.exceptions import ObjectValidationError

from .helpers import MockClient


class TestTriangleMesh(TestWithConnector):
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
            patch("evo.objects.typed.attributes.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed._data.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.DownloadedObject.from_context", mock_client.from_reference),
        ):
            yield mock_client

    # A simple tetrahedron mesh (4 vertices, 4 triangles)
    example_mesh = TriangleMeshData(
        name="Test Triangle Mesh",
        vertices=pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.5, 0.5],
                "y": [0.0, 0.0, 1.0, 0.5],
                "z": [0.0, 0.0, 0.0, 1.0],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        triangles=pd.DataFrame(
            {
                "n0": [0, 0, 0, 1],
                "n1": [1, 2, 3, 2],
                "n2": [2, 3, 1, 3],
                "area": [0.5, 0.6, 0.7, 0.8],
            }
        ),
    )

    def _assert_bounding_box_equal(
        self, bbox: BoundingBox, min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float
    ):
        self.assertAlmostEqual(bbox.min_x, min_x)
        self.assertAlmostEqual(bbox.max_x, max_x)
        self.assertAlmostEqual(bbox.min_y, min_y)
        self.assertAlmostEqual(bbox.max_y, max_y)
        self.assertAlmostEqual(bbox.min_z, min_z)
        self.assertAlmostEqual(bbox.max_z, max_z)

    @parameterized.expand([BaseObject, TriangleMesh])
    async def test_create(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=self.example_mesh)
        self.assertIsInstance(result, TriangleMesh)
        self.assertEqual(result.name, "Test Triangle Mesh")
        self.assertEqual(result.num_vertices, 4)
        self.assertEqual(result.num_triangles, 4)

        vertex_df = await result.triangles.get_vertices_dataframe()
        pd.testing.assert_frame_equal(vertex_df, self.example_mesh.vertices)

        triangle_df = await result.triangles.get_indices_dataframe()
        pd.testing.assert_frame_equal(triangle_df, self.example_mesh.triangles)

    @parameterized.expand([BaseObject, TriangleMesh])
    async def test_replace(self, class_to_call):
        # Create a mesh with only coordinates and indices (no attributes)
        vertices = pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.5],
                "y": [0.0, 0.0, 1.0],
                "z": [0.0, 0.0, 0.0],
            }
        )
        triangles = pd.DataFrame(
            {
                "n0": [0],
                "n1": [1],
                "n2": [2],
            }
        )
        data = TriangleMeshData(
            name="Simple Triangle",
            vertices=vertices,
            triangles=triangles,
        )
        with self._mock_geoscience_objects():
            result = await class_to_call.replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=data,
            )
        self.assertIsInstance(result, TriangleMesh)
        self.assertEqual(result.name, "Simple Triangle")
        self.assertEqual(result.num_vertices, 3)
        self.assertEqual(result.num_triangles, 1)

        actual_vertices = await result.triangles.get_vertices_dataframe()
        pd.testing.assert_frame_equal(actual_vertices, vertices)

    @parameterized.expand([BaseObject, TriangleMesh])
    async def test_create_or_replace(self, class_to_call):
        with self._mock_geoscience_objects():
            result = await class_to_call.create_or_replace(
                context=self.context,
                reference=ObjectReference.new(
                    environment=self.context.get_environment(),
                    object_id=uuid.uuid4(),
                ),
                data=self.example_mesh,
            )
        self.assertIsInstance(result, TriangleMesh)
        self.assertEqual(result.name, "Test Triangle Mesh")
        self.assertEqual(result.num_vertices, 4)
        self.assertEqual(result.num_triangles, 4)

    @parameterized.expand([BaseObject, TriangleMesh])
    async def test_from_reference(self, class_to_call):
        with self._mock_geoscience_objects():
            original = await TriangleMesh.create(context=self.context, data=self.example_mesh)

            result = await class_to_call.from_reference(context=self.context, reference=original.metadata.url)
            self.assertEqual(result.name, "Test Triangle Mesh")
            self.assertEqual(result.num_vertices, 4)
            self.assertEqual(result.num_triangles, 4)

            vertex_df = await result.triangles.get_vertices_dataframe()
            pd.testing.assert_frame_equal(vertex_df, self.example_mesh.vertices)

    def test_bounding_box_from_data(self):
        """Test that the bounding box is computed correctly from the vertex data."""
        bbox = self.example_mesh.compute_bounding_box()
        self._assert_bounding_box_equal(bbox, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    async def test_bounding_box_from_object(self):
        """Test that the bounding box is stored correctly on the created object."""
        with self._mock_geoscience_objects() as mock_client:
            obj = await TriangleMesh.create(context=self.context, data=self.example_mesh)

            bbox = obj.bounding_box
            self._assert_bounding_box_equal(bbox, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

            # Verify it was saved to the document
            bbox_dict = mock_client.objects[str(obj.metadata.url.object_id)]["bounding_box"]
            self.assertAlmostEqual(bbox_dict["min_x"], 0.0)
            self.assertAlmostEqual(bbox_dict["max_x"], 1.0)

    def test_vertices_validation(self):
        """Test that vertices validation works correctly."""
        triangles = pd.DataFrame({"n0": [0], "n1": [1], "n2": [2]})

        # Missing x column
        with self.assertRaises(ObjectValidationError):
            TriangleMeshData(
                name="Bad Mesh",
                vertices=pd.DataFrame({"y": [0.0, 1.0, 0.5], "z": [0.0, 0.0, 0.0]}),
                triangles=triangles,
            )

        # Missing y and z columns
        with self.assertRaises(ObjectValidationError):
            TriangleMeshData(
                name="Bad Mesh",
                vertices=pd.DataFrame({"x": [0.0, 1.0, 0.5]}),
                triangles=triangles,
            )

    def test_triangles_validation(self):
        """Test that triangle indices validation works correctly."""
        vertices = pd.DataFrame({"x": [0.0, 1.0, 0.5], "y": [0.0, 0.0, 1.0], "z": [0.0, 0.0, 0.0]})

        # Missing n0 column
        with self.assertRaises(ObjectValidationError):
            TriangleMeshData(
                name="Bad Mesh",
                vertices=vertices,
                triangles=pd.DataFrame({"n1": [1], "n2": [2]}),
            )

        # Missing n1 and n2 columns
        with self.assertRaises(ObjectValidationError):
            TriangleMeshData(
                name="Bad Mesh",
                vertices=vertices,
                triangles=pd.DataFrame({"n0": [0]}),
            )

    def test_triangle_index_out_of_range(self):
        """Test that validation fails when triangle indices reference non-existent vertices."""
        vertices = pd.DataFrame({"x": [0.0, 1.0, 0.5], "y": [0.0, 0.0, 1.0], "z": [0.0, 0.0, 0.0]})

        # Index 5 is out of range (only 3 vertices)
        with self.assertRaises(ObjectValidationError):
            TriangleMeshData(
                name="Bad Mesh",
                vertices=vertices,
                triangles=pd.DataFrame({"n0": [0], "n1": [1], "n2": [5]}),
            )

    async def test_create_with_geometry_only(self):
        """Test creating a mesh with only geometry (no attributes)."""
        data = TriangleMeshData(
            name="Geometry Only Mesh",
            vertices=pd.DataFrame(
                {
                    "x": [0.0, 1.0, 0.5],
                    "y": [0.0, 0.0, 1.0],
                    "z": [0.0, 0.0, 0.0],
                }
            ),
            triangles=pd.DataFrame(
                {
                    "n0": [0],
                    "n1": [1],
                    "n2": [2],
                }
            ),
        )
        with self._mock_geoscience_objects():
            result = await TriangleMesh.create(context=self.context, data=data)
        self.assertEqual(result.num_vertices, 3)
        self.assertEqual(result.num_triangles, 1)

    async def test_description_and_tags(self):
        """Test setting and getting description and tags."""
        data = TriangleMeshData(
            name="Test Mesh",
            vertices=self.example_mesh.vertices,
            triangles=self.example_mesh.triangles,
            description="A test triangle mesh for testing",
            tags={"category": "test", "priority": "high"},
        )
        with self._mock_geoscience_objects():
            result = await TriangleMesh.create(context=self.context, data=data)

        self.assertEqual(result.description, "A test triangle mesh for testing")
        self.assertEqual(result.tags, {"category": "test", "priority": "high"})

    async def test_json(self):
        """Test the JSON structure of the created object."""
        with self._mock_geoscience_objects() as mock_client:
            obj = await TriangleMesh.create(context=self.context, data=self.example_mesh)

            # Get the JSON that was stored (would be sent to the API)
            object_json = mock_client.objects[str(obj.metadata.url.object_id)]

            # Verify schema
            self.assertEqual(object_json["schema"], "/objects/triangle-mesh/2.2.0/triangle-mesh.schema.json")

            # Verify base properties
            self.assertEqual(object_json["name"], "Test Triangle Mesh")
            self.assertIn("uuid", object_json)
            self.assertIn("bounding_box", object_json)
            self.assertEqual(object_json["coordinate_reference_system"], "unspecified")

            # Verify triangles structure
            self.assertIn("triangles", object_json)
            self.assertIn("vertices", object_json["triangles"])
            self.assertIn("indices", object_json["triangles"])

            # Verify vertices structure
            self.assertIn("data", object_json["triangles"]["vertices"])
            self.assertEqual(object_json["triangles"]["vertices"]["length"], 4)

            # Verify indices structure
            self.assertIn("data", object_json["triangles"]["indices"])
            self.assertEqual(object_json["triangles"]["indices"]["length"], 4)

            # Verify vertex attributes structure
            self.assertEqual(len(object_json["triangles"]["vertices"]["attributes"]), 1)
            self.assertEqual(object_json["triangles"]["vertices"]["attributes"][0]["name"], "value")
            self.assertEqual(object_json["triangles"]["vertices"]["attributes"][0]["attribute_type"], "scalar")

            # Verify triangle attributes structure
            self.assertEqual(len(object_json["triangles"]["indices"]["attributes"]), 1)
            self.assertEqual(object_json["triangles"]["indices"]["attributes"][0]["name"], "area")
            self.assertEqual(object_json["triangles"]["indices"]["attributes"][0]["attribute_type"], "scalar")

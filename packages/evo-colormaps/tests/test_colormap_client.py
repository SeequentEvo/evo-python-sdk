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

import json
from unittest import mock
from uuid import UUID

from data import load_test_data
from evo.colormaps import ColormapAPIClient
from evo.colormaps.data import (
    Association,
    AssociationMetadata,
    CategoryColormap,
    ColormapMetadata,
    ContinuousColormap,
    DiscreteColormap,
)
from evo.colormaps.exceptions import UnknownColormapType
from evo.common import Environment, HealthCheckType, RequestMethod, ServiceUser
from evo.common.exceptions import NotFoundException
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector, utc_datetime
from evo.common.utils import get_header_metadata


class TestColormapApiClient(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.colormap_api_client = ColormapAPIClient(connector=self.connector, environment=self.environment)
        self.setup_universal_headers(get_header_metadata(ColormapAPIClient.__module__))

    @property
    def base_path(self) -> str:
        return f"colormap/orgs/{self.environment.org_id}/workspaces/{self.environment.workspace_id}"

    async def test_check_service_health(self) -> None:
        """Test service health check implementation."""
        with mock.patch("evo.colormaps.client.get_service_health", spec_set=True) as mock_get_service_health:
            await self.colormap_api_client.get_service_health()
        mock_get_service_health.assert_called_once_with(
            self.connector, "visualization", check_type=HealthCheckType.FULL
        )

    async def test_create_continuous_colormap(self) -> None:
        post_colormap_response = load_test_data("continuous_colormap_response.json")
        colormap = ContinuousColormap(
            colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
            attribute_controls=[0.1, 0.2, 0.3],
            gradient_controls=[0.4, 0.5, 0.6],
        )
        colormap_name = "continuous colormap 1"
        expected_colormap_metadata = ColormapMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            colormap=colormap,
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name=colormap_name,
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{UUID(int=6)}",
            id=UUID(int=6),
        )
        with self.transport.set_http_response(
            status_code=201,
            content=json.dumps(post_colormap_response),
            headers={"Content-Type": "application/json"},
        ):
            colormap_metadata = await self.colormap_api_client.create_colormap(colormap=colormap, name=colormap_name)

        self.assertIsInstance(colormap_metadata, ColormapMetadata)
        self.assertEqual(colormap_metadata, expected_colormap_metadata)
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/colormaps",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "attribute_controls": colormap.attribute_controls,
                "colors": colormap.colors,
                "dtype": "continuous",
                "gradient_controls": colormap.gradient_controls,
                "name": colormap_name,
            },
        )

    async def test_create_discrete_colormap(self) -> None:
        post_colormap_response = load_test_data("discrete_colormap_response.json")
        colormap = DiscreteColormap(
            colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
            end_inclusive=[True, False],
            end_points=[0.1, 0.2],
        )
        colormap_name = "discrete colormap 1"
        expected_colormap_metadata = ColormapMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            colormap=colormap,
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name=colormap_name,
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{UUID(int=7)}",
            id=UUID(int=7),
        )
        with self.transport.set_http_response(
            status_code=201,
            content=json.dumps(post_colormap_response),
            headers={"Content-Type": "application/json"},
        ):
            colormap_metadata = await self.colormap_api_client.create_colormap(colormap=colormap, name=colormap_name)

        self.assertIsInstance(colormap_metadata, ColormapMetadata)
        self.assertEqual(colormap_metadata, expected_colormap_metadata)
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/colormaps",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "colors": colormap.colors,
                "dtype": "discrete",
                "end_inclusive": colormap.end_inclusive,
                "end_points": colormap.end_points,
                "name": colormap_name,
            },
        )

    async def test_create_category_colormap(self) -> None:
        post_colormap_response = load_test_data("category_colormap_response.json")
        colormap = CategoryColormap(
            colors=[[0, 0, 219], [0, 217, 0], [217, 0, 0]],
            map=["a", "b", "c"],
        )
        colormap_name = "category colormap 1"
        expected_colormap_metadata = ColormapMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            colormap=colormap,
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name=colormap_name,
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{UUID(int=8)}",
            id=UUID(int=8),
        )
        with self.transport.set_http_response(
            status_code=201,
            content=json.dumps(post_colormap_response),
            headers={"Content-Type": "application/json"},
        ):
            colormap_metadata = await self.colormap_api_client.create_colormap(colormap=colormap, name=colormap_name)

        self.assertIsInstance(colormap_metadata, ColormapMetadata)
        self.assertEqual(colormap_metadata, expected_colormap_metadata)
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/colormaps",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "colors": colormap.colors,
                "dtype": "category",
                "map": colormap.map,
                "name": colormap_name,
            },
        )

    async def test_create_duplicate_colourmap(self) -> None:
        post_colormap_response = load_test_data("continuous_colormap_response.json")
        colormap = ContinuousColormap(
            colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
            attribute_controls=[0.1, 0.2, 0.3],
            gradient_controls=[0.4, 0.5, 0.6],
        )
        colormap_name = "continuous colormap 1"
        expected_colormap_metadata = ColormapMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            colormap=colormap,
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name=colormap_name,
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{UUID(int=6)}",
            id=UUID(int=6),
        )
        with self.transport.set_http_response(
            status_code=201,
            content=json.dumps(post_colormap_response),
            headers={"Content-Type": "application/json"},
        ):
            colormap_metadata = await self.colormap_api_client.create_colormap(colormap=colormap, name=colormap_name)

        self.assertIsInstance(colormap_metadata, ColormapMetadata)
        self.assertEqual(colormap_metadata, expected_colormap_metadata)
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/colormaps",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "attribute_controls": colormap.attribute_controls,
                "colors": colormap.colors,
                "dtype": "continuous",
                "gradient_controls": colormap.gradient_controls,
                "name": colormap_name,
            },
        )

    async def test_create_colormap_unknown_type(self) -> None:
        colormap = "not a colormap"
        with self.assertRaises(UnknownColormapType):
            await self.colormap_api_client.create_colormap(colormap=colormap, name="colormap name")

    async def assert_get_colormap_by_id(self, colormap_id, expected_colormap_metadata, get_colormap_response):
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_colormap_response),
            headers={"Content-Type": "application/json"},
        ):
            colormap_metadata = await self.colormap_api_client.get_colormap_by_id(colormap_id)
        self.assertIsInstance(colormap_metadata, ColormapMetadata)
        self.assertEqual(colormap_metadata, expected_colormap_metadata)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/colormaps/{colormap_id}",
            headers={"Accept": "application/json"},
        )

    async def test_get_colormap_by_id_continuous(self) -> None:
        get_colormap_response = load_test_data("continuous_colormap_response.json")
        colormap_id = UUID(int=6)
        expected_colormap_metadata = ColormapMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            colormap=ContinuousColormap(
                colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
                attribute_controls=[0.1, 0.2, 0.3],
                gradient_controls=[0.4, 0.5, 0.6],
            ),
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name="continuous colormap 1",
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
            id=colormap_id,
        )
        await self.assert_get_colormap_by_id(colormap_id, expected_colormap_metadata, get_colormap_response)

    async def test_get_colormap_by_id_discrete(self) -> None:
        get_colormap_response = load_test_data("discrete_colormap_response.json")
        colormap_id = UUID(int=7)
        expected_colormap_metadata = ColormapMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            colormap=DiscreteColormap(
                colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
                end_inclusive=[True, False],
                end_points=[0.1, 0.2],
            ),
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name="discrete colormap 1",
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
            id=colormap_id,
        )
        await self.assert_get_colormap_by_id(colormap_id, expected_colormap_metadata, get_colormap_response)

    async def test_get_colormap_by_id_category(self) -> None:
        get_colormap_response = load_test_data("category_colormap_response.json")
        colormap_id = UUID(int=8)
        expected_colormap_metadata = ColormapMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            colormap=CategoryColormap(
                colors=[[0, 0, 219], [0, 217, 0], [217, 0, 0]],
                map=["a", "b", "c"],
            ),
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name="category colormap 1",
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
            id=colormap_id,
        )
        await self.assert_get_colormap_by_id(colormap_id, expected_colormap_metadata, get_colormap_response)

    async def test_get_colormap_by_id_does_not_exist(self) -> None:
        colormap_id = UUID(int=9)
        with self.transport.set_http_response(status_code=404, content=""):
            with self.assertRaises(NotFoundException):
                await self.colormap_api_client.get_colormap_by_id(colormap_id)

    async def test_get_colormap_collection(self) -> None:
        get_colormap_collection_response = load_test_data("colormap_collection_response.json")
        expected_colormap_metadata = [
            ColormapMetadata(
                environment=self.environment,
                created_at=utc_datetime(2024, 9, 16, 1, 30),
                created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                colormap=ContinuousColormap(
                    colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
                    attribute_controls=[0.1, 0.2, 0.3],
                    gradient_controls=[0.4, 0.5, 0.6],
                ),
                modified_at=utc_datetime(2024, 9, 16, 1, 30),
                modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                name="continuous colormap 1",
                self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{UUID(int=6)}",
                id=UUID(int=6),
            ),
            ColormapMetadata(
                environment=self.environment,
                created_at=utc_datetime(2024, 9, 16, 1, 30),
                created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                colormap=DiscreteColormap(
                    colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
                    end_inclusive=[True, False],
                    end_points=[0.1, 0.2],
                ),
                modified_at=utc_datetime(2024, 9, 16, 1, 30),
                modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                name="discrete colormap 1",
                self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{UUID(int=7)}",
                id=UUID(int=7),
            ),
            ColormapMetadata(
                environment=self.environment,
                created_at=utc_datetime(2024, 9, 16, 1, 30),
                created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                colormap=CategoryColormap(
                    colors=[[255, 240, 219], [238, 217, 196], [217, 185, 155]],
                    map=["a", "b", "c"],
                ),
                modified_at=utc_datetime(2024, 9, 16, 1, 30),
                modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                name="category colormap 1",
                self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{UUID(int=8)}",
                id=UUID(int=8),
            ),
        ]
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(get_colormap_collection_response),
            headers={"Content-Type": "application/json"},
        ):
            colormap_metadata = await self.colormap_api_client.get_colormap_collection()

        self.assertIsInstance(colormap_metadata, list)
        self.assertEqual(colormap_metadata, expected_colormap_metadata)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/colormaps",
            headers={"Accept": "application/json"},
        )

    async def test_create_association(self) -> None:
        colormap_association_response = load_test_data("colormap_association_response.json")
        colormap_id = UUID(int=6)
        attribute_id = "a very unique ID"
        association = Association(attribute_id=attribute_id, colormap_id=colormap_id)
        object_id = UUID(int=20)
        association_id = UUID(int=30)

        association_metadata = AssociationMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name=f"Association between {attribute_id} and {colormap_id}",
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/objects/{object_id}/associations/{association_id}",
            id=association_id,
            attribute_id=attribute_id,
            object_id=object_id,
            colormap_id=colormap_id,
            colormap_uri=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
        )
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(colormap_association_response),
            headers={"Content-Type": "application/json"},
        ):
            association = await self.colormap_api_client.create_association(object_id, association)

        self.assertIsInstance(association, AssociationMetadata)
        self.assertEqual(association, association_metadata)
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{object_id}/associations",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={"attribute_id": str(attribute_id), "colormap_id": str(colormap_id)},
        )

    async def test_create_batch_associations(self) -> None:
        colormap_association_response = load_test_data("colormap_association_collection_response.json")
        object_id = UUID(int=20)
        association_ids = [UUID(int=30), UUID(int=31), UUID(int=32)]
        attribute_ids = ["a fairly unique ID", str(UUID(int=2)), "another fairly unique ID"]
        colormap_ids = [UUID(int=6), UUID(int=7), UUID(int=8)]
        associations = [
            Association(attribute_id=attribute_id, colormap_id=colormap_id)
            for attribute_id, colormap_id in zip(attribute_ids, colormap_ids)
        ]

        expected_associations = [
            AssociationMetadata(
                environment=self.environment,
                created_at=utc_datetime(2024, 9, 16, 1, 30),
                created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                modified_at=utc_datetime(2024, 9, 16, 1, 30),
                modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                name=f"Association between {attribute_id} and {colormap_id}",
                self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/objects/{object_id}/associations/{association_id}",
                id=association_id,
                colormap_id=colormap_id,
                attribute_id=attribute_id,
                object_id=object_id,
                colormap_uri=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
            )
            for association_id, attribute_id, colormap_id in zip(association_ids, attribute_ids, colormap_ids)
        ]
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(colormap_association_response),
            headers={"Content-Type": "application/json"},
        ):
            associations = await self.colormap_api_client.create_batch_associations(object_id, associations)

        self.assertIsInstance(associations, list)
        self.assertEqual(associations, expected_associations)
        self.assert_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{object_id}/associations/batch",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "associations": [
                    {"attribute_id": attribute_id, "colormap_id": str(colormap_id)}
                    for attribute_id, colormap_id in zip(attribute_ids, colormap_ids)
                ]
            },
        )

    async def test_create_batch_associations_chunked(self) -> None:
        colormap_association_response = load_test_data("colormap_association_collection_response.json")
        object_id = UUID(int=20)
        associations = [Association(attribute_id=f"attribute_id_{i}", colormap_id=UUID(int=i)) for i in range(500)]
        association_ids = [UUID(int=30), UUID(int=31), UUID(int=32)]
        attribute_ids = ["a fairly unique ID", str(UUID(int=2)), "another fairly unique ID"]
        colormap_ids = [UUID(int=6), UUID(int=7), UUID(int=8)]
        expected_associations = [
            AssociationMetadata(
                environment=self.environment,
                created_at=utc_datetime(2024, 9, 16, 1, 30),
                created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                modified_at=utc_datetime(2024, 9, 16, 1, 30),
                modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                name=f"Association between {attribute_id} and {colormap_id}",
                self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/objects/{object_id}/associations/{association_id}",
                id=association_id,
                colormap_id=colormap_id,
                attribute_id=attribute_id,
                object_id=object_id,
                colormap_uri=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
            )
            for association_id, attribute_id, colormap_id in zip(association_ids, attribute_ids, colormap_ids)
        ] * 4  # The requests should be chunked 4 times and stitch the results back together
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(colormap_association_response),
            headers={"Content-Type": "application/json"},
        ):
            associations = await self.colormap_api_client.create_batch_associations(object_id, associations)

        self.assertIsInstance(associations, list)
        self.assertEqual(associations, expected_associations)
        self.assert_any_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{object_id}/associations/batch",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "associations": [
                    {"attribute_id": f"attribute_id_{i}", "colormap_id": str(UUID(int=i))} for i in range(128)
                ]
            },
        )
        self.assert_any_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{object_id}/associations/batch",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "associations": [
                    {"attribute_id": f"attribute_id_{i}", "colormap_id": str(UUID(int=i))} for i in range(128, 256)
                ]
            },
        )
        self.assert_any_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{object_id}/associations/batch",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "associations": [
                    {"attribute_id": f"attribute_id_{i}", "colormap_id": str(UUID(int=i))} for i in range(256, 384)
                ]
            },
        )
        self.assert_any_request_made(
            method=RequestMethod.POST,
            path=f"{self.base_path}/objects/{object_id}/associations/batch",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body={
                "associations": [
                    {"attribute_id": f"attribute_id_{i}", "colormap_id": str(UUID(int=i))} for i in range(384, 500)
                ]
            },
        )

    async def test_create_batch_associations_empty(self) -> None:
        object_id = UUID(int=20)
        associations = []
        with self.assertRaises(ValueError):
            await self.colormap_api_client.create_batch_associations(object_id, associations)

    async def test_get_association(self) -> None:
        colormap_association_response = load_test_data("colormap_association_response.json")
        colormap_id = UUID(int=6)
        attribute_id = "a very unique ID"
        object_id = UUID(int=20)
        association_id = UUID(int=30)

        expected_association_metadata = AssociationMetadata(
            environment=self.environment,
            created_at=utc_datetime(2024, 9, 16, 1, 30),
            created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            modified_at=utc_datetime(2024, 9, 16, 1, 30),
            modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
            name=f"Association between {attribute_id} and {colormap_id}",
            self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/objects/{object_id}/associations/{association_id}",
            id=association_id,
            colormap_id=colormap_id,
            attribute_id=attribute_id,
            object_id=object_id,
            colormap_uri=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
        )
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(colormap_association_response),
            headers={"Content-Type": "application/json"},
        ):
            association = await self.colormap_api_client.get_association(object_id, association_id)

        self.assertIsInstance(association, AssociationMetadata)
        self.assertEqual(association, expected_association_metadata)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{object_id}/associations/{association_id}",
            headers={"Accept": "application/json"},
        )

    async def test_get_association_collection(self) -> None:
        colormap_association_response = load_test_data("colormap_association_collection_response.json")
        object_id = UUID(int=20)
        association_ids = [UUID(int=30), UUID(int=31), UUID(int=32)]
        attribute_ids = ["a fairly unique ID", str(UUID(int=2)), "another fairly unique ID"]
        colormap_ids = [UUID(int=6), UUID(int=7), UUID(int=8)]

        expected_associations = [
            AssociationMetadata(
                environment=self.environment,
                created_at=utc_datetime(2024, 9, 16, 1, 30),
                created_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                modified_at=utc_datetime(2024, 9, 16, 1, 30),
                modified_by=ServiceUser(id=UUID(int=16), name=None, email=None),
                name=f"Association between {attribute_id} and {colormap_id}",
                self_link=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/objects/{object_id}/associations/{association_id}",
                id=association_id,
                colormap_id=colormap_id,
                attribute_id=attribute_id,
                object_id=object_id,
                colormap_uri=f"{BASE_URL}colormap/orgs/{ORG.id}/workspaces/{WORKSPACE_ID}/colormaps/{colormap_id}",
            )
            for association_id, attribute_id, colormap_id in zip(association_ids, attribute_ids, colormap_ids)
        ]
        with self.transport.set_http_response(
            status_code=200,
            content=json.dumps(colormap_association_response),
            headers={"Content-Type": "application/json"},
        ):
            associations = await self.colormap_api_client.get_association_collection(object_id)

        self.assertIsInstance(associations, list)
        self.assertEqual(associations, expected_associations)
        self.assert_request_made(
            method=RequestMethod.GET,
            path=f"{self.base_path}/objects/{object_id}/associations",
            headers={"Accept": "application/json"},
        )

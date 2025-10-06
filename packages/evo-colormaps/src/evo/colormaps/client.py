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

from datetime import datetime, timezone
from uuid import UUID

from evo import logging
from evo.common import (
    APIConnector,
    BaseAPIClient,
    Environment,
    HealthCheckType,
    ServiceHealth,
    ServiceUser,
)
from evo.common.utils import get_service_health

from .data import (
    Association,
    AssociationMetadata,
    CategoryColormap,
    ColormapMetadata,
    ContinuousColormap,
    DiscreteColormap,
)
from .endpoints.api import AssociationsApi, ColormapsApi
from .endpoints.models import (
    AttributeAssociationData,
    AttributeAssociationsData,
    AttributeColormapAssociation,
    CategoryColormapData,
    CategoryColormapResponse,
    ColormapData,
    ColormapTypeEnum,
    ContinuousColormapData,
    ContinuousColormapResponse,
    DiscreteColormapData,
    DiscreteColormapResponse,
)
from .exceptions import UnknownColormapResponse, UnknownColormapType

logger = logging.getLogger("colormap.client")

__all__ = [
    "ColormapAPIClient",
]


def _colors_from_endpoint_model(
    model: ContinuousColormapResponse | DiscreteColormapResponse | CategoryColormapResponse,
) -> list[list[int]]:
    """Get colors list from an endpoint model."""
    return [[channel.root for channel in color.root] for color in model.colors]


def _with_timezone(dt: datetime) -> datetime:
    """Add timestamps to naive datetimes returned from the API."""
    if dt.tzinfo is not None:  # future proofing
        return dt
    return dt.replace(tzinfo=timezone.utc)


class ColormapAPIClient(BaseAPIClient):
    def __init__(self, environment: Environment, connector: APIConnector) -> None:
        """
        :param environment: The environment object
        :param connector: The connector object.
        """
        super().__init__(environment, connector)
        self._colourmaps_api = ColormapsApi(connector=connector)
        self._associations_api = AssociationsApi(connector=connector)

    def _colormap_metadata_from_endpoint_model(
        self, model: ContinuousColormapResponse | DiscreteColormapResponse | CategoryColormapResponse
    ) -> ColormapMetadata:
        """Creates a ColormapMetadata object from an endpoint model.

        :param model: Colormap response model from the endpoint.

        :return: A ColormapMetadata object.

        :raises UnknownColormapResponse: If the response model is not recognized.
        """
        match model:
            case ContinuousColormapResponse():
                colormap = ContinuousColormap(
                    colors=_colors_from_endpoint_model(model),
                    attribute_controls=model.attribute_controls,
                    gradient_controls=model.gradient_controls,
                )
            case DiscreteColormapResponse():
                colormap = DiscreteColormap(
                    colors=_colors_from_endpoint_model(model),
                    end_inclusive=model.end_inclusive,
                    end_points=model.end_points,
                )
            case CategoryColormapResponse():
                colormap = CategoryColormap(
                    colors=_colors_from_endpoint_model(model),
                    map=model.map,
                )
            case _:
                raise UnknownColormapResponse(f"Unexpected colormap response type: {type(model)}")

        return ColormapMetadata(
            environment=self._environment,
            id=model.id,
            name=model.name,
            created_at=_with_timezone(model.created_at),
            created_by=ServiceUser(id=model.created_by, name=None, email=None),
            modified_at=_with_timezone(model.modified_at),
            modified_by=ServiceUser(id=model.modified_by, name=None, email=None),
            colormap=colormap,
            self_link=str(model.self_link),
        )

    def _association_metadata_from_endpoint_model(
        self, model: AttributeColormapAssociation, object_id: UUID
    ) -> AssociationMetadata:
        """Creates an AssociationMetadata object from an endpoint model.

        :param model: Association response model from the endpoint.

        :return: An AssociationMetadata object.
        """
        return AssociationMetadata(
            environment=self._environment,
            id=model.id,
            created_at=_with_timezone(model.created_at),
            created_by=ServiceUser(id=model.created_by, name=None, email=None),
            modified_at=_with_timezone(model.modified_at),
            modified_by=ServiceUser(id=model.modified_by, name=None, email=None),
            attribute_id=model.attribute_id,
            colormap_id=model.colormap_id,
            object_id=object_id,
            colormap_uri=str(model.colormap_uri),
            self_link=str(model.self_link),
            name=f"Association between {model.attribute_id} and {model.colormap_id}",
        )

    async def get_service_health(self, check_type: HealthCheckType = HealthCheckType.FULL) -> ServiceHealth:
        """Get the health of the service.

        :param check_type: The type of health check to perform.

        :return: A ServiceHealth object.

        :raises EvoApiException: If the API returns an unexpected status code.
        :raises ClientValueError: If the response is not a valid service health check response.
        """
        return await get_service_health(self._connector, "visualization", check_type=check_type)

    async def create_colormap(
        self,
        colormap: ContinuousColormap | DiscreteColormap | CategoryColormap,
        name: str,
    ) -> ColormapMetadata:
        """Create a new colormap.

        :param colormap: The colormap data.
        :param name: The name of the colormap.

        :return: The created colormap.

        :raises UnknownColormapType: If the colormap type is not recognized.
        :raises EvoApiException: If the API returns an unexpected status code.
        """
        match colormap:
            case ContinuousColormap():
                colormap_data = ContinuousColormapData.model_validate(
                    {
                        "colors": colormap.colors,
                        "attribute_controls": colormap.attribute_controls,
                        "gradient_controls": colormap.gradient_controls,
                        "name": name,
                        "dtype": ColormapTypeEnum.continuous,
                    }
                )
            case DiscreteColormap():
                colormap_data = DiscreteColormapData.model_validate(
                    {
                        "colors": colormap.colors,
                        "end_inclusive": colormap.end_inclusive,
                        "end_points": colormap.end_points,
                        "name": name,
                        "dtype": ColormapTypeEnum.discrete,
                    }
                )
            case CategoryColormap():
                colormap_data = CategoryColormapData.model_validate(
                    {
                        "colors": colormap.colors,
                        "map": colormap.map,
                        "name": name,
                        "dtype": ColormapTypeEnum.category,
                    }
                )
            case _:
                raise UnknownColormapType(f"Unexpected colormap type: {type(colormap)}")

        response = await self._colourmaps_api.post_colormap(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            colormap_data=ColormapData.model_validate(colormap_data),
        )
        return self._colormap_metadata_from_endpoint_model(response.root)

    async def get_colormap_by_id(self, colormap_id: UUID) -> ColormapMetadata:
        """Get a colormap by ID.

        :param colormap_id: The UUID of the colormap.

        :return: A ColormapMetadata object representation of the colormap.

        :raises EvoApiException: If the API returns an unexpected status code.
        """
        response = await self._colourmaps_api.get_colormap_by_id(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            colormap_id=str(colormap_id),
        )
        return self._colormap_metadata_from_endpoint_model(response.root)

    async def get_colormap_collection(self) -> list[ColormapMetadata]:
        """Get all the colormaps in the current workspace.

        :return: A list of ColormapMetadata object representations of the colormaps.

        :raises EvoApiException: If the API returns an unexpected status code.
        """
        response = await self._colourmaps_api.get_colormap_collection(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
        )
        return [self._colormap_metadata_from_endpoint_model(model) for model in response.colormaps]

    async def create_association(self, object_id: UUID, association: Association) -> AssociationMetadata:
        """Associate an existing colormap with a geoscience object.

        :param object_id: The UUID of the geoscience object.
        :param association: The association data

        :return: An AssociationMetadata object representation of the association.

        :raises EvoApiException: If the API returns an unexpected status code.
        """
        response = await self._associations_api.post_association(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            attribute_association_data=AttributeAssociationData.model_validate(
                {"attribute_id": association.attribute_id, "colormap_id": str(association.colormap_id)}
            ),
            object_id=str(object_id),
        )
        return self._association_metadata_from_endpoint_model(response, object_id)

    async def create_batch_associations(
        self,
        object_id: UUID,
        associations: list[Association],
    ) -> list[AssociationMetadata]:
        """Create multiple associations for a colormap to multiple object attributes.

        :param object_id: The UUID of the geoscience object.
        :param associations: A list of dictionaries containing attribute_id and colormap_id pairs.

        :return: A list of AssociationMetadata object representations of the associations.

        :raises EvoApiException: If the API returns an unexpected status code.
        """
        if len(associations) == 0:
            raise ValueError("Associations list must not be empty.")

        def _chunk_list(lst):
            """Yield successive chunks from list."""
            chunk_size = 128
            for i in range(0, len(lst), chunk_size):
                yield lst[i : i + chunk_size]

        all_associations_metadata = []
        for chunk in _chunk_list(associations):
            attribute_associations_data = AttributeAssociationsData.model_validate(
                {
                    "associations": [
                        AttributeAssociationData.model_validate(
                            {"attribute_id": association.attribute_id, "colormap_id": str(association.colormap_id)}
                        )
                        for association in chunk
                    ]
                }
            )

            response = await self._associations_api.post_batch_associations(
                org_id=str(self._environment.org_id),
                workspace_id=str(self._environment.workspace_id),
                object_id=str(object_id),
                attribute_associations_data=attribute_associations_data,
            )
            all_associations_metadata.extend(
                self._association_metadata_from_endpoint_model(model, object_id) for model in response.associations
            )

        return all_associations_metadata

    async def get_association(self, object_id: UUID, association_id: UUID) -> AssociationMetadata:
        """Get the specific colormap association for a geoscience object given an association UUID.

        :param object_id: The UUID of the geoscience object.
        :param association_id: The UUID of the association.

        :return: An AssociationMetadata object representation of the association.

        :raises EvoApiException: If the API returns an unexpected status code.
        """
        response = await self._associations_api.get_association_by_id(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            object_id=str(object_id),
            association_id=str(association_id),
        )
        return self._association_metadata_from_endpoint_model(response, object_id)

    async def get_association_collection(self, object_id: UUID) -> list[AssociationMetadata]:
        """Get all associations for a geoscience object.

        :param object_id: The UUID of the geoscience object.

        :return: A list of AssociationMetadata object representations of the associations.

        :raises EvoApiException: If the API returns an unexpected status code.
        """
        response = await self._associations_api.get_association_collection(
            org_id=str(self._environment.org_id),
            workspace_id=str(self._environment.workspace_id),
            object_id=str(object_id),
        )
        return [self._association_metadata_from_endpoint_model(model, object_id) for model in response.associations]

"""
Workspaces API
=============

The Workspaces API enables users to organize, maintain, and store project data, but does not use or process the data. The workspace APIs allow you to manage:
- Workspaces
- User roles within workspaces
- Workspace thumbnails

There are three pre-defined roles within workspaces:

- Owner: can perform all actions in the workspace
- Editor: can perform all actions excluding deleting of a workspace
- Viewer: can view the workspace

These user roles can be assigned to users in a workspace. Once a role has been assigned it can be replaced or removed.
Users can also retrieve user roles, the role of a particular user, and their own role if applicable.
For more information on using the Workspaces API, see the [Workspaces API overview](/docs/workspaces/overview), or the API references here.


This code is generated from the OpenAPI specification for Workspaces API.
API version: 1.0
"""

from evo.common.connector import ApiConnector
from evo.common.data import EmptyResponse, RequestMethod

from ..models import *  # noqa: F403

__all__ = ["AdminApi"]


class AdminApi:
    """Api client for the Admin endpoint.

    NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    :param connector: Client for communicating with the API.
    """

    def __init__(self, connector: ApiConnector):
        self.connector = connector

    async def assign_user_role_admin(
        self,
        workspace_id: str,
        org_id: str,
        assign_role_request: AssignRoleRequest,  # noqa: F405
        preview_api: str | None = None,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> UserRole:  # noqa: F405
        """Assign user role

        Assigns a user a role in a workspace. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param workspace_id:
            Format: `uuid`
            Example: `'workspace_id_example'`
        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param assign_role_request:
            Example: `endpoints.AssignRoleRequest()`
        :param preview_api: (optional) Set to \"opt-in\" to enable adding user by email
            Example: `'preview_api_example'`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "workspace_id": workspace_id,
            "org_id": org_id,
        }

        # Prepare the header parameters.
        _header_params = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if preview_api is not None:
            _header_params["preview-api"] = preview_api
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "201": UserRole,  # noqa: F405
        }

        return await self.connector.call_api(
            method=RequestMethod.POST,
            resource_path="/workspace/admin/orgs/{org_id}/workspaces/{workspace_id}/users",
            path_params=_path_params,
            header_params=_header_params,
            body=assign_role_request,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def bulk_assign_roles_admin(
        self,
        org_id: str,
        bulk_user_role_assignments_request: BulkUserRoleAssignmentsRequest,  # noqa: F405
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> dict:
        """Bulk assign roles

        Assign multiple users a role in a workspace in a single request. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param bulk_user_role_assignments_request:
            Example: `endpoints.BulkUserRoleAssignmentsRequest()`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "org_id": org_id,
        }

        # Prepare the header parameters.
        _header_params = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": dict,
        }

        return await self.connector.call_api(
            method=RequestMethod.POST,
            resource_path="/workspace/admin/orgs/{org_id}/action/bulk_assign_roles",
            path_params=_path_params,
            header_params=_header_params,
            body=bulk_user_role_assignments_request,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def delete_user_role_admin(
        self,
        workspace_id: str,
        user_id: str,
        org_id: str,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> EmptyResponse:
        """Remove user from workspace

        Removes a user's role from a workspace. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param workspace_id:
            Format: `uuid`
            Example: `'workspace_id_example'`
        :param user_id:
            Format: `uuid`
            Example: `'user_id_example'`
        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "org_id": org_id,
        }

        # Prepare the header parameters.
        _header_params = {}
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "204": EmptyResponse,
        }

        return await self.connector.call_api(
            method=RequestMethod.DELETE,
            resource_path="/workspace/admin/orgs/{org_id}/workspaces/{workspace_id}/users/{user_id}",
            path_params=_path_params,
            header_params=_header_params,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def get_organization_settings(
        self,
        org_id: str,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> OrganizationSettingsResponse:  # noqa: F405
        """Get organization settings

        Returns org settings, provided your user has access to the org.

        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "org_id": org_id,
        }

        # Prepare the header parameters.
        _header_params = {
            "Accept": "application/json",
        }
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": OrganizationSettingsResponse,  # noqa: F405
        }

        return await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="/workspace/admin/orgs/{org_id}/settings",
            path_params=_path_params,
            header_params=_header_params,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def get_thumbnail_admin(
        self,
        workspace_id: str,
        org_id: str,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> bytearray:
        """Get thumbnail

        Returns the thumbnail image for a specified workspace. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param workspace_id:
            Format: `uuid`
            Example: `'workspace_id_example'`
        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "workspace_id": workspace_id,
            "org_id": org_id,
        }

        # Prepare the header parameters.
        _header_params = {
            "Accept": "image/jpeg",
        }
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": bytearray,
        }

        return await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="/workspace/admin/orgs/{org_id}/workspaces/{workspace_id}/thumbnail",
            path_params=_path_params,
            header_params=_header_params,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def get_workspace_admin(
        self,
        org_id: str,
        workspace_id: str,
        deleted: bool | None = None,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> WorkspaceRoleOptionalResponse:  # noqa: F405
        """Get workspace

        Get a workspace by its ID. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param workspace_id:
            Format: `uuid`
            Example: `'workspace_id_example'`
        :param deleted: (optional) Only list workspaces that have been deleted.
            Example: `False`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "org_id": org_id,
            "workspace_id": workspace_id,
        }

        # Prepare the query parameters.
        _query_params = {}
        if deleted is not None:
            _query_params["deleted"] = deleted

        # Prepare the header parameters.
        _header_params = {
            "Accept": "application/json",
        }
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": WorkspaceRoleOptionalResponse,  # noqa: F405
        }

        return await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="/workspace/admin/orgs/{org_id}/workspaces/{workspace_id}",
            path_params=_path_params,
            query_params=_query_params,
            header_params=_header_params,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def list_user_roles_admin(
        self,
        workspace_id: str,
        org_id: str,
        filter_user_id: str | None = None,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> ListUserRoleResponse:  # noqa: F405
        """List users

        List all users and their roles within a workspace. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param workspace_id:
            Format: `uuid`
            Example: `'workspace_id_example'`
        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param filter_user_id: (optional) Filter to see the role of a specific user ID.
            Format: `uuid`
            Example: `'filter_user_id_example'`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "workspace_id": workspace_id,
            "org_id": org_id,
        }

        # Prepare the query parameters.
        _query_params = {}
        if filter_user_id is not None:
            _query_params["filter[user_id]"] = filter_user_id

        # Prepare the header parameters.
        _header_params = {
            "Accept": "application/json",
        }
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": ListUserRoleResponse,  # noqa: F405
        }

        return await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="/workspace/admin/orgs/{org_id}/workspaces/{workspace_id}/users",
            path_params=_path_params,
            query_params=_query_params,
            header_params=_header_params,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def list_user_workspaces_admin(
        self,
        org_id: str,
        user_id: str,
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
        order_by: str | None = None,
        filter_created_by: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        filter_name: str | None = None,
        name: str | None = None,
        deleted: bool | None = None,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> ListUserWorkspacesResponse:  # noqa: F405
        """List user workspaces

        Returns a list of all workspaces that a specified user has a role i. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace..

        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param user_id:
            Format: `uuid`
            Example: `'user_id_example'`
        :param limit: (optional) The maximum number of results to return.
            Example: `20`
        :param offset: (optional) The (zero-based) offset of the first item returned in the collection.
            Example: `0`
        :param sort: (optional) An optional comma separated list of fields to sort the results by. Options are: `name`, `-name`, `created_at`, `-created_at`, `updated_at`, `-updated_at`, `user_role`, `-user_role`.
            Example: `'sort_example'`
        :param order_by: (optional) An optional, comma-separated list of fields by which to order the results. Each field could be prefixed with an order operator: `asc:` for ascending order or `desc:` for descending order, default is ascending order. The sortable fields are: `name`, `created_at`, `updated_at`, and `user_role`.
            Example: `'order_by_example'`
        :param filter_created_by: (optional) Filter by workspace that a user has created, by user ID.
            Format: `uuid`
            Example: `'filter_created_by_example'`
        :param created_at: (optional) Filter by the time workspace has created.
            Example: `'created_at_example'`
        :param updated_at: (optional) Filter by the latest time workspace was updated.
            Example: `'updated_at_example'`
        :param filter_name: (optional) Filter by workspace name.
            Example: `'filter_name_example'`
        :param name: (optional) Filter by workspace name.
            Example: `'name_example'`
        :param deleted: (optional) Include workspaces that have been deleted.
            Example: `True`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "org_id": org_id,
            "user_id": user_id,
        }

        # Prepare the query parameters.
        _query_params = {}
        if limit is not None:
            _query_params["limit"] = limit
        if offset is not None:
            _query_params["offset"] = offset
        if sort is not None:
            _query_params["sort"] = sort
        if order_by is not None:
            _query_params["order_by"] = order_by
        if filter_created_by is not None:
            _query_params["filter[created_by]"] = filter_created_by
        if created_at is not None:
            _query_params["created_at"] = created_at
        if updated_at is not None:
            _query_params["updated_at"] = updated_at
        if filter_name is not None:
            _query_params["filter[name]"] = filter_name
        if name is not None:
            _query_params["name"] = name
        if deleted is not None:
            _query_params["deleted"] = deleted

        # Prepare the header parameters.
        _header_params = {
            "Accept": "application/json",
        }
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": ListUserWorkspacesResponse,  # noqa: F405
        }

        return await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="/workspace/admin/orgs/{org_id}/users/{user_id}/workspaces",
            path_params=_path_params,
            query_params=_query_params,
            header_params=_header_params,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def list_workspaces_admin(
        self,
        org_id: str,
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
        order_by: str | None = None,
        filter_created_by: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        filter_name: str | None = None,
        name: str | None = None,
        deleted: bool | None = None,
        filter_user_id: str | None = None,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> ListWorkspacesResponse:  # noqa: F405
        """List workspaces

        Get a list of all workspaces. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param limit: (optional) The maximum number of results to return.
            Example: `20`
        :param offset: (optional) The (zero-based) offset of the first item returned in the collection.
            Example: `0`
        :param sort: (optional) An optional comma separated list of fields to sort the results by. Options are: `name`, `-name`, `created_at`, `-created_at`, `updated_at`, `-updated_at`, `user_role`, `-user_role`.
            Example: `'sort_example'`
        :param order_by: (optional) An optional, comma-separated list of fields by which to order the results. Each field could be prefixed with an order operator: `asc:` for ascending order or `desc:` for descending order, default is ascending order. The sortable fields are: `name`, `created_at`, `updated_at`, and `user_role`.
            Example: `'order_by_example'`
        :param filter_created_by: (optional) Filter by workspace that a user has created, by user ID.
            Format: `uuid`
            Example: `'filter_created_by_example'`
        :param created_at: (optional) Filter by the time workspace has created.
            Example: `'created_at_example'`
        :param updated_at: (optional) Filter by the latest time workspace was updated.
            Example: `'updated_at_example'`
        :param filter_name: (optional) Filter by workspace name.
            Example: `'filter_name_example'`
        :param name: (optional) Filter by workspace name.
            Example: `'name_example'`
        :param deleted: (optional) Include workspaces that have been deleted.
            Example: `True`
        :param filter_user_id: (optional) Filter by workspaces that a user ID has access to.
            Format: `uuid`
            Example: `'filter_user_id_example'`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "org_id": org_id,
        }

        # Prepare the query parameters.
        _query_params = {}
        if limit is not None:
            _query_params["limit"] = limit
        if offset is not None:
            _query_params["offset"] = offset
        if sort is not None:
            _query_params["sort"] = sort
        if order_by is not None:
            _query_params["order_by"] = order_by
        if filter_created_by is not None:
            _query_params["filter[created_by]"] = filter_created_by
        if created_at is not None:
            _query_params["created_at"] = created_at
        if updated_at is not None:
            _query_params["updated_at"] = updated_at
        if filter_name is not None:
            _query_params["filter[name]"] = filter_name
        if name is not None:
            _query_params["name"] = name
        if deleted is not None:
            _query_params["deleted"] = deleted
        if filter_user_id is not None:
            _query_params["filter[user_id]"] = filter_user_id

        # Prepare the header parameters.
        _header_params = {
            "Accept": "application/json",
        }
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": ListWorkspacesResponse,  # noqa: F405
        }

        return await self.connector.call_api(
            method=RequestMethod.GET,
            resource_path="/workspace/admin/orgs/{org_id}/workspaces",
            path_params=_path_params,
            query_params=_query_params,
            header_params=_header_params,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

    async def update_ml_enablement_admin(
        self,
        org_id: str,
        ml_enablement_request: MlEnablementRequest,  # noqa: F405
        preview_api: str | None = None,
        additional_headers: dict[str, str] | None = None,
        request_timeout: int | float | tuple[int | float, int | float] | None = None,
    ) -> MlEnablementRequest:  # noqa: F405
        """Update ML enablements

        Update ML (machine learning) enablement status for a group of given workspaces. Admin endpoints allow organization admin users to access any workspace, regardless of their role or lack thereof within the workspace.

        :param org_id:
            Format: `uuid`
            Example: `'org_id_example'`
        :param ml_enablement_request:
            Example: `endpoints.MlEnablementRequest()`
        :param preview_api: (optional) Set to \"opt-in\" to enable adding user by email
            Example: `'preview_api_example'`
        :param additional_headers: (optional) Additional headers to send with the request.
        :param request_timeout: (optional) Timeout setting for this request. If one number is provided, it will be the
            total request timeout. It can also be a pair (tuple) of (connection, read) timeouts.

        :return: Returns the result object.

        :raise evo.common.exceptions.BadRequestException: If the server responds with HTTP status 400.
        :raise evo.common.exceptions.UnauthorizedException: If the server responds with HTTP status 401.
        :raise evo.common.exceptions.ForbiddenException: If the server responds with HTTP status 403.
        :raise evo.common.exceptions.NotFoundException: If the server responds with HTTP status 404.
        :raise evo.common.exceptions.BaseTypedError: If the server responds with any other HTTP status between
            400 and 599, and the body of the response has a type.
        :raise evo.common.exceptions.EvoApiException: If the server responds with any other HTTP status between 400
            and 599, and the body of the response does not conform to RFC 87.
        :raise evo.common.exceptions.UnknownResponseError: For other HTTP status codes with no corresponding response
            type in `response_types_map`.
        """
        # Prepare the path parameters.
        _path_params = {
            "org_id": org_id,
        }

        # Prepare the header parameters.
        _header_params = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if preview_api is not None:
            _header_params["preview-api"] = preview_api
        if additional_headers is not None:
            _header_params.update(additional_headers)

        # Define the collection formats.
        _collection_formats = {}

        _response_types_map = {
            "200": MlEnablementRequest,  # noqa: F405
        }

        return await self.connector.call_api(
            method=RequestMethod.POST,
            resource_path="/workspace/admin/orgs/{org_id}/action/update_ml_enablement",
            path_params=_path_params,
            header_params=_header_params,
            body=ml_enablement_request,
            collection_formats=_collection_formats,
            response_types_map=_response_types_map,
            request_timeout=request_timeout,
        )

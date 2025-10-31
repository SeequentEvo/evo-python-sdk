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

from uuid import UUID

from evo.common.exceptions import CustomTypedError, EvoClientException


class SchemaIDFormatError(EvoClientException):
    """Unrecognized schema id format."""


class ObjectUUIDError(EvoClientException):
    """Object UUID is an unexpected value."""


class UnknownSchemaError(EvoClientException):
    """Unknown object schema."""

    def __init__(self, schema_id: str | None, detail: str):
        self.schema_id = schema_id
        self.detail = detail

    def __str__(self) -> str:
        return f"Could not resolve schema with id: '{self.schema_id}'\ndetail: {self.detail}"


class TableFormatError(EvoClientException):
    """The table structure does not match the expected Geoscience Object Schema data structure"""


class SchemaValidationError(EvoClientException):
    """Failed to validate data against a geoscience object schema"""


class ObjectAlreadyExistsError(CustomTypedError):
    TYPE_ID = "/geoscienceobject/objects/already-exists"

    @property
    def existing_id(self) -> UUID | None:
        if "existing_id" in self.content:
            return UUID(self.content["existing_id"])
        else:
            return None

    @property
    def object_path(self) -> str | None:
        if "object_path" in self.content:
            return str(self.content["object_path"])
        else:
            return None

    def __str__(self) -> str:
        error_message = super().__str__()
        if object_path := self.object_path:
            error_message += f"\nPath: {object_path}"
        if existing_id := self.existing_id:
            error_message += f"\nExisting ID: {existing_id}"
        return error_message


class ObjectModifiedError(CustomTypedError):
    """The object has been modified.

    Thia occurs when using the "If-Match" header with an outdated version.
    """

    TYPE_ID = "/geoscienceobject/objects/modified"

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

import sys

from .connector import APIConnector
from .data import Environment
from .interfaces import IContext

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ["BaseAPIClient"]


class BaseAPIClient:
    """Base class that may be used for individual API clients.

    Defines including cache management and environment variables.
    """

    def __init__(self, environment: Environment, connector: APIConnector) -> None:
        """
        :param environment: The environment object
        :param connector: The connector object.
        """
        self._environment = environment
        self._connector = connector

    @classmethod
    def from_context(cls, context: IContext) -> Self:
        """Creates an API client from the given context.

        :param context: The context to create the client from.
        :return: The created API client.
        """
        return cls(environment=context.get_environment(), connector=context.get_connector())

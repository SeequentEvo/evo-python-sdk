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

import functools
from importlib import metadata
from typing import NamedTuple

__all__ = ["get_header_metadata"]


class PackageDetails(NamedTuple):
    name: str
    version: str


@functools.cache
def get_header_metadata(candidate: str) -> dict[str, str]:
    """Get package name and version given the module __name__ for use in headers.

    :param candidate: The module __name__ to start searching from.

    :return: A dictionary containing the package name and version in a single entry.
    """
    package_details = get_package_details(candidate)
    return {package_details.name: package_details.version}


@functools.cache
def get_package_details(candidate: str) -> PackageDetails:
    """Get package name and version given the module __name__.

    Falls back to `evo-sdk-common` if no package is found for the provided module name.

    :param candidate: The module __name__ to start searching from.

    :return: A dictionary containing the package name and version.
    """
    while candidate:
        try:
            package_metadata = metadata.metadata(candidate)
        except metadata.PackageNotFoundError:
            candidate, *_ = candidate.rpartition(".")
        else:
            return PackageDetails(
                name=package_metadata["name"],
                version=package_metadata["version"],
            )

    return get_package_details("evo-sdk-common")

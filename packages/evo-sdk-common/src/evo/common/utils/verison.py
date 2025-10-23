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

__all__ = ["get_package_details"]


@functools.cache
def get_package_details(candidate: str) -> dict:
    """Get package name and version given the module __name__.

    :param candidate: The module __name__ to start searching from.

    :return: A dictionary containing the package name and version.

    :raises metadata.PackageNotFoundError: If no package could be found.
    """
    while candidate:
        try:
            package_metadata = metadata.metadata(candidate)
        except metadata.PackageNotFoundError:
            candidate, *_ = candidate.rpartition(".")
        else:
            return {
                "name": package_metadata["name"],
                "version": package_metadata["version"],
            }

    raise metadata.PackageNotFoundError(__package__)

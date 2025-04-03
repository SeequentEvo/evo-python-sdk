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

from importlib.abc import Traversable
from importlib.resources import files

_ROOT = files(__name__)


def get(filename: str) -> Traversable:
    """Get the path to a file in this directory.

    :param filename: The name of the file.

    :return: A Traversable object representing the file.
    """
    return _ROOT / filename

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

import os
from pathlib import Path
from typing import TypeAlias

from evo import logging

from .exceptions import FileNameTooLongError

__all__ = [
    "PathLike",
    "resolve_path",
]

logger = logging.getLogger("types")

PathLike: TypeAlias = str | os.PathLike | Path


MAX_WINDOWS_PATH_LEN = 259
FILE_NAME_TOO_LONG_ERR_NO = 63


def _check_path_length(path: Path) -> None:
    if os.name == "posix":
        try:
            # https://linux.die.net/man/3/pathconf
            max_path_length = os.pathconf(path, "PC_PATH_MAX")
        except FileNotFoundError:
            # If the path does not exist, we cannot check its length
            logger.info("File does not exist.", exc_info=True)
            max_path_length = 1024  # Default value for POSIX systems
        except ValueError:
            logger.error("Error checking path length. Assuming path is not too long.", exc_info=True)
            return
        except OSError as e:
            if e.errno == FILE_NAME_TOO_LONG_ERR_NO:
                raise FileNameTooLongError() from e

            logger.error("Error checking path length. Assuming path is not too long.", exc_info=True)
            return

    elif os.name == "nt":
        max_path_length = MAX_WINDOWS_PATH_LEN
    else:
        logger.error("Unsupported operating system. Assuming path is not too long.")
        return

    if len(str(path)) > max_path_length:
        raise FileNameTooLongError()


def resolve_path(path: PathLike, *, check_path_length: bool = False) -> Path:
    """Resolve the given path and check for maximum path length.

    :param path: The path to resolve.
    :param check_path_length: If True, check the path length.

    :returns: The resolved path.

    :raises TypeError: If the path is not a PathLike object.
    :raises FileNameTooLongError: If check_path_length is True and the path is too long.
    """
    if not isinstance(path, Path):
        try:
            path = Path(path)
        except TypeError:
            raise TypeError(f"Expected a PathLike object, got {path!r}")

    resolved = path.resolve()
    if check_path_length:
        _check_path_length(resolved)
    return resolved

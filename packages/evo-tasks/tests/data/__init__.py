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
from pathlib import Path

_THIS_DIR = Path(__file__).parent.resolve()


def _load_json_data(target_file: Path) -> list | dict:
    with target_file.open("r") as json_file:
        return json.load(json_file)


def load_test_data(filename: str) -> list | dict:
    target_file = (_THIS_DIR / filename).resolve()

    # Test data must live in test data directory.
    if not target_file.is_relative_to(_THIS_DIR):
        raise RuntimeError("Cannot access files outside test data directory.")

    # File must exist.
    if not target_file.exists():
        raise FileNotFoundError(f"Unknown file '{target_file}'")

    match target_file.suffix.lower():
        case ".json":
            return _load_json_data(target_file)
        case ext:
            raise ValueError(f"Unsupported data file type '{ext}'")

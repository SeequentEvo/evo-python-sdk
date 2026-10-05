#  Copyright © 2026 Bentley Systems, Incorporated
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

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"


def find_packages() -> list[Path]:
    return sorted(package_file.parent for package_file in PACKAGES_DIR.glob("*/pyproject.toml"))


def main() -> int:
    failed_packages: list[str] = []

    for package_dir in find_packages():
        print(f"\n=== {package_dir.name} ===", flush=True)
        result = subprocess.run(["uv", "run", "pytest"], cwd=package_dir, check=False)
        if result.returncode:
            failed_packages.append(package_dir.name)

    if failed_packages:
        print(f"\nFailed packages: {', '.join(failed_packages)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

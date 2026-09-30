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

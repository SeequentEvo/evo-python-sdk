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

import re
import sys
from pathlib import Path

PYPI = "https://pypi.org/simple"

_SOURCE = re.compile(
    r"""
    ^\s*
    source \s* = \s*
    \{ \s* (?P<body>.*?) \s* }
    \s*$
""",
    re.VERBOSE,
)

_ALLOWED = re.compile(
    r"""
    registry \s* = \s* "(?P<registry>[^"]+)"
    | (?:editable|virtual) \s* = \s* "[^"]+"
""",
    re.VERBOSE,
)


def validate_lock_file(lockfile: Path) -> bool:
    for line in lockfile.read_text(encoding="utf-8").splitlines():
        match = _SOURCE.match(line)
        if match is None:
            continue
        body = match.group("body").strip()
        allowed_match = _ALLOWED.fullmatch(body)
        if allowed_match and (allowed_match.group("registry") is None or allowed_match.group("registry") == PYPI):
            continue
        return False
    return True


def main() -> int:
    lockfiles = sorted(Path.cwd().rglob("uv.lock"))
    bad_files = [f for f in lockfiles if not validate_lock_file(f)]
    if bad_files:
        print("uv.lock validation failed:", file=sys.stderr)
        for f in bad_files:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"Validated {len(lockfiles)} uv.lock file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

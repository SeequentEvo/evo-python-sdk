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

"""The override seam: one task at a time, a hand-written runner in place of the generic proxy.

The generic engine shapes every call from the task's published JSON Schema, which is all
most tasks need. A few want more than the schema can say -- a check that spans two
parameters, a result type with helpers of its own, a signature someone wrote and reviewed.
This is how such a task opts out without the engine growing a special case for it.

A module at ``evo/compute/overrides/<topic>/<task>.py`` exposing ``bind`` claims that task::

    def bind(client: ComputeClient, topic: str, task: str) -> Any: ...

``client.<topic>.<task>`` then returns whatever ``bind`` builds instead of the generic
:class:`~evo.compute.engine._TaskProxy`. Named calls through
:meth:`~evo.compute.engine.ComputeClient.arun` call the runner's ``run_parameters`` method,
which shares preparation and checks with its typed ``run`` method.

The module path is the registration: no decorator, no import of the overrides package from
the engine at start-up, and nothing to keep in step with the catalogue. ``<task>`` is the
task name as Python spells it (``normal-score`` -> ``normal_score``), which is also how the
caller spells it on the client.

An override uses the client's internal generic pipeline, so a job still carries
the same authorization, the same validation and the same reference resolution as any other.
What it replaces is the surface, not the plumbing.
"""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from pathlib import Path
from types import ModuleType

__all__ = [
    "load_override",
    "overridden_tasks",
]


def overridden_tasks() -> list[tuple[str, str]]:
    """Every ``(topic, task)`` a hand-written runner claims, as Python spells the names.

    Read from this package rather than from a task catalogue, because an override is a
    decision made in code: it applies to whatever the platform advertises under that name,
    and it has to be describable without anyone having fetched a catalogue first.
    """
    root = Path(__file__).resolve().parent
    claimed: list[tuple[str, str]] = []
    for topic in sorted(module.name for module in pkgutil.iter_modules([str(root)]) if module.ispkg):
        for task in sorted(module.name for module in pkgutil.iter_modules([str(root / topic)])):
            if load_override(topic, task) is not None:
                claimed.append((topic, task))
    return claimed


@lru_cache(maxsize=None)
def load_override(topic: str, task: str) -> ModuleType | None:
    """Import the override module for ``topic``/``task``, or return ``None`` if there is none.

    A :exc:`ModuleNotFoundError` naming a *different* module is re-raised: that is an
    override whose own imports are broken, and silently falling back to the generic proxy
    would turn a bug into a task that quietly behaves differently.

    Results are cached because this is called on every ``client.<topic>.<task>`` attribute
    access, and the common answer -- no override -- otherwise costs a failed import each
    time. Call ``load_override.cache_clear()`` after adding a module at runtime.
    """
    module_name = f"{__name__}.{topic}.{task}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if module_name == missing or module_name.startswith(f"{missing}."):
            return None
        raise
    return module if hasattr(module, "bind") else None

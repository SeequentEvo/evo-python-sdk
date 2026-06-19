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

"""Geostatistics compute tasks.

This package groups all geostatistics-topic compute tasks:

- **kriging** — Kriging interpolation
- **break_ties** — Spatial tie-breaking
- **conditioned_simulator** — Conditional turning-band simulation
- **conditional_turning_bands** — Conditional turning-band simulation (pre-computed distribution)
- **declustering** — Grid-based declustering weights
- **location_wise** — Per-location ensemble statistics
- **continuous_distribution** — Continuous non-parametric cumulative distribution
- **loss_calculation** — Classification by expected economic loss
- **normal_score** — Normal score transformation
- **profit_calculation** — Classification by expected economic profit
- **simulation_report** — Simulation result reporting

Task modules are imported here to trigger runner registration with the
central :class:`~evo.compute.tasks.common.runner.TaskRegistry`.

Example:
    >>> from evo.compute.tasks import run, SearchNeighborhood
    >>> from evo.compute.tasks.geostatistics.kriging import KrigingParameters
    >>>
    >>> result = await run(manager, KrigingParameters(...), preview=True)
"""

from . import break_ties as _break_ties_module  # noqa: F401
from . import conditional_turning_bands as _conditional_turning_bands_module  # noqa: F401
from . import conditioned_simulator as _conditioned_simulator_module  # noqa: F401
from . import continuous_distribution as _continuous_distribution_module  # noqa: F401
from . import declustering as _declustering_module  # noqa: F401
from . import kriging as _kriging_module  # noqa: F401
from . import location_wise as _location_wise_module  # noqa: F401
from . import loss_calculation as _loss_calculation_module  # noqa: F401
from . import normal_score as _normal_score_module  # noqa: F401
from . import profit_calculation as _profit_calculation_module  # noqa: F401
from . import simulation_report as _simulation_report_module  # noqa: F401
from .break_ties import BreakTiesResult
from .conditional_turning_bands import ConditionalTurningBandsResult
from .declustering import DeclusteringResult
from .kriging import (
    BlockDiscretisation,
    KrigingResult,
    RegionFilter,
)
from .location_wise import LocationWiseResult

__all__ = [
    "BlockDiscretisation",
    "BreakTiesResult",
    "ConditionalTurningBandsResult",
    "DeclusteringResult",
    "KrigingResult",
    "LocationWiseResult",
    "RegionFilter",
]

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

"""Profit calculation compute task client.

Classifies grid cells into material categories (ore/waste) based on ensemble
simulation data and economic parameters, maximising expected profit.

Uses the same parameter schema as loss-calculation but applies a different
economic calculation.

Example::

    >>> from evo.compute.tasks import run, Target
    >>> from evo.compute.tasks.geostatistics.profit_calculation import (
    ...     ProfitCalculationParameters,
    ...     MaterialCategory,
    ... )
    >>>
    >>> params = ProfitCalculationParameters(
    ...     source=grid.attributes["simulations"],
    ...     target=Target.new_attribute(grid, "profit_category"),
    ...     material_categories=[
    ...         MaterialCategory(cutoff_grade=0.0, metal_price=0, label="waste"),
    ...         MaterialCategory(cutoff_grade=0.5, metal_price=200, label="ore"),
    ...     ],
    ...     processing_cost=20.0,
    ...     mining_waste_cost=5.0,
    ...     mining_ore_cost=10.0,
    ...     metal_recovery_fraction=0.85,
    ... )
    >>> result = await run(manager, params)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from evo.common import IContext
from evo.objects.typed import BaseSpatialObject
from pydantic import BaseModel, Field, model_validator

from ..common import AnySourceAttribute, AnyTargetAttribute, SingleTargetTaskResult, TaskParameters
from ..common.results import TaskTarget
from ..common.runner import TaskRunner
from .conditioned_simulator import _validate_material_category_sequence
from .loss_calculation import MaterialCategory

__all__ = [
    "MaterialCategory",
    "ProfitCalculationParameters",
    "ProfitCalculationResult",
    "ProfitCalculationResultModel",
    "ProfitCalculationRunner",
]


class ProfitCalculationParameters(TaskParameters):
    """Parameters for the profit-calculation task.

    Same parameter schema as loss-calculation: classifies grid cells into
    material categories based on ensemble data and economic parameters.
    """

    _object_roles: ClassVar[Mapping[str, tuple[type, ...]]] = {
        "source": (BaseSpatialObject,),
        "target": (BaseSpatialObject,),
    }

    source: AnySourceAttribute
    """The source object and attribute containing ensemble simulation values."""

    target: AnyTargetAttribute
    """The target object and attribute to create or update with grade categories."""

    material_categories: list[MaterialCategory] = Field(min_length=1)
    """List of material categories, ordered from lowest to highest cutoff grade."""

    processing_cost: float = Field(ge=0.0)
    """Processing cost per unit of material."""

    mining_waste_cost: float = Field(ge=0.0)
    """Mining cost per unit of waste material."""

    mining_ore_cost: float = Field(ge=0.0)
    """Mining cost per unit of ore material."""

    metal_recovery_fraction: float = Field(ge=0.0, le=1.0)
    """Fraction of metal expected to be recovered during processing (0.0–1.0)."""

    @model_validator(mode="after")
    def _check_categories(self) -> ProfitCalculationParameters:
        _validate_material_category_sequence(self.material_categories)
        return self


class ProfitCalculationResultModel(BaseModel):
    """Pydantic model for the raw profit-calculation task result."""

    message: str
    target: TaskTarget


class ProfitCalculationResult(SingleTargetTaskResult[TaskTarget]):
    TASK_DISPLAY_NAME: ClassVar[str] = "Profit Calculation"

    def __init__(self, context: IContext, model: ProfitCalculationResultModel) -> None:
        super().__init__(context, model)

    def __str__(self) -> str:
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message:   {self.message}",
            f"  Target:    {self.target_name}",
            f"  Attribute: {self.attribute_name}",
        ]
        return "\n".join(lines)


class ProfitCalculationRunner(
    TaskRunner[ProfitCalculationParameters, ProfitCalculationResultModel, ProfitCalculationResult],
    topic="geostatistics",
    task="profit-calculation",
):
    """Runner for profit-calculation compute tasks.

    Automatically registered — used by ``run()`` for dispatch, or directly::

        result = await ProfitCalculationRunner(context, params)
    """

    async def _get_result(self, raw_result: ProfitCalculationResultModel) -> ProfitCalculationResult:
        return ProfitCalculationResult(self._context, raw_result)

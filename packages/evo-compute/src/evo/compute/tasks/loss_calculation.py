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

"""Loss calculation compute task client.

Classifies grid cells into material categories (ore/waste) based on ensemble
simulation data and economic parameters.  Each cell is assigned the category
that minimises the expected economic loss.

Example:
    >>> from evo.compute.tasks import run
    >>> from evo.compute.tasks.loss_calculation import (
    ...     LossCalculationParameters,
    ...     MaterialCategory,
    ... )
    >>>
    >>> params = LossCalculationParameters(
    ...     source=grid.attributes["simulations"],
    ...     target=Target.new_attribute(grid, "loss_category"),
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

from typing import ClassVar, Protocol, runtime_checkable

import pandas as pd
from evo.common import IContext, IFeedback
from evo.objects import ObjectSchema
from evo.objects.typed import BaseObject, object_from_reference
from pydantic import BaseModel, Field

from .common import AnySourceAttribute, AnyTargetAttribute
from .common.results import TaskAttribute, TaskTarget
from .common.runner import TaskRunner

__all__ = [
    "LossCalculationParameters",
    "LossCalculationResult",
    "LossCalculationResultModel",
    "LossCalculationRunner",
    "MaterialCategory",
]


class MaterialCategory(BaseModel):
    """A material category for grade calculations.

    Each category is defined by a cutoff grade, metal price, and label.
    Categories should be ordered from lowest to highest cutoff grade.
    """

    cutoff_grade: float = Field(0.0, ge=0.0)
    """The cutoff grade for this category."""

    metal_price: float = Field(0.0, ge=0.0)
    """The metal price per unit for this category."""

    label: str = "waste"
    """Label for the material category (e.g. 'ore', 'waste')."""


class LossCalculationParameters(BaseModel):
    """Parameters for the loss-calculation task.

    Classifies grid cells into material categories based on ensemble data
    and economic parameters, minimising expected economic loss.
    """

    source: AnySourceAttribute
    """The source object and attribute containing ensemble simulation values."""

    target: AnyTargetAttribute
    """The target object and attribute to create or update with grade categories."""

    material_categories: list[MaterialCategory]
    """List of material categories, ordered from lowest to highest cutoff grade."""

    processing_cost: float = Field(ge=0.0)
    """Processing cost per unit of material."""

    mining_waste_cost: float = Field(ge=0.0)
    """Mining cost per unit of waste material."""

    mining_ore_cost: float = Field(ge=0.0)
    """Mining cost per unit of ore material."""

    metal_recovery_fraction: float = Field(ge=0.0, le=1.0)
    """Fraction of metal expected to be recovered during processing (0.0–1.0)."""


class LossCalculationResultModel(BaseModel):
    """Pydantic model for the raw loss-calculation task result."""

    message: str
    target: TaskTarget


@runtime_checkable
class _ObjToDataframeProtocol(Protocol):
    async def to_dataframe(self, *keys: str, fb: IFeedback = ...) -> pd.DataFrame: ...


class LossCalculationResult:
    TASK_DISPLAY_NAME: ClassVar[str] = "Loss Calculation"

    def __init__(self, context: IContext, model: LossCalculationResultModel) -> None:
        self._target = model.target
        self._message = model.message
        self._context = context

    @property
    def message(self) -> str:
        return self._message

    @property
    def target_name(self) -> str:
        return self._target.name

    @property
    def target_reference(self) -> str:
        return self._target.reference

    @property
    def attribute_name(self) -> str:
        return self._target.attribute.name

    @property
    def schema(self) -> ObjectSchema:
        return ObjectSchema.from_id(self._target.schema_id)

    async def get_target_object(self) -> BaseObject:
        return await object_from_reference(self._context, self._target.reference)

    async def to_dataframe(self, *columns: str) -> pd.DataFrame:
        target_obj = await self.get_target_object()
        if isinstance(target_obj, _ObjToDataframeProtocol):
            return await target_obj.to_dataframe(*columns)
        raise TypeError(
            f"Don't know how to get DataFrame from {type(target_obj).__name__}. "
            "Use get_target_object() and access the data manually."
        )

    def __str__(self) -> str:
        lines = [
            f"✓ {self.TASK_DISPLAY_NAME} Result",
            f"  Message:   {self.message}",
            f"  Target:    {self.target_name}",
            f"  Attribute: {self.attribute_name}",
        ]
        return "\n".join(lines)


class LossCalculationRunner(
    TaskRunner[LossCalculationParameters, LossCalculationResultModel, LossCalculationResult],
    topic="geostatistics",
    task="loss-calculation",
):
    async def _get_result(self, raw_result: LossCalculationResultModel) -> LossCalculationResult:
        return LossCalculationResult(self._context, raw_result)

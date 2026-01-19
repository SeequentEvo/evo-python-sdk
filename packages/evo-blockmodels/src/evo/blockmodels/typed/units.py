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

"""Common unit IDs for block model attributes.

Unit IDs must match the values supported by the Block Model Service.
Use `get_available_units()` to retrieve the full list of available units from the service.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evo.common import IContext

__all__ = [
    "UnitType",
    "UnitInfo",
    "Units",
    "get_available_units",
]


class UnitType(Enum):
    """Types of units supported by the Block Model Service."""

    LENGTH = "LENGTH"
    MASS = "MASS"
    VOLUME = "VOLUME"
    VALUE = "VALUE"
    MASS_PER_VOLUME = "MASS_PER_VOLUME"
    MASS_PER_MASS = "MASS_PER_MASS"
    VOLUME_PER_VOLUME = "VOLUME_PER_VOLUME"
    VALUE_PER_MASS = "VALUE_PER_MASS"


@dataclass(frozen=True)
class UnitInfo:
    """Information about a unit."""

    unit_id: str
    """The unit ID to use when setting column units."""

    symbol: str
    """Display symbol for the unit."""

    description: str
    """Human-readable description of the unit."""

    unit_type: UnitType
    """The type/category of this unit."""

    conversion_factor: float
    """Conversion factor to the reference unit for this unit type."""


class Units:
    """Common unit IDs for block model attributes.

    These are the most commonly used unit IDs. For a complete list,
    use `get_available_units()` to query the Block Model Service.

    Example usage:
        from evo.blockmodels.typed import Units

        # Create block model with units
        bm_data = RegularBlockModelData(
            ...
            units={
                "grade": Units.GRAMS_PER_TONNE,
                "density": Units.TONNES_PER_CUBIC_METRE,
            },
        )

        # Add attribute with unit
        await bm_ref.add_attribute(df, "metal_content", unit=Units.KG_PER_CUBIC_METRE)
    """

    # Length units
    METRES = "m"
    CENTIMETRES = "cm"
    MILLIMETRES = "mm"
    KILOMETRES = "km"
    FEET = "ft"
    INCHES = "in"
    YARDS = "yd"
    MILES = "mi"

    # Mass units
    GRAMS = "g"
    KILOGRAMS = "kg"
    TONNES = "t"
    POUNDS = "lb"
    OUNCES = "oz"
    TROY_OUNCES = "oz_tr"

    # Volume units
    CUBIC_METRES = "m3"
    CUBIC_CENTIMETRES = "cm3"
    LITRES = "L"
    CUBIC_FEET = "ft3"
    GALLONS_US = "gal_us"

    # Mass per volume (density) units
    KG_PER_CUBIC_METRE = "kg/m3"
    GRAMS_PER_CUBIC_CENTIMETRE = "g/cm3"
    TONNES_PER_CUBIC_METRE = "t/m3"
    POUNDS_PER_CUBIC_FOOT = "lb/ft3"

    # Mass per mass (grade) units
    GRAMS_PER_TONNE = "g/t"
    KILOGRAMS_PER_TONNE = "kg/t"
    PERCENT = "%"
    PPM = "ppm"
    PPB = "ppb"
    OUNCES_PER_TONNE = "oz/t"
    TROY_OUNCES_PER_TONNE = "oz_tr/t"
    POUNDS_PER_TONNE = "lb/t"

    # Volume per volume units
    PERCENT_VOLUME = "%_vol"

    # Value per mass units
    DOLLARS_PER_TONNE = "$/t"
    DOLLARS_PER_OUNCE = "$/oz"


async def get_available_units(context: IContext) -> list[UnitInfo]:
    """Get the list of available units from the Block Model Service.

    :param context: The context containing environment and connector.
    :return: List of available units.

    Example:
        units = await get_available_units(manager)
        for unit in units:
            print(f"{unit.unit_id}: {unit.description} ({unit.symbol})")
    """
    from evo.blockmodels import BlockModelAPIClient

    client = BlockModelAPIClient.from_context(context)
    units_api = client._units_api

    units_response = await units_api.get_units(
        org_id=str(context.get_environment().org_id),
    )

    return [
        UnitInfo(
            unit_id=u.unit_id,
            symbol=u.symbol,
            description=u.description,
            unit_type=UnitType(u.unit_type.value),
            conversion_factor=u.conversion_factor,
        )
        for u in units_response
    ]


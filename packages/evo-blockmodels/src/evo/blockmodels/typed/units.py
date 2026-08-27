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

from evo.blockmodels import BlockModelAPIClient
from evo.common import IContext

__all__ = [
    "UnitInfo",
    "UnitType",
    "Units",
    "get_available_units",
]


class UnitType(Enum):
    """Types of units supported by the Block Model Service."""

    MASS_PER_MASS = "MASS_PER_MASS"
    MASS_PER_VOLUME = "MASS_PER_VOLUME"
    DIMENSIONLESS = "DIMENSIONLESS"
    MASS = "MASS"
    LENGTH = "LENGTH"
    VOLUME = "VOLUME"
    VALUE_PER_MASS = "VALUE_PER_MASS"
    VALUE = "VALUE"
    CATEGORY = "CATEGORY"
    ANGLE = "ANGLE"


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


class Units(str, Enum):
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
        await bm_ref.add_attribute(df, "metal_content", unit=Units.KILOS_PER_CUBIC_METRE)
    """

    # Mass per mass (grade) units
    CARATS_PER_HUNDRED_TONNE = "0.01 ct/t"
    CARATS_PER_TONNE = "ct/t"
    GRAMS_PER_TONNE = "g/t"
    MICROGRAMS_PER_GRAM = "ug/g"
    MICROGRAMS_PER_KILOGRAM = "ug/kg"
    MILLIGRAMS_PER_GRAM = "mg/g"
    MILLIGRAMS_PER_KILOGRAM = "mg/kg"
    PARTS_PER_BILLION = "ppb[mass]"
    PARTS_PER_MILLION = "ppm[mass]"
    PERCENT = "%[mass]"
    TROY_OUNCES_PER_SHORT_TON = "oz t/ton[US]"

    # Mass per volume (density) units
    GRAMS_PER_CUBIC_CENTIMETRE = "g/cm3"
    KILOS_PER_CUBIC_METRE = "kg/m3"
    POUNDS_PER_CUBIC_FOOT = "lbm/ft3"
    SHORT_TON_PER_CUBIC_FOOT = "ton[US]/ft3"
    TONNES_PER_CUBIC_METRE = "t/m3"

    # Dimensionless units
    COUNT = "count"
    PERCENTAGE = "%"
    RATIO = "ratio"

    # Mass units
    CARATS = "ct"
    GRAMS = "g"
    KILOGRAMS = "kg"
    KILOTONNES = "kt"
    MEGATONNES = "Mt"
    MICROGRAMS = "ug"
    MILLIGRAMS = "mg"
    MILLION_POUNDS = "Mlbm"
    MILLION_SHORT_TONS = "Mton[US]"
    MILLION_TROY_OUNCES = "1000000 ozm[troy]"
    POUNDS = "lbm"
    SHORT_TONS = "ton[US]"
    THOUSAND_CARATS = "1000 ct"
    THOUSAND_POUNDS = "klbm"
    THOUSAND_SHORT_TONS = "kton[US]"
    THOUSAND_TROY_OUNCES = "1000 ozm[troy]"
    TONNES = "t"
    TROY_OUNCES = "ozm[troy]"

    # Length units
    CENTIMETRES = "cm"
    FEET = "ft"
    METRES = "m"

    # Volume units
    CUBIC_CENTIMETRES = "cm3"
    CUBIC_FEET = "ft3"
    CUBIC_METRES = "m3"

    # Value units
    DOLLARS = "$"
    DOLLARS_PER_SHORT_TON = "$/ton[US]"
    DOLLARS_PER_TONNE = "$/t"

    # Category units
    NUMERIC_CATEGORY = "numeric_category"

    # Angle units
    DEGREES = "dega"
    RADIANS = "rad"
    REVOLUTIONS = "rev"


async def get_available_units(context: IContext) -> list[UnitInfo]:
    """Get the list of available units from the Block Model Service.

    :param context: The context containing environment and connector.
    :return: List of available units.

    Example:
        units = await get_available_units(manager)
        for unit in units:
            print(f"{unit.unit_id}: {unit.description} ({unit.symbol})")
    """
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

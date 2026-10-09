"""Block discretization shared by kriging and simulation tasks."""

from pydantic import BaseModel, Field

__all__ = ["BlockDiscretisation", "BlockDiscretization"]


class BlockDiscretization(BaseModel):
    """Subdivide each target block into nx * ny * nz sub-cells.

    Each dimension accepts integers from 1 to 9. The default of one in each
    direction is equivalent to point support.
    """

    nx: int = Field(1, ge=1, le=9)
    ny: int = Field(1, ge=1, le=9)
    nz: int = Field(1, ge=1, le=9)


# alias for British spelling used in legacy items
BlockDiscretisation = BlockDiscretization

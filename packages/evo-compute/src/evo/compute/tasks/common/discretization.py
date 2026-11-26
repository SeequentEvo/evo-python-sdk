"""Block discretization shared by kriging and simulation tasks."""

from pydantic import Field, StrictInt

from .models import TaskInputModel

__all__ = ["BlockDiscretization"]


class BlockDiscretization(TaskInputModel):
    """Subdivide each target block into nx * ny * nz sub-cells.

    Each dimension accepts integers from 1 to 9. The default of one in each
    direction is equivalent to point support.
    """

    nx: StrictInt = Field(1, ge=1, le=9)
    ny: StrictInt = Field(1, ge=1, le=9)
    nz: StrictInt = Field(1, ge=1, le=9)

from math import ceil
from typing import Any

import numpy as np
import pandas as pd

import evo.objects.typed as evo_objs
from evo.common import IContext
from evo.compute.tasks.geostatistics.conditioned_simulator import ConSimResult

STRUCTURE_TYPES: dict[str, type[evo_objs.VariogramStructure]] = {
    "spherical": evo_objs.SphericalStructure,
    "exponential": evo_objs.ExponentialStructure,
    "gaussian": evo_objs.GaussianStructure,
    "cubic": evo_objs.CubicStructure,
    "linear": evo_objs.LinearStructure,
    "spheroidal": evo_objs.SpheroidalStructure,
    "generalisedcauchy": evo_objs.GeneralisedCauchyStructure,
}


def variogram_structure(
    structure_type: str,
    *,
    contribution: float,
    major: float,
    semi_major: float,
    minor: float,
    rotation: tuple[float, float, float],
    **structure_options: Any,
) -> evo_objs.VariogramStructure:
    """Create a variogram structure by name, forwarding type-specific options."""
    name = structure_type.strip().lower()
    try:
        structure_class = STRUCTURE_TYPES[name]
    except KeyError:
        supported = ", ".join(STRUCTURE_TYPES)
        raise ValueError(f"Unknown structure type {structure_type!r}. Choose from: {supported}") from None

    return structure_class(
        contribution=contribution,
        anisotropy=evo_objs.Ellipsoid(
            ranges=evo_objs.EllipsoidRanges(
                major=major,
                semi_major=semi_major,
                minor=minor,
            ),
            rotation=evo_objs.Rotation(*rotation),
        ),
        **structure_options,
    )


def create_regular_grid_from_block_model(
    block_model: evo_objs.BlockModel,
    *,
    name: str,
    cell_size_scale: float | tuple[float, float, float] = 1.0,
    description: str | None = None,
) -> evo_objs.Regular3DGridData:
    """Build empty grid data covering a regular block model's extent without uploading.

    Preserve origin, rotation, and CRS. Scale cell sizes uniformly or per axis,
    rounding cell counts up to cover the model. No attribute data is copied.
    """
    scale = np.asarray(cell_size_scale, dtype=float)
    if scale.ndim == 0:
        scale = np.repeat(scale, 3)
    if scale.shape != (3,) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("cell_size_scale must be a positive finite number or three positive finite numbers")

    geometry = block_model.geometry
    return evo_objs.Regular3DGridData(
        name=name,
        description=description,
        origin=geometry.origin,
        rotation=geometry.rotation,
        coordinate_reference_system=block_model.coordinate_reference_system,
        cell_size=evo_objs.Size3d(
            dx=geometry.block_size.dx * scale[0],
            dy=geometry.block_size.dy * scale[1],
            dz=geometry.block_size.dz * scale[2],
        ),
        size=evo_objs.Size3i(
            nx=ceil(geometry.n_blocks.nx / scale[0]),
            ny=ceil(geometry.n_blocks.ny / scale[1]),
            nz=ceil(geometry.n_blocks.nz / scale[2]),
        ),
    )


def create_masked_grid_from_block_model(
    block_model: evo_objs.BlockModel,
    block_indices: pd.DataFrame,
    *,
    name: str,
    description: str | None = None,
) -> evo_objs.RegularMasked3DGridData:
    """Build masked grid data whose active cells are the selected block model blocks.

    Preserve origin, rotation, CRS, cell size, and extent, so each active cell is
    exactly one block. ``block_indices`` supplies the ``i``, ``j``, and ``k`` columns
    of the blocks to activate; every other cell stays inactive.
    """
    geometry = block_model.geometry
    size = evo_objs.Size3i(nx=geometry.n_blocks.nx, ny=geometry.n_blocks.ny, nz=geometry.n_blocks.nz)
    mask = np.zeros(size.total_size, dtype=bool)
    # Grid cells are stored X-fastest, so i/j/k map onto the mask in Fortran order.
    mask[
        np.ravel_multi_index(
            tuple(block_indices[axis].to_numpy() for axis in ("i", "j", "k")),
            (size.nx, size.ny, size.nz),
            order="F",
        )
    ] = True
    if not mask.any():
        raise ValueError("No blocks selected, so the masked grid would have no active cells")

    return evo_objs.RegularMasked3DGridData(
        name=name,
        description=description,
        origin=geometry.origin,
        rotation=geometry.rotation,
        coordinate_reference_system=block_model.coordinate_reference_system,
        cell_size=evo_objs.Size3d(
            dx=geometry.block_size.dx,
            dy=geometry.block_size.dy,
            dz=geometry.block_size.dz,
        ),
        size=size,
        mask=mask,
    )


async def simulation_summary_for_block_model(
    context: IContext,
    result: ConSimResult,
    block_model: evo_objs.BlockModel,
    *,
    prefix: str = "Cu_sim",
) -> pd.DataFrame:
    """Download simulation summaries with IJK columns for a matching block model.

    Read only: include mean, variance, min/max, quantiles, and cutoff products,
    preserving the masked grid's X-fastest active-cell order. Values retain their
    simulation units; variance has squared grade units. No realisations are copied.
    """
    grid = await evo_objs.object_from_reference(context, result.target_reference)
    if not isinstance(grid, evo_objs.RegularMasked3DGrid):
        raise ValueError("Expected a masked simulation grid")
    geometry = block_model.geometry
    if (
        grid.origin != geometry.origin
        or grid.rotation != geometry.rotation
        or grid.size != geometry.n_blocks
        or grid.cell_size != geometry.block_size
        or grid.coordinate_reference_system != block_model.coordinate_reference_system
    ):
        raise ValueError("Simulation grid and BlockSync model geometry must match")

    product_names = {
        getattr(result.summary_attributes, statistic).name: f"{prefix}_{statistic}"
        for statistic in ("mean", "variance", "min", "max")
    }
    product_names.update(
        {attribute.name: f"{prefix}_P{100 * attribute.quantile:g}" for attribute in result.quantile_attributes}
    )
    product_names.update(
        {
            attribute.name: f"{prefix}_prob_above_{attribute.cutoff:g}"
            for attribute in result.probability_above_cutoff_attributes
        }
    )
    product_names.update(
        {
            attribute.name: f"{prefix}_mean_above_{attribute.cutoff:g}"
            for attribute in result.mean_above_cutoff_attributes
        }
    )
    products = await grid.to_dataframe(*product_names)
    products = products.rename(columns=product_names).reset_index(drop=True)
    active_cells = np.flatnonzero(await grid.cells.get_mask())
    indices = np.column_stack(np.unravel_index(active_cells, (grid.size.nx, grid.size.ny, grid.size.nz), order="F"))
    if len(products) != len(indices):
        raise ValueError("Simulation product rows do not match the active mask cells")
    return pd.concat([pd.DataFrame(indices, columns=["i", "j", "k"], dtype="uint32"), products], axis=1)


def regular_grid_cell_coordinates(grid: evo_objs.Regular3DGrid) -> pd.DataFrame:
    """Return world-space cell centers in the grid's X-fastest attribute row order."""
    indices = np.column_stack(
        np.unravel_index(
            np.arange(grid.size.total_size),
            (grid.size.nx, grid.size.ny, grid.size.nz),
            order="F",
        )
    )
    coordinates = (indices + 0.5) * np.array([grid.cell_size.dx, grid.cell_size.dy, grid.cell_size.dz])
    if grid.rotation is not None:
        coordinates = coordinates @ grid.rotation.as_rotation_matrix().T
    coordinates += np.array([grid.origin.x, grid.origin.y, grid.origin.z])
    return pd.DataFrame(coordinates, columns=["x", "y", "z"])


def regular_grid_corners(grid: evo_objs.Regular3DGrid) -> np.ndarray:
    """Return the eight world-space corners of a regular grid, ordered by (x, y, z) bit pattern."""
    extent = np.array(
        [
            grid.size.nx * grid.cell_size.dx,
            grid.size.ny * grid.cell_size.dy,
            grid.size.nz * grid.cell_size.dz,
        ]
    )
    unit = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    corners = unit * extent
    if grid.rotation is not None:
        corners = corners @ grid.rotation.as_rotation_matrix().T
    return corners + np.array([grid.origin.x, grid.origin.y, grid.origin.z])


def compare_weighted_summary(values: pd.Series, weights: pd.Series) -> pd.DataFrame:
    """Compare original and IDW-declustered summary statistics."""
    mean = np.average(values, weights=weights)
    return pd.DataFrame(
        {
            "Original": [len(values), values.min(), values.mean(), values.var(ddof=0), values.max()],
            "IDW declustered": [
                len(values),
                values.min(),
                mean,
                np.average((values - mean) ** 2, weights=weights),
                values.max(),
            ],
        },
        index=["count", "min", "mean", "variance", "max"],
    )

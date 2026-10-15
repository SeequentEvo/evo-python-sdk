from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from evo.objects.typed import Variogram


def variogram_model(variogram: Variogram) -> go.Figure:
    """Plot variogram curves in each principal direction."""
    directions = variogram.get_principal_directions()
    return px.line(
        {
            "distance": np.concatenate([curve.distance for curve in directions]),
            "semivariance": np.concatenate([curve.semivariance for curve in directions]),
            "direction": np.concatenate([np.repeat(curve.direction, len(curve.distance)) for curve in directions]),
        },
        x="distance",
        y="semivariance",
        facet_col="direction",
        labels={"distance": "Distance", "semivariance": "Semivariance"},
        title="Variogram by principal direction",
        template="simple_white",
    )


def histogram_comparison(
    data: pd.DataFrame,
    *,
    x: str,
    color: str,
    weights: str | None = None,
    category_order: list[str] | None = None,
    colors: list[str] | None = None,
    title: str | None = None,
    labels: dict[str, str] | None = None,
    nbins: int = 50,
    histnorm: Literal["", "percent", "probability", "density", "probability density"] = "probability density",
    opacity: float = 0.8,
) -> go.Figure:
    """Compare distributions using shared bins, optionally summing a weights column."""
    fig = px.histogram(
        data,
        x=x,
        y=weights,
        histfunc="sum" if weights is not None else "count",
        color=color,
        category_orders={color: category_order} if category_order is not None else None,
        color_discrete_sequence=colors,
        barmode="overlay",
        opacity=opacity,
        nbins=nbins,
        histnorm=histnorm,
        title=title,
        labels=labels,
        template="simple_white",
    )
    fig.update_traces(bingroup="comparison")
    return fig


def swath_comparison(
    data: pd.DataFrame,
    *,
    x: str,
    value: str,
    color: str,
    bin_width: float,
    weights: str | None = None,
    title: str | None = None,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Compare weighted means in shared coordinate swaths, leaving empty bins as gaps.

    Exclude nonfinite coordinates, values, and weights, and zero-weight rows.
    Negative weights are rejected. Counts represent contributing rows, not support.
    """
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be positive and finite")
    frame = pd.DataFrame(
        {
            "coordinate": data[x],
            "value": data[value],
            "series": data[color],
            "weight": data[weights] if weights is not None else 1.0,
        }
    )
    if (frame["weight"] < 0).any():
        raise ValueError("Swath weights must be nonnegative")
    valid = np.isfinite(frame[["coordinate", "value", "weight"]]).all(axis=1)
    frame = frame.loc[valid & frame["series"].notna() & (frame["weight"] > 0)].copy()
    if frame.empty:
        raise ValueError("No finite, positive-weight data available for swath plotting")

    origin = np.floor(frame["coordinate"].min() / bin_width) * bin_width
    frame["swath"] = np.floor((frame["coordinate"] - origin) / bin_width).astype(int)
    frame["weighted_value"] = frame["value"] * frame["weight"]
    summary = frame.groupby(["series", "swath"], observed=True).agg(
        weighted_sum=("weighted_value", "sum"),
        weight_sum=("weight", "sum"),
        count=("value", "size"),
    )
    bins = pd.MultiIndex.from_product(
        [frame["series"].unique(), range(frame["swath"].max() + 1)], names=["series", "swath"]
    )
    summary = summary.reindex(bins).reset_index()
    summary["mean"] = summary["weighted_sum"] / summary["weight_sum"]
    summary["coordinate"] = origin + (summary["swath"] + 0.5) * bin_width
    summary["count"] = summary["count"].fillna(0).astype(int)
    fig = px.line(
        summary,
        x="coordinate",
        y="mean",
        color="series",
        markers=True,
        hover_data={"count": True, "weight_sum": ":.3f"},
        labels=labels or {"coordinate": x, "mean": value, "series": color},
        title=title,
        template="simple_white",
    )
    fig.update_traces(connectgaps=False)
    return fig


_BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def grid_and_samples_3d(
    corners: np.ndarray,
    samples: pd.DataFrame,
    *,
    value: str,
    cells: pd.DataFrame | None = None,
    cell_value: str | None = None,
    max_cells: int = 20000,
    title: str | None = None,
) -> go.Figure:
    """Show a grid's extent with sample points, optionally overlaying cell values on a shared colour scale."""
    edges = np.full((len(_BOX_EDGES) * 3, 3), np.nan)
    for position, (start, end) in enumerate(_BOX_EDGES):
        edges[3 * position] = corners[start]
        edges[3 * position + 1] = corners[end]

    overlay = cells is not None and cell_value is not None
    colored = [samples[value].to_numpy(dtype=float)]
    if overlay:
        colored.append(cells[cell_value].to_numpy(dtype=float))
    finite = np.concatenate(colored)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values available to colour the 3D plot")
    cmin, cmax = float(finite.min()), float(finite.max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=edges[:, 0],
            y=edges[:, 1],
            z=edges[:, 2],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Grid extent",
            hoverinfo="skip",
        )
    )
    if overlay:
        shown = cells if len(cells) <= max_cells else cells.sample(max_cells, random_state=0)
        fig.add_trace(
            go.Scatter3d(
                x=shown["x"],
                y=shown["y"],
                z=shown["z"],
                mode="markers",
                marker={
                    "size": 2,
                    "color": shown[cell_value],
                    "colorscale": "Viridis",
                    "cmin": cmin,
                    "cmax": cmax,
                    "opacity": 0.35,
                    "colorbar": {"title": cell_value},
                },
                name=f"Grid cells (n={len(shown)})",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=samples["x"],
            y=samples["y"],
            z=samples["z"],
            mode="markers",
            marker={
                "size": 4,
                "color": samples[value],
                "colorscale": "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "line": {"color": "black", "width": 0.5},
            },
            name="Composites",
        )
    )
    fig.update_layout(
        title=title,
        template="simple_white",
        scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
    )
    return fig

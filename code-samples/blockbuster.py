from dataclasses import dataclass
from math import floor
from uuid import UUID, uuid4

import asyncio
import random
import ipywidgets as widgets
import pandas as pd
import pyarrow
from evo.blockmodels import BlockModelAPIClient
from evo.blockmodels.endpoints import models
from evo.blockmodels.typed import MassUnits, ReportCategorySpec, ReportSpecificationData, Units
from evo.blockmodels.endpoints.models import ColumnHeaderType
from evo.common import Environment
from evo.notebooks import ServiceManagerWidget
from evo.objects.data import ObjectReference
from evo.objects.typed import BlockModel
from IPython.display import clear_output, display


@dataclass(frozen=True)
class GameConfig:
    """Fixed configuration for the Blockbuster game model."""

    model_origin: tuple[float, float, float] = (1478500, 5174500, 100)
    block_size: tuple[float, float, float] = (25, 25, 25)
    model_dimensions: tuple[int, int, int] = (48, 68, 40)
    target_column: str = "Classification"
    dirt_category: str = "Dirt"
    secret_domain_names: frozenset[str] = frozenset({"Unobtainium"})


DEFAULT_TEST_PLAYER_NAMES = (
    "Alex", "Blake", "Casey", "Devon", "Ellis",
    "Frankie", "Gray", "Harper", "Indigo", "Jordan",
)


async def connect_to_evo(client_id: str) -> tuple[ServiceManagerWidget, Environment, BlockModelAPIClient]:
    """Sign in and create a Block Model API client for the selected workspace."""
    manager = await ServiceManagerWidget.with_auth_code(client_id=client_id).login()
    environment = manager.get_environment()
    service_client = BlockModelAPIClient(environment, manager.get_connector(), manager.cache)
    return manager, environment, service_client


def validate_player_category(player_category: str, config: GameConfig) -> str:
    """Validate and normalize a player's category name."""
    normalized_category = player_category.strip()
    if not normalized_category:
        raise ValueError("Enter a participant name.")
    if normalized_category in config.secret_domain_names | {config.dirt_category}:
        raise ValueError("Choose a name that is not a secret domain or reserved category.")
    return normalized_category


@dataclass
class TurnResult:
    """Display-ready outcome of a Blockbuster turn."""

    min_point: tuple[float, float, float]
    max_point: tuple[float, float, float]
    dirt_blocks_cleared: int
    special_blocks_claimed: int
    remaining_blocks: pd.DataFrame


async def play_turn(
    service_client: BlockModelAPIClient,
    block_model_id: UUID,
    player_category: str,
    x: float,
    y: float,
    z: float,
    buffer_blocks: int,
    model_origin: list[float],
    block_size: list[float],
    model_dimensions: list[int],
    target_column: str,
    dirt_category: str,
    secret_domain_names: set[str],
) -> TurnResult:
    """Clear dirt and claim special targets within a clamped search box."""
    if buffer_blocks < 0:
        raise ValueError("BUFFER_BLOCKS must be zero or greater.")

    coordinates = (x, y, z)
    indices = tuple(
        floor((coordinate - model_origin[axis]) / block_size[axis])
        for axis, coordinate in enumerate(coordinates)
    )
    if any(index < 0 or index >= model_dimensions[axis] for axis, index in enumerate(indices)):
        raise ValueError("Choose coordinates inside the block model.")

    min_indices = tuple(max(0, index - buffer_blocks) for index in indices)
    max_indices = tuple(
        min(model_dimensions[axis] - 1, index + buffer_blocks)
        for axis, index in enumerate(indices)
    )

    def centroid(indices: tuple[int, int, int]) -> tuple[float, float, float]:
        return tuple(
            model_origin[axis] + indices[axis] * block_size[axis] + block_size[axis] / 2
            for axis in range(3)
        )

    min_point = centroid(min_indices)
    max_point = centroid(max_indices)

    scan = await service_client.query_block_model_as_table(
        bm_id=block_model_id,
        columns=[target_column],
        column_headers=ColumnHeaderType.name,
    )
    scan_df = scan.to_pandas()
    in_search_box = (
        scan_df["x"].between(min_point[0], max_point[0])
        & scan_df["y"].between(min_point[1], max_point[1])
        & scan_df["z"].between(min_point[2], max_point[2])
    )
    search_rows = scan_df.loc[in_search_box]
    dirt_blocks_cleared = int((search_rows[target_column] == dirt_category).sum())
    special_blocks_claimed = int(search_rows[target_column].isin(secret_domain_names).sum())

    target_updates = [
        None if in_box and target == dirt_category else
        player_category if in_box and target in secret_domain_names else
        None if pd.isna(target) else str(target)
        for target, in_box in zip(scan_df[target_column], in_search_box)
    ]
    conversion_data = pyarrow.table(
        {
            "x": scan_df["x"].tolist(),
            "y": scan_df["y"].tolist(),
            "z": scan_df["z"].tolist(),
            target_column: target_updates,
        },
        schema=pyarrow.schema(
            {
                "x": pyarrow.float64(),
                "y": pyarrow.float64(),
                "z": pyarrow.float64(),
                target_column: pyarrow.string(),
            }
        ),
    )
    await service_client.update_block_model_columns(
        bm_id=block_model_id,
        data=conversion_data,
        new_columns=[],
        update_columns={target_column},
        update_type=models.UpdateType.replace,
    )

    remaining_scan = await service_client.query_block_model_as_table(
        bm_id=block_model_id,
        columns=[target_column],
        column_headers=ColumnHeaderType.name,
    )
    remaining_df = remaining_scan.to_pandas()
    remaining_blocks = (
        remaining_df[remaining_df[target_column].isin(secret_domain_names)]
        .groupby(target_column)
        .size()
        .reindex(sorted(secret_domain_names), fill_value=0)
        .rename("Blocks still to claim")
        .reset_index()
    )
    return TurnResult(
        min_point=min_point,
        max_point=max_point,
        dirt_blocks_cleared=dirt_blocks_cleared,
        special_blocks_claimed=special_blocks_claimed,
        remaining_blocks=remaining_blocks,
    )


class BlockModelSelectorWidget(widgets.VBox):
    """Select an available block model and expose its UUID through ``value``."""

    def __init__(self, block_models: list[object]) -> None:
        options = [
            (f"{block_model.name} ({block_model.id})", block_model.id)
            for block_model in block_models
        ]
        self.selector = widgets.Dropdown(
            description="Block model:",
            options=options,
            layout=widgets.Layout(width="auto"),
        )
        super().__init__([self.selector])

    @property
    def value(self) -> UUID:
        """Return the UUID of the selected block model."""
        return self.selector.value

    @classmethod
    async def create(cls, service_client: BlockModelAPIClient) -> "BlockModelSelectorWidget":
        """Load available block models from Evo and build a selector."""
        block_models = await service_client.list_all_block_models()
        if not block_models:
            raise RuntimeError("No block models are available in the selected Evo workspace.")
        return cls(sorted(block_models, key=lambda block_model: block_model.name.casefold()))


async def fire_selected_turn(
    service_client: BlockModelAPIClient,
    block_model_selector: BlockModelSelectorWidget,
    player_category: str,
    x: float,
    y: float,
    z: float,
    buffer_blocks: int,
    config: GameConfig,
) -> TurnResult:
    """Run a turn against the block model currently selected in the dropdown."""
    return await play_turn(
        service_client=service_client,
        block_model_id=block_model_selector.value,
        player_category=player_category,
        x=x,
        y=y,
        z=z,
        buffer_blocks=buffer_blocks,
        model_origin=list(config.model_origin),
        block_size=list(config.block_size),
        model_dimensions=list(config.model_dimensions),
        target_column=config.target_column,
        dirt_category=config.dirt_category,
        secret_domain_names=set(config.secret_domain_names),
    )


def display_turn_result(turn_result: TurnResult, player_category: str) -> None:
    """Display the outcome of one turn."""
    min_x, min_y, min_z = turn_result.min_point
    max_x, max_y, max_z = turn_result.max_point
    print(
        f"Search box: X={min_x:,.1f}-{max_x:,.1f}, "
        f"Y={min_y:,.1f}-{max_y:,.1f}, Z={min_z:,.1f}-{max_z:,.1f}"
    )
    if turn_result.dirt_blocks_cleared == 0 and turn_result.special_blocks_claimed == 0:
        print("No change. This search area was already cleared or claimed.")
    else:
        print(
            f"Cleared {turn_result.dirt_blocks_cleared} dirt block(s) and claimed "
            f"{turn_result.special_blocks_claimed} special deposit block(s) for {player_category!r}."
        )
    print("\nRemaining blocks by special deposit:")
    display(turn_result.remaining_blocks)


async def show_selected_block_model(
    manager: ServiceManagerWidget,
    service_client: BlockModelAPIClient,
    block_model_selector: BlockModelSelectorWidget,
    config: GameConfig,
) -> None:
    """Display the currently selected model in the Evo visualizer."""
    from evo.widgets import EvoObjectViewer, download_tileset_bundle

    block_model = await service_client.get_block_model(block_model_selector.value)
    object_id = block_model.geoscience_object_id
    if object_id is None:
        raise RuntimeError("The block model is not linked to a geoscience object.")
    bundle = await download_tileset_bundle(manager, object_id, name=block_model.name)
    viewer = EvoObjectViewer()
    viewer.add_bundle(bundle, category_options={config.target_column: {}})
    display(viewer)


async def run_automated_turns(
    service_client: BlockModelAPIClient,
    block_model_selector: BlockModelSelectorWidget,
    config: GameConfig,
    buffer_blocks: int,
    turn_interval_seconds: float,
    player_names: tuple[str, ...] = DEFAULT_TEST_PLAYER_NAMES,
) -> None:
    """Run randomized turns until the displayed Stop button is pressed."""
    stop_button = widgets.Button(description="Stop", button_style="danger")
    status_output = widgets.Output()
    stop_requested = asyncio.Event()

    def stop_firing(_: widgets.Button) -> None:
        stop_requested.set()
        stop_button.disabled = True
        stop_button.description = "Stopping..."

    stop_button.on_click(stop_firing)
    display(stop_button, status_output)
    turn_number = 0
    while not stop_requested.is_set():
        indices = [random.randrange(dimension) for dimension in config.model_dimensions]
        coordinates = tuple(
            config.model_origin[axis] + indices[axis] * config.block_size[axis] + config.block_size[axis] / 2
            for axis in range(3)
        )
        player_name = random.choice(player_names)
        turn_result = await fire_selected_turn(
            service_client,
            block_model_selector,
            player_name,
            *coordinates,
            buffer_blocks,
            config,
        )
        turn_number += 1
        with status_output:
            clear_output(wait=True)
            print(
                f"Turn {turn_number} ({player_name}): "
                f"X={coordinates[0]:,.1f}, Y={coordinates[1]:,.1f}, Z={coordinates[2]:,.1f}"
            )
            print(
                f"Cleared {turn_result.dirt_blocks_cleared} dirt block(s) and claimed "
                f"{turn_result.special_blocks_claimed} special deposit block(s)."
            )
        try:
            await asyncio.wait_for(stop_requested.wait(), timeout=turn_interval_seconds)
        except asyncio.TimeoutError:
            pass

    with status_output:
        clear_output(wait=True)
        print(f"Stopped after {turn_number} turn(s).")
        if turn_number:
            print("\nRemaining blocks by special deposit:")
            display(turn_result.remaining_blocks)


async def create_final_report(
    manager: ServiceManagerWidget,
    environment: Environment,
    service_client: BlockModelAPIClient,
    block_model_selector: BlockModelSelectorWidget,
    config: GameConfig,
) -> pd.DataFrame:
    """Create and return final block volumes grouped by player category."""
    api_block_model = await service_client.get_block_model(block_model_selector.value)
    object_id = api_block_model.geoscience_object_id
    if object_id is None:
        raise RuntimeError("The block model is not linked to a geoscience object.")
    object_reference = ObjectReference.new(environment, object_id=object_id)
    report_block_model = await BlockModel.from_reference(manager, object_reference)
    final_report = await report_block_model.create_report(
        ReportSpecificationData(
            name=f"Blockbuster Final Volumes {uuid4().hex[:8]}",
            description="Block volume grouped by Classification category.",
            columns=[],
            categories=[ReportCategorySpec(column_name=config.target_column, label="Classification")],
            mass_unit_id=MassUnits.TONNES,
            density_value=2.5,
            density_unit_id=Units.TONNES_PER_CUBIC_METRE,
            run_now=True,
        )
    )
    final_volumes = (await final_report.refresh()).to_dataframe()
    display_targets = final_volumes[config.target_column].astype("string").str.strip()
    return final_volumes[
        display_targets.notna()
        & display_targets.ne("")
        & display_targets.str.lower().ne("nan")
    ].sort_values("Volume", ascending=False)


def display_final_report(final_volumes: pd.DataFrame) -> None:
    """Display final volumes with readable numeric formatting."""
    display(final_volumes.style.format({"Volume": "{:,.2f}", "Mass": "{:,.2f}"}))
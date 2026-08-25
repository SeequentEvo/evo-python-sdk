from __future__ import annotations

from uuid import UUID

import ipywidgets as widgets

from evo.blockmodels import BlockModelAPIClient


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
    async def create(cls, service_client: BlockModelAPIClient) -> BlockModelSelectorWidget:
        """Load available block models from Evo and build a selector."""
        block_models = await service_client.list_all_block_models()
        if not block_models:
            raise RuntimeError("No block models are available in the selected Evo workspace.")
        return cls(sorted(block_models, key=lambda block_model: block_model.name.casefold()))
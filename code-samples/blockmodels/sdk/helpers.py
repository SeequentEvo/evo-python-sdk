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


class BlockModelAttributeSelectorWidget(widgets.VBox):
    """Select a source attribute and enter a name for its transformed attribute."""

    def __init__(self, attributes: list[object]) -> None:
        attribute_names = sorted({attribute.name for attribute in attributes}, key=str.casefold)
        if not attribute_names:
            raise RuntimeError("The selected block model has no attributes to transform.")
        self.selector = widgets.Dropdown(
            description="Attribute:",
            options=attribute_names,
            layout=widgets.Layout(width="max-content", margin="0 0 6px 0"),
        )
        self.selector.style.description_width = "7em"
        self.new_attribute_input = widgets.Text(
            description="New attribute:",
            placeholder="Enter a new attribute name",
            value=f"{self.selector.value}_transformed",
            layout=widgets.Layout(width="40ch"),
        )
        self.new_attribute_input.style.description_width = "7em"
        self.selector.observe(self._update_new_attribute_name, names="value")
        super().__init__(
            [self.selector, self.new_attribute_input],
            layout=widgets.Layout(padding="4px 0"),
        )

    def _update_new_attribute_name(self, change: dict[str, object]) -> None:
        self.new_attribute_input.value = f"{change['new']}_transformed"

    @property
    def value(self) -> str:
        """Return the name of the selected block model attribute."""
        return self.selector.value

    @property
    def new_attribute(self) -> str:
        """Return the requested name for the transformed attribute."""
        return self.new_attribute_input.value.strip()
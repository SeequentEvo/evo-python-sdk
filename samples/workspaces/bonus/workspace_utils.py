from typing import Optional
from uuid import UUID

import ipywidgets as widgets
from IPython.display import display


def collect_data_references(obj, data_refs=None):
    """Recursively collect all 'data' field values from the object dictionary."""
    if data_refs is None:
        data_refs = set()

    if isinstance(obj, dict):
        # If this dict has a 'data' key with a string value, it's a data reference
        if "data" in obj and isinstance(obj["data"], str):
            data_refs.add(obj["data"])
        # Recursively search through all values
        for value in obj.values():
            collect_data_references(value, data_refs)
    elif isinstance(obj, list):
        # Recursively search through all items
        for item in obj:
            collect_data_references(item, data_refs)

    return data_refs


"""Custom widgets for workspace maintenance operations."""


class ObjectSelectorWidget:
    """Widget for selecting geoscience objects with filtering by object type."""

    def __init__(self, object_client):
        self.object_client = object_client
        self.all_objects = []
        self.filtered_objects = []

        # Create widgets
        self.refresh_btn = widgets.Button(
            description="Refresh Objects",
            button_style="info",
            icon="refresh",
            layout=widgets.Layout(margin="5px 5px 5px 5px", align_self="center"),
        )
        self.refresh_btn.style.button_color = "#265C7F"
        self.refresh_btn.on_click(self._on_refresh_click)

        self.loading_label = widgets.Label(value="")

        # Object type filter
        self.object_type_filter = widgets.Dropdown(
            options=[("All Types", "")],
            value="",
            description="Object Type",
            style={"description_width": "100px"},
        )
        self.object_type_filter.observe(self._on_filter_changed, names="value")

        # Object selector
        self.object_selector = widgets.Dropdown(
            options=[("Select an object...", None)],
            value=None,
            description="Object",
            disabled=True,
            style={"description_width": "100px"},
        )

        # Info display
        self.info_display = widgets.HTML(value="")

        # Layout
        self.widget = widgets.VBox(
            [
                widgets.HBox([self.refresh_btn, self.loading_label]),
                self.object_type_filter,
                self.object_selector,
                self.info_display,
            ]
        )

    def _on_refresh_click(self, btn):
        """Handle refresh button click."""
        import asyncio

        asyncio.create_task(self.refresh())

    async def refresh(self):
        """Load all objects from the workspace."""
        self.loading_label.value = "Loading objects..."
        self.refresh_btn.disabled = True
        self.object_type_filter.disabled = True
        self.object_selector.disabled = True

        try:
            # Fetch all objects
            self.all_objects = await self.object_client.list_all_objects()

            # Extract unique object types
            object_types = set()
            for obj in self.all_objects:
                # Get the schema classification (e.g., "objects/pointset")
                schema_type = obj.schema_id.classification
                object_types.add(schema_type)

            # Update object type filter options
            type_options = [("All Types", "")]
            type_options.extend(sorted([(t, t) for t in object_types]))
            self.object_type_filter.options = type_options

            # Update filtered objects
            self._update_filtered_objects()

            self.loading_label.value = f"Loaded {len(self.all_objects)} object(s)"

        except Exception as e:
            self.loading_label.value = f"Error: {str(e)}"

        finally:
            self.refresh_btn.disabled = False
            self.object_type_filter.disabled = False
            self.object_selector.disabled = False

    def _on_filter_changed(self, change):
        """Handle filter change."""
        self._update_filtered_objects()

    def _format_object_name(self, obj):
        """Format the object name by removing .json extension."""
        name = obj.name
        if name.endswith(".json"):
            name = name[:-5]
        return name

    def _update_filtered_objects(self):
        """Update the object selector based on the current filter."""
        selected_type = self.object_type_filter.value

        # Filter objects
        if selected_type == "":
            self.filtered_objects = self.all_objects
        else:
            self.filtered_objects = [obj for obj in self.all_objects if obj.schema_id.classification == selected_type]

        # Update object selector options
        if self.filtered_objects:
            object_options = [("Select an object...", None)]
            object_options.extend(
                [
                    (f"{self._format_object_name(obj)} ({obj.schema_id.sub_classification})", obj.id)
                    for obj in self.filtered_objects
                ]
            )
            self.object_selector.options = object_options
            self.object_selector.value = None
        else:
            self.object_selector.options = [("No objects found", None)]
            self.object_selector.value = None

        # Update info display when selection changes
        self.object_selector.observe(self._on_object_selected, names="value")
        self._update_info_display()

    def _on_object_selected(self, change):
        """Handle object selection change."""
        self._update_info_display()

    def _update_info_display(self):
        """Update the info display with details about the selected object."""
        selected_id = self.object_selector.value

        if selected_id is None:
            self.info_display.value = ""
            return

        # Find the selected object
        selected_obj = next((obj for obj in self.filtered_objects if obj.id == selected_id), None)

        if selected_obj is None:
            self.info_display.value = ""
            return

        # Build info HTML
        info_html = f"""
        <div style="margin-top: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
            <b>Object Details:</b><br/>
            <b>Name:</b> {self._format_object_name(selected_obj)}<br/>
            <b>Path:</b> {selected_obj.path}<br/>
            <b>ID:</b> {selected_obj.id}<br/>
            <b>Schema:</b> {selected_obj.schema_id}<br/>
            <b>Version:</b> {selected_obj.version_id}<br/>
            <b>Created:</b> {selected_obj.created_at.strftime("%Y-%m-%d %H:%M:%S")}<br/>
            <b>Modified:</b> {selected_obj.modified_at.strftime("%Y-%m-%d %H:%M:%S")}<br/>
        </div>
        """
        self.info_display.value = info_html

    async def display(self):
        """Display the widget and auto-load objects."""
        display(self.widget)
        # Auto-load objects on display
        await self.refresh()

    def get_selected_object(self):
        """Get the currently selected object metadata."""
        selected_id = self.object_selector.value
        if selected_id is None:
            return None
        return next((obj for obj in self.filtered_objects if obj.id == selected_id), None)

    def get_selected_id(self) -> Optional[UUID]:
        """Get the UUID of the currently selected object."""
        return self.object_selector.value


class TargetWorkspaceSelectorWidget:
    """Widget for selecting a target workspace and operation type (copy/move)."""

    def __init__(self, manager_widget):
        self.manager_widget = manager_widget
        # Get the selected workspace ID
        selected_id = manager_widget._workspace_selector.selected

        _NULL_UUID = UUID(int=0)
        if selected_id != _NULL_UUID:
            self.current_workspace_id = selected_id
        else:
            raise ValueError("No source workspace is currently selected.")

        # Create widgets
        self.refresh_btn = widgets.Button(
            description="Refresh Workspaces",
            button_style="info",
            icon="refresh",
            layout=widgets.Layout(margin="5px 5px 5px 5px", align_self="center", width="auto"),
        )
        self.refresh_btn.style.button_color = "#265C7F"
        self.refresh_btn.on_click(self._on_refresh_click)

        self.loading_label = widgets.Label(value="")

        # Operation section - label and checkbox
        self.operation_label = widgets.HTML(value="<b>Copy object</b>")
        self.delete_original_checkbox = widgets.Checkbox(
            value=False,
            description="Delete original after copy",
            indent=False,
            style={"description_width": "auto"},
        )

        # Workspace selector
        self.workspace_selector = widgets.Dropdown(
            options=[("Select target workspace...", None)],
            value=None,
            description="To Workspace",
            disabled=True,
            style={"description_width": "100px"},
        )

        # Info display
        self.info_display = widgets.HTML(value="")

        # Layout
        self.widget = widgets.VBox(
            [
                widgets.HBox([self.refresh_btn, self.loading_label]),
                self.operation_label,
                self.delete_original_checkbox,
                self.workspace_selector,
                self.info_display,
            ]
        )

        # Observe selection changes
        self.workspace_selector.observe(self._on_workspace_selected, names="value")
        self.delete_original_checkbox.observe(self._on_checkbox_changed, names="value")

    def _on_refresh_click(self, btn):
        """Handle refresh button click."""
        import asyncio

        asyncio.create_task(self.refresh())

    async def refresh(self):
        """Load all workspaces."""
        self.loading_label.value = "Loading workspaces..."
        self.refresh_btn.disabled = True
        self.workspace_selector.disabled = True

        try:
            # Get all workspaces from the service manager
            workspaces = self.manager_widget.workspaces

            # Filter out the current workspace
            other_workspaces = [ws for ws in workspaces if ws.id != self.current_workspace_id]

            # Update workspace selector options
            if other_workspaces:
                workspace_options = [("Select target workspace...", None)]
                workspace_options.extend(
                    [(ws.display_name, ws.id) for ws in sorted(other_workspaces, key=lambda w: w.display_name)]
                )
                self.workspace_selector.options = workspace_options
                self.workspace_selector.value = None
                self.loading_label.value = f"Loaded {len(other_workspaces)} workspace(s)"
            else:
                self.workspace_selector.options = [("No other workspaces available", None)]
                self.workspace_selector.value = None
                self.loading_label.value = "No other workspaces found"

        except Exception as e:
            self.loading_label.value = f"Error: {str(e)}"

        finally:
            self.refresh_btn.disabled = False
            self.workspace_selector.disabled = False

    def _on_workspace_selected(self, change):
        """Handle workspace selection change."""
        self._update_info_display()

    def _on_checkbox_changed(self, change):
        """Handle delete original checkbox change."""
        self._update_info_display()

    def _update_info_display(self):
        """Update the info display with details about the selected workspace."""
        selected_id = self.workspace_selector.value

        if selected_id is None:
            self.info_display.value = ""
            return

        # Find the selected workspace
        workspaces = self.manager_widget.workspaces
        selected_ws = next((ws for ws in workspaces if ws.id == selected_id), None)

        if selected_ws is None:
            self.info_display.value = ""
            return

        # Get current workspace info
        current_ws = next((ws for ws in workspaces if ws.id == self.current_workspace_id), None)
        operation = "Move" if self.delete_original_checkbox.value else "Copy"

        # Build info HTML
        info_html = f"""
        <div style="margin-top: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px;">
            <b>Operation Summary:</b><br/>
            <b>Action:</b> {operation} object<br/>
            <b>From:</b> {current_ws.display_name}<br/>
            <b>To:</b> {selected_ws.display_name}<br/>
            <b>Target Workspace ID:</b> {selected_ws.id}<br/>
        </div>
        """
        self.info_display.value = info_html

    async def display(self):
        """Display the widget and auto-load workspaces."""
        display(self.widget)
        # Auto-load workspaces on display
        await self.refresh()

    def get_selected_workspace_id(self) -> Optional[UUID]:
        """Get the UUID of the currently selected target workspace."""
        return self.workspace_selector.value

    def get_operation(self) -> str:
        """Get the selected operation type ('Copy' or 'Move')."""
        return "Move" if self.delete_original_checkbox.value else "Copy"

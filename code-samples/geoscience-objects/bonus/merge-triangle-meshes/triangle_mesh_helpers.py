#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Helpers for listing, selecting and downloading triangle meshes, preserving their CRS, and viewing published
objects in Evo."""

import re

import ipywidgets as widgets
import numpy as np
from evo_schemas.components import Crs_V1_0_1_EpsgCode, Crs_V1_0_1_OgcWkt
from IPython.display import HTML, display

from evo.notebooks import FeedbackWidget


class MeshSelectorWidget(widgets.VBox):
    """Widget for selecting triangle meshes to merge with checkboxes, and either naming a new merged mesh or choosing
    an existing mesh to update with a new version."""

    def __init__(self, meshes, default_name="Merged_Mesh_SDK_demo", default_path="Jupyter_Example/Mesh"):
        """
        Args:
            meshes (list[ObjectMetadata]): The triangle mesh objects to list, in display order.
            default_name (str): The initial name of the merged mesh.
            default_path (str): The initial path of the merged mesh.

        Tick "Update existing merged mesh" to choose an existing mesh from a dropdown; the merge is then published as
        a new version of that mesh instead of as a new mesh.
        """
        self.meshes = list(meshes)

        widget_style = {"description_width": "120px"}
        widget_layout = widgets.Layout(width="90%")

        self.mesh_checkboxes = [
            widgets.Checkbox(value=False, description=f"{obj.path}  ({obj.id})", indent=False, layout=widget_layout)
            for obj in self.meshes
        ]
        self.select_all_checkbox = widgets.Checkbox(value=False, description="Select all", indent=False)
        self.selection_label = widgets.Label()

        self.merged_name_text = widgets.Text(
            value=default_name, description="Merged name:", style=widget_style, layout=widget_layout
        )
        self.merged_path_text = widgets.Text(
            value=default_path, description="Merged path:", style=widget_style, layout=widget_layout
        )
        self.new_mesh_fields = widgets.VBox([self.merged_name_text, self.merged_path_text])

        self.update_existing_checkbox = widgets.Checkbox(
            value=False, description="Update existing merged mesh", indent=False
        )
        self.target_mesh_dropdown = widgets.Dropdown(
            description="Target mesh:",
            style=widget_style,
            layout=widgets.Layout(width="90%", display="none"),
        )
        self.update_existing_checkbox.observe(self._toggle_update_existing, names="value")

        self.select_all_checkbox.observe(self._toggle_all, names="value")
        for checkbox in self.mesh_checkboxes:
            checkbox.observe(self._on_selection_changed, names="value")
        self._on_selection_changed()

        # Scroll the mesh list if the workspace contains many meshes.
        mesh_list = widgets.VBox(
            self.mesh_checkboxes,
            layout=widgets.Layout(max_height="300px", overflow="auto", border="1px solid lightgray", padding="4px"),
        )

        super().__init__(
            [
                widgets.HBox([self.select_all_checkbox, self.selection_label]),
                mesh_list,
                self.update_existing_checkbox,
                self.target_mesh_dropdown,
                self.new_mesh_fields,
            ]
        )

    @property
    def selected_meshes(self):
        """list[ObjectMetadata]: The meshes whose checkboxes are ticked, in display order."""
        return [obj for obj, checkbox in zip(self.meshes, self.mesh_checkboxes) if checkbox.value]

    @property
    def merged_name(self):
        """str: The name entered for the merged mesh."""
        return self.merged_name_text.value

    @property
    def merged_path(self):
        """str: The path entered for the merged mesh."""
        return self.merged_path_text.value

    @property
    def update_existing(self):
        """bool: Whether to publish the merge as a new version of an existing mesh instead of a new mesh."""
        return self.update_existing_checkbox.value

    @property
    def target_mesh(self):
        """ObjectMetadata | None: The existing mesh to update when `update_existing` is ticked. Selected meshes are
        excluded from the choices."""
        return self.target_mesh_dropdown.value

    def _toggle_update_existing(self, change):
        # Show the target mesh dropdown when updating, otherwise show the name and path fields for a new mesh.
        self.target_mesh_dropdown.layout.display = None if change["new"] else "none"
        self.new_mesh_fields.layout.display = "none" if change["new"] else None

    def _on_selection_changed(self, _=None):
        selected_count = sum(checkbox.value for checkbox in self.mesh_checkboxes)
        self.selection_label.value = f"{selected_count} of {len(self.mesh_checkboxes)} meshes selected"
        self._update_target_options()

    def _update_target_options(self):
        """Offer only the meshes that are not selected for merging as targets to update."""
        available = [obj for obj, checkbox in zip(self.meshes, self.mesh_checkboxes) if not checkbox.value]
        current = self.target_mesh_dropdown.value
        # Changing the options resets the dropdown, so restore the current target if it is still available.
        self.target_mesh_dropdown.options = [("Select a target mesh...", None)] + [
            (f"{obj.path}  ({obj.id})", obj) for obj in available
        ]
        self.target_mesh_dropdown.value = current if current in available else None

    def _toggle_all(self, change):
        for checkbox in self.mesh_checkboxes:
            checkbox.value = change["new"]


class TriangleMeshHelper:
    """Helper for working with triangle mesh geoscience objects."""

    def __init__(self, object_client, data_client):
        """
        Args:
            object_client (ObjectAPIClient): Client used to download geoscience objects.
            data_client (ObjectDataClient): Client used to download the Parquet data referenced by objects.
        """
        self.object_client = object_client
        self.data_client = data_client

    async def list_triangle_meshes(self):
        """
        List all triangle mesh objects in the workspace.

        Returns:
            list[ObjectMetadata]: The triangle mesh objects, sorted by path.
        """
        all_objects = await self.object_client.list_all_objects()
        return sorted(
            (obj for obj in all_objects if obj.schema_id.sub_classification == "triangle-mesh"),
            key=lambda obj: obj.path.lower(),
        )

    async def download_triangle_mesh_object(self, object_metadata):
        """
        Download the definition of a triangle mesh object, without its vertex and index data.

        Args:
            object_metadata (ObjectMetadata): Metadata of the triangle mesh object to download.

        Returns:
            dict: The object definition, including its `coordinate_reference_system` and `triangles` table references.
        """
        downloaded_object = await self.object_client.download_object_by_id(
            object_id=object_metadata.id, version=object_metadata.version_id
        )
        object_dict = downloaded_object.as_dict()

        triangles = object_dict.get("triangles")
        if not triangles or "vertices" not in triangles or "indices" not in triangles:
            raise ValueError(f"Object '{object_metadata.name}' does not contain triangle vertices and indices.")
        return object_dict

    async def download_triangle_mesh_data(self, object_metadata, object_dict):
        """
        Download the vertex and index data of a triangle mesh object.

        Args:
            object_metadata (ObjectMetadata): Metadata of the triangle mesh object.
            object_dict (dict): The object definition returned by `download_triangle_mesh_object`.

        Returns:
            tuple: An (N, 3) float64 array of vertices and an (M, 3) uint64 array of triangle indices.
        """
        triangles = object_dict["triangles"]
        vertices_table = await self.data_client.download_table(
            object_id=object_metadata.id,
            version_id=object_metadata.version_id,
            table_info=triangles["vertices"],
            fb=FeedbackWidget(f"Downloading vertices of '{object_metadata.name}'"),
        )
        indices_table = await self.data_client.download_table(
            object_id=object_metadata.id,
            version_id=object_metadata.version_id,
            table_info=triangles["indices"],
            fb=FeedbackWidget(f"Downloading indices of '{object_metadata.name}'"),
        )

        vertices = vertices_table.to_pandas().to_numpy(dtype=np.float64)
        indices = indices_table.to_pandas().to_numpy(dtype=np.uint64)
        return vertices, indices

    @staticmethod
    def to_crs_model(crs):
        """
        Convert a coordinate reference system from an object definition into the equivalent evo-schemas model.

        Args:
            crs (dict | str): The `coordinate_reference_system` value of a downloaded object.

        Returns:
            Crs_V1_0_1_EpsgCode | Crs_V1_0_1_OgcWkt | str: The CRS model to use for the new object.
        """
        if isinstance(crs, dict) and "epsg_code" in crs:
            return Crs_V1_0_1_EpsgCode(epsg_code=crs["epsg_code"])
        if isinstance(crs, dict) and "ogc_wkt" in crs:
            return Crs_V1_0_1_OgcWkt(ogc_wkt=crs["ogc_wkt"])
        return crs if isinstance(crs, str) else "unspecified"

    @staticmethod
    def describe_crs(crs):
        """
        Describe a coordinate reference system from an object definition in a human-readable form.

        Args:
            crs (dict | str): The `coordinate_reference_system` value of a downloaded object.

        Returns:
            str: For example `EPSG:32650`, `WKT "WGS 84 / UTM zone 50N"` or `Unspecified`.
        """
        if isinstance(crs, dict) and "epsg_code" in crs:
            return f"EPSG:{crs['epsg_code']}"
        if isinstance(crs, dict) and "ogc_wkt" in crs:
            # The first quoted string in a WKT definition is the name of the CRS.
            wkt_name = re.search(r'"([^"]+)"', crs["ogc_wkt"])
            return f'WKT "{wkt_name.group(1)}"' if wkt_name else f"WKT {crs['ogc_wkt'][:60]}..."
        return "Unspecified" if crs == "unspecified" else str(crs)

    @staticmethod
    def build_portal_url(object_metadata):
        """
        Build the URL to view the object in the Evo Portal and display it as a link.

        Args:
            object_metadata (ObjectMetadata): The object metadata object returned after the object is created.
        """
        hub_url = object_metadata.environment.hub_url
        hub_name = hub_url.split("://")[1].split(".")[0]
        org_id = object_metadata.environment.org_id
        workspace_id = object_metadata.environment.workspace_id
        object_id = object_metadata.id

        url = f"https://evo.seequent.com/{org_id}/workspaces/{hub_name}/{workspace_id}/viewer?id={object_id}"

        display(HTML(f'<a href="{url}" target="_blank">View object in the Evo Portal</a>'))

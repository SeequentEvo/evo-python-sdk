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

"""Evo SDK Widgets.

This package provides HTML rendering and IPython formatters for displaying
Evo SDK objects in Jupyter notebooks.

Usage:
    In a Jupyter notebook, load the extension to enable rich HTML rendering:

        %load_ext evo.widgets

    After loading, any Evo SDK object will automatically render with styled HTML,
    including Portal and Viewer links.

Manual API:
    from evo.widgets import get_viewer_url_for_objects

    # View multiple objects together
    url = get_viewer_url_for_objects(manager, [pointset, grid])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .formatters import (
    format_attributes_collection,
    format_base_object,
    format_block_model,
    format_block_model_attributes,
    format_block_model_version,
    format_report,
    format_report_result,
    format_variogram,
)
from .urls import (
    get_blocksync_base_url,
    get_blocksync_block_model_url,
    get_blocksync_block_model_url_from_environment,
    get_blocksync_report_url,
    get_evo_base_url,
    get_hub_code,
    get_portal_url,
    get_portal_url_for_object,
    get_portal_url_from_reference,
    get_viewer_url,
    get_viewer_url_for_object,
    get_viewer_url_for_objects,
    get_viewer_url_from_reference,
    parse_object_reference_url,
    serialize_object_reference,
)

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

__all__ = [
    "format_attributes_collection",
    "format_base_object",
    "format_block_model",
    "format_block_model_attributes",
    "format_block_model_version",
    "format_report",
    "format_report_result",
    "format_variogram",
    "get_blocksync_base_url",
    "get_blocksync_block_model_url",
    "get_blocksync_block_model_url_from_environment",
    "get_blocksync_report_url",
    "get_evo_base_url",
    "get_hub_code",
    "get_portal_url",
    "get_portal_url_for_object",
    "get_portal_url_from_reference",
    "get_viewer_url",
    "get_viewer_url_for_object",
    "get_viewer_url_for_objects",
    "get_viewer_url_from_reference",
    "load_ipython_extension",
    "parse_object_reference_url",
    "serialize_object_reference",
    "unload_ipython_extension",
]


def _register_formatters(ipython: InteractiveShell) -> None:
    """Register HTML formatters for Evo SDK types.

    Uses `for_type_by_name` to avoid hard imports of model classes,
    which keeps the presentation layer decoupled from the data models.

    :param ipython: The IPython shell instance.
    """
    html_formatter = ipython.display_formatter.formatters["text/html"]

    # Register formatter for BaseObject and all subclasses (typed objects like PointSet, TensorGrid)
    # Using for_type_by_name avoids importing the class directly
    html_formatter.for_type_by_name(
        "evo.objects.typed.base",
        "_BaseObject",
        format_base_object,
    )

    # Register formatter for Variogram (overrides BaseObject for variogram-specific rendering)
    html_formatter.for_type_by_name(
        "evo.objects.typed.variogram",
        "Variogram",
        format_variogram,
    )

    # Register formatter for Attributes collection
    html_formatter.for_type_by_name(
        "evo.objects.typed.attributes",
        "Attributes",
        format_attributes_collection,
    )

    # Register formatters for block model types
    html_formatter.for_type_by_name(
        "evo.blockmodels.data",
        "Version",
        format_block_model_version,
    )

    html_formatter.for_type_by_name(
        "evo.blockmodels.typed.report",
        "Report",
        format_report,
    )

    html_formatter.for_type_by_name(
        "evo.blockmodels.typed.report",
        "ReportResult",
        format_report_result,
    )

    # Register formatters for BlockModel from evo-objects
    html_formatter.for_type_by_name(
        "evo.objects.typed.block_model_ref",
        "BlockModel",
        format_block_model,
    )

    html_formatter.for_type_by_name(
        "evo.objects.typed.block_model_ref",
        "BlockModelAttributes",
        format_block_model_attributes,
    )


def _unregister_formatters(ipython: InteractiveShell) -> None:
    """Unregister HTML formatters for Evo SDK types.

    :param ipython: The IPython shell instance.
    """
    html_formatter = ipython.display_formatter.formatters["text/html"]

    # Remove registered formatters by type name
    # Note: IPython doesn't have a direct "unregister by name" method,
    # so we need to work with the type_printers dict
    try:
        # Try to get the actual types and remove them
        from evo.objects.typed.attributes import Attributes
        from evo.objects.typed.base import _BaseObject
        from evo.objects.typed.variogram import Variogram

        html_formatter.type_printers.pop(_BaseObject, None)
        html_formatter.type_printers.pop(Variogram, None)
        html_formatter.type_printers.pop(Attributes, None)
    except ImportError:
        # If types can't be imported, try to clean up by string name
        # This is a best-effort cleanup
        pass


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Load the Evo presentation IPython extension.

    This function is called when the user runs `%load_ext evo.presentation`.
    It registers HTML formatters for all Evo SDK types, enabling rich display
    of objects like PointSet, Regular3DGrid, TensorGrid, etc.

    :param ipython: The IPython shell instance.

    Example:
        In a Jupyter notebook::

            %load_ext evo.presentation

            # Now typed objects display with rich HTML formatting
            grid = await object_from_reference(manager, grid_url)
            grid  # Shows formatted HTML with Portal/Viewer links
    """
    _register_formatters(ipython)


def unload_ipython_extension(ipython: InteractiveShell) -> None:
    """Unload the Evo presentation IPython extension.

    This function is called when the user runs `%unload_ext evo.presentation`.

    :param ipython: The IPython shell instance.
    """
    _unregister_formatters(ipython)

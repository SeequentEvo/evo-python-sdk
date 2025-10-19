import json
from operator import itemgetter

import ipywidgets as widgets
from IPython.display import display

from evo.common.data import HTTPResponse


class BlockModelSelector:
    """
    A class used to select and display block models from a service client.
    Attributes
    ----------
    connector : object
        An object to handle API calls to the Evo service.
    evo_hub_url : string
        The Evo Hub URL.
    org_id : string
        The org ID of the user.
    workspace_id : string
        The workspace ID of the user.
    blockmodel_dropdown : widgets.Dropdown
        Dropdown widget for selecting pointsets.
    all_blockmodels : list
        List of all block models fetched from the service client.
    Methods
    -------
    async display_blockmodels()
        Fetches block models from the service client and displays them in dropdown widgets.
    get_selected_items() -> dict
        Returns a dictionary of the selected items from the dropdown widgets.
    """

    def __init__(self, environment, connector):
        """
        Initializes the BlockModelSelector with environment and connector objects.

        Parameters:
            environment (object): An object containing Evo Hub URL, org ID, and workspace ID.
            connector (object): An object to handle API calls.
        """
        self.connector = connector
        self.evo_hub_url = environment.hub_url
        self.org_id = environment.org_id
        self.workspace_id = environment.workspace_id
        self.blockmodel_dropdown = None
        self.all_blockmodels = None

    async def display_blockmodels(self):
        """
        Asynchronously retrieves and displays block models in a dropdown widget.
        This method fetches all block models from the service client with a limit of 50 objects per request.
        These block models are then displayed in dropdown widgets for user selection.
        Attributes:
            all_blockmodels (list): A list of all blockmodels retrieved from the service client.
            blockmodel_dropdown (widgets.Dropdown): Dropdown widget for selecting block models.
        Displays:
            A label and a dropdown widget for selecting the block model.
        """

        resource_path = "/blockmodel/orgs/{org_id}/workspaces/{workspace_id}/block-models"

        path_params = {
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
        }

        api_response = await self.connector.call_api(
            method="GET",
            resource_path=resource_path,
            path_params=path_params,
            response_types_map={
                "200": HTTPResponse,
            },
        )

        # Parse the response data
        response = api_response.data.decode("utf-8")
        response_json = json.loads(response)

        self.all_blockmodels = []
        for obj in response_json["results"]:
            self.all_blockmodels.append((obj["name"], obj))

        layout = widgets.Layout(
            display="flex",
            flex_flow="row",
            align_items="stretch",
        )
        self.blockmodel_dropdown = widgets.Dropdown(
            options=sorted(self.all_blockmodels, key=itemgetter(0), reverse=False),
            description="",
            disabled=False,
            layout=layout,
        )

        display(widgets.Label("Select a block model:"))
        display(self.blockmodel_dropdown)

    def get_selected_item(self):
        """
        Returns the selected block model from the dropdown widget.
        Returns
        -------
        obj
            A JSON object representing a block model.
        """
        return self.blockmodel_dropdown.value

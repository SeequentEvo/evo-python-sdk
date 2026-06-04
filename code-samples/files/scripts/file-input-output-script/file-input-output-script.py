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


import asyncio
import os
import tempfile
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv

from evo.aio import AioTransport
from evo.common import APIConnector, Environment
from evo.files import FileAPIClient
from evo.oauth import ClientCredentialsAuthorizer, EvoScopes, OAuthConnector

load_dotenv()

# Configuration
CONFIG = {
    "evo": {
        "USER_AGENT": "Evo File I/O Script",
        "CLIENT_ID": os.environ["CLIENT_ID"],
        "CLIENT_SECRET": os.environ["CLIENT_SECRET"],
        "service_host": os.environ["HUB_URL"],
        "org_id": os.environ["ORG_ID"],
        "workspace_id": os.environ["WORKSPACE_ID"],
        "evo_input_file_path": os.environ["EVO_INPUT_FILE_PATH"],
        "evo_output_file_path": os.environ["EVO_OUTPUT_FILE_PATH"],
    }
}


async def download_csv_file(temp_dir, source_csv_filename, file_client):
    """
    Downloads a CSV file from Evo workspace using the file_client and saves it to the temp directory.
    Returns True if successful, False otherwise.
    """
    success = True
    try:
        ctx = await file_client.prepare_download_by_path(source_csv_filename)
        download_url = await ctx.get_download_url()

        response = requests.get(download_url)
        if response.status_code == 200:
            output_file = Path(temp_dir) / source_csv_filename
            output_file.write_bytes(response.content)
            print(f"File saved to: {output_file}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            success = False

    except Exception as e:
        print(f"Error downloading {source_csv_filename}: {e}")
        success = False
    if success:
        print(f"{source_csv_filename} file downloaded successfully.")
    return success


async def upload_csv_files(temp_dir, processed_csv_filename, file_client, connector):
    """
    Uploads the processed CSV file from the temp directory to Evo using the file_client.
    Returns True if successful, False otherwise.
    """
    success = True
    try:
        output_file = Path(temp_dir) / processed_csv_filename
        ctx = await file_client.prepare_upload_by_path(processed_csv_filename)
        await ctx.upload_from_path(str(output_file), connector.transport)
        print(f"File {processed_csv_filename} uploaded successfully.")
    except Exception as e:
        print(f"Error uploading {processed_csv_filename}: {e}")
        success = False
    return success


def main():
    evo_cfg = CONFIG["evo"]

    environment = Environment(
        hub_url=evo_cfg["service_host"],
        org_id=uuid.UUID(evo_cfg["org_id"]),
        workspace_id=uuid.UUID(evo_cfg["workspace_id"]),
    )
    transport = AioTransport(user_agent=evo_cfg["USER_AGENT"])
    authorizer = ClientCredentialsAuthorizer(
        oauth_connector=OAuthConnector(
            transport=transport,
            client_id=evo_cfg["CLIENT_ID"],
            client_secret=evo_cfg["CLIENT_SECRET"],
        ),
        scopes=EvoScopes.all_evo,
    )
    connector = APIConnector(environment.hub_url, transport, authorizer)
    file_client = FileAPIClient(connector=connector, environment=environment)
    source_csv_filename = evo_cfg["evo_input_file_path"]
    processed_csv_filename = evo_cfg["evo_output_file_path"]

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    script_dir = Path(__file__).parent
    with tempfile.TemporaryDirectory(dir=script_dir) as temp_dir:
        asyncio.run(download_csv_file(temp_dir, source_csv_filename, file_client))

        # Process the file downloaded from Evo and generate a new output file to be uploaded to Evo.

        asyncio.run(upload_csv_files(temp_dir, processed_csv_filename, file_client, connector))


if __name__ == "__main__":
    main()

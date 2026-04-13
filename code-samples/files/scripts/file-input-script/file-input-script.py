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
import uuid
from pathlib import Path

from dotenv import load_dotenv
from evo.aio import AioTransport
from evo.common import APIConnector, Environment
from evo.files import FileAPIClient
from evo.oauth import ClientCredentialsAuthorizer, EvoScopes, OAuthConnector

load_dotenv()

# Configuration
CONFIG = {
    "evo": {
        "USER_AGENT": "Evo CSV File Input Script",
        "CLIENT_ID": os.environ["CLIENT_ID"],
        "CLIENT_SECRET": os.environ["CLIENT_SECRET"],
        "service_host": os.environ["HUB_URL"],
        "org_id": os.environ["ORG_ID"],
        "workspace_id": os.environ["WORKSPACE_ID"],
        "file_path": os.environ["DIR_PATH"],
    }
}


async def upload_csv_files(script_dir, file_client, connector):
    success = True
    for file_path in script_dir.glob("*.csv"):
        print(f"Uploading file: {file_path.name}")
        try:
            ctx = await file_client.prepare_upload_by_path(file_path.name)
            await ctx.upload_from_path(str(file_path), connector.transport)
        except Exception as e:
            print(f"Error uploading {file_path.name}: {e}")
            success = False
    if success:
        print("All CSV files uploaded successfully.")
    return success


def main():
    evo_cfg = CONFIG["evo"]
    script_dir = Path(evo_cfg["file_path"])

    environment = Environment(
        hub_url=evo_cfg["service_host"],
        org_id=uuid.UUID(evo_cfg["org_id"]),
        workspace_id=uuid.UUID(evo_cfg["workspace_id"]),
    )
    transport = AioTransport(user_agent=evo_cfg["USER_AGENT"])
    authorizer = ClientCredentialsAuthorizer(
        oauth_connector=OAuthConnector(
            transport=transport, client_id=evo_cfg["CLIENT_ID"], client_secret=evo_cfg["CLIENT_SECRET"]
        ),
        scopes=EvoScopes.all_evo,
    )
    connector = APIConnector(environment.hub_url, transport, authorizer)
    file_client = FileAPIClient(connector=connector, environment=environment)

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(upload_csv_files(script_dir, file_client, connector))


if __name__ == "__main__":
    main()

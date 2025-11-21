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
import datetime
import json
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import requests

from evo.aio import AioTransport
from evo.common import APIConnector, Environment
from evo.files import FileAPIClient
from evo.oauth import ClientCredentialsAuthorizer, EvoScopes, OAuthConnector

# Configuration
CONFIG = {
    "mx": {
        "url": "https://app.mxdeposit.net/api/v3/collars/export/",
        "project_id": "<project_id>",
        "template_code": "<template_code>",
        "auth_token": "<api_key>",
        "client_id": "<client_id>",
    },
    "evo": {
        "USER_AGENT": "MXDepositToEvoScript",
        "CLIENT_ID": "<client_id>",
        "CLIENT_SECRET": "<client_secret>",
        "service_host": "<hub_url>",
        "org_id": "<org_id>",
        "workspace_id": "<workspace_id>",
    },
}


def export_collars(config):
    payload = json.dumps(
        {
            "project": config["project_id"],
            "template_code": config["template_code"],
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": config["auth_token"],
        "Client-ID": config["client_id"],
    }
    print("Request made at (UTC):", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    response = requests.post(config["url"], headers=headers, data=payload)
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.text}")
    print(response.json())
    try:
        data = response.json()
        operation_uid = data["jobs"][0]["parameters"]["body"]["operation_uid"]
        print("operation_uid:", operation_uid)
        return operation_uid
    except Exception as e:
        print("Could not extract operation_uid:", e)
        return None


def poll_export_status(operation_uid, config, interval=30, timeout=8 * 60 * 60):
    headers = {
        "Content-Type": "application/json",
        "Authorization": config["auth_token"],
        "Client-ID": config["client_id"],
    }
    status_url = f"https://app.mxdeposit.net/export-status/{operation_uid}"
    start_time = time.time()
    while True:
        response = requests.get(status_url, headers=headers)
        try:
            data = response.json()
            print(f"Polling: {data}")
            if data.get("state") == "done":
                return data.get("url")
        except Exception as e:
            print("Error parsing response:", e)
        if time.time() - start_time > timeout:
            print("Polling timed out.")
            return None
        time.sleep(interval)


def download_and_extract_zip(download_url, temp_dir):
    export_file = temp_dir / "export.zip"
    response = requests.get(download_url)
    export_file.write_bytes(response.content)
    with zipfile.ZipFile(export_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    print(f"Files extracted to: {temp_dir}")


async def upload_csv_files(temp_dir, file_client, connector):
    success = True
    for file_path in temp_dir.glob("*.csv"):
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
    mx_cfg = CONFIG["mx"]
    evo_cfg = CONFIG["evo"]

    operation_uid = export_collars(mx_cfg)
    if not operation_uid:
        return

    download_url = poll_export_status(operation_uid, mx_cfg)
    if not download_url:
        return

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

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    script_dir = Path(__file__).parent
    with tempfile.TemporaryDirectory(dir=script_dir) as temp_dir:
        temp_path = Path(temp_dir)
        download_and_extract_zip(download_url, temp_path)
        asyncio.run(upload_csv_files(temp_path, file_client, connector))


if __name__ == "__main__":
    main()

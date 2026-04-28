# File Input/Output Script

This script downloads a CSV file from an Evo workspace, allows you to process it locally, and then uploads the processed result back to the workspace.

## Prerequisites

- Python 3.10+
- The following Python packages:
  - `python-dotenv`
  - `requests`
  - `evo-sdk-common`
  - `evo-files`
- A registered Evo service application with a client ID and client secret

## Setup

1. Copy the example environment file to create your own `.env` file:

   On macOS and Linux:

   ```bash
   cp .env.example .env
   ```

   On Windows (PowerShell):

   ```powershell
   Copy-Item .env.example .env
   ```

2. Open the `.env` file and fill in your values:

   | Variable              | Description                                                |
   |-----------------------|------------------------------------------------------------|
   | `CLIENT_ID`           | Your Evo service application client ID                     |
   | `CLIENT_SECRET`       | Your Evo service application client secret                 |
   | `HUB_URL`             | The Evo hub URL                                            |
   | `ORG_ID`              | Your organisation ID (UUID)                                |
   | `WORKSPACE_ID`        | The target workspace ID (UUID)                             |
   | `EVO_INPUT_FILE_PATH` | Path of the CSV file to download from the Evo workspace    |
   | `EVO_OUTPUT_FILE_PATH`| Path to upload the processed CSV file to the Evo workspace |

3. Install dependencies:

   ```bash
   pip install python-dotenv requests evo-sdk-common evo-files
   ```

## Usage

Run the script from the `file-input-output-script` directory:

```bash
python file-input-output-script.py
```

The script will:

1. Download the CSV file specified by `EVO_INPUT_FILE_PATH` from the Evo workspace into a temporary directory.
2. *(Add your processing logic in the marked section of the script.)*
3. Upload the processed file as `EVO_OUTPUT_FILE_PATH` back to the Evo workspace.

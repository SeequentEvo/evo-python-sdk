# File Input Script

This script uploads all CSV files from a specified directory to an Evo workspace using the Evo File API.

## Prerequisites

- Python 3.10+
- The following Python packages:
  - `python-dotenv`
  - `evo-sdk-common`
  - `evo-files`
- A registered Evo service application with a client ID and client secret

## Setup

1. Copy the example environment file to create your own `.env` file:

   On macOS and Linux (bash):

   ```bash
   cp .env.example .env
   ```

   On Windows (PowerShell):

   ```powershell
   Copy-Item .env.example .env
   ```

2. Open the `.env` file and fill in your values:

   | Variable        | Description                                        |
   |-----------------|----------------------------------------------------|
   | `CLIENT_ID`     | Your Evo service application client ID             |
   | `CLIENT_SECRET` | Your Evo service application client secret         |
   | `HUB_URL`       | The Evo hub URL                                    |
   | `ORG_ID`        | Your organisation ID (UUID)                        |
   | `WORKSPACE_ID`  | The target workspace ID (UUID)                     |
   | `DIR_PATH`      | Local directory path containing the CSV files      |

3. Install dependencies:

   ```bash
   pip install python-dotenv evo-sdk-common evo-files
   ```

## Usage

Run the script from the `file-input-script` directory:

```bash
python file-input-script.py
```

The script will scan the directory specified in `DIR_PATH` for `*.csv` files and upload each one to the configured Evo workspace.

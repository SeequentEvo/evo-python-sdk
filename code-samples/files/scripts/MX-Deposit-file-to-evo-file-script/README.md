# MX Deposit File to Evo File Script

This script exports collar data from MX Deposit, downloads the exported CSV files, and uploads them to an Evo workspace.

## Prerequisites

- Python 3.10+
- The following Python packages:
  - `python-dotenv`
  - `requests`
  - `evo-sdk-common`
  - `evo-files`
- An MX Deposit account
- A registered Evo service application with a client ID and client secret

## Setup

1. Copy the example environment file to create your own `.env` file:

   ```bash
   cp .env.example .env
   ```

   On Windows (PowerShell):

   ```powershell
   Copy-Item .env.example .env
   ```

2. Open the `.env` file and fill in your values:

   **MX Deposit settings:**

   | Variable           | Description                        |
   |--------------------|------------------------------------|
   | `MX_PROJECT_ID`    | Your MX Deposit project ID         |
   | `MX_TEMPLATE_CODE` | Your MX Deposit template code      |
   | `MX_AUTH_TOKEN`    | Your MX Deposit API key            |
   | `MX_CLIENT_ID`     | Your MX Deposit client ID          |

   **Evo settings:**

   | Variable        | Description                                |
   |-----------------|--------------------------------------------|
   | `CLIENT_ID`     | Your Evo service application client ID     |
   | `CLIENT_SECRET` | Your Evo service application client secret |
   | `HUB_URL`       | The Evo hub URL                            |
   | `ORG_ID`        | Your organisation ID (UUID)                |
   | `WORKSPACE_ID`  | The target workspace ID (UUID)             |

3. Install dependencies:

   ```bash
   pip install python-dotenv requests evo-sdk-common evo-files
   ```

## Usage

Run the script from the `MX-Deposit-file-to-evo-file-script` directory:

```bash
python MX-Deposit-file-to-evo-file-script.py
```

The script will:

1. Trigger a collar export from MX Deposit.
2. Poll the export status until the data is ready.
3. Download and extract the exported ZIP file into a temporary directory.
4. Upload all extracted CSV files to the configured Evo workspace.

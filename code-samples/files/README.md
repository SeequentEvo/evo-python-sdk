# File API Samples

These samples demonstrate how to upload, download, list, and delete files in an Evo workspace using the Evo File API.

## Start Here

1. [SDK Examples](sdk-examples.ipynb)
	- Use the high-level `evo.files` interfaces for standard file workflows.
2. [API Examples](api-examples.ipynb)
	- Use direct API calls to inspect requests and responses or build custom integrations.

## Automated File Workflows

- [File Input Script](scripts/file-input-script/README.md) uploads all CSV files from a local directory to an Evo workspace.
- [File Input/Output Script](scripts/file-input-output-script/README.md) downloads a CSV file, provides a place to process it, and uploads the result.
- [MX Deposit File to Evo File Script](scripts/MX-Deposit-file-to-evo-file-script/README.md) exports collar data from MX Deposit and uploads the resulting CSV files to Evo.

## Sample Data

`sample-data/` contains files used by the notebook examples.

## Requirements

See the [code-samples setup guide](../README.md#before-you-start) for supported Python versions and environment setup. Each script directory contains its own dependency and credential instructions.

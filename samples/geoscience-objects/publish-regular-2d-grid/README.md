# Getting Started

## Geosoft GX for Python

This notebook uses [Geosoft GX for Python](https://github.com/GeosoftInc/gxpy) to extract data from a grid file and re-package it as an Evo-compatible geoscience object.

Visit the [Geosoft Github page](https://github.com/GeosoftInc) to learn more about Geosoft GX, including developer toolkits for .NET and C/C++ and the extensive documentation.

## System requirements

### Operating system

Geosoft GX for Python is only compatible with Windows-based computers - users of Linux or macOS will not be able to run this notebook or use the toolkit.

TIP: macOS users with an Apple silicon computer (M1 or higher) can install Windows in a virtual environment by using the **free version** of [VMware Fusion Pro](https://blogs.vmware.com/cloud-foundation/2024/11/11/vmware-fusion-and-workstation-are-now-free-for-all-users/).

### Python versions

Geosoft GX is compatible with Python versions 3.8 or higher.

Th `requirements.txt` file found in the `2d-grids` folder is unique for these examples. In your Python environment use `pip` to install `requirements.txt`, eg.

``` bash
pip install -r requirements.txt
````

Python package notes:

- The version of `numpy` installed must be **less than 1.24**. This is due to the GX toolkit using some numpy features that have been removed in more recent versions.
- The version of `pyzmq` installed must be **25.1.2**. This is due to a bug in more recent versions that cause the Jupyter notebook kernel to crash in Windows. NOTE: If you are not using Jupyter you can ignore this requirement.
- A Python wheel (found in the root `geoscience-objects` folder) will install the geoscience object dataclasses written by the Seequent. These dataclasses are used extensive in all Geoscience Object API notebooks. 

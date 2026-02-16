# Jupyter Notebooks

The `publish-*` directories contain Jupyter notebooks with sample code for uploading geoscience objects to Evo. For example, `publish-triangular-mesh/publish-triangular-mesh.ipynb` Jupyter notebook will demonstrate how to upload a triangular mesh object.

## Recommended Starting Point

**New users should start with `simplified-object-interactions/`** - This example demonstrates the recommended approach for most users and geologists using the typed objects API (`PointSet`, `Regular3DGrid`, etc.) with the `evo.widgets` extension for rich HTML display. It provides a simpler, more intuitive way to upload and download geoscience objects.

The `publish-*` examples use the lower-level `evo-schemas` approach, which offers more control but requires more boilerplate code.

## Requirements

* Python ^3.10

## Creating a virtual environment
To run the a Jupyter notebook we recommend first creating a Python virtual environment. 

NOTE: The steps below assume you have a compatible copy of Python installed on your system.

1. In the root directory of the notebook you want to work with, install `virtualenv` and initialize a virtual environment:
```shell
pip install virtualenv
python -m venv my_virtual_env
```

1. Activate the virtual environment from the root directory.

On Windows:

```shell
my_virtual_env\Scripts\activate
```

On macOS or Linux:

```shell
source my_virtual_env/bin/activate
```

## Install the Python dependencies

Each notebook may have it's own unique set of requirements. For example, `publish-regular-2d-grid` requires the `geosoft` package which only works on Windows.
For this reason, each notebook is bundled with it's own `requirements.txt` file.

```shell
pip install -r requirements.txt
```

## Running the Jupyter notebook

1. The first cell of every notebook requires you to enter the `client ID` of your Evo app. Update the default value of `redirect_url` too, if required.
1. Save and run the first cell and the notebook will launch your web browser and ask you to sign in with your Bentley ID. 
1. Once you've signed in, return to the notebook and select your Evo workspace using the widget on screen.
1. Continue working in the notebook by running the remaining cells in order.


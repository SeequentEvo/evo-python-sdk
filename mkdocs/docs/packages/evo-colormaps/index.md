# evo-colormaps

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-colormaps/src/evo/colormaps/)

The Colormap API is a key feature of the Evo platform, providing a mechanism to create colour mappings and associate them to geoscience data.

## Usage

```python
from evo.colormaps import ColormapAPIClient

colormap_client = ColormapAPIClient(environment, connector)
colormaps = await colormap_client.list_colormaps()
```

See the [ColormapAPIClient](evo-colormaps/ColormapAPIClient) page for the full API reference.

# evo-blockmodels

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-blockmodels/src/evo/blockmodels/)

The `evo-blockmodels` package provides both a low-level API client and typed Python classes for working with block models in Evo.

!!! tip "Using block models from typed objects"
    The full functionality of `evo-blockmodels` — creating, retrieving, updating attributes, running reports — is accessible directly from the [`BlockModel`](../evo-objects/Introduction.md#blockmodel-via-evo-blockmodels) object in `evo.objects.typed`. When `evo-blockmodels` is installed, `BlockModel` acts as a proxy and delegates data operations to the Block Model Service automatically.

    ```python
    from evo.objects.typed import object_from_path

    # Load any block model — full evo-blockmodels functionality is available
    bm = await object_from_path(manager, "my-folder/block-model")
    df = await bm.to_dataframe()
    await bm.add_attribute(data_df, "new_col")
    report = await bm.create_report(spec)
    ```

    See the [evo-objects Introduction](../evo-objects/Introduction.md#blockmodel-via-evo-blockmodels) for the full API.

See the [Typed Objects](TypedObjects.md) page for the full typed API reference.

## Typed Block Models

The typed module provides intuitive classes for creating, retrieving, and updating regular block models with pandas DataFrame support.

### Creating a block model

```python
from evo.blockmodels.typed import RegularBlockModel, RegularBlockModelData, Point3, Size3d, Size3i

data = RegularBlockModelData(
    name="My Block Model",
    origin=Point3(0, 0, 0),
    n_blocks=Size3i(10, 10, 10),
    block_size=Size3d(1.0, 1.0, 1.0),
    cell_data=my_dataframe,
)
block_model = await RegularBlockModel.create(context, data)
```

### Retrieving a block model

```python
block_model = await RegularBlockModel.get(context, block_model_id)
df = block_model.cell_data  # pandas DataFrame with all cell attributes
```

### Updating attributes

```python
new_version = await block_model.update_attributes(
    updated_dataframe,
    new_columns=["new_col"],
)
```

## Reports

Reports provide resource estimation summaries for block models — calculating tonnages, grades, and metal content grouped by category (e.g., geological domains, rock types).

### Creating and running a report

```python
from evo.blockmodels.typed import (
    Report, ReportSpecificationData, ReportColumnSpec, ReportCategorySpec,
    Aggregation, MassUnits, Units,
)

spec = ReportSpecificationData(
    name="Grade Report",
    category=ReportCategorySpec(column_name="domain"),
    columns=[
        ReportColumnSpec(
            column_name="Au",
            aggregation=Aggregation.MASS_AVERAGE,
            output_unit_id="g/t",
        ),
    ],
    mass_units=MassUnits.TONNES,
)

report = await Report.create(context, block_model, spec)
result = await report.run(context)
df = result.to_dataframe()  # Tonnages and grades by domain
```


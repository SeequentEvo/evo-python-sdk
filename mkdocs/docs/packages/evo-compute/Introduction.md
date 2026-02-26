# evo-compute

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-compute/src/evo/compute/)

The `evo-compute` package provides a client for running compute tasks on Evo. Tasks are submitted to the Compute Tasks API and polled for results.

See the [Typed Objects](TypedObjects.md) page for the full typed API reference.

## Running Compute Tasks

The `run()` function is the main entry point for executing compute tasks. It supports running a single task or multiple tasks concurrently.

### Single task

```python
from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
from evo.compute.tasks.kriging import KrigingParameters

params = KrigingParameters(
    source=pointset.attributes["grade"],
    target=block_model.attributes["kriged_grade"],  # Creates if new, updates if exists
    variogram=variogram,
    search=SearchNeighborhood(
        ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)),
        max_samples=20,
    ),
)
result = await run(manager, params, preview=True)
```

### Multiple tasks

Run multiple kriging tasks concurrently — for example, estimating different attributes or using different parameters:

```python
from evo.compute.tasks import run, SearchNeighborhood
from evo.compute.tasks.kriging import KrigingParameters

results = await run(manager, [
    KrigingParameters(
        source=pointset.attributes["Au"],
        target=block_model.attributes["Au_kriged"],
        variogram=au_variogram,
        search=SearchNeighborhood(...),
    ),
    KrigingParameters(
        source=pointset.attributes["Cu"],
        target=block_model.attributes["Cu_kriged"],
        variogram=cu_variogram,
        search=SearchNeighborhood(...),
    ),
], preview=True)

results[0]  # First kriging result
results[1]  # Second kriging result
```

### Working with results

Task results provide convenient methods to access the output:

```python
# Pretty-print the result
result  # Shows ✓ Kriging Result with target and attribute info

# Get the target object
target = await result.get_target_object()

# Get data as a DataFrame
df = await result.to_dataframe()
```

For complete examples, see the [kriging notebook](https://github.com/SeequentEvo/evo-python-sdk/blob/main/code-samples/geoscience-objects/running-kriging-compute/running-kriging-compute.ipynb) and the [multiple kriging notebook](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-compute/docs/examples/kriging_multiple.ipynb).

## FAQ

### How do I run parallel tasks that update the same attribute?

You can set the target for a compute task using `block_model.attributes["name"]`. If the attribute does not yet exist, it will be **created**; if it already exists, it will be **updated**. This is determined by the local state of the object.

When the first task creates a new attribute on the server, your **local** object doesn't know about it yet. If you then try to run more tasks targeting the same attribute name, the local object still thinks it doesn't exist and will try to create it again — causing a conflict.

To avoid this:

1. Run the **first** task to create the attribute.
2. **Refresh** the local object so it sees the newly created attribute.
3. Run the **remaining** tasks — now `block_model.attributes["kriged_grade"]` resolves to the existing attribute and will update it.

```python
from evo.compute.tasks import run, SearchNeighborhood
from evo.compute.tasks.kriging import KrigingParameters, RegionFilter

# Step 1: Run the first task — attribute "kriged_grade" does not exist yet, so it is created
first_result = await run(manager, KrigingParameters(
    source=pointset.attributes["grade"],
    target=block_model.attributes["kriged_grade"],
    variogram=variogram,
    search=SearchNeighborhood(...),
    target_region_filter=RegionFilter(
        attribute=block_model.attributes["domain"],
        names=["LMS1"],
    ),
), preview=True)

# Step 2: Refresh so the local object recognises the newly created attribute
block_model = await block_model.refresh()

# Step 3: Now "kriged_grade" exists locally — remaining tasks will update it
results = await run(manager, [
    KrigingParameters(
        source=pointset.attributes["grade"],
        target=block_model.attributes["kriged_grade"],  # Exists → update
        variogram=variogram,
        search=SearchNeighborhood(...),
        target_region_filter=RegionFilter(
            attribute=block_model.attributes["domain"],
            names=["LMS2"],
        ),
    ),
    KrigingParameters(
        source=pointset.attributes["grade"],
        target=block_model.attributes["kriged_grade"],  # Exists → update
        variogram=variogram,
        search=SearchNeighborhood(...),
        target_region_filter=RegionFilter(
            attribute=block_model.attributes["domain"],
            names=["LMS3"],
        ),
    ),
], preview=True)
```

!!! tip
    If each task writes to a **different** attribute name, they can all run in parallel without refreshing — the compute service handles concurrent attribute creation on the same target object. See the [multiple kriging notebook](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-compute/docs/examples/kriging_multiple.ipynb) for an example.

!!! note "Preview APIs"
    Kriging and other compute tasks are currently preview features. You must pass `preview=True` when calling `run()`.
    Preview APIs may change between releases. For more details, see:

    - [Preview APIs](https://developer.seequent.com/docs/api/fundamentals/preview-apis) — how to opt in and what to expect
    - [API Lifecycle](https://developer.seequent.com/docs/api/fundamentals/lifecycle) — how Evo APIs evolve from preview to stable


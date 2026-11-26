# evo-compute

[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-compute/src/evo/compute/)

The `evo-compute` package provides clients for discovering and running compute tasks on Evo:

- **`ComputeClient`** — a discovery-driven async client (`client.<topic>.<task>.run(...)`, `submit(...)`, `arun(...)`, and `asubmit(...)`) that runs any task advertised in your organization's compute catalogue without requiring a dedicated hand-written task module.
- **`TaskJob`** — typed async job handles for non-blocking submission, status checks, cancellation, deferred result hydration, and reopening jobs from saved URLs (`TaskJob.from_url(...)`).
- **`evo.compute.tasks.run()`** — typed Pydantic parameter models and result helpers for geostatistics workflows, routed concurrently through `ComputeClient`.
- **`DiscoveryClient` and `JobClient`** — direct access to task catalogue discovery (`TaskResource`) and low-level job submission/polling.

## Running Any Catalogued Task with `ComputeClient`

`ComputeClient` resolves topics, tasks, and keyword arguments live from the discovery catalogue (replacing hyphens with underscores in Python identifiers):

```python
from evo.compute import ComputeClient, DiscoveryClient

# Inspect available topics and task schemas
discovery = DiscoveryClient.from_context(manager)
topics = await discovery.get_topics()

# Run any advertised task using typed objects, attributes, and SDK models
client = ComputeClient(manager)
result = await client.geostatistics.declustering.run(
    source=pointset,
    grid=block_model,
    target=pointset.attributes["IDW declus"],
    neighborhood=neighborhood,
    preview=True,
)

# Results return a TaskResult (a dict subclass with ResultNode loaders)
updated_pointset = await result.target.load()
weights_df = await result.target.attribute.to_dataframe()
```

`ReferenceResolver` automatically converts typed geoscience objects, attributes, file handles, and target attribute slots into the wire references each task's schema declares (and verifies `supported_schemas` unless `check_schemas=False`). Parameters are validated before submission (`validate=True` by default; `deep_validation=True` for full JSON Schema Draft 2020-12 checks, raising `ParameterValidationError` on failure), and output references hydrate into `TaskResult` / `ResultNode` objects supporting `.load()` and `.to_dataframe()`. Tasks with custom SDK runners (such as `client.geostatistics.kriging`, backed by `KrigingOverride`) expose additional cross-field checks, convenience parameters (`source_filter`, `target_filter`, `diagnostics`), and native result wrappers (`KrigingResult`) across both attribute access and `client.arun(...)`.

### Submitting and Managing Jobs with `TaskJob`

Every task also exposes `submit(...)` (and `client.asubmit(...)` by name) to return a `TaskJob` without waiting, and `TaskJob.from_url(...)` to restore a job later from its saved status URL:

```python
from evo.compute import TaskJob

job = await client.geostatistics.declustering.submit(
    source=pointset,
    grid=block_model,
    target=pointset.attributes["IDW declus"],
    neighborhood=neighborhood,
    preview=True,
)
progress = await job.status()

# Reopen a saved status URL later and wait for hydrated results
reopened = await TaskJob.from_url(manager, job.url, preview=True)
result = await reopened.results()
```

### Static Typing and Task Catalogue Reference

`evo-compute` ships a generated static type stub (`evo/compute/engine.pyi`) and `py.typed` marker built from a snapshot of discoverable compute task JSON schemas, giving editors and `mypy` / `pyright` full autocomplete, parameter `TypedDict` checking, and typed `TaskResult` / `ResultNode` / `TaskJob` hints on `client.<topic>.<task>.run(...)` and `submit(...)` while keeping runtime execution live.

In the SDK reference sidebar under **`tasks`**, each topic page (`converter`, `cpt-to-borehole`, `geology`, `geophysics`, `geostatistics`, `geotech`, `isosurfacing`, `rbf`, `utilities`) documents the snapshotted tasks' `run(...)` and `submit(...)` signatures, accepted SDK input shorthands, nested dictionary/union shapes, and hydrated result loaders. Tasks published to the platform after the snapshot remain callable via `client.<topic>.<task>.run(...)`, `submit(...)`, `arun(...)`, or `asubmit(...)`.

## Running Typed Geostatistics Tasks

The `run()` function executes typed task parameter models — either a single task or multiple tasks concurrently.

### Single task

```python
from evo.compute.tasks import run, SearchNeighborhood, Ellipsoid, EllipsoidRanges
from evo.compute.tasks.geostatistics.kriging import KrigingParameters

params = KrigingParameters(
    source=pointset.attributes["grade"],
    target=block_model.attributes["kriged_grade"],  # Creates if new, updates if exists
    variogram=variogram,
    neighborhood=SearchNeighborhood(
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
from evo.compute.tasks.geostatistics.kriging import KrigingParameters

results = await run(manager, [
    KrigingParameters(
        source=pointset.attributes["Au"],
        target=block_model.attributes["Au_kriged"],
        variogram=au_variogram,
        neighborhood=SearchNeighborhood(...),
    ),
    KrigingParameters(
        source=pointset.attributes["Cu"],
        target=block_model.attributes["Cu_kriged"],
        variogram=cu_variogram,
        neighborhood=SearchNeighborhood(...),
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
from evo.compute.tasks import run, Filter, FilterCondition, SearchNeighborhood
from evo.compute.tasks.geostatistics.kriging import KrigingParameters

# Step 1: Run the first task — attribute "kriged_grade" does not exist yet, so it is created
first_result = await run(manager, KrigingParameters(
    source=pointset.attributes["grade"],
    target=block_model.attributes["kriged_grade"],
    variogram=variogram,
    neighborhood=SearchNeighborhood(...),
    target_filter=Filter(
        where=FilterCondition(
            attribute=block_model.attributes["domain"],
            operator="in",
            values=["LMS1"],
        ),
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
        neighborhood=SearchNeighborhood(...),
        target_filter=Filter(
            where=FilterCondition(
                attribute=block_model.attributes["domain"],
                operator="in",
                values=["LMS2"],
            ),
        ),
    ),
    KrigingParameters(
        source=pointset.attributes["grade"],
        target=block_model.attributes["kriged_grade"],  # Exists → update
        variogram=variogram,
        neighborhood=SearchNeighborhood(...),
        target_filter=Filter(
            where=FilterCondition(
                attribute=block_model.attributes["domain"],
                operator="in",
                values=["LMS3"],
            ),
        ),
    ),
], preview=True)
```

### Can I run multiple tasks in parallel that update the same attribute?
Block models support a fully parallelised workflow. If each task writes to a **different** attribute name, they can all run in parallel without refreshing. If the attribute was not created yet, run a task to create it and then run parallel tasks to update it with compute results. See the [multiple kriging notebook](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-compute/docs/examples/kriging_multiple.ipynb) for an example.
Storing computation results on Pointsets and 3D grids is supported but the tasks can't run in parallel.

### Preview APIs
Kriging and other compute tasks are currently preview features. You must pass `preview=True` when calling `run()`.
Preview APIs may change between releases. For more details, see:

- [Preview APIs](https://developer.seequent.com/docs/api/fundamentals/preview-apis) — how to opt in and what to expect
- [API Lifecycle](https://developer.seequent.com/docs/api/fundamentals/lifecycle) — how Evo APIs evolve from preview to stable


<p align="center"><a href="https://seequent.com" target="_blank"><picture><source media="(prefers-color-scheme: dark)" srcset="https://developer.seequent.com/img/seequent-logo-dark.svg" alt="Seequent logo" width="400" /><img src="https://developer.seequent.com/img/seequent-logo.svg" alt="Seequent logo" width="400" /></picture></a></p>
<p align="center">
    <a href="https://pypi.org/project/evo-compute/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-compute" /></a>
    <a href="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml"><img src="https://github.com/SeequentEvo/evo-python-sdk/actions/workflows/run-all-tests.yaml/badge.svg" alt="" /></a>
</p>
<p align="center">
    <a href="https://developer.seequent.com/" target="_blank">Seequent Developer Portal</a>
    &bull; <a href="https://community.seequent.com/group/19-evo" target="_blank">Seequent Community</a>
    &bull; <a href="https://seequent.com" target="_blank">Seequent website</a>
</p>

# Evo Compute Task Client

The Compute Task API provides the ability to execute computation tasks in Evo that require variable, on-demand processing power. Building the Compute Task API into your application can enable fast processing of long running, or resource intensive operations, without depending on the end user's physical hardware.

Tasks are created, triggering a job to be executed asynchronously within a specific topic for your organization, and can be monitored throughout their execution lifecycle.

## Pre-requisites

* Python ^3.10
* An [application registered in Bentley](https://developer.bentley.com/register/?product=seequent-evo)

## Installation

```shell
pip install evo-compute
```

## Usage

See [the evo-sdk-common documentation](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-sdk-common/README.md)
for information on how to authenticate, then select the organisation, hub and workspace that you would like to use.

### Interacting with the Compute Task API

To get up and running quickly with the Evo Compute Task SDK, start by configuring your
[environment and API connector](https://github.com/SeequentEvo/evo-python-sdk/blob/main/packages/evo-sdk-common/docs/quickstart.md).

For some interactive Jupyter notebook examples, see the [examples folder](docs/examples).

### Running tasks through the generic engine

Authenticate and load inputs through the existing async SDK APIs, then use a client on
the same event loop:

```python
from evo.compute import ComputeClient

client = ComputeClient(manager)
result = await client.geostatistics.kriging.run(
    source=samples.attributes["grade"],
    target=target_model.attributes["kriged_grade"],
    variogram=variogram,
    neighborhood=search,
    preview=True,
)
dataframe = await result.to_dataframe()
```

Kriging opts into a handwritten runner with additional checks and native `KrigingResult`.
Its arguments use the same names as other estimation tasks: `neighborhood` and
`kriging_method`, with `block_discretization` for optional sub-block settings.
The engine adapts the latter to older catalogues that publish `block_discretisation`.
Other tasks use the schema-driven generic surface.
`client.arun(topic, task, parameters)` uses the same runner, argument names and result type
as the task's `run(...)` method, including for tasks absent from the snapshot.
Preview is off by default on every path; pass `preview=True` to opt in.
The existing lowercase task modules and `await tasks.run(manager, parameters)` API are unchanged.

Existing parameter models also use the generic engine through `tasks.run`:

```python
from evo.compute import tasks

results = await tasks.run(
    manager, [kriging_parameters, idw_parameters, knn_parameters], preview=True
)
```

The models keep their existing serialization, including defaults, filters and diagnostics.
The engine shares discovery across direct calls and repeated batches in the same context
until the catalogue cache expires, and submits batch jobs concurrently. Registered
runners convert their results back to the native SDK classes. A single input still returns
its native result, and a list still returns an ordered `TaskResultList` with notebook
formatting and aggregate progress. Preview remains an explicit opt-in on this facade.
These calls now require access to the task catalogue as well as job submission. They also
check the parameters against the task schema and, where the task lists supported object
schemas, load the metadata of each referenced object that is not already loaded to confirm
its schema. That adds requests and can reject inputs an older runner would have submitted.
This facade always runs those checks; to skip the object checks while keeping parameter
validation, call the task through `ComputeClient(context, check_schemas=False)`.
Direct `TaskRunner` usage follows the same engine path, checks and result conversion.

The [estimation engine notebook](docs/examples/estimation_engine.ipynb) follows the
existing estimation workflow, showing both the unchanged `tasks.run(...)` call and
explicit async job submission with the same parameter models, result queries and plots.

### Submitting and managing jobs

Tasks expose async `submit(...)` to return the accepted job without waiting:

```python
job = await client.geostatistics.declustering.submit(
    source=samples,
    grid=grid,
    target=weights,
    neighborhood=search,
    preview=True,
)
progress = await job.status()
result = await job.results()
dataframe = await result.target.attribute.to_dataframe()
```

`TaskJob` preserves the job's `id`, `url`, `topic` and `task`. Use `await job.cancel()`
to request cancellation, and `await job.status()` to observe it. `job.results()` accepts
feedback, polling interval and retry overrides; feedback set with `ComputeClient(manager,
fb=feedback)` is inherited by each job. Callbacks run on the caller's async event loop.

`run(...)` still waits for results using the same preparation and hydration path. The
Kriging override also provides typed submission: `await client.geostatistics.kriging.submit(...)`
takes the same arguments as `run`, and `await job.results()` returns `KrigingResult` with
its dataframe and object helpers. Named calls through `client.asubmit(topic, task, parameters)`
use the same override checks and native result conversion, not a bypass.

Progress feedback belongs on `ComputeClient(..., fb=feedback)` or on
`await job.results(fb=feedback)`, not among the task's parameters. This keeps local
wait controls separate from service inputs and avoids reserving a task keyword.
Client feedback defaults to `NoFeedback`; the existing `tasks.run(..., fb=feedback)`
batch interface retains its aggregate notebook progress.

Save `job.url` to reopen the job later:

```python
from evo.compute import TaskJob

job = await TaskJob.from_url(manager, saved_job_url, preview=True)
result = await job.results()
```

Restoration checks the hub and organization, reads the task's current result schema,
and restores native conversion for custom tasks such as Kriging. Preview is not stored
in the URL, so opt in again only for preview jobs. A URL cannot determine the result
type statically; use the typed task's `submit` method when static result hints matter.
Custom adapters can use `job.map(async_converter)` to convert results without changing
the underlying job, status, cancellation or wait options.

### Static typing for the generic engine

`ComputeClient` resolves `client.<topic>.<task>.run(...)` from the live task catalogue, so
there is no hand-written code for a type checker to read. A generated type stub gives
editors completion, signature help and parameter checking for every task in a checked-in
catalogue snapshot, while execution stays fully generic. See
[stubs/README.md](stubs/README.md) for how to regenerate it and refresh the snapshot.

For compatible neighborhood schemas, generated client signatures accept the
existing `SearchNeighborhood` model alongside schema-shaped dictionaries. Its native
`Ellipsoid`, ranges and rotation objects keep their existing serialization; the type hints
describe behavior the resolver already supports. Matching is based on serialized field
structure, not a task or schema name. Unrelated models and malformed dictionaries are still
rejected by the type checkers.

## Contributing

For instructions on contributing to the development of this library, please refer to the [evo-python-sdk documentation](https://github.com/seequentevo/evo-python-sdk).

## License

The Python SDK for Evo is open source and licensed under the [Apache 2.0 license.](./LICENSE.md).

Copyright © 2025 Bentley Systems, Incorporated.

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

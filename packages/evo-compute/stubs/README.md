# Static typing for the generic compute engine

`ComputeClient` builds its `client.<topic>.<task>.run(...)` namespace at runtime from the
live discovery catalogue. Nothing about that surface exists in `engine.py`, so a type
checker sees `Any` and an editor offers no completion.

This directory closes that gap. An **offline** generator turns a checked-in snapshot of the
task catalogue into [`src/evo/compute/engine.pyi`](../src/evo/compute/engine.pyi), which
type checkers read instead of `engine.py`. Every snapshotted task then gets completion,
signature help, hover documentation, parameter type-checking and a typed result through
the async client.

```
stubs/
  snapshot/
    manifest.json                    # provenance: source endpoint, capture date, task versions
    <topic>/<task>.json              # the discovery payload for one task, verbatim
  checks/
    usage_ok.py                      # must type-check clean
    usage_bad.py                     # must be rejected, one way per mistake
```

## Using the generated types

Passing a dictionary literal needs no import — the checker knows the expected shape from
the call site, and the editor completes the keys:

```python
result = await client.geostatistics.declustering.run(
    source={"object": pointset},
    grid={"object": grid_url},
    target={"object": pointset, "attribute": {"operation": "create", "name": "weight"}},
    neighborhood={"ellipsoid": ..., "max_samples": 20},
)
print(result["message"])           # the raw payload is still a dict
weights = await result.target.attribute.to_dataframe()   # and the references load
```

To name a shape in your own annotations, import it under `TYPE_CHECKING`. The generated
types exist only in the stub — they are not runtime objects:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evo.compute.engine import GeostatisticsDeclusteringResult
```

## Regenerating

```shell
make stubs-compute                        # rewrite the compute engine stub
make check-stubs-compute                  # fail if the stub is stale
make check-types-compute                  # check good and bad examples with mypy and pyright
python -m evo.compute._stubgen generate   # the same thing, without uv
```

Generation reads the snapshot and existing SDK declarations: no credentials, no network, and
no generated execution wrappers. It writes only `engine.pyi`. The normal generator tests
compare it against its committed version, so snapshot or declaration changes require regeneration.

## Refreshing the snapshot

Capture is the one online step and is run by hand, not by CI:

```shell
EVO_ACCESS_TOKEN=... EVO_HUB_URL=... EVO_ORG_ID=... \
    python -m evo.compute._stubgen capture
make stubs-compute
```

The snapshot is versioned as source: the task payloads and `manifest.json` are committed,
so the diff shows exactly which task changed and how, and the stub is reproducible from
any commit. That only holds if a refresh writes the files the same way it found them, so
the serialiser is pinned and a test asserts the committed files are already in capture
format. `manifest.json` records the endpoint the snapshot came from, the date it was
captured, and each task's version; a test asserts it still agrees with the files beside it.

## Decisions

**One definition of the client, in `engine.pyi`.** A stub for the compute package's `__init__`
would declare a *second* `ComputeClient` that has nothing to do with
`evo.compute.engine.ComputeClient`, so the two would disagree depending on how the class
was imported. Stubbing the module that defines the class keeps one type, and
`evo/compute/__init__.py` re-exports it as it already does.

**Client namespace stubs declare `__getattr__`, so unlisted tasks are untyped rather than forbidden.**
The catalogue is live, scoped per organization, and moves between SDK releases, so a shipped
artifact can never be all of it — a snapshot is always simultaneously too small and too
specific to wherever it was captured. Execution is live while hints are point-in-time: a
task added after the snapshot still **runs**, it just does not appear in autocomplete.
Declaring `__getattr__` is what keeps that true. Without it, unlisted tasks in the
catalogue become false type errors on valid code, undermining full catalogue coverage.

The cost is that a misspelled *task name* is no longer a static error — it resolves to the
generic proxy and fails at run time, when discovery cannot find it. Everything else the
stub checks is unaffected: parameter names, scalar types, enum members and result shapes all
still come from the generated types for any task in the snapshot. This also matches the rest
of the SDK, where a task with no hand-written client falls back to `JobClient.submit` rather
than being refused.

**The snapshot is a curated investment, not a mirror.** Because nothing breaks when a task
is absent, which tasks to snapshot is a cost/benefit choice rather than a correctness one:
more tasks means more precise hints and a larger generated artifact. `manifest.json` records
exactly what was taken and from where, and
`tests/test_schema_conformance.py::test_the_snapshot_still_describes_tasks_the_catalogue_advertises`
reports drift against a live catalogue when run with credentials.

**Results are `TaskResult` / `ResultNode` subclasses, not `TypedDict`s.** The engine hydrates
what the platform sends into nodes that are still dictionaries but additionally load what
their references point at. Generating the result tree as classes is what makes
`result.target.attribute.name` typed *and* `await result.target.load()` available. The cost
is that those classes inherit `__getattr__`, so an unknown key is no longer a static error —
`stubs/checks/usage_bad.py` pins the guarantee that survives, which is that a declared key
carries its real type all the way down.

**Reference parameters are typed by what you may pass, not by what goes on the wire.** A
property annotated `reference_to: geoscience-object` is a URL string on the wire, but
`resolution.ReferenceResolver` runs before submission and accepts any handle it can derive
that URL from, so the stub says `ObjectInput` — `str | UUID | BaseObject | DownloadedObject |
ObjectMetadata | ObjectReference`. Deep validation still holds the leaf to the declared
string, because by then resolution has already run. The two describe different moments, and
`TestDeepValidationAgreement` checks the part they share: a value neither side could make
sense of, an `int` where a reference belongs, is rejected by both.

**Shorthands are typed too.** `ReferenceResolver` expands a few abbreviations the typed
tasks already accept, so the generated signatures offer them rather than insisting on the
spelled-out shape: a parameter wrapping a single object reference takes the object
(`grid=block_model`), one shaped `{object, attribute}` takes a typed attribute, which knows
both (`source=pointset.attributes["grade"]`), and a `target: attribute` slot takes the name
of the attribute to create. Each is detected from the same schema shape the resolver
branches on, and unioned with the declared `TypedDict`.

**Names are topic/task-scoped and de-duplicated.** Every generated type is prefixed with its
topic and task, so two tasks never fight over `Source`; result-side shapes take a further `Result`
prefix so an input and an output that share a schema title stay distinct. Within a task,
structurally identical objects collapse onto one type — the published schemas inline the
same filter shape at four different depths. Only shapes with the same base collapse, so an
input `TypedDict` is never reused for a result that has to hydrate.

## The runtime half

The stub is one half of a contract the engine enforces at run time from the same schema:

| Mistake | Caught statically by | Caught at run time by |
|---|---|---|
| unknown topic or task | *nothing — by design* | discovery lookup in `arun` |
| unknown parameter | the stub's `run(...)` signature | signature binding at the top level; JSON Schema rules under deep validation |
| missing required parameter | the stub's `run(...)` signature | signature binding, then `validate_parameters` |
| wrong scalar type, bad enum member, missing nested field | the stub's `TypedDict`s and `Literal`s | `validate_parameters(..., deep=True)` |
| a reference nothing could resolve | the stub's `ObjectInput` / `AttributeInput` unions | `ReferenceResolver`, then `validate_parameters(..., deep=True)` |

Deep validation is opt-in — `ComputeClient(context, deep_validation=True)`, or
`arun(..., deep_validation=True)` for a single call — because it costs a JSON Schema
pass. `tests/test_stubgen.py::TestDeepValidationAgreement` asserts the two halves agree:
the payload from `usage_ok.py` passes deep validation, and its known-value and reference
mistakes are rejected at run time too.

## Verified with

| Checker | Result |
|---|---|
| pyright | `usage_ok.py` clean; `usage_bad.py` reports all seven mistakes |
| mypy | `usage_ok.py` clean; `usage_bad.py` reports all seven mistakes |
| PyCharm / VS Code | completion, signature help and hover text on `client.<topic>.<task>.run(...)` |

`tests/test_stubgen.py` runs both checkers as required test dependencies.
The pull request workflow also runs `make check-types-compute` explicitly, checking that
the good examples pass and every expected mistake in the bad examples is rejected.

#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Correct use of the generated stub. A type checker must report **zero** errors here.

Run ``pyright stubs/checks/usage_ok.py`` (or ``mypy``) from ``packages/evo-compute``, or
let ``tests/test_stubgen.py`` do it when the checker is installed.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

from evo.common import IContext
from evo.objects.typed import Attribute, BaseObject, PendingAttribute
from typing_extensions import assert_type

from evo.compute import ComputeClient, JobProgress, TaskJob, TaskResult
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges, Rotation, SearchNeighborhood
from evo.compute.tasks.geostatistics.kriging import KrigingMethod, KrigingResult

if TYPE_CHECKING:
    from evo.compute.engine import GeostatisticsDeclusteringResult


def _model_neighborhood() -> SearchNeighborhood:
    return SearchNeighborhood(
        ellipsoid=Ellipsoid(
            ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            rotation=Rotation(dip_azimuth=45.0, dip=10.0, pitch=5.0),
        ),
        max_samples=20,
        min_samples=4,
        max_empty_octants=None,
        max_samples_per_octant=None,
        max_empty_quadrants=None,
        max_samples_per_quadrant=None,
        max_samples_per_drillhole=None,
        max_drillholes_per_estimate=None,
    )


async def neighborhood_objects_async(context: IContext) -> None:
    client = ComputeClient(context)
    result = await client.geostatistics.declustering.run(
        source="https://example.com/objects/samples",
        grid="https://example.com/objects/grid",
        target={"object": "https://example.com/objects/target", "attribute": "weights"},
        neighborhood=_model_neighborhood(),
        power=2.5,
    )
    await result.target.attribute.to_dataframe()


async def submit_then_wait(context: IContext) -> None:
    client = ComputeClient(context)
    job = await client.geostatistics.declustering.submit(
        source="https://example.com/objects/samples",
        grid="https://example.com/objects/grid",
        target={"object": "https://example.com/objects/target", "attribute": "weights"},
        neighborhood=_model_neighborhood(),
        power=2.5,
    )
    assert_type(job, TaskJob[GeostatisticsDeclusteringResult])
    assert_type(job.id, UUID)
    assert_type(job.url, str)
    assert_type(await job.status(), JobProgress)
    result = await job.results(polling_interval_seconds=0.0)
    assert_type(result, GeostatisticsDeclusteringResult)
    assert_type(result.target.attribute.name, str)
    await result.target.attribute.to_dataframe()
    await job.cancel()


async def submit_unknown_tasks(context: IContext) -> None:
    client = ComputeClient(context)
    job = await client.geology.some_task.submit(anything=1)
    assert_type(job, TaskJob[TaskResult])
    assert_type(await job.results(), TaskResult)
    generic = await client.asubmit("new-topic", "new-task", {"source": "..."})
    assert_type(generic, TaskJob[Any])


async def submit_native_kriging(context: IContext, source: Attribute, target: PendingAttribute) -> None:
    job = await ComputeClient(context).geostatistics.kriging.submit(
        source=source,
        target=target,
        variogram="https://example.com/objects/variogram",
        neighborhood=_model_neighborhood(),
        kriging_method=KrigingMethod.ORDINARY,
    )
    assert_type(job, TaskJob[KrigingResult])
    result = await job.results()
    assert_type(result, KrigingResult)
    assert_type(result.attribute_name, str)
    await result.get_target_object()
    await result.to_dataframe()

    async def target_name(native: KrigingResult) -> str:
        return native.target_name

    mapped = job.map(target_name)
    assert_type(mapped, TaskJob[str])
    assert_type(await mapped.results(), str)


async def declustering(context: IContext) -> None:
    client = ComputeClient(context)
    result = await client.geostatistics.declustering.run(
        source={"object": "https://example.com/objects/samples"},
        grid={"object": "https://example.com/objects/grid"},
        target={
            "object": "https://example.com/objects/samples",
            "attribute": {"operation": "create", "name": "declustering_weight"},
        },
        neighborhood={
            "ellipsoid": {
                "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 50.0},
                "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
            },
            "max_samples": 20,
        },
        power=2.0,
    )
    # The raw payload is still a dict, and the declared keys are typed attributes on top.
    print(result["message"], result.target.attribute.name)
    # What the result refers to is loadable, because the node carries an ``output`` annotation.
    await result.target.load()
    await result.target.attribute.to_dataframe()


async def normal_score(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.normal_score.run(
        method="forward",
        source={
            "object": "https://example.com/objects/samples",
            "attribute": "grade",
        },
        distribution="https://example.com/objects/distribution",
        target={
            "object": "https://example.com/objects/samples",
            "attribute": {"operation": "create", "name": "grade_ns"},
        },
    )


async def not_in_the_snapshot(context: IContext) -> None:
    """A task published after the snapshot still runs -- ``arun`` is the typed escape hatch."""
    client = ComputeClient(context)
    result: dict = await client.arun("geostatistics", "some-new-task", {"source": "..."})
    print(result)


async def a_topic_the_snapshot_has_never_seen(context: IContext) -> None:
    """Execution is live; hints are point-in-time. An unlisted name is untyped, not wrong.

    The catalogue is per-organization and moves between SDK releases, so the snapshot is
    never all of it. These calls reach the engine exactly as a snapshotted task does; the
    only thing missing is the parameter and result types.
    """
    client = ComputeClient(context)
    await client.geology.some_task.run(anything=1)
    await client.geostatistics.a_task_added_last_week.run(source="...")


async def typed_handles(context: IContext, pointset: BaseObject, weights: PendingAttribute) -> None:
    """Reference resolution takes the handles the typed tasks take, so the stub does too.

    A parameter that wraps a single object reference accepts the object; one shaped
    ``{object, attribute}`` accepts a typed attribute, which knows both; and a ``target:
    attribute`` slot accepts the name of the attribute to create.
    """
    client = ComputeClient(context)
    await client.geostatistics.declustering.run(
        source=pointset,
        grid=pointset,
        target=weights,
        neighborhood={
            "ellipsoid": {
                "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 50.0},
                "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
            },
            "max_samples": 20,
        },
    )


async def typed_attribute_source(context: IContext, grade: Attribute, kriged: PendingAttribute) -> None:
    client = ComputeClient(context)
    result = await client.geostatistics.kriging.run(
        source=grade,
        target=kriged,
        variogram="https://example.com/objects/variogram",
        neighborhood=SearchNeighborhood(
            ellipsoid=Ellipsoid(ranges=EllipsoidRanges(major=100.0, semi_major=100.0, minor=50.0)),
            max_samples=20,
        ),
        kriging_method=KrigingMethod.ORDINARY,
    )
    assert_type(result, KrigingResult)
    print(result.target_name, result.attribute_name)
    await result.get_target_object()

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

"""Incorrect use of the generated stub. Every call below must be rejected statically.

``tests/test_stubgen.py`` asserts that the checker's report mentions each name listed in
``EXPECTED_ERRORS``, which is what makes this file a regression test for the stub rather
than a pile of broken code.
"""

from evo.common import IContext
from pydantic import BaseModel

from evo.compute import ComputeClient
from evo.compute.tasks.common import Ellipsoid, EllipsoidRanges, SearchNeighborhood, Source, Target

EXPECTED_ERRORS = [
    "neighborhood",  # required parameter omitted
    "sourse",  # misspelled parameter
    "power",  # wrong scalar type
    "method",  # value outside the schema's enum
    "upper_case",  # result attribute used as something other than the type it declares
    "NotANeighborhood",
    "target",
    "attribute_name",
]


class NotANeighborhood(BaseModel):
    value: str


async def unrelated_neighborhood_model_async(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.declustering.run(
        source="https://example.com/objects/samples",
        grid="https://example.com/objects/grid",
        target={"object": "https://example.com/objects/target", "attribute": "weights"},
        neighborhood=NotANeighborhood(value="invalid"),
    )


async def malformed_neighborhood_dictionary(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.declustering.run(
        source="https://example.com/objects/samples",
        grid="https://example.com/objects/grid",
        target={"object": "https://example.com/objects/target", "attribute": "weights"},
        neighborhood={
            "ellipsoid": {
                "ellipsoid_ranges": {"major": 200.0, "semi_major": 150.0, "minor": 100.0},
                "rotation": {"dip_azimuth": 45.0, "dip": 10.0, "pitch": 5.0},
            },
            "max_samples": "invalid",
        },
    )


async def unrelated_neighborhood_model_submit(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.declustering.submit(
        source="https://example.com/objects/samples",
        grid="https://example.com/objects/grid",
        target={"object": "https://example.com/objects/target", "attribute": "weights"},
        neighborhood=NotANeighborhood(value="invalid"),
    )


async def unawaited_job_results(context: IContext) -> None:
    client = ComputeClient(context)
    job = await client.geology.some_task.submit(anything=1)
    result = job.results()
    print(result.target)


async def misused_native_submission(context: IContext) -> None:
    job = await ComputeClient(context).geostatistics.kriging.submit(
        source=Source(object="https://example.com/objects/source", attribute="grade"),
        target=Target.new_attribute("https://example.com/objects/target", "estimate"),
        variogram="https://example.com/objects/variogram",
        neighborhood=SearchNeighborhood(ellipsoid=Ellipsoid(ranges=EllipsoidRanges(200, 150, 100)), max_samples=20),
    )
    pending_result = job.results()
    print(pending_result.attribute_name)
    result = await job.results()
    attribute_name: int = result.attribute_name
    print(attribute_name)


async def missing_required_parameter(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.declustering.run(
        source={"object": "https://example.com/objects/samples"},
        grid={"object": "https://example.com/objects/grid"},
        target={
            "object": "https://example.com/objects/samples",
            "attribute": {"operation": "create", "name": "declustering_weight"},
        },
    )


async def misspelled_parameter(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.declustering.run(
        sourse={"object": "https://example.com/objects/samples"},
        grid={"object": "https://example.com/objects/grid"},
        target={
            "object": "https://example.com/objects/samples",
            "attribute": {"operation": "create", "name": "declustering_weight"},
        },
        neighborhood={
            "ellipsoid": {
                "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 50.0},
                "rotation": {},
            },
            "max_samples": 20,
        },
    )


async def wrong_scalar_type(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.declustering.run(
        source={"object": "https://example.com/objects/samples"},
        grid={"object": "https://example.com/objects/grid"},
        target={
            "object": "https://example.com/objects/samples",
            "attribute": {"operation": "create", "name": "declustering_weight"},
        },
        neighborhood={
            "ellipsoid": {
                "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 50.0},
                "rotation": {},
            },
            "max_samples": 20,
        },
        power="strong",
    )


async def value_outside_the_enum(context: IContext) -> None:
    client = ComputeClient(context)
    await client.geostatistics.normal_score.run(
        method="sideways",
        source={"object": "https://example.com/objects/samples", "attribute": "grade"},
        distribution="https://example.com/objects/distribution",
        target={
            "object": "https://example.com/objects/samples",
            "attribute": {"operation": "create", "name": "grade_ns"},
        },
    )


async def misused_result_attribute(context: IContext) -> None:
    """A result node inherits ``__getattr__``, so an unknown key cannot be caught.

    What the generated types do guarantee is that the keys the schema *does* declare carry
    their real types all the way down, which is what this checks.
    """
    client = ComputeClient(context)
    result = await client.geostatistics.normal_score.run(
        method="forward",
        source={"object": "https://example.com/objects/samples", "attribute": "grade"},
        distribution="https://example.com/objects/distribution",
        target={
            "object": "https://example.com/objects/samples",
            "attribute": {"operation": "create", "name": "grade_ns"},
        },
    )
    print(result.target.attribute.name.upper_case())

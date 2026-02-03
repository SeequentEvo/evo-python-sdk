#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .attributes import Attribute, Attributes, PendingAttribute
from .base import object_from_path, object_from_reference, object_from_uuid
from .block_model_ref import (
    BlockModel,
    BlockModelAttribute,
    BlockModelAttributes,
    BlockModelData,
    BlockModelGeometry,
    RegularBlockModelData,
)
from .ellipsoid import Ellipsoid
from .ellipsoid import EllipsoidRanges as EllipsoidRanges  # For search/visualization
from .ellipsoid import Rotation as EllipsoidRotation  # For search/visualization
from .pointset import (
    Locations,
    PointSet,
    PointSetData,
)
from .regular_grid import (
    Regular3DGrid,
    Regular3DGridData,
)
from .regular_masked_grid import (
    MaskedCells,
    RegularMasked3DGrid,
    RegularMasked3DGridData,
)
from .tensor_grid import (
    Tensor3DGrid,
    Tensor3DGridData,
)
from .types import BoundingBox, CoordinateReferenceSystem, EpsgCode, Point3, Rotation, Size3d, Size3i
from .variogram import (
    Anisotropy,
    CubicStructure,
    EllipsoidRanges as VariogramEllipsoidRanges,  # For variogram structure data definition
    ExponentialStructure,
    GaussianStructure,
    GeneralisedCauchyStructure,
    LinearStructure,
    SphericalStructure,
    SpheroidalStructure,
    Variogram,
    VariogramCurveData,
    VariogramData,
    VariogramRotation,
    VariogramStructure,
)

__all__ = [
    "Attribute",
    "Attributes",
    "Anisotropy",
    "BlockModel",
    "BlockModelAttribute",
    "BlockModelAttributes",
    "BlockModelData",
    "BlockModelGeometry",
    "BoundingBox",
    "CoordinateReferenceSystem",
    "CubicStructure",
    "Ellipsoid",
    "EllipsoidRanges",
    "EllipsoidRotation",
    "EpsgCode",
    "ExponentialStructure",
    "GaussianStructure",
    "GeneralisedCauchyStructure",
    "LinearStructure",
    "Locations",
    "MaskedCells",
    "PendingAttribute",
    "Point3",
    "PointSet",
    "PointSetData",
    "Regular3DGrid",
    "Regular3DGridData",
    "RegularBlockModelData",
    "RegularMasked3DGrid",
    "RegularMasked3DGridData",
    "Rotation",
    "Size3d",
    "Size3i",
    "SphericalStructure",
    "SpheroidalStructure",
    "Tensor3DGrid",
    "Tensor3DGridData",
    "Variogram",
    "VariogramCurveData",
    "VariogramData",
    "VariogramEllipsoidRanges",
    "VariogramRotation",
    "VariogramStructure",
    "object_from_path",
    "object_from_reference",
    "object_from_uuid",
]

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

"""Utilities for the indexed tables used by downhole collections."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["expand_hole_index", "hole_chunks_from_ids"]


def hole_chunks_from_ids(hole_ids: pd.Series, *, dtype: pd.CategoricalDtype) -> pd.DataFrame:
    """Run-length encode contiguous hole IDs using their categorical codes.

    A zero-count entry is emitted for categories absent from ``hole_ids``.  An ID
    outside ``dtype`` or a repeated non-contiguous run is rejected.
    """
    unknown_mask = hole_ids.notna() & ~hole_ids.isin(dtype.categories)
    if unknown_mask.any():
        unknown = hole_ids[unknown_mask].unique().tolist()
        raise ValueError(f"hole_ids contains values absent from dtype: {unknown}")
    categorical = hole_ids.astype(dtype)
    codes = categorical.cat.codes.to_numpy(dtype=np.int32)
    chunks: dict[int, tuple[int, int]] = {}
    start = 0
    while start < len(codes):
        code = int(codes[start])
        if code < 0:
            raise ValueError("hole_ids cannot contain missing values")
        end = start + 1
        while end < len(codes) and codes[end] == code:
            end += 1
        if code in chunks:
            raise ValueError("Rows for each hole_id must be contiguous")
        chunks[code] = (start, end - start)
        start = end
    return pd.DataFrame(
        {
            "hole_index": np.arange(len(dtype.categories), dtype=np.int32),
            "offset": np.array([chunks.get(code, (0, 0))[0] for code in range(len(dtype.categories))], dtype=np.uint64),
            "count": np.array([chunks.get(code, (0, 0))[1] for code in range(len(dtype.categories))], dtype=np.uint64),
        }
    )


def expand_hole_index(holes: pd.DataFrame, num_rows: int) -> pd.Series:
    """Expand ``hole_index`` chunks into a per-row nullable integer Series."""
    required = {"hole_index", "offset", "count"}
    if missing := required - set(holes.columns):
        raise ValueError(f"holes is missing columns: {sorted(missing)}")
    result = pd.Series(pd.array([pd.NA] * num_rows, dtype="Int32"))
    for chunk in holes.itertuples(index=False):
        offset, count = int(chunk.offset), int(chunk.count)
        if offset < 0 or count < 0 or offset + count > num_rows:
            raise ValueError("Hole chunk offsets and counts must be within num_rows")
        result.iloc[offset : offset + count] = int(chunk.hole_index)
    return result

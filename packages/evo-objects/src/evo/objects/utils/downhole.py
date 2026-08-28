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

from collections.abc import Mapping

import numpy as np
import pandas as pd

__all__ = ["expand_hole_index", "hole_chunks_from_ids"]


def hole_chunks_from_ids(hole_ids: pd.Series, *, hole_indices: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Run-length encode contiguous hole IDs into ``[hole_index, offset, count]`` chunks.

    ``hole_indices`` maps hole IDs to their lookup-table keys. When omitted, the
    keys are dense, zero-based, and assigned in sorted ID order. Explicit mappings
    may use other unique keys. Only IDs present in ``hole_ids`` produce chunks; an
    empty collection therefore produces no chunks.
    """
    if hole_ids.isna().any():
        raise ValueError("hole_ids cannot contain missing values")
    ids = hole_ids.astype(str)
    if hole_indices is None:
        indices = {hole_id: index for index, hole_id in enumerate(sorted(set(ids)))}
    else:
        indices = dict(hole_indices)
        unknown = sorted(set(ids) - set(indices))
        if unknown:
            raise ValueError(f"hole_ids contains values absent from hole_indices: {unknown}")
        if len(set(indices.values())) != len(indices):
            raise ValueError("hole_indices must map each hole_id to a unique hole_index")
    codes = ids.map(indices).to_numpy(dtype=np.int32)
    chunks: list[tuple[int, int, int]] = []
    start = 0
    while start < len(codes):
        code = int(codes[start])
        end = start + 1
        while end < len(codes) and codes[end] == code:
            end += 1
        if any(chunk_code == code for chunk_code, _, _ in chunks):
            raise ValueError("Rows for each hole_id must be contiguous")
        chunks.append((code, start, end - start))
        start = end
    return pd.DataFrame(
        {
            "hole_index": np.array([code for code, _, _ in chunks], dtype=np.int32),
            "offset": np.array([offset for _, offset, _ in chunks], dtype=np.uint64),
            "count": np.array([count for _, _, count in chunks], dtype=np.uint64),
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

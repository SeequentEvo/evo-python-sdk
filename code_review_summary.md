## Code Review Summary

**Files Reviewed**:
- [packages/evo-objects/src/evo/objects/typed/regular_masked_grid.py](packages/evo-objects/src/evo/objects/typed/regular_masked_grid.py)
- [packages/evo-objects/src/evo/objects/typed/tensor_grid.py](packages/evo-objects/src/evo/objects/typed/tensor_grid.py)
- [packages/evo-objects/src/evo/objects/typed/regular_grid.py](packages/evo-objects/src/evo/objects/typed/regular_grid.py)
- [packages/evo-objects/src/evo/objects/typed/types.py](packages/evo-objects/src/evo/objects/typed/types.py)
- [packages/evo-objects/src/evo/objects/typed/__init__.py](packages/evo-objects/src/evo/objects/typed/__init__.py)
- [packages/evo-objects/tests/typed/test_regular_masked_grid.py](packages/evo-objects/tests/typed/test_regular_masked_grid.py)
- [packages/evo-objects/tests/typed/test_tensor_grid.py](packages/evo-objects/tests/typed/test_tensor_grid.py)
- [packages/evo-objects/tests/typed/test_types.py](packages/evo-objects/tests/typed/test_types.py)
- [packages/evo-objects/tests/typed/helpers.py](packages/evo-objects/tests/typed/helpers.py)
- [packages/evo-objects/docs/examples/typed-objects.ipynb](packages/evo-objects/docs/examples/typed-objects.ipynb)

**Package(s) Affected**: evo-objects

**Overall Assessment**: APPROVE

### Brief Summary of Changes

This PR adds two new typed geoscience objects to the evo-objects package:

1. **RegularMasked3DGrid** - A regular 3D grid with a boolean mask indicating which cells are "active". Only active cells have attribute values, enabling efficient storage of sub-regions.

2. **Tensor3DGrid** - A 3D grid where cells can have different sizes along each axis, useful for variable-resolution grids.

Additionally, the PR refactors bounding box calculation into reusable class methods on `BoundingBox`.

### Changes Overview

1. **New `RegularMasked3DGrid` typed object** ([regular_masked_grid.py](packages/evo-objects/src/evo/objects/typed/regular_masked_grid.py)):
   - `RegularMasked3DGridData` dataclass with mask validation
   - `MaskedCells` class for managing active cells and attributes
   - Mask stored as bool-attribute with `BOOL_ARRAY_1` format
   - `number_of_active_cells` tracked in schema
   - `set_dataframe` supports optional mask update
   - Schema version 1.3.0

2. **New `Tensor3DGrid` typed object** ([tensor_grid.py](packages/evo-objects/src/evo/objects/typed/tensor_grid.py)):
   - `Tensor3DGridData` dataclass with cell size array validation
   - `NumpyFloat1D` custom Pydantic type for numpy array serialization
   - Reuses `Cells` and `Vertices` from regular_grid
   - Bounding box calculated from sum of cell sizes
   - Schema version 1.3.0

3. **BoundingBox enhancements** ([types.py](packages/evo-objects/src/evo/objects/typed/types.py)):
   - Added `BoundingBox.from_box(origin, extent, rotation)` class method
   - Added `BoundingBox.from_regular_grid(origin, size, cell_size, rotation)` class method
   - Added `Size3i.total_size` property
   - Rotation matrix correctly handles all 3 rotation angles (dip_azimuth, dip, pitch)

4. **Refactored regular_grid.py** ([regular_grid.py](packages/evo-objects/src/evo/objects/typed/regular_grid.py)):
   - Removed duplicate `_calculate_bounding_box` function
   - Now uses `BoundingBox.from_regular_grid()` class method

5. **Test coverage**:
   - 15 tests for `RegularMasked3DGrid` covering CRUD, mask update, validation, JSON structure
   - 14 tests for `Tensor3DGrid` covering CRUD, cell size validation, bounding box, varying sizes
   - 4 tests for new `BoundingBox` class methods

6. **Documentation** ([typed-objects.ipynb](packages/evo-objects/docs/examples/typed-objects.ipynb)):
   - Added examples for creating and downloading `RegularMasked3DGrid`
   - Added examples for creating and downloading `Tensor3DGrid`

### Critical Issues 🔴

None identified.

### Major Issues 🟠

1. **Missing version bump** - The `pyproject.toml` version for `evo-objects` is not updated. Since this PR adds new public API (`RegularMasked3DGrid`, `RegularMasked3DGridData`, `MaskedCells`, `Tensor3DGrid`, `Tensor3DGridData`), the version should be incremented (suggest `0.4.0` for new features).

### Minor Issues 🟡

1. **`MaskedCells.number_active` setter is a raw assignment** ([regular_masked_grid.py#L83](packages/evo-objects/src/evo/objects/typed/regular_masked_grid.py#L83)): The line `self.number_active = number_active` directly assigns to what appears to be a `SchemaProperty` descriptor. This works because `SchemaProperty.__set__` handles it, but it's inconsistent with other property patterns. Consider using the internal `_set_property_value` pattern or documenting this behavior.

2. **Missing `Vertices` support in `RegularMasked3DGrid`**: The `Regular3DGrid` supports both cell and vertex attributes, but `RegularMasked3DGrid` only supports cell attributes. If this is intentional (masked grids don't have vertex data), consider documenting this limitation.

3. **Validation tests could be non-async**: Tests like `test_mask_size_validation`, `test_cell_data_size_validation`, `test_cell_sizes_x_validation` don't use `await` and could be regular synchronous tests.

### Suggestions 💡

1. **Consider empty mask validation**: Add validation or tests for edge cases like:
   - Empty grids (size 0x0x0)
   - All-inactive masks (all False)
   - The all-inactive case is already tested (`test_all_inactive_mask`) ✓

2. **Consider `BaseObject` parameterized tests**: Similar to the PointSet tests, consider adding parameterized tests that verify `BaseObject.create()` and `BaseObject.from_reference()` correctly dispatch to the specific grid types.

3. **Document rotation convention**: The rotation matrix implementation in `Rotation.as_rotation_matrix()` uses a specific convention (clockwise intrinsic rotations: dip_azimuth → dip → pitch). This is correct for Evo but could benefit from a docstring explaining the convention matches Evo API expectations.

4. **Consider `Tensor3DGrid` inheriting from `DynamicBoundingBoxObject`**: Currently both `Regular3DGrid` and `Tensor3DGrid` inherit from `BaseSpatialObject`, but their bounding boxes are computed from data. Consider whether `DynamicBoundingBoxObject` should be used for consistency (though current approach works correctly).

### Positive Observations ✅

1. **Excellent test coverage**: 74 total typed object tests pass, with comprehensive coverage of CRUD operations, validation edge cases, and JSON structure verification.

2. **Clean refactoring of bounding box**: Moving the bounding box calculation to `BoundingBox.from_box()` and `BoundingBox.from_regular_grid()` class methods eliminates code duplication and provides a reusable API for any grid-based objects.

3. **Proper use of pyarrow for mask storage**: The mask is uploaded using `upload_table` with `BOOL_ARRAY_1` format, following the established pattern for boolean arrays.

4. **Well-designed `NumpyFloat1D` type**: The custom Pydantic type for numpy arrays with proper validation and serialization is cleanly implemented and reusable.

5. **Comprehensive validation**: Both new objects validate input data thoroughly:
   - Cell size array lengths match grid dimensions
   - Cell sizes are positive (tensor grid)
   - Mask length matches grid size
   - Cell data length matches active cell count (masked grid)
   - Vertex data length matches vertex count (tensor grid)

6. **Good documentation**: The notebook examples clearly demonstrate both creating and downloading the new grid types with realistic use cases.

7. **Backwards compatible**: All 19 existing `Regular3DGrid` tests pass, confirming the refactoring doesn't break existing functionality.

8. **Consistent schema versions**: Both new objects use schema version 1.3.0, matching expected Evo API versions.

9. **Proper mask data tracking**: The `_context.mark_modified()` pattern prevents reading stale mask data after modification.

10. **Rotation handling is correct**: The bounding box calculation correctly applies rotation to compute axis-aligned bounding box from rotated box corners.

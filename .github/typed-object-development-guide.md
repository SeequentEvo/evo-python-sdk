# Typed Object Development Guide

This guide explains how to create new typed object wrappers in `evo.objects.typed`. Use this when adding support for new Geoscience Object types.

## Overview

Typed objects provide a high-level, user-friendly interface for working with Geoscience Objects. Each typed object wraps a JSON schema-based object and provides:

- **Data classes** (`*Data`) - Frozen dataclasses for creating new objects
- **Wrapper classes** - Classes that wrap downloaded objects and provide typed property access
- **Pretty printing** - HTML representation for Jupyter notebooks with Portal/Viewer links
- **Datasets** - Access to tabular data and attributes within objects

## Existing Examples

| Object Type | Data Class | Wrapper Class | Key Features |
|-------------|------------|---------------|--------------|
| PointSet | `PointSetData` | `PointSet` | Locations dataset, coordinates + attributes |
| Variogram | `VariogramData` | `Variogram` | Typed structures, anisotropy classes |
| Regular3DGrid | `Regular3DGridData` | `Regular3DGrid` | Cells/vertices datasets |
| RegularMasked3DGrid | `RegularMasked3DGridData` | `RegularMasked3DGrid` | Boolean mask, active cells |
| Tensor3DGrid | `Tensor3DGridData` | `Tensor3DGrid` | Variable cell sizes |
| BlockModel | `RegularBlockModelData` | `BlockModel` | Block Model Service integration |

## Step-by-Step Guide

### Step 1: Understand the Schema

First, review the Geoscience Object JSON schema for your object type:

1. Find the schema in `evo-schemas` or the API documentation
2. Identify:
   - **Schema path**: e.g., `/objects/variogram/1.1.0/variogram.schema.json`
   - **Sub-classification**: e.g., `"variogram"`, `"pointset"`, `"regular-3d-grid"`
   - **Required fields**: Properties that must be provided when creating
   - **Optional fields**: Properties with defaults or that can be omitted
   - **Datasets**: Embedded parquet data (locations, cells, vertices, etc.)
   - **Nested structures**: Complex objects like variogram structures, anisotropy, etc.

### Step 2: Create the Data Class

Create a frozen dataclass that holds all the data needed to create a new object.

```python
from dataclasses import dataclass, field
from typing import Any, Literal

from .base import BaseObjectData, BaseSpatialObjectData


@dataclass(kw_only=True, frozen=True)
class MyObjectData(BaseObjectData):
    """Data for creating a MyObject.

    Detailed docstring explaining what this object represents
    and any important notes for users.

    Example:
        >>> data = MyObjectData(
        ...     name="Example Object",
        ...     required_field=42,
        ...     optional_field="value",
        ... )
    """

    # Required fields (no default value)
    required_field: int
    """Description of this field."""

    another_required: list[str]
    """Another required field."""

    # Optional fields (with default values)
    optional_field: str | None = None
    """Optional description field."""

    optional_with_default: float = 0.0
    """Field with a default value."""
```

**Key patterns:**

- Use `@dataclass(kw_only=True, frozen=True)` - immutable, keyword-only args
- Inherit from `BaseObjectData` (basic) or `BaseSpatialObjectData` (with CRS)
- Document each field with a docstring below it
- Put required fields first, optional fields after
- Use `field(default_factory=...)` for mutable defaults (lists, dicts)

### Step 3: Create Helper Classes (if needed)

For complex nested structures, create typed dataclasses:

```python
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, kw_only=True)
class EllipsoidRanges:
    """Ellipsoid ranges for spatial correlation."""
    
    major: float
    """Range in the major direction."""
    
    semi_major: float
    """Range in the semi-major direction."""
    
    minor: float
    """Range in the minor direction."""

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "major": self.major,
            "semi_major": self.semi_major,
            "minor": self.minor,
        }


@dataclass(frozen=True, kw_only=True)
class VariogramStructure:
    """Base class for variogram structures."""
    
    contribution: float
    """Contribution of this structure to total variance."""
    
    anisotropy: Anisotropy
    """Anisotropy definition for this structure."""
    
    variogram_type: str = field(init=False)  # Set by subclasses
    """Type identifier (set automatically by subclasses)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "variogram_type": self.variogram_type,
            "contribution": self.contribution,
            "anisotropy": self.anisotropy.to_dict(),
        }


@dataclass(frozen=True, kw_only=True)
class SphericalStructure(VariogramStructure):
    """Spherical variogram structure."""
    
    variogram_type: str = field(default="spherical", init=False)
```

**Key patterns:**

- Add `to_dict()` method for JSON serialization
- Use inheritance for structure type variants
- Use `field(init=False)` for computed/constant fields
- Subclasses set type discriminator with `field(default="value", init=False)`

### Step 4: Create the Wrapper Class

Create the main wrapper class that provides typed access to downloaded objects.

```python
from typing import Any, Literal

from pydantic import TypeAdapter

from evo.objects import SchemaVersion

from ._property import SchemaProperty
from .base import BaseSpatialObject, ConstructableObject, DatasetProperty
from .dataset import Dataset


class MyObject(BaseSpatialObject, ConstructableObject[MyObjectData]):
    """A GeoscienceObject representing [description].

    This class provides typed access to [object type] objects stored in Evo.
    
    Example:
        >>> obj = await MyObject.from_reference(context, reference)
        >>> print(obj.name)
        >>> print(obj.my_property)
    """

    # Link to the data class
    _data_class = MyObjectData

    # Schema identification (REQUIRED for creating new objects)
    sub_classification = "my-object"  # Must match schema sub-classification
    creation_schema_version = SchemaVersion(major=1, minor=0, patch=0)

    # Define schema properties (simple fields from JSON)
    my_property: int = SchemaProperty("my_property", TypeAdapter(int))
    """Description of this property."""

    optional_property: str | None = SchemaProperty(
        "optional_property", 
        TypeAdapter(str | None),
        default_factory=lambda: None,
    )
    """Optional property with default."""

    # For nested/complex properties, specify JMESPath
    nested_value: float = SchemaProperty(
        "nested.deeply.value",
        TypeAdapter(float),
    )

    # Define dataset properties (for parquet-based data)
    locations: MyDataset = DatasetProperty(
        MyDataset,  # Dataset subclass
        value_adapters=[
            TableAdapter(
                min_major_version=1,
                max_major_version=1,
                column_names=("x", "y", "z"),
                values_path="locations.coordinates",
                table_formats=[FLOAT_ARRAY_3],
            ),
        ],
        attributes_adapters=[
            AttributesAdapter(
                min_major_version=1, 
                max_major_version=1, 
                attribute_list_path="locations.attributes",
            )
        ],
        extract_data=lambda data: data.locations,  # Extract from data class
    )
```

**Key patterns:**

- Inherit from base classes:
  - `BaseSpatialObject` - for objects with bounding box and CRS
  - `ConstructableObject[DataClass]` - enables `create()` method
- Set `_data_class` to link wrapper to data class
- Set `sub_classification` to match the schema's sub-classification
- Set `creation_schema_version` to the schema version for new objects
- Use `SchemaProperty` for simple JSON fields
- Use `DatasetProperty` for embedded parquet datasets

### Step 5: Override `_data_to_dict` (if needed)

For complex serialization (e.g., typed structures), override `_data_to_dict`:

```python
from dataclasses import replace


class Variogram(BaseSpatialObject, ConstructableObject[VariogramData]):
    # ... properties ...

    @classmethod
    async def _data_to_dict(cls, data: VariogramData, context: Any) -> dict[str, Any]:
        """Convert VariogramData to dictionary for creating the object.

        Override to handle typed structure conversion before Pydantic serialization.
        """
        # Convert typed structures to dicts BEFORE calling super()
        # This avoids Pydantic serialization warnings
        converted_structures = data.get_structures_as_dicts()
        modified_data = replace(data, structures=converted_structures)

        # Let parent class handle the rest
        return await super()._data_to_dict(modified_data, context)
```

### Step 6: Add Pretty Printing

Override `_repr_html_` for Jupyter notebook display:

```python
class Variogram(BaseSpatialObject, ConstructableObject[VariogramData]):
    # ... properties ...

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter notebooks."""
        from .._html_styles import (
            STYLESHEET,
            build_nested_table,
            build_table_row,
            build_title,
        )

        # Build HTML table rows
        rows = [
            build_table_row("Sill:", str(self.sill)),
            build_table_row("Nugget:", str(self.nugget)),
            build_table_row("Rotation Fixed:", str(self.is_rotation_fixed)),
        ]

        # Add optional fields if present
        if self.attribute:
            rows.append(build_table_row("Attribute:", self.attribute))

        # Build structures table
        structure_rows = []
        for i, struct in enumerate(self.structures, 1):
            # Format structure details...
            structure_rows.append(...)

        # Combine with title and links from metadata
        content = build_nested_table(rows)
        return STYLESHEET + build_title(self.name, self.metadata) + content
```

### Step 7: Export from `__init__.py`

Add exports to `evo/objects/typed/__init__.py`:

```python
from .my_object import (
    MyObject,
    MyObjectData,
    EllipsoidRanges,  # Helper classes if public
    StructureBase,
)

__all__ = [
    # ... existing exports ...
    "MyObject",
    "MyObjectData",
    "EllipsoidRanges",
    "StructureBase",
]
```

### Step 8: Write Tests

Create comprehensive tests in `tests/typed/test_my_object.py`:

```python
import contextlib
import dataclasses
import uuid
from unittest.mock import patch

from parameterized import parameterized

from evo.common import Environment, StaticContext
from evo.common.test_tools import BASE_URL, ORG, WORKSPACE_ID, TestWithConnector
from evo.objects import ObjectReference
from evo.objects.typed import MyObject, MyObjectData
from evo.objects.typed.base import BaseObject

from .helpers import MockClient


class TestMyObject(TestWithConnector):
    def setUp(self) -> None:
        TestWithConnector.setUp(self)
        self.environment = Environment(hub_url=BASE_URL, org_id=ORG.id, workspace_id=WORKSPACE_ID)
        self.context = StaticContext.from_environment(
            environment=self.environment,
            connector=self.connector,
        )

    @contextlib.contextmanager
    def _mock_geoscience_objects(self):
        mock_client = MockClient(self.environment)
        with (
            patch("evo.objects.typed.dataset.get_data_client", lambda _: mock_client),
            patch("evo.objects.typed.base.create_geoscience_object", mock_client.create_geoscience_object),
            patch("evo.objects.typed.base.replace_geoscience_object", mock_client.replace_geoscience_object),
            patch("evo.objects.typed.base.download_geoscience_object", mock_client.from_reference),
        ):
            yield mock_client

    # Example test data
    example_data = MyObjectData(
        name="Test Object",
        required_field=42,
        optional_field="test",
    )

    @parameterized.expand([BaseObject, MyObject])
    async def test_create(self, class_to_call):
        """Test creating a new object."""
        with self._mock_geoscience_objects():
            result = await class_to_call.create(context=self.context, data=self.example_data)
        self.assertIsInstance(result, MyObject)
        self.assertEqual(result.name, "Test Object")

    async def test_from_reference(self):
        """Test loading an existing object."""
        with self._mock_geoscience_objects():
            original = await MyObject.create(context=self.context, data=self.example_data)
            result = await MyObject.from_reference(context=self.context, reference=original.metadata.url)
            self.assertEqual(result.name, "Test Object")

    def test_helper_to_dict(self):
        """Test helper class serialization."""
        helper = EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0)
        result = helper.to_dict()
        self.assertEqual(result["major"], 200.0)
        self.assertEqual(result["semi_major"], 150.0)
        self.assertEqual(result["minor"], 100.0)
```

## Checklist for New Typed Objects

- [ ] **Schema research**: Understand the JSON schema structure
- [ ] **Data class**: Create `*Data` class with all creation fields
- [ ] **Helper classes**: Create typed dataclasses for nested structures
- [ ] **Wrapper class**: Create main class inheriting from appropriate bases
- [ ] **Schema properties**: Define `SchemaProperty` for JSON fields
- [ ] **Dataset properties**: Define `DatasetProperty` for parquet data
- [ ] **Serialization**: Override `_data_to_dict` if needed for complex types
- [ ] **Pretty printing**: Implement `_repr_html_` for Jupyter display
- [ ] **Exports**: Add to `__init__.py` with proper `__all__`
- [ ] **Tests**: Write comprehensive unit tests
- [ ] **Documentation**: Add to typed-objects-reference.md

## Common Pitfalls

### Pydantic Serialization Warnings

If you see warnings like:
```
PydanticSerializationUnexpectedValue(Expected `dict[str, any]` - serialized value may not be as expected...)
```

This means Pydantic's `TypeAdapter` is receiving a dataclass when it expects a dict. Fix by converting to dicts **before** calling `super()._data_to_dict()`:

```python
@classmethod
async def _data_to_dict(cls, data: MyObjectData, context: Any) -> dict[str, Any]:
    # Convert complex types to dicts first
    modified_data = replace(data, structures=data.get_structures_as_dicts())
    return await super()._data_to_dict(modified_data, context)
```

### Schema Version Compatibility

When the schema version changes, ensure:
1. The `creation_schema_version` matches what the backend expects
2. The `TableAdapter` and `AttributesAdapter` version ranges are correct
3. Test with both old and new schema versions if backward compatibility is needed

### Registering for `object_from_uuid`

The `sub_classification` class variable automatically registers your class in `_BaseObject._sub_classification_lookup`. This enables `object_from_uuid()` and `object_from_path()` to return the correct typed class.

```python
# This automatically works:
obj = await object_from_uuid(context, uuid)
# Returns MyObject if the object's sub_classification is "my-object"
```

## Reference: Base Classes

| Base Class | Purpose | Key Features |
|------------|---------|--------------|
| `BaseObject` | Any object | `name`, `description`, `metadata` |
| `BaseSpatialObject` | Objects with location | `bounding_box`, `coordinate_reference_system` |
| `ConstructableObject[T]` | Objects that can be created | `create()`, `replace()` class methods |
| `DynamicBoundingBoxSpatialObject` | Computed bounding box | Overrides `bounding_box` property |

## Reference: Property Descriptors

| Descriptor | Purpose | Example |
|------------|---------|---------|
| `SchemaProperty` | Simple JSON fields | `name: str = SchemaProperty("name", TypeAdapter(str))` |
| `DatasetProperty` | Parquet datasets | `locations: Locations = DatasetProperty(...)` |

## Reference: Adapters

| Adapter | Purpose |
|---------|---------|
| `TableAdapter` | Maps parquet table to columns with specific format |
| `AttributesAdapter` | Maps attribute list from JSON to `Attributes` object |
| `CategoryTableAdapter` | For categorical/discrete data tables |


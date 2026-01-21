# Compute Task Notebook Generation Guide

This guide provides patterns for generating Jupyter notebooks that run compute tasks (like kriging) using the `evo-compute` package.

## Preferred Notebook Structure

When users ask to run compute tasks (especially kriging), generate notebooks following this structure:

### 1. Authentication
```python
from evo.notebooks import ServiceManagerWidget

manager = await ServiceManagerWidget.with_auth_code(
    client_id="your-client-id",  # Replace with your client ID
    base_uri="https://qa-ims.bentley.com",
    discovery_url="https://int-discover.test.api.seequent.com",
    cache_location="./notebook-data",
).login()
```

### 2. Load Source Objects and Review
Load inputs using `object_from_uuid()` or `object_from_path()` and display them using pretty printing:

```python
from evo.objects.typed import object_from_uuid, object_from_path

# Load by UUID (preferred when UUID is known)
source_pointset = await object_from_uuid(manager, "YOUR-POINTSET-UUID")
# Alternative: load by path
# source_pointset = await object_from_path(manager, "path/to/pointset.json")

# Pretty-print to review (shows Portal/Viewer links)
source_pointset
```

```python
# View the source attributes
source_pointset.attributes
```

For variograms - load existing or create new:
```python
# Load existing variogram by UUID
variogram = await object_from_uuid(manager, "YOUR-VARIOGRAM-UUID")
# Alternative: load by path
# variogram = await object_from_path(manager, "path/to/variogram.json")

# Pretty-print the variogram
variogram
```

Or create a new variogram:
```python
from evo.objects.typed import Variogram, VariogramData, SphericalStructure, Anisotropy, EllipsoidRanges

variogram_data = VariogramData(
    name="My Variogram",
    sill=1.0,
    nugget=0.1,
    is_rotation_fixed=True,
    modelling_space="data",  # Required: "data" or "normalscore"
    data_variance=1.0,       # Required: should match sill for non-normalized data
    structures=[
        SphericalStructure(
            contribution=0.9,
            anisotropy=Anisotropy(
                ellipsoid_ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            ),
        )
    ],
    attribute="grade",
)
variogram = await Variogram.create(manager, variogram_data)
variogram
```

### 3. Create Target Block Model (Preferred) or Use Existing
**Block Models are preferred as targets** because they:
- Support multiple attributes from different kriging scenarios
- Integrate with Block Model Service for efficient data access
- Allow concurrent attribute creation

```python
import uuid
from evo.objects.typed import BlockModel, RegularBlockModelData, Point3, Size3i, Size3d
from evo.blockmodels import Units

run_uuid = uuid.uuid4()

# Create a Block Model to hold kriging results
bm_data = RegularBlockModelData(
    name=f"Kriging Results - {run_uuid}",
    description="Block model with kriging results",
    origin=Point3(x=10000, y=100000, z=200),
    n_blocks=Size3i(nx=40, ny=40, nz=40),
    block_size=Size3d(dx=25.0, dy=25.0, dz=10.0),
    crs="EPSG:32632",
    size_unit_id=Units.METRES,
)

block_model = await BlockModel.create_regular(manager, bm_data)

# Pretty-print the created block model
block_model
```

**Alternative: Use existing block model**
```python
block_model = await object_from_uuid(manager, "YOUR-BLOCK-MODEL-UUID")
block_model
```

### 4. Define Kriging Parameters
For **multiple scenarios** (preferred pattern):
```python
from evo.compute.tasks import (
    KrigingParameters,
    Target,
    SearchNeighbourhood,
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
)

# Define scenario variations (e.g., different max_samples)
max_samples_values = [5, 10, 15, 20]

# Search ellipsoid configuration
search_ellipsoid = Ellipsoid(
    ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
    rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
)

# Create parameter sets for each scenario
# Note: method defaults to ordinary kriging
parameter_sets = []
for max_samples in max_samples_values:
    params = KrigingParameters(
        source=source_pointset.attributes["grade"],  # Access attribute from pointset
        target=Target.new_attribute(block_model, attribute_name=f"Samples={max_samples}"),
        variogram=variogram,
        search=SearchNeighbourhood(ellipsoid=search_ellipsoid, max_samples=max_samples),
    )
    parameter_sets.append(params)
    print(f"Prepared scenario with max_samples={max_samples}")
```

For **single run** (alternative):
```python
from evo.compute.tasks import (
    KrigingParameters,
    Target,
    SearchNeighbourhood,
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
)

kriging_params = KrigingParameters(
    source=source_pointset.attributes["grade"],  # Access attribute from pointset
    target=Target.new_attribute(target_object, attribute_name="kriged_grade"),
    variogram=variogram,
    search=SearchNeighbourhood(
        ellipsoid=Ellipsoid(
            ranges=EllipsoidRanges(major=200.0, semi_major=150.0, minor=100.0),
            rotation=Rotation(dip_azimuth=0.0, dip=0.0, pitch=0.0),
        ),
        max_samples=20,
    ),
)
```

### 5. Run Kriging Tasks
For **multiple scenarios** (preferred):
```python
from evo.compute.tasks import run_kriging_multiple
from evo.notebooks import FeedbackWidget

print(f"Submitting {len(parameter_sets)} kriging tasks in parallel...")
fb = FeedbackWidget("Kriging Scenarios")

results = await run_kriging_multiple(manager, parameter_sets, fb=fb)

print(f"\nAll {len(results)} scenarios completed!")
```

For **single run**:
```python
from evo.compute.tasks import run_kriging
from evo.notebooks import FeedbackWidget

print("Submitting kriging task...")
fb = FeedbackWidget("Kriging task")
job_result = await run_kriging(manager, kriging_params, fb=fb)

# Pretty-print the result
job_result
```

### 6. Review Results
**Refresh the block model** to see new attributes:
```python
# Refresh to see new attributes added by kriging
block_model = await block_model.refresh()

# Pretty-print shows updated state
block_model
```

```python
# View all attributes (pretty-printed table)
block_model.attributes
```

**Query the data**:
```python
# Query scenario columns using to_dataframe()
scenario_columns = [f"Samples={ms}" for ms in max_samples_values]
df = await block_model.to_dataframe(columns=scenario_columns)

print(f"Retrieved {len(df)} blocks with {len(scenario_columns)} scenario columns")
df.head(10)
```

**Get data from job result directly**:
```python
# Simplest approach - get data directly from the job result
df = await job_result.to_dataframe()
df.head()
```

**Basic analysis**:
```python
# Statistics by scenario
print("Statistics by max_samples:")
print(df[scenario_columns].describe())
```

## Key Patterns

### DO:
- Use `object_from_uuid()` or `object_from_path()` to load objects
- Use pretty printing (just output the object) to display objects - it shows Portal/Viewer links
- Use `BlockModel` as the target for kriging (preferred over grids)
- Use `run_kriging_multiple()` for scenario analysis
- Use `block_model.refresh()` after modifications to see new attributes
- Use `block_model.to_dataframe()` to get data (preferred over `get_data()`)
- Use `job_result.to_dataframe()` to get kriging results directly
- Use `FeedbackWidget` for progress display

### DON'T:
- Don't use `display_object_links()` - pretty printing is preferred
- Don't use `TypedClass.from_reference()` directly - use `object_from_uuid()` or `object_from_path()`
- Don't forget to refresh objects after kriging to see new attributes

## Example: Complete Kriging Notebook Outline

```
1. Title and description
2. Authentication (ServiceManagerWidget)
3. Setup for local development (sys.path if needed)
4. Load source pointset (object_from_uuid) + pretty print + view attributes
5. Load or create variogram (object_from_uuid or Variogram.create) + pretty print
6. Create target block model (BlockModel.create_regular) + pretty print
7. Define kriging scenarios (KrigingParameters with SearchNeighbourhood)
8. Run kriging tasks (run_kriging_multiple with FeedbackWidget)
9. Refresh block model (block_model.refresh()) + pretty print
10. View block model attributes
11. Query results (to_dataframe())
12. Basic analysis (describe, optional plotly visualization)
```

## Compute Task Imports Reference

```python
# Core compute imports
from evo.compute.tasks import (
    run_kriging,
    run_kriging_multiple,
    KrigingParameters,
    Target,
    OrdinaryKriging,
    SimpleKriging,
    KrigingSearch,
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
)

# Object loading (preferred methods)
from evo.objects.typed import object_from_uuid, object_from_path

# Block model creation
from evo.objects.typed import BlockModel, RegularBlockModelData, Point3, Size3i, Size3d
from evo.blockmodels import Units

# Variogram creation (using typed structure classes)
from evo.objects.typed import (
    Variogram, VariogramData,
    SphericalStructure, ExponentialStructure, GaussianStructure, CubicStructure,
    LinearStructure, SpheroidalStructure, GeneralisedCauchyStructure,
    Anisotropy,
    EllipsoidRanges as VariogramEllipsoidRanges,  # Disambiguate from compute
    VariogramRotation,
)

# Notebooks utilities
from evo.notebooks import ServiceManagerWidget, FeedbackWidget
```


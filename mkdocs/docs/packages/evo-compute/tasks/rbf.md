[GitHub source](https://github.com/SeequentEvo/evo-python-sdk/tree/main/packages/evo-compute/stubs/snapshot/rbf)

# RBF Tasks

Discoverable compute tasks in the `rbf` topic. Call any task below through `ComputeClient` as `await client.rbf.<task>.run(...)` (or `submit(...)` for a non-blocking `TaskJob`), or by name via `await client.arun("rbf", "<task>", parameters)` / `await client.asubmit("rbf", "<task>", parameters)`.

## Input Shorthands

| Alias | Accepted Python Types | Description |
| --- | --- | --- |
| `ObjectInput` | `str | UUID | BaseObject | DownloadedObject | ObjectMetadata | ObjectReference` | Any handle `ReferenceResolver` can derive a geoscience object URL from. |
| `AttributeInput` | `str | AnyTypedAttribute` | A typed attribute, a JMESPath expression, or a bare attribute name. |
| `AnyTypedAttribute` | `Attribute | BlockModelAttribute | PendingAttribute | BlockModelPendingAttribute` | A typed attribute handle carrying both its parent geoscience object and attribute name. |

## Tasks

### rbf_evaluation

`client.rbf.rbf_evaluation` · Task: `rbf/rbf-evaluation` · Version: `0.1.0` · Preview (`preview=True` required)

Evaluates a radial basis function (RBF) interpolant defined by centres, coefficients, and a basic function at specified target locations. Supports all turbo-rbf basic function types including aspherical, biharmonic, spherical, cubic, exponential, gaussian, multiquadric, normalized multiquadric, and triharmonic.

```python
await client.rbf.rbf_evaluation.run(
    *,
    source: RbfRbfEvaluationSource,
    basic_function: RbfRbfEvaluationAsphericalFunction | RbfRbfEvaluationSphericalFunction | RbfRbfEvaluationBiharmonicFunction | RbfRbfEvaluationCubicFunction | RbfRbfEvaluationExponentialFunction | RbfRbfEvaluationGaussianFunction | RbfRbfEvaluationMultiquadricFunction | RbfRbfEvaluationNormalizedMultiquadricFunction | RbfRbfEvaluationTriharmonicFunction,
    target: RbfRbfEvaluationTarget | AnyTypedAttribute,
    polynomial: RbfRbfEvaluationPolynomial | None = ...,
    preview: bool = False,
) -> RbfRbfEvaluationResult

await client.rbf.rbf_evaluation.submit(
    *,
    source: RbfRbfEvaluationSource,
    basic_function: RbfRbfEvaluationAsphericalFunction | RbfRbfEvaluationSphericalFunction | RbfRbfEvaluationBiharmonicFunction | RbfRbfEvaluationCubicFunction | RbfRbfEvaluationExponentialFunction | RbfRbfEvaluationGaussianFunction | RbfRbfEvaluationMultiquadricFunction | RbfRbfEvaluationNormalizedMultiquadricFunction | RbfRbfEvaluationTriharmonicFunction,
    target: RbfRbfEvaluationTarget | AnyTypedAttribute,
    polynomial: RbfRbfEvaluationPolynomial | None = ...,
    preview: bool = False,
) -> TaskJob[RbfRbfEvaluationResult]
```

Parameters:

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `RbfRbfEvaluationSource` | RBF model centres and coefficients. | _required_ |
| `basic_function` | `RbfRbfEvaluationAsphericalFunction | RbfRbfEvaluationSphericalFunction | RbfRbfEvaluationBiharmonicFunction | RbfRbfEvaluationCubicFunction | RbfRbfEvaluationExponentialFunction | RbfRbfEvaluationGaussianFunction | RbfRbfEvaluationMultiquadricFunction | RbfRbfEvaluationNormalizedMultiquadricFunction | RbfRbfEvaluationTriharmonicFunction` | Basic function type and parameters. | _required_ |
| `target` | `RbfRbfEvaluationTarget | AnyTypedAttribute` | Target object and attribute for evaluation output. | _required_ |
| `polynomial` | `RbfRbfEvaluationPolynomial | None` | Optional polynomial configuration. Required when basic function degree >= 0. | `None` |
| `preview` | `bool` | Opt in to preview execution (required for this preview task: pass `preview=True`). | `False` |

Results (`RbfRbfEvaluationResult`):

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `target` | `RbfRbfEvaluationResultResultTarget` | `await result.target.load()`, `await result.target.to_dataframe(*keys)` | Target geoscience object containing the evaluated values. |

#### Nested Shapes

**`RbfRbfEvaluationSource`** (`TypedDict`)

RBF model centres and coefficients.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Geoscience object whose coordinates represent the RBF centres. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `coefficients` | `AttributeInput` | Yes | Numeric attribute on the source object containing the RBF coefficients (one per centre). |

**`RbfRbfEvaluationAsphericalFunction`** (`TypedDict`)

Aspherical basic function: phi(r) = (r^2 + c^2)^(k/2).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["aspherical"]` | Yes | Type of basic function. |
| `c` | `float` | Yes | Distance scaling parameter. Bounds: `> 0`. |
| `k` | `int` | Yes | Shape exponent (negative odd integer, e.g. -1, -3, -5). Bounds: `<= -1`. |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationSphericalFunction`** (`TypedDict`)

Spherical basic function.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["spherical"]` | Yes | Type of basic function. |
| `c` | `float` | Yes | Distance scaling parameter. Bounds: `> 0`. |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationBiharmonicFunction`** (`TypedDict`)

Biharmonic basic function: phi(r) = r.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["biharmonic"]` | Yes | Type of basic function. |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationCubicFunction`** (`TypedDict`)

Cubic basic function: phi(r) = (r^2 + c^2)^(3/2).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["cubic"]` | Yes | Type of basic function. |
| `c` | `float` | Yes | Distance scaling parameter. Bounds: `> 0`. |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationExponentialFunction`** (`TypedDict`)

Exponential basic function: phi(r) = exp(-r/c).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["exponential"]` | Yes | Type of basic function. |
| `c` | `float` | Yes | Distance scaling parameter. Bounds: `> 0`. |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationGaussianFunction`** (`TypedDict`)

Gaussian basic function: phi(r) = exp(-r^2/c^2).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["gaussian"]` | Yes | Type of basic function. |
| `c` | `float` | Yes | Distance scaling parameter. Bounds: `> 0`. |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationMultiquadricFunction`** (`TypedDict`)

Multiquadric basic function: phi(r) = (r^2 + c^2)^(k/2).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["multiquadric"]` | Yes | Type of basic function. |
| `c` | `float` | Yes | Distance scaling parameter. Bounds: `> 0`. |
| `k` | `int` | Yes | Shape exponent (positive odd integer, e.g. 1, 3, 5). |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationNormalizedMultiquadricFunction`** (`TypedDict`)

Normalized multiquadric basic function: phi(r) = (r^2 + c^2)^(k/2) / c^k.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["normalized_multiquadric"]` | Yes | Type of basic function. |
| `c` | `float` | Yes | Distance scaling parameter. Bounds: `> 0`. |
| `k` | `int` | Yes | Shape exponent (positive odd integer, e.g. 1, 3, 5). |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationTriharmonicFunction`** (`TypedDict`)

Triharmonic basic function: phi(r) = r^3.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `Literal["triharmonic"]` | Yes | Type of basic function. |
| `degree` | `int` | Yes | Polynomial degree. -1 for no polynomial. Bounds: `>= -1`. |

**`RbfRbfEvaluationTargetAttributeCreate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["create"]` | Yes | The operation to perform on the attribute. |
| `name` | `str` | Yes | The name of the attribute to create. |

**`RbfRbfEvaluationTargetAttributeUpdate`** (`TypedDict`)

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `operation` | `Literal["update"]` | Yes | The operation to perform on the attribute. |
| `reference` | `AttributeInput` | Yes | Reference to an existing attribute to update. |

**`RbfRbfEvaluationTarget`** (`TypedDict`)

Target object and attribute for evaluation output.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `object` | `ObjectInput` | Yes | Geoscience object with locations where the RBF will be evaluated. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `attribute` | `RbfRbfEvaluationTargetAttributeCreate | RbfRbfEvaluationTargetAttributeUpdate | AttributeInput` | Yes | Attribute to create or update on the target object with evaluated values. Accepts a typed attribute, a bare attribute name (updates if present on the target object, otherwise creates), or `{"operation": "create" | "update", ...}`. |

**`RbfRbfEvaluationPoint3D`** (`TypedDict`)

The 3D polynomial base point.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | `float` | Yes | The x coordinate. |
| `y` | `float` | Yes | The y coordinate. |
| `z` | `float` | Yes | The z coordinate. |

**`RbfRbfEvaluationPolynomial`** (`TypedDict`)

Optional polynomial configuration. Required when basic function degree >= 0.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `coefficients` | `list[float]` | Yes | Polynomial coefficient values. |
| `base` | `RbfRbfEvaluationPoint3D` | Yes | The 3D polynomial base point. |

**`RbfRbfEvaluationResultResultTargetAttribute`** (`ResultNode`)

Numeric attribute containing the evaluated RBF values.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to the attribute in the geoscience object. |
| `name` | `str` | — | The name of the output attribute. |

**`RbfRbfEvaluationResultResultTarget`** (`ResultNode`)

Target geoscience object containing the evaluated values.

| Field | Type | Hydrated Loader | Description |
| --- | --- | --- | --- |
| `reference` | `str` | — | Reference to a geoscience object. Supported schemas: `pointset/[>=1.2,<2]`, `block-model/[>=1.0,<2]`, `regular-3d-grid/[>=1.2,<2]`, `regular-masked-3d-grid/[>=1.2,<2]`, `tensor-3d-grid/[>=1.2,<2]`, `downhole-intervals/[>=1.2,<2]`. |
| `name` | `str` | — | The name of the geoscience object. |
| `description` | `str | None` | — | The description of the geoscience object. |
| `schema_id` | `str` | — | The ID of the Geoscience Object schema. |
| `attribute` | `RbfRbfEvaluationResultResultTargetAttribute` | `await node.attribute.load()`, `await node.attribute.to_dataframe()` | Numeric attribute containing the evaluated RBF values. |

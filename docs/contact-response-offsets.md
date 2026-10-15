# Contact response offsets

Adapter 0.10.28 accepts valid authored PhysxCollisionAPI contactOffset/restOffset
without changing USD geometry. Edit a versioned asset and rebuild the world; this
feature does not provide live in-place editing.

Contact offset controls predictive contact generation, not geometric mesh inflation.
Rest offset controls equilibrium separation: positive permits a gap, negative permits
overlap. Pair distances are the sum of both shapes' offsets. Planning against exact
collision geometry therefore does not promise avoidance of all contact envelopes or
controller convergence. No planner margin is silently introduced.

Finite contact offsets must be nonnegative and strictly exceed finite rest offsets,
except the backward-compatible authored (0, 0) pair. Installed native USD PhysX
accepts that pair and resolves a positive effective contact offset (measured 0.1 mm
on the fixture). Provenance records authored zero, not this shape-dependent result.
Bool, nonnumeric, NaN, positive infinity and out-of-float-range values are rejected.
The SDK's negative-infinity automatic sentinel remains symbolic, including when only
one parameter is authored; shape-dependent effective values remain SDK-owned.

Authored parameters and composed-stage metres-per-unit enter geometry provenance
and the catalog digest. Referenced source-layer units are not incorrectly multiplied
into the composed values. Geometry dimensions, vertices, representation and transforms
remain unchanged. Immutable mesh-byte caches can be reused; descriptors/catalogs must
be rebuilt after asset edits. System ground and clone admission use the same validation.

[PhysX shape API](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/class_px_shape.html)
and [contact documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html)
define these semantics. The native conformance script checks actual readbacks and
resting heights for zero, positive, negative and automatic offsets. This fixture test
is separate from production-robot acceptance.

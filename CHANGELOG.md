# Changelog

## 0.1.0a0

- Add a lazy, capability-declared Isaac Lab 3.0 provider.
- Add transactional session/world lifecycle and strict public command validation.
- Add native Isaac Lab runtime for rigid entities, USD articulations, and exact surface/volume meshes.
- Require rigid USD assets to contain exactly one native `UsdPhysics.RigidBodyAPI` prim.
- Isolate Kit in a restartable worker process and clean its complete Linux process session on close.

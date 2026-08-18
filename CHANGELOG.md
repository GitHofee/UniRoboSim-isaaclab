# Changelog

## 0.2.0a0

- Add batched rigid-body root-link pose and twist state.
- Add persistent selected-environment force/torque commands in environment-local world axes.
- Add PhysX aggregated net-normal-force and binary-contact state.
- Upgrade the exact core dependency and provider contract to UniRoboSim `0.3.0a0` / `v0alpha3`.

## 0.1.0a0

- Add a lazy, capability-declared Isaac Lab 3.0 provider.
- Add transactional session/world lifecycle and strict public command validation.
- Add native Isaac Lab runtime for rigid entities, USD articulations, and exact surface/volume meshes.
- Require rigid USD assets to contain exactly one native `UsdPhysics.RigidBodyAPI` prim.
- Isolate Kit in a restartable worker process and clean its complete Linux process session on close.

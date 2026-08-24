# Changelog

## 0.9.3

- Align the required PyTorch runtime profile and fail-closed preflight with
  `torch==2.11.0`, `torchvision==0.26.0`, and `torchaudio==2.11.0`.
- Keep RGB conversion and layout normalization on the GPU, then transfer one
  contiguous packed byte buffer across the native worker boundary instead of
  millions of Python integers.
- Add explicit render cadence control and stop rendering every physics tick in the
  headless camera profile; camera reads still render on demand.
- Add the `headless-physics` launch profile for physics-only batch production and
  cap the visible EasyAPI profile at 60 rendered frames per second.
- Require UniRoboSim Core 0.9.1 for compatible packed RGB `ArrayValue` transport.

## 0.9.2

- Anchor the clean native worker to the exact Core and adapter package roots loaded by
  the parent process, independent of the caller's working directory or `PYTHONPATH`.
- Add a fail-closed startup handshake for worker protocol, package versions, and
  canonical package origins.
- Start the clean worker with Python safe-path mode and preserve owned-process cleanup
  on spawn or handshake failure.

## 0.9.1

- Add the public `UNIROBOSIM_ISAACLAB_LAUNCH_PROFILE` entry-point profile selector.
- Preserve the headless camera/render batch default while allowing an exact `visible`
  opt-in for normal backend discovery.
- Reject unknown profile values before loading the native Isaac SDK, without reflecting
  untrusted environment content in errors.

## 0.7.0

- Join the coordinated UniRoboSim 0.7 release train.
- Accept released Core versions `>=0.7.0,<0.8` and align package/provider identity.
- Retain the verified Isaac Lab 3.0 / Isaac Sim 6.0.1 feature and fidelity profile.
- Launch Kit through a clean interpreter bootstrap so MCP and other dependency-heavy
  parent processes cannot preload libraries into the Isaac runtime.

## 0.6.1a0

- Default camera capture to explicit FXAA and verify the effective RTX anti-aliasing mode after
  Isaac Lab rendering presets have loaded.
- Disable headless texture streaming by default so sensor frames use full-resolution source textures;
  retain an opt-in streaming mode for memory-constrained workloads.
- Expose the effective camera fidelity profile in provider metadata.

## 0.6.0a0

- Declare the provider-owned `isaaclab.dynamic-rigid-usd@1` semantic normalization target for rigid USD.
- Keep adapter preflight strict while allowing the optional asset normalizer to prepare visual-only USD
  before native build.
- Use neutral package authorship and UniRoboSim core `0.7.0a0`.

## 0.3.0a0

- Add real fixed-count PhysX PBD particle-fluid authoring, position/velocity commands, state readback,
  stepping, and deterministic reset.
- Add capability-gated Isaac Lab RTX RGB/depth camera samples without advancing the physics tick.
- Add stable native USD point/line debug overlays with layer/ID replacement, filtering, and lifetimes.
- Add a USD rigid-body bridge for verified rigid + particle-fluid mixed scenes.
- Upgrade the exact core dependency and provider contract to UniRoboSim `0.4.0a0` / `v0alpha4`.
- Document and fail explicitly for unsupported bridged contact-force and particle + articulation/deformable
  same-world combinations in this Isaac Sim 6.0.1 profile.

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

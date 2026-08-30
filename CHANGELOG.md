# Changelog

## 0.10.6

- Add the `static` composite unbound-body mode. It disables private joints and
  removes `RigidBodyAPI` only from unbound bodies while preserving authored mesh
  collision and all explicitly embedded bodies/joints.
- Keep static composite bodies out of articulation/rigid tensor views and reset
  caches. Repeated commands in the same joint control mode no longer rewrite
  unchanged stiffness/damping; mode changes and reset still restore exact gains.
- Enable Isaac Sim 6's legacy RTX Real-Time implementation at the Kit startup
  boundary when ordinary cameras select `RaytracedLighting`. Isaac Sim 6 keeps
  that implementation disabled by default, so selecting it later through
  `SimulationApp` was silently mapped back to `RealTimePathTracing`.
- Keep fluid-isosurface worlds on the explicitly selected RTPT implementation
  and retain fail-closed renderer readback after startup.

## 0.10.5

- Select `RaytracedLighting` for ordinary camera worlds instead of inheriting
  Isaac Sim 6's `RealTimePathTracing` default. This prevents single-sample RTX
  noise when a headless runtime cannot initialize the NGX/DLSS Ray
  Reconstruction denoiser, while retaining explicit real-time path tracing for
  fluid-isosurface rendering.
- Re-apply and verify the selected renderer after Isaac Lab rendering presets
  load so they cannot silently restore a different mode.

## 0.10.4

- Admit the exact official NGC Isaac Lab 3.0 / Isaac Sim 6.0.1 bundle as a
  second fail-closed runtime profile. Its probe validates release files, module
  locations, adapter-required API modules, PhysX, Torch, and debug-draw layout
  without importing or launching Kit; the existing 6.1.17 source profile remains
  unchanged and exact.
- Resolve `isaacsim.util.debug_draw` from both the pip SDK layout and the official
  NGC bundle layout without an unbounded filesystem search.
- Keep the source profile at a configurable 90-second Kit idle allowance inside
  a 120-second hard limit. Give only the fully fingerprinted official NGC profile
  a finite 300-second allowance for first-run RTX pipeline compilation. Zero-config
  provider factories select the matching budget automatically; explicitly supplied
  configurations remain authoritative.

## 0.10.3

- Share one global RTX render across every camera read at the same world
  revision. A visible physics-step render is reused, while reset, direct pose,
  particle, debug, and non-rendering physics mutations invalidate the cached
  revision before the next camera capture.
- Report a fixed, strictly ordered native-worker startup phase sequence before,
  during, and after Kit launch. Phase progress cannot act as a heartbeat and
  cannot extend the existing 30-second hard startup limit.
- Fail and clean up a worker after 8 seconds without pre-Kit progress, while
  retaining a 15-second Kit-launch idle allowance. The latter leaves almost
  7 seconds of margin above the 8.136-second maximum across seven instrumented
  visible launches on the 2026-08-26 acceptance host.
- Execute the stdlib-only worker bootstrap by exact file path so it can report
  interpreter connectivity before importing the adapter or Isaac SDK. Timeout
  diagnostics now identify the last completed phase, and retry still creates
  exactly one clean replacement worker.
- Forward the frozen FastSim launch profile through the DROID acceptance plan
  and make its injected provider factory accept and verify FastSim's
  `launch_profile` keyword contract.

## 0.10.2

- Bound the isolated native-worker startup wait to 30 seconds and retry once
  after fully cleaning up a worker that never returns from Isaac Kit startup.
- Keep deterministic startup errors fail-fast; only a live worker that produces
  no startup reply is retried.
- Align the DROID acceptance composition with FastSim's explicit recording
  capture boundary while leaving capture disabled for the adapter-only run.

## 0.10.1

- Preserve authored per-joint stiffness and damping by default instead of applying
  one global gain pair to every articulation DoF.
- Restore authored gains across reset and control-mode changes; position commands
  fall back locally only for explicitly commanded authored-zero drives.
- Keep raw articulation state and gain tensor indexing on their respective devices,
  including CUDA state with CPU-resident PhysX gain tables.
- Expose complete composite-scene planning geometry and tolerate degenerate USD
  faces only when they contain fewer than three vertices; malformed valid faces and
  meshes with no usable triangles still fail closed.

## 0.10.0

- Add UniRoboSim Core 0.10 composite USD scene support. The adapter composes one
  scene layer per environment and binds declared rigid bodies and articulations to
  existing Prims without respawning them.
- Resolve embedded articulation joints by their complete USD Prim paths so duplicate
  short joint names remain unambiguous.
- Capture and deterministically restore embedded rigid-body and articulation state on
  reset, including authored articulation roots that are not exposed as entities.

## 0.9.6

- Let the installed backend factory accept the canonical `visible`, `headless`, or
  `headless-physics` launch profile as an explicit keyword-only argument.
- Keep zero-argument EasyAPI environment compatibility while ensuring an explicit
  FastSim profile never reads or inherits process environment state.

## 0.9.5

- Accept a PhysX `convexHull` collision carrier whose Prim is itself a
  `UsdGeom.Mesh`, while retaining the existing container + one descendant Mesh
  representation.
- Use the same exact-one-Mesh selection rule for PhysX cooking and multi-environment
  clone signatures; ambiguous multi-Mesh and nested-collider subtrees still fail
  closed.

## 0.9.4

- Add immutable static-scene USD composition, physical asset scaling and mounted
  RGB/depth/normals cameras.
- Reject static scenes before native allocation when a complete
  `planning.scene@2` catalog is required but cannot yet be represented.

## 0.9.3

- Provide a reproducible Isaac Lab 6.1.17 source-profile patch that removes
  the test-only Coverage runtime pin and keeps the complete Torch 2.11 stack.
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

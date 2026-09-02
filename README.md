# UniRoboSim Isaac Lab adapter

[English](README.md) | [简体中文](README.zh-CN.md)

`unirobosim-isaaclab` connects UniRoboSim `0.10.x` to Isaac Lab 3.0 / Isaac Sim 6.0.1. Import and `probe()` are side-effect free. `open()` starts Kit in an adapter-owned worker process, so Isaac Sim cannot take over or terminate the application process.

Worker startup reports one fixed, strictly ordered phase sequence. Pre-Kit phases
retain short fail-fast idle limits. The source profile uses a 90-second Kit idle
budget inside a 120-second per-worker hard limit. The exact official NGC profile uses
a 300-second finite budget because its first GPU launch may compile RTX pipelines
without emitting worker-protocol progress. Both budgets are validated and capped at 300 seconds;
unused allowance adds no delay to warm startup. A timeout is diagnosed, fully cleaned
up, and retried once. Progress never extends the hard limit, while deterministic
configuration, protocol, and package-fingerprint failures remain fail-fast.

## Compatibility

Python `>=3.12,<3.13`, UniRoboSim `>=0.10,<0.11`, and runtime contract
`v0alpha6` are common to both admitted profiles:

| Profile | Isaac Lab | Isaac Sim | Torch stack |
| --- | --- | --- | --- |
| `source-isaaclab-3.0.0-beta2` | source release, distribution `6.1.17`; PhysX `1.1.3` | distribution `6.0.1.0` | Torch `2.11.0`, TorchVision `0.26.0`, TorchAudio `2.11.0` |
| `ngc-isaaclab-3.0.0` | official NGC bundle, `VERSION=3.0.0`, distribution `6.1.11`; PhysX `1.1.3` | module bundle, `VERSION=6.0.1` | Torch `2.10.0`, TorchVision `0.25.0`; TorchAudio is not required |

The NGC profile is not a relaxed version range. It recognizes the exact official
bundle layout and verifies the Isaac Lab and Isaac Sim version files, top-level
module locations, every adapter-required API module, and the debug-draw extension.
The source profile retains its complete exact distribution gate. The larger NGC
startup budget is selected only after this complete fingerprint passes and adds no
delay to a warm launch.

## Installation

Install the NVIDIA runtime in a dedicated Python 3.12 Conda environment. The upstream
Isaac Lab 6.1.17 source incorrectly declares the test-only `coverage==7.6.1` as a
runtime dependency, while Isaac Sim Kernel 6.0.1.0 requires `coverage==7.4.4`. This
repository provides a small, reviewable patch against the exact tested upstream commit.
It removes the Isaac Lab runtime pin and aligns its installer with the complete Torch
2.11 profile.

```bash
conda create -n unirobosim-isaaclab3 python=3.12 pip -y
conda activate unirobosim-isaaclab3

git clone https://github.com/GitHofee/UniRoboSim-isaaclab.git
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout 2e44ddb2e19536579140496023b5ccb060bc4152
git -C IsaacLab apply ../UniRoboSim-isaaclab/patches/isaaclab-6.1.17-runtime-profile.patch

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install -e ./IsaacLab/source/isaaclab
python -m pip install -e ./IsaacLab/source/isaaclab_physx

git clone https://github.com/GitHofee/UniRoboSim.git
git clone https://github.com/GitHofee/UniRoboSim-usd-converter.git

python -m pip install ./UniRoboSim
python -m pip install ./UniRoboSim-isaaclab
python -m pip install ./UniRoboSim-usd-converter
```

Validate the installed dependency graph and the complete runtime profile without launching Kit:

```bash
python -m pip check
python -c "from unirobosim_isaaclab import create_provider; print(create_provider().probe())"
```

The CUDA 12.8 command above is the tested Linux x86-64 source profile. CUDA wheels with
the same public versions and a compatible local tag are accepted. The adapter
intentionally keeps the large NVIDIA and Torch packages out of its own wheel metadata.
The side-effect-free probe selects exactly one supported profile. It reads distribution
metadata for the source profile; for the NGC profile it additionally reads the official
bundle's module specs, version files, required module tree, and extension layout without
importing or launching Kit. It fails closed if neither complete fingerprint matches.
`pip check` must finish with `No broken requirements found` for the source profile.

## EasyAPI

The installed entry point makes switching explicit and minimal:

```python
from unirobosim import Sim

with Sim(backend="isaaclab", world_id="isaac-demo") as sim:
    box = sim.add_box(
        "red_box",
        size_m=0.1,
        color_rgba=(1.0, 0.0, 0.0, 1.0),
        position_m=(0.0, 0.0, 0.5),
    )
    camera = sim.add_camera("camera", resolution=(640, 360), outputs=("rgb", "depth"))
    sim.start()
    sim.step(30)
    print(box.state)
    print(camera.read("rgb").shape)
```

The installed `unirobosim.backends/isaaclab` entry point uses a headless camera/render
profile by default. Opt into a real Kit window for an interactive FastSim or EasyAPI
run with one exact environment value:

```bash
UNIROBOSIM_ISAACLAB_LAUNCH_PROFILE=visible python run_simulation.py
```

The complete public profile contract is:

| Value | Launch behavior |
| --- | --- |
| unset | Headless, cameras enabled, rendering enabled |
| `headless` | Headless, cameras enabled, rendering enabled |
| `headless-physics` | Headless, cameras and rendering disabled |
| `visible` | Visible Kit window, cameras enabled, rendering enabled |

Use `headless` for offscreen debug capture. It loads no desktop window, but native
debug points, lines, frames, labels, boxes, and paths are rendered into scene-camera
images. `headless-physics` deliberately has no renderer and therefore does not
advertise the native-overlay capability; an attempted draw reports an actionable
unsupported-capability error instead of failing inside an Isaac plugin.

Values are case-sensitive. Any other value fails before the Isaac SDK is loaded.
The adapter does not infer a profile from `DISPLAY`, so batch behavior is stable across
machines. This selector applies only to normal installed entry-point discovery;
`create_provider(IsaacLabAdapterConfig(...))` remains the explicit programmatic API.
Calling `create_provider()` without a configuration uses the same exact-profile
startup-budget selection as the installed entry point. Passing a configuration is
authoritative, including both worker startup budgets.
FastSim passes its canonical Plan value directly to the same entry-point factory, for
example `create_easy_provider(launch_profile="headless-physics")`. An explicit keyword
never reads `UNIROBOSIM_ISAACLAB_LAUNCH_PROFILE`; the environment remains a
zero-argument EasyAPI compatibility path only.

Use provider injection for launch settings that do not belong in the portable world:

```python
from unirobosim import Sim
from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider

provider = create_provider(
    IsaacLabAdapterConfig(
        headless=True,
        device="cuda:0",
        enable_cameras=True,
        render=True,
        anti_aliasing="fxaa",
        texture_streaming=False,
        render_on_step=False,
        worker_startup_hard_timeout_s=120,
        worker_kit_launch_idle_timeout_s=90,
    )
)

with Sim(provider=provider) as sim:
    sim.add_camera("camera", resolution=(1920, 1080), outputs=("rgb",))
    sim.start()
```

## Implemented native features

- USD articulations, including robots and non-robot articulated objects;
- composite USD scenes composed once per environment, with explicitly declared
  embedded rigid bodies and articulations bound to their existing Prims;
- optional static composite semantics that retain authored support collision while
  excluding undeclared bodies and private joints from dynamic solve/reset views;
- immutable static-scene USD composition without rigid-object or contact-sensor wrappers;
- physical XYZ rigid/static-scene scaling, uniform articulation scaling, and
  capability-gated composite-scene scaling;
- joint position, velocity, and effort control;
- cached control-mode gains, with exact rewrites only on mode changes and reset;
- rigid-body pose/twist, persistent wrench, aggregated normal contact, reset, and scene pose writes;
- triangle surface deformables and tetrahedral volume deformables;
- volume-deformable kinematic-node position control;
- fixed-count PhysX PBD particle-fluid state and commands;
- RTX RGB/depth/normals camera sensors, including articulation-root and named-link mounts;
- native point/line/axes/text/bounding-box/trajectory debug overlays;
- scene snapshots, scene deltas, and idempotent browser drag transactions;
- multi-environment batching and restartable process-isolated lifecycle.

## Assets and rendering fidelity

Native rigid USD must contain exactly one `UsdPhysics.RigidBodyAPI` prim. With `unirobosim-usd-converter` installed, visual-only rigid USD is normalized to `isaaclab.dynamic-rigid-usd@1`: the derived layer references the original visual/material composition and adds Z-up metre units, mass, inertia, collision, and physics material.

Articulations, skinned meshes, and empty stages are rejected by the rigid normalizer. Concave containers such as cups and bowls require an authored collision representation or an intentional convex-decomposition policy; a single convex hull will close the cavity.

The EasyAPI profile defaults to FXAA and disables texture streaming so camera sensors keep full-resolution textures. `anti_aliasing` accepts `off`, `taa`, `fxaa`, `dlss`, or `dlaa`. Enable `texture_streaming=True` only when the memory saving is worth reduced texture fidelity.

The headless camera profile renders on sensor reads instead of every physics tick. All camera reads at one world revision share that global RTX render, and visible runs reuse a render already produced by the physics step. RGB frames cross the worker boundary as compact C-order NHWC bytes; use `camera.read("rgb").to_bytes()` in recorders to avoid expanding a frame into Python integers. Programmatic visible runs can set `max_render_hz` to cap viewport rendering independently of physics frequency.

## Explicit limitations

In this Isaac Sim profile, particle state is read through PhysX/USD rather than a public particle tensor API. Rigid + fluid + camera + debug is verified together. Contact-force readback for bridged rigid bodies, and a same-world mix of particle fluid with tensor-backed articulations or deformables, fail explicitly instead of returning misleading data.

Non-uniform composite USD scale is supported only after authoring produces a static
scene with Mesh or Cube collision. Composite scenes with embedded rigid bodies or
articulations use uniform scale, because a non-uniform transform cannot preserve
their rigid poses, joint anchors, and inertia consistently. Pre-cook such an asset at
the target dimensions when non-uniform scaling of a mechanism is required.

## Planning-scene compatibility

Adapter 0.10.16 requires UniRoboSim Core `>=0.10.5,<0.11`, adds physical checkpoints, and defines entity pose as the imported asset-root USD Prim pose. Physical root-link and child-link poses remain explicit planning link states. Initial spawn captures the authored entity-to-physical-root transform and converts the public entity pose into Isaac Lab's physical-root `init_state`; runtime `set_pose` uses the same retargeting rule, and reset restores both frames. Pose-preserving runtime attachments write USD FixedJoint frames relative to the targeted rigid-body Prims, so authored center-of-mass offsets cannot snap an attached child. Entity-level pose reads and writes remove authored scale and shear before extracting rotation and normalize the quaternion explicitly; they target the same Prim and move the associated physical bodies coherently without changing articulation joints or authored scale. The adapter also renders resource-backed debug meshes as filled native USD overlays and exposes `planning.scene@2`. Named planning frames are physical references declared with `name`, `owner_link_name`, and `source`; they do not encode grasp, place, handle, or task semantics. Older declaration schemas fail explicitly. A PhysX `convexHull` collision carrier may be the Mesh itself or a container with exactly one descendant Mesh that has no nested `PhysicsCollisionAPI`; ambiguous multi-Mesh and nested-collider carriers still fail closed.

Static and composite USD scenes expose immutable planning resources plus live world
transforms. Degenerate faces with fewer than three vertices are skipped; invalid
indices, inconsistent orientation data, and meshes without a usable triangle fail
closed instead of producing incomplete planner geometry.

## Verification

```bash
python -m pip install -e '.[dev]'
ruff format --check src tests
ruff check src tests
mypy src
pytest -q
python scripts/native_conformance.py --output result.json
```

The release gate covers the SDK-free source suite, deterministic artifacts, isolated
wheel installation, and installed entry-point discovery. The native acceptance matrix
covers rigid/contact, surface/volume deformables, robot and non-robot articulations,
particle fluid, RGB/depth/normals, mounted cameras, static USD scenes, native debug, provider reopen, planning-scene
catalog/state/delta/resource reads, multiple environments, and clean-interpreter
startup.

## Repository relationship

This package contains only the Isaac Lab adapter. Portable contracts and EasyAPI live in [UniRoboSim Core](https://github.com/GitHofee/UniRoboSim.git); browser and MCP interfaces remain separate packages.

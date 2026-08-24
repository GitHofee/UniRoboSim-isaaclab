# UniRoboSim Isaac Lab adapter

[English](README.md) | [简体中文](README.zh-CN.md)

`unirobosim-isaaclab` connects UniRoboSim `0.10.x` to Isaac Lab 3.0 / Isaac Sim 6.0.1. Import and `probe()` are side-effect free. `open()` starts Kit in an adapter-owned worker process, so Isaac Sim cannot take over or terminate the application process.

## Compatibility

| Item | Required profile |
| --- | --- |
| Python | `>=3.12,<3.13` |
| UniRoboSim | `>=0.10,<0.11` |
| Isaac Lab profile | `release/3.0.0-beta2` (`isaaclab==6.1.17`) |
| Isaac Lab PhysX | `1.1.3` |
| Isaac Sim | `6.0.1.0` |
| PyTorch | `torch==2.11.0` |
| TorchVision | `torchvision==0.26.0` |
| TorchAudio | `torchaudio==2.11.0` |
| Runtime contract | `v0alpha6` |

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

The CUDA 12.8 command above is the tested Linux x86-64 profile. CUDA wheels with the
same public versions and a compatible local tag are accepted. The adapter intentionally
keeps the large NVIDIA and Torch packages out of its own wheel metadata. The probe reads
distribution metadata for Isaac Lab, Isaac Lab PhysX, Isaac Sim, PyTorch, TorchVision,
and TorchAudio, and fails closed on a missing or incompatible package. `pip check` must
finish with `No broken requirements found`.

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

Values are case-sensitive. Any other value fails before the Isaac SDK is loaded.
The adapter does not infer a profile from `DISPLAY`, so batch behavior is stable across
machines. This selector applies only to normal installed entry-point discovery;
`create_provider(IsaacLabAdapterConfig(...))` remains the explicit programmatic API.
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
- immutable static-scene USD composition without rigid-object or contact-sensor wrappers;
- physical rigid, uniform articulation, and static-scene asset scaling;
- joint position, velocity, and effort control;
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

The headless camera profile renders on sensor reads instead of every physics tick. RGB frames cross the worker boundary as compact C-order NHWC bytes; use `camera.read("rgb").to_bytes()` in recorders to avoid expanding a frame into Python integers. Programmatic visible runs can set `max_render_hz` to cap viewport rendering independently of physics frequency.

## Explicit limitations

In this Isaac Sim profile, particle state is read through PhysX/USD rather than a public particle tensor API. Rigid + fluid + camera + debug is verified together. Contact-force readback for bridged rigid bodies, and a same-world mix of particle fluid with tensor-backed articulations or deformables, fail explicitly instead of returning misleading data.

## Planning-scene compatibility

Adapter 0.10.0 requires UniRoboSim Core `>=0.10,<0.11` and exposes `planning.scene@2`. Named planning frames are physical references declared with `name`, `owner_link_name`, and `source`; they do not encode grasp, place, handle, or task semantics. Applications that used the earlier semantic frame-role draft must migrate those meanings to their task or annotation layer and keep only the neutral physical frame declaration in the simulator contract. Older declaration schemas fail explicitly. A PhysX `convexHull` collision carrier may be the Mesh itself or a container with exactly one descendant Mesh that has no nested `PhysicsCollisionAPI`; ambiguous multi-Mesh and nested-collider carriers still fail closed.

A world that demands `planning.scene@2` currently rejects `STATIC_SCENE` and
`COMPOSITE_SCENE` before native allocation. This fail-closed gate prevents a planner
from receiving a silently incomplete room collider set.

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

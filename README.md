# UniRoboSim Isaac Lab adapter

[English](README.md) | [简体中文](README.zh-CN.md)

`unirobosim-isaaclab` connects UniRoboSim `0.9.x` to Isaac Lab 3.0 / Isaac Sim 6.0.1. Import and `probe()` are side-effect free. `open()` starts Kit in an adapter-owned worker process, so Isaac Sim cannot take over or terminate the application process.

## Compatibility

| Item | Verified version |
| --- | --- |
| Python | `>=3.12,<3.13` |
| UniRoboSim | `>=0.9,<0.10` |
| Isaac Lab profile | `release/3.0.0-beta2` (`isaaclab==6.1.17`) |
| Isaac Lab PhysX | `1.1.3` |
| Isaac Sim | `6.0.1.0` |
| PyTorch | `2.10.0+cu128` |
| Runtime contract | `v0alpha5` |

## Installation

Install the verified NVIDIA Isaac Sim/Isaac Lab stack in a dedicated Python 3.12 Conda environment first. Confirm that `import isaaclab` and `import isaacsim` work in that environment, then install Core, the adapter, and the optional USD converter:

```bash
conda create -n unirobosim-isaaclab3 python=3.12 pip -y
conda activate unirobosim-isaaclab3

git clone https://github.com/GitHofee/UniRoboSim.git
git clone https://github.com/GitHofee/UniRoboSim-isaaclab.git
git clone https://github.com/GitHofee/UniRoboSim-usd-converter.git

python -m pip install ./UniRoboSim
python -m pip install ./UniRoboSim-isaaclab
python -m pip install ./UniRoboSim-usd-converter
```

Validate the environment without launching Kit:

```bash
python -c "from unirobosim_isaaclab import create_provider; print(create_provider().probe())"
```

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
| `visible` | Visible Kit window, cameras enabled, rendering enabled |

Values are case-sensitive. Any other value fails before the Isaac SDK is loaded.
The adapter does not infer a profile from `DISPLAY`, so batch behavior is stable across
machines. This selector applies only to normal installed entry-point discovery;
`create_provider(IsaacLabAdapterConfig(...))` remains the explicit programmatic API.

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
    )
)

with Sim(provider=provider) as sim:
    sim.add_camera("camera", resolution=(1920, 1080), outputs=("rgb",))
    sim.start()
```

## Implemented native features

- USD articulations, including robots and non-robot articulated objects;
- joint position, velocity, and effort control;
- rigid-body pose/twist, persistent wrench, aggregated normal contact, reset, and scene pose writes;
- triangle surface deformables and tetrahedral volume deformables;
- volume-deformable kinematic-node position control;
- fixed-count PhysX PBD particle-fluid state and commands;
- RTX RGB/depth camera sensors;
- native point/line/axes/text/bounding-box/trajectory debug overlays;
- scene snapshots, scene deltas, and idempotent browser drag transactions;
- multi-environment batching and restartable process-isolated lifecycle.

## Assets and rendering fidelity

Native rigid USD must contain exactly one `UsdPhysics.RigidBodyAPI` prim. With `unirobosim-usd-converter` installed, visual-only rigid USD is normalized to `isaaclab.dynamic-rigid-usd@1`: the derived layer references the original visual/material composition and adds Z-up metre units, mass, inertia, collision, and physics material.

Articulations, skinned meshes, and empty stages are rejected by the rigid normalizer. Concave containers such as cups and bowls require an authored collision representation or an intentional convex-decomposition policy; a single convex hull will close the cavity.

The EasyAPI profile defaults to FXAA and disables texture streaming so camera sensors keep full-resolution textures. `anti_aliasing` accepts `off`, `taa`, `fxaa`, `dlss`, or `dlaa`. Enable `texture_streaming=True` only when the memory saving is worth reduced texture fidelity.

## Explicit limitations

In this Isaac Sim profile, particle state is read through PhysX/USD rather than a public particle tensor API. Rigid + fluid + camera + debug is verified together. Contact-force readback for bridged rigid bodies, and a same-world mix of particle fluid with tensor-backed articulations or deformables, fail explicitly instead of returning misleading data.

## Planning-scene compatibility

Adapter 0.9.2 requires UniRoboSim Core `>=0.9,<0.10` and exposes `planning.scene@2`. Named planning frames are physical references declared with `name`, `owner_link_name`, and `source`; they do not encode grasp, place, handle, or task semantics. Applications that used the earlier semantic frame-role draft must migrate those meanings to their task or annotation layer and keep only the neutral physical frame declaration in the simulator contract. Older declaration schemas fail explicitly.

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
particle fluid, RGB/depth, native debug, provider reopen, planning-scene
catalog/state/delta/resource reads, multiple environments, and clean-interpreter
startup.

## Repository relationship

This package contains only the Isaac Lab adapter. Portable contracts and EasyAPI live in [UniRoboSim Core](https://github.com/GitHofee/UniRoboSim.git); browser and MCP interfaces remain separate packages.

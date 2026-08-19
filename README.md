# UniRoboSim Isaac Lab adapter

The optional Isaac Lab 3.0 / Isaac Sim 6.0.1 backend for UniRoboSim. Import and `probe()` are
side-effect free; `open()` launches Kit in an adapter-owned worker process so Isaac Sim shutdown
cannot terminate or corrupt the caller.

```python
from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider

provider = create_provider(IsaacLabAdapterConfig(headless=True, device="cuda:0"))
report = provider.probe()
if report.available:
    session = provider.open()
    try:
        # world = session.build(...)
        ...
    finally:
        session.close()
```

This alpha profile targets Python 3.12, Isaac Lab `release/3.0.0-beta2`
(`isaaclab==6.1.17`, `isaaclab_physx==1.1.3`), Isaac Sim `6.0.1.0`, and the
officially pinned PyTorch `2.10.0` profile.

Supported native paths are USD articulations, rigid-body state and persistent wrench control, aggregated
normal-contact state, rigid-body USD assets, fixed-topology triangle/tetrahedral deformables, articulation
position/velocity/effort control, volume-deformable kinematic-node position control, fixed-count PhysX PBD
particle fluid position/velocity state and commands, RTX RGB/depth cameras, and native USD debug overlays.
Camera capabilities are advertised only when both `enable_cameras=True` and `render=True` are selected.
The EasyAPI camera profile defaults to explicit FXAA with texture streaming disabled. This keeps
full-resolution source textures resident for sensor fidelity and avoids Isaac Lab's headless rendering
preset silently restoring DLSS. Large multi-environment workloads may trade fidelity for memory by using
`IsaacLabAdapterConfig(texture_streaming=True)`; `anti_aliasing` accepts `off`, `taa`, `fxaa`, `dlss`, or
`dlaa`. The native worker reapplies these settings after rendering presets load and rejects a silent
anti-aliasing or texture-streaming mismatch.

The adapter declares the semantic target `isaaclab.dynamic-rigid-usd@1`. With the optional
`unirobosim-usd-converter` package installed, EasyAPI inspects a visual-only rigid USD and derives a
physics-ready USD before the adapter build. The derived stage keeps the source USD visual/material
composition by reference and adds normalized Z-up metre units, mass, inertia, collision and physics
material. Articulations, skinned meshes and empty stages are rejected by this rigid-only profile.

Isaac Sim 6.0.1 does not expose a public particle tensor view in this profile. A world containing particle
fluid therefore uses PhysX-to-USD particle readback and a USD rigid-body bridge. Rigid + fluid + camera +
debug is verified together; contact-force readback for bridged rigid bodies and a same-world combination of
particle fluid with tensor-backed articulations or deformables fail explicitly. Rigid assets must contain
exactly one `UsdPhysics.RigidBodyAPI` prim.

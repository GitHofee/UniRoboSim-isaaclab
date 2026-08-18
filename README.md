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
normal-contact state, rigid-body USD assets, fixed-topology triangle/tetrahedral
deformables, articulation position/velocity/effort control, and volume-deformable kinematic-node position
control. Rigid assets must contain exactly one `UsdPhysics.RigidBodyAPI` prim; unsupported mappings fail
explicitly instead of falling back to an approximation. Particle fluids are not advertised by this adapter
milestone.

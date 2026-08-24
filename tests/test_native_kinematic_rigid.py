from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from unirobosim import EntityPath

from unirobosim_isaaclab.native import IsaacLabNativeWorld, _is_kinematic_rigid


class FakeTensor:
    def __init__(self, label: str) -> None:
        self.label = label
        self.device = "cpu"
        self.dtype = "float32"

    def clone(self) -> FakeTensor:
        return FakeTensor(f"{self.label}.clone")

    def __getitem__(self, key: object) -> FakeTensor:
        return FakeTensor(f"{self.label}[{key!r}]")

    def __setitem__(self, key: object, value: object) -> None:
        del key, value

    def __add__(self, other: object) -> FakeTensor:
        del other
        return FakeTensor(f"{self.label}.add")

    def __iadd__(self, other: object) -> FakeTensor:
        del other
        return self


class FakeTorch:
    int64 = "int64"
    float32 = "float32"

    @staticmethod
    def tensor(data: object, *, device: object, dtype: object) -> FakeTensor:
        del data, device, dtype
        return FakeTensor("tensor")

    @staticmethod
    def zeros(shape: object, *, device: object, dtype: object) -> FakeTensor:
        del shape, device, dtype
        return FakeTensor("zeros")


class FakeSimulation:
    device = "cpu"

    def __init__(self) -> None:
        self.forward_count = 0

    def forward(self) -> None:
        self.forward_count += 1

    @staticmethod
    def get_physics_dt() -> float:
        return 1.0 / 120.0


class FakeRawRigidView:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_transforms(self) -> FakeTensor:
        return FakeTensor("current_transforms")

    def set_transforms(self, transforms: object, indices: object) -> None:
        del transforms, indices
        self.calls.append("pose")

    def set_velocities(self, velocities: object, indices: object) -> None:
        del velocities, indices
        self.calls.append("velocity")


class FakeHighLevelRigid:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def reset(self, *, env_ids: object) -> None:
        del env_ids
        self.calls.append("reset_buffers")

    def write_root_pose_to_sim_index(self, *, root_pose: object, env_ids: object) -> None:
        del root_pose, env_ids
        self.calls.append("pose")

    def write_root_link_velocity_to_sim_index(self, *, root_velocity: object, env_ids: object) -> None:
        del root_velocity, env_ids
        self.calls.append("velocity")

    def update(self, dt: float) -> None:
        del dt


class FakeContact:
    def reset(self, *, env_ids: object) -> None:
        del env_ids

    def update(self, dt: float) -> None:
        del dt


def _bare_world(*, kinematic: bool, raw: bool) -> tuple[IsaacLabNativeWorld, Any]:
    path = EntityPath("/objects/funnel")
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=FakeTorch())
    world._sim = FakeSimulation()
    world._origins_cpu = ((0.0, 0.0, 0.0),)
    world._origins = FakeTensor("origins")
    world._articulations = {}
    world._usd_articulation_views = {}
    world._deformables = {}
    world._fluids = {}
    world._cameras = {}
    world._mounted_cameras = {}
    world._debug_lifetimes = {}
    world._kinematic_rigids = {path: kinematic}
    if raw:
        native = FakeRawRigidView()
        world._rigids = {}
        world._contacts = {}
        world._initial_rigid = {}
        world._usd_rigid_views = {path: native}
        world._initial_usd_rigid = {path: (FakeTensor("initial_transforms"), FakeTensor("initial_velocities"))}
        world._usd_rigid_wrenches = {path: (FakeTensor("forces"), FakeTensor("torques"))}
    else:
        native = FakeHighLevelRigid()
        world._rigids = {path: native}
        world._contacts = {path: FakeContact()}
        world._initial_rigid = {path: (FakeTensor("initial_pose"), FakeTensor("initial_velocity"))}
        world._usd_rigid_views = {}
        world._initial_usd_rigid = {}
        world._usd_rigid_wrenches = {}
    return world, native


@pytest.mark.parametrize("raw", [False, True], ids=["high-level", "raw-usd"])
def test_kinematic_rigid_reset_and_pose_update_never_write_velocity(raw: bool) -> None:
    world, native = _bare_world(kinematic=True, raw=raw)
    path = EntityPath("/objects/funnel")

    world.reset((0,))
    assert "pose" in native.calls
    assert "velocity" not in native.calls

    native.calls.clear()
    world.set_rigid_body_pose(path, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 0)
    assert native.calls == ["pose"]


def test_native_physics_diagnostics_read_live_simulation_context_dt() -> None:
    world = object.__new__(IsaacLabNativeWorld)
    world._sim = FakeSimulation()
    world._spec = SimpleNamespace(physics=SimpleNamespace(time_step_seconds=1.0 / 60.0, substeps=2))

    diagnostics = world.physics_diagnostics()

    assert diagnostics.native_step_dt_seconds == 1.0 / 120.0
    assert diagnostics.substeps == 2
    assert diagnostics.world_step_dt_seconds == 1.0 / 60.0
    assert "SimulationContext.get_physics_dt" in diagnostics.source


@pytest.mark.parametrize("raw", [False, True], ids=["high-level", "raw-usd"])
def test_dynamic_rigid_reset_and_pose_update_preserve_velocity_writes(raw: bool) -> None:
    world, native = _bare_world(kinematic=False, raw=raw)
    path = EntityPath("/objects/funnel")

    world.reset((0,))
    assert native.calls[-2:] == ["pose", "velocity"]

    native.calls.clear()
    world.set_rigid_body_pose(path, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 0)
    assert native.calls == ["pose", "velocity"]


@pytest.mark.parametrize(("authored", "expected"), [(None, False), (False, False), (True, True)])
def test_kinematic_motion_mode_is_read_from_the_usd_rigid_api(authored: bool | None, expected: bool) -> None:
    class Attribute:
        def Get(self) -> bool | None:
            return authored

    class RigidBodyAPI:
        def GetKinematicEnabledAttr(self) -> Attribute:
            return Attribute()

    usd_physics = SimpleNamespace(RigidBodyAPI=lambda prim: RigidBodyAPI())
    assert _is_kinematic_rigid(object(), usd_physics) is expected

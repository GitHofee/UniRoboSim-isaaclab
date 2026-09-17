from __future__ import annotations

import pytest
from unirobosim import EntityKind, EntityPath, EntitySpec, KinematicTarget, ValidationError, WorldSpec

from unirobosim_isaaclab import IsaacLabProvider

from .helpers import FakeNativeRuntime, available_probe, make_articulation_asset


def test_native_batch_transfers_each_articulation_once_and_rereads_actual_poses():
    from types import SimpleNamespace as NS

    from unirobosim_isaaclab.native import IsaacLabNativeWorld

    from .test_planning_state_layout import _Rows

    path = EntityPath("/robot")
    positions = _Rows([[1., 0., 0., 0., 0., 0., 1.], [2., 0., 0., 0., 0., 1., 0.]])
    velocities = _Rows([[0.] * 6, [1.] * 6])
    asset = NS(body_names=("base", "tip"), data=NS(
        body_link_pose_w=NS(torch=[positions]), body_link_vel_w=NS(torch=[velocities])))
    world = object.__new__(IsaacLabNativeWorld)
    world._spec = NS(environments=NS(count=1))
    world._origins_cpu = [(0.5, 0., 0.)]
    world._articulations = {path: asset}
    targets = (KinematicTarget("tip", path, "tip"), KinematicTarget("base", path, "base"),
               KinematicTarget("tip-again", path, "tip"))
    states = world.read_selected_kinematics(targets)
    assert positions.reads == velocities.reads == 1
    assert tuple(item.target_id for item in states) == ("tip", "base", "tip-again")
    assert states[0].position_m == (1.5, 0., 0.)
    assert states[0].orientation_xyzw == (0., 0., 1., 0.)
    assert states[1].orientation_xyzw == (0., 0., 0., 1.)
    positions.values[1][0] = 3.
    assert world.read_selected_kinematics(targets)[0].position_m == (2.5, 0., 0.)
    assert positions.reads == velocities.reads == 2
    with pytest.raises(KeyError, match="exactly one body"):
        world.read_selected_kinematics((KinematicTarget("missing", path, "missing"),))


def test_selected_link_read_preserves_identity_and_does_not_advance(tmp_path) -> None:
    asset = make_articulation_asset(tmp_path / "arm.usda")
    arm = EntitySpec(
        EntityPath("/articulations/pour_arm"),
        EntityKind.ARTICULATION,
        joint_names=("tilt",),
        initial_joint_positions=(0.0,),
        asset_uri=str(asset),
    )
    runtime = FakeNativeRuntime()
    provider = IsaacLabProvider(runtime_factory=lambda config: runtime, probe_function=available_probe)
    session = provider.open()
    world = session.build(WorldSpec("selected-link", (arm,)))
    target = KinematicTarget("kettle-outlet", arm.path, "kettle_link")
    before = world.tick

    states = world.read_selected_kinematics((target,))

    assert world.tick == before
    assert states[0].tick == before
    assert states[0].target_id == "kettle-outlet"
    assert states[0].pose.position == (0.4, -0.2, 0.9)
    assert runtime.worlds[0].calls[-1][0] == "read_selected_kinematics"
    session.close()


def test_selected_link_read_rejects_duplicate_ids_before_native_call(tmp_path) -> None:
    asset = make_articulation_asset(tmp_path / "arm.usda")
    arm = EntitySpec(
        EntityPath("/articulations/pour_arm"),
        EntityKind.ARTICULATION,
        joint_names=("tilt",),
        initial_joint_positions=(0.0,),
        asset_uri=str(asset),
    )
    runtime = FakeNativeRuntime()
    provider = IsaacLabProvider(runtime_factory=lambda config: runtime, probe_function=available_probe)
    session = provider.open()
    world = session.build(WorldSpec("selected-link", (arm,)))
    target = KinematicTarget("same", arm.path, "kettle_link")
    before = len(runtime.worlds[0].calls)

    with pytest.raises(ValidationError, match="unique"):
        world.read_selected_kinematics((target, target))

    assert len(runtime.worlds[0].calls) == before
    session.close()

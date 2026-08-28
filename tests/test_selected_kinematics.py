from __future__ import annotations

import pytest
from unirobosim import EntityKind, EntityPath, EntitySpec, KinematicTarget, ValidationError, WorldSpec

from unirobosim_isaaclab import IsaacLabProvider

from .helpers import FakeNativeRuntime, available_probe, make_articulation_asset


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

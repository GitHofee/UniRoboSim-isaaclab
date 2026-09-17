from dataclasses import replace

import pytest
from unirobosim import PlanningSceneContractError

from unirobosim_isaaclab.native_protocols import NativePlanningPoseState

from .test_planning_scene import _FakePlanningRuntime, _make_planning_fixture, _open_session
from .test_planning_state_layout import _admission


def test_native_pose_only_branch_reads_actual_bodies_without_frames_geometry_or_joints():
    admission, path, poses, _ = _admission()
    admission._world._articulations[path] = admission._world._rigids.pop(path)
    # No joint arrays exist on this fake; the pose-only branch must not access them.
    admission._frames = {}
    admission._ordered_geometry_bindings = None
    first = admission.state(0, poses_only=True)
    assert type(first) is NativePlanningPoseState
    assert tuple(item.link_id for item in first.links) == ("link.base", "link.tip")
    poses.values[0][0] = 0.25
    second = admission.state(0, poses_only=True)
    assert second.step_index == first.step_index
    assert second.links[0].pose.position_m[0] == 0.25
    assert admission.state(1, poses_only=True).links[0].pose.position_m[0] == -1.75


def test_public_pose_read_preserves_history_and_rejects_incomplete_or_stale(tmp_path):
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    world = session.build(fixture.spec)
    try:
        native = runtime.worlds[0]
        pose = NativePlanningPoseState(0, fixture.state.entities, fixture.state.links)
        native.planning_pose_state = lambda environment: pose
        native.planning_calls.clear()
        first = world.planning_scene_pose_state()
        first.validate_against(world.planning_scene_catalog())
        assert native.planning_calls == []
        assert world._planning_environments[0].state.sequence == 1
        pose = replace(pose, links=pose.links[:-1])
        with pytest.raises(PlanningSceneContractError):
            world.planning_scene_pose_state()
        pose = replace(pose, step_index=1)
        with pytest.raises(PlanningSceneContractError, match="stale"):
            world.planning_scene_pose_state()
    finally:
        session.close()


@pytest.mark.parametrize("environment", [True, -1, 1, 0.0])
def test_pose_environment_requires_exact_admitted_index(tmp_path, environment):
    fixture = _make_planning_fixture(tmp_path)
    session = _open_session(_FakePlanningRuntime(fixture))
    world = session.build(fixture.spec)
    try:
        with pytest.raises(PlanningSceneContractError, match="environment"):
            world.planning_scene_pose_state(environment)
    finally:
        session.close()


def test_pose_extension_unavailable_is_explicit_without_stale_fallback(tmp_path):
    fixture = _make_planning_fixture(tmp_path)
    session = _open_session(_FakePlanningRuntime(fixture))
    world = session.build(fixture.spec)
    try:
        with pytest.raises(PlanningSceneContractError, match="unavailable"):
            world.planning_scene_pose_state()
    finally:
        session.close()

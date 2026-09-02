from __future__ import annotations

import math

import pytest
from unirobosim import Pose

from unirobosim_isaaclab.native import (
    _attachment_joint_frames,
    _compose_pose,
    _relative_pose,
)


def _assert_same_pose(actual: Pose, expected: Pose) -> None:
    assert actual.position == pytest.approx(expected.position, abs=1.0e-12)
    dot = abs(sum(actual.orientation_xyzw[index] * expected.orientation_xyzw[index] for index in range(4)))
    assert dot == pytest.approx(1.0, abs=1.0e-12)


def test_default_attachment_joint_frames_are_body_prim_local_and_world_coincident() -> None:
    half_parent_yaw = math.radians(37.0) / 2.0
    half_child_roll = math.radians(-28.0) / 2.0
    parent_endpoint_pose = Pose(
        (0.31, -0.27, 1.12),
        (0.0, 0.0, math.sin(half_parent_yaw), math.cos(half_parent_yaw)),
    )
    child_endpoint_pose = Pose(
        (0.46, -0.08, 0.93),
        (math.sin(half_child_roll), 0.0, 0.0, math.cos(half_child_roll)),
    )
    relative, parent_body_T_joint, child_body_T_joint = _attachment_joint_frames(
        parent_endpoint_pose,
        child_endpoint_pose,
        None,
    )

    _assert_same_pose(relative, _relative_pose(parent_endpoint_pose, child_endpoint_pose))
    _assert_same_pose(_compose_pose(parent_endpoint_pose, parent_body_T_joint), child_endpoint_pose)
    _assert_same_pose(_compose_pose(child_endpoint_pose, child_body_T_joint), child_endpoint_pose)
    _assert_same_pose(child_body_T_joint, Pose((0.0, 0.0, 0.0)))


def test_explicit_attachment_relation_remains_an_intentional_retarget() -> None:
    parent_body_pose = Pose((0.1, 0.2, 0.3))
    child_body_pose = Pose((1.0, 2.0, 3.0))
    requested = Pose((0.4, -0.2, 0.7))

    relative, parent_body_T_joint, child_body_T_joint = _attachment_joint_frames(
        parent_body_pose,
        child_body_pose,
        requested,
    )

    assert relative is requested
    _assert_same_pose(parent_body_T_joint, requested)
    _assert_same_pose(child_body_T_joint, _relative_pose(child_body_pose, _compose_pose(parent_body_pose, requested)))

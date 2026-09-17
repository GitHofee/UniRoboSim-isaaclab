from types import SimpleNamespace as NS

import pytest
from unirobosim import EntityKind, EntityPath, PlanningJointDescriptor, PlanningJointType

from unirobosim_isaaclab.native_joint_limits import effective_joint_drive_limits
from unirobosim_isaaclab.native_protocols import NativePlanningError

from .test_planning_state_layout import _Rows


def _fixture(raw=False):
    path = EntityPath("/robot")
    joint = PlanningJointDescriptor(
        "joint.slide",
        "entity.robot",
        "slide",
        "link.base",
        "link.tip",
        PlanningJointType.PRISMATIC,
        "frame.joint",
        (1.0, 0.0, 0.0),
        "m",
        0.0,
        0.02,
    )
    velocity = _Rows([[0.04, 2.0], [0.04, 2.0]])
    effort = _Rows([[100.0, 200.0], [100.0, 200.0]])
    asset = NS(
        joint_names=("slide", "other"),
        data=NS(joint_vel_limits=NS(torch=velocity), joint_effort_limits=NS(torch=effort)),
    )
    view = NS(get_dof_max_velocities=lambda: velocity, get_dof_max_forces=lambda: effort)
    world = NS(
        _spec=NS(environments=NS(count=2)),
        _articulations={} if raw else {path: asset},
        _usd_articulation_views={path: view} if raw else {},
        _joint_maps={path: (0,)},
    )
    spec = NS(kind=EntityKind.ARTICULATION, path=path, joint_names=("slide",))
    binding = NS(joint_bindings=(NS(movable_name="slide", descriptor=joint),))
    return world, spec, binding, joint, velocity, effort


@pytest.mark.parametrize("raw", [False, True])
def test_effective_values_come_from_native_limits_and_preserve_position_contract(raw):
    world, spec, binding, joint, velocity, effort = _fixture(raw)
    (updated,) = effective_joint_drive_limits(world, spec, binding, (joint,))
    assert updated.max_velocity == 0.04
    assert updated.max_effort == 100.0
    assert (updated.lower, updated.upper) == (joint.lower, joint.upper)
    assert velocity.reads == effort.reads == 1
    assert joint.max_velocity is None


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_invalid_limits_fail_admission(value):
    world, spec, binding, joint, velocity, _ = _fixture()
    velocity.values[0][0] = value
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))


def test_unbounded_and_passive_limits_do_not_claim_positive_caps():
    world, spec, binding, joint, velocity, effort = _fixture()
    for row in velocity.values:
        row[0] = float("inf")
    for row in effort.values:
        row[0] = 0.0
    (updated,) = effective_joint_drive_limits(world, spec, binding, (joint,))
    assert updated.max_velocity is updated.max_effort is None


def test_mismatched_environment_limits_fail_admission():
    world, spec, binding, joint, velocity, _ = _fixture()
    velocity.values[1][0] = 0.01
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))


@pytest.mark.parametrize("change", ["ragged", "different_channel_width", "missing_environment"])
def test_malformed_limit_layout_is_rejected(change):
    world, spec, binding, joint, velocity, effort = _fixture()
    if change == "ragged":
        velocity.values[1].append(1.0)
    elif change == "different_channel_width":
        for row in effort.values:
            row.append(1.0)
    else:
        velocity.values.pop()
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))

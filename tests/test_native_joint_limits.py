import math
from dataclasses import replace
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
    position = _Rows([[[0.0, 0.019999999552965164], [-1.0, 1.0]], [[0.0, 0.019999999552965164], [-1.0, 1.0]]])
    asset = NS(
        joint_names=("slide", "other"),
        data=NS(
            joint_pos_limits=NS(torch=position),
            joint_vel_limits=NS(torch=velocity),
            joint_effort_limits=NS(torch=effort),
        ),
    )
    view = NS(
        get_dof_limits=lambda: position, get_dof_max_velocities=lambda: velocity, get_dof_max_forces=lambda: effort
    )
    world = NS(
        _spec=NS(environments=NS(count=2)),
        position_limits=position,
        _articulations={} if raw else {path: asset},
        _usd_articulation_views={path: view} if raw else {},
        _joint_maps={path: (0,)},
    )
    spec = NS(kind=EntityKind.ARTICULATION, path=path, joint_names=("slide",))
    binding = NS(joint_bindings=(NS(movable_name="slide", descriptor=joint),))
    return world, spec, binding, joint, velocity, effort


@pytest.mark.parametrize("raw", [False, True])
def test_effective_values_come_from_native_limits(raw):
    world, spec, binding, joint, velocity, effort = _fixture(raw)
    (updated,) = effective_joint_drive_limits(world, spec, binding, (joint,))
    assert updated.max_velocity == 0.04
    assert updated.max_effort == 100.0
    assert (updated.lower, updated.upper) == (0.0, 0.019999999552965164)
    assert world.position_limits.reads == 1
    assert joint.upper == 0.02
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


@pytest.mark.parametrize("raw", [False, True])
def test_sdk_revolute_limit_contains_measured_float32_boundary(raw):
    world, spec, binding, joint, _, _ = _fixture(raw)
    joint = replace(
        joint, joint_type=PlanningJointType.REVOLUTE, position_unit="rad", lower=-2 * math.pi, upper=2 * math.pi
    )
    binding.joint_bindings[0].descriptor = joint
    actual = [-6.2831854820251465, 6.2831854820251465]
    for row in world.position_limits.values:
        row[0] = actual[:]
    (updated,) = effective_joint_drive_limits(world, spec, binding, (joint,))
    assert [updated.lower, updated.upper] == actual
    assert updated.lower <= actual[0] <= updated.upper
    assert updated.lower - actual[0] == 0.0
    assert joint.lower - actual[0] > 0.0


@pytest.mark.parametrize("raw", [False, True])
def test_position_limits_follow_native_dof_mapping(raw):
    world, spec, binding, joint, _, _ = _fixture(raw)
    if raw:
        world._joint_maps[spec.path] = (1,)
    else:
        world._articulations[spec.path].joint_names = ("other", "slide")
    (updated,) = effective_joint_drive_limits(world, spec, binding, (joint,))
    assert (updated.lower, updated.upper) == (-1.0, 1.0)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize(
    "pair",
    [(float("nan"), 1.0), (2.0, 1.0), (-float("inf"), 1.0), (0.0, float("inf")), (float("inf"), float("inf")), (0.0,)],
)
def test_invalid_or_unrepresentable_position_limits_fail(raw, pair):
    world, spec, binding, joint, _, _ = _fixture(raw)
    for row in world.position_limits.values:
        row[0] = list(pair)
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))


@pytest.mark.parametrize("raw", [False, True])
def test_native_unbounded_position_pair_remains_unbounded(raw):
    world, spec, binding, joint, _, _ = _fixture(raw)
    for row in world.position_limits.values:
        row[0] = [-math.inf, math.inf]
    (updated,) = effective_joint_drive_limits(world, spec, binding, (joint,))
    assert updated.lower is updated.upper is None


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("change", ["environment_count", "width", "environment_mismatch"])
def test_position_layout_and_environment_mismatch_rejected(raw, change):
    world, spec, binding, joint, _, _ = _fixture(raw)
    rows = world.position_limits.values
    if change == "environment_count":
        rows.pop()
    elif change == "width":
        rows[0].pop()
    else:
        rows[1][0][1] = 0.01
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))

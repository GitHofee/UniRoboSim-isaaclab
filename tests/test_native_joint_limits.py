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


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("bad", [True, False, "0", None, 10**1000])
def test_position_endpoint_type_and_overflow_rejected(raw, bad):
    world, spec, binding, joint, _, _ = _fixture(raw)
    world.position_limits.values[0][0][0] = bad
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("case", ["nonarticulation", "fixed", "empty"])
def test_nonmovable_descriptor_is_preserved(raw, case):
    world, spec, binding, joint, _, _ = _fixture(raw)
    if case == "nonarticulation":
        spec.kind = EntityKind.RIGID_BODY
    elif case == "fixed":
        binding.joint_bindings[0].movable_name = None
    else:
        binding.joint_bindings = ()
    original = (joint,)
    assert effective_joint_drive_limits(world, spec, binding, original) is original
    assert world.position_limits.reads == 0


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("case", ["duplicate_names", "missing_name", "out_of_range"])
def test_native_dof_identity_rejects_ambiguity(raw, case):
    world, spec, binding, joint, _, _ = _fixture(raw)
    if raw:
        if case == "duplicate_names":
            spec.joint_names = ("slide", "slide")
            world._joint_maps[spec.path] = (0, 1)
        elif case == "missing_name":
            spec.joint_names = ("other",)
        else:
            world._joint_maps[spec.path] = (7,)
    else:
        world._articulations[spec.path].joint_names = {
            "duplicate_names": ("slide", "slide"),
            "missing_name": ("other", "unused"),
            "out_of_range": ("other", "unused", "slide"),
        }[case]
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))


@pytest.mark.parametrize("mapping", [None, (), (0, 1), (True,), (0, 0)])
def test_raw_mapping_shape_type_and_duplicates_rejected(mapping):
    world, spec, binding, joint, _, _ = _fixture(True)
    world._joint_maps[spec.path] = mapping
    if mapping == (0, 0):
        spec.joint_names = ("slide", "other")
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        effective_joint_drive_limits(world, spec, binding, (joint,))


def test_catalog_admission_and_binding_share_effective_descriptor():
    from unirobosim import PlanningGeometryLocalPose

    from unirobosim_isaaclab.native_planning import _EntityBinding, _JointBinding, _PlanningAdmission

    world, spec, _, joint, _, _ = _fixture()
    spec.embedded_binding = None
    world._spec.entities = (spec,)
    binding = _EntityBinding(
        spec.path,
        "entity.robot",
        "base",
        "link.base",
        {},
        (_JointBinding(joint, "base", "tip", PlanningGeometryLocalPose(), "slide"),),
        ("/robot",),
    )
    admission = object.__new__(_PlanningAdmission)
    admission._world = world
    admission._frames = {}
    admission._geometries = {}
    admission._entities = {}
    admission._ground_geometry = lambda: None
    admission._entity_catalog = lambda _: (NS(entity_id="entity.robot"), (), (joint,), (), (), binding)
    admission._verify_complete_native_world = lambda: None
    catalog = admission._build_catalog()
    effective = catalog.joints[0]
    assert effective is admission._entities[spec.path].joint_bindings[0].descriptor
    assert effective.upper == 0.019999999552965164
    assert joint.upper == 0.02


def test_public_catalog_preserves_effective_native_precision(tmp_path):
    from unirobosim_isaaclab.config import IsaacLabAdapterConfig
    from unirobosim_isaaclab.descriptor import descriptor_for_config
    from unirobosim_isaaclab.provider import IsaacLabSession

    from .test_planning_scene import _FakePlanningRuntime, _make_planning_fixture

    fixture = _make_planning_fixture(tmp_path)
    updated = replace(fixture.catalog.joints[0], lower=-6.2831854820251465, upper=6.2831854820251465)
    fixture = replace(fixture, catalog=replace(fixture.catalog, joints=(updated, *fixture.catalog.joints[1:])))
    runtime = _FakePlanningRuntime(fixture)
    config = IsaacLabAdapterConfig()
    session = IsaacLabSession(descriptor_for_config(config), runtime, config=config, on_close=lambda _: None)
    try:
        world = session.build(fixture.spec)
        actual = next(j for j in world.planning_scene_catalog().joints if j.joint_id == updated.joint_id)
        assert actual is updated
        assert (actual.lower, actual.upper) == (-6.2831854820251465, 6.2831854820251465)
    finally:
        session.close()

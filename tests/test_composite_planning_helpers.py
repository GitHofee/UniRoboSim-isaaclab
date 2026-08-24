from __future__ import annotations

import math
from pathlib import Path

import pytest
from unirobosim import (
    COMPOSITE_WORLD_SCHEMA_VERSION,
    PHYSICAL_WORLD_SCHEMA_VERSION,
    CapabilityId,
    CapabilityRequirement,
    EntityKind,
    EntityPath,
    EntitySpec,
    PlanningGeometryLocalPose,
    PlanningJointType,
    PlanningSceneIncompleteError,
    WorldSpec,
)

from unirobosim_isaaclab.native_planning import (
    _bake_mesh_linear_transform,
    _compose_scaled_local_pose,
    _cylinder_pose_scale,
    _exact_filtered_pair_encoding,
    _MeshInput,
    _path_is_at_or_under,
    _PlanningAdmission,
    _reflect_mesh_input,
    _rotate,
    _triangulate_faces,
)
from unirobosim_isaaclab.native_protocols import NativePlanningError
from unirobosim_isaaclab.planning_scene import validate_planning_build_spec


def _planning_requirement() -> CapabilityRequirement:
    return CapabilityRequirement(CapabilityId("planning.scene@2"))


def test_planning_preflight_admits_composite_but_keeps_static_scene_fail_closed(tmp_path: Path) -> None:
    asset = tmp_path / "room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    composite = WorldSpec(
        "composite-planning",
        (EntitySpec(EntityPath("/room"), EntityKind.COMPOSITE_SCENE, asset_uri=asset.as_uri()),),
        requirements=(_planning_requirement(),),
        schema_version=COMPOSITE_WORLD_SCHEMA_VERSION,
        build_resource_manifest_sha256="a" * 64,
    )
    validate_planning_build_spec(composite, backend_id="nvidia.isaaclab")

    static = WorldSpec(
        "static-planning",
        (EntitySpec(EntityPath("/room"), EntityKind.STATIC_SCENE, asset_uri=asset.as_uri()),),
        requirements=(_planning_requirement(),),
        schema_version=PHYSICAL_WORLD_SCHEMA_VERSION,
        build_resource_manifest_sha256="b" * 64,
    )
    with pytest.raises(PlanningSceneIncompleteError, match="static-scene collider forest"):
        validate_planning_build_spec(static, backend_id="nvidia.isaaclab")


def test_embedded_moving_subtree_matching_does_not_claim_anchor_siblings() -> None:
    moving = "/World/env_0/room/mechanism/door"
    assert _path_is_at_or_under(moving, moving)
    assert _path_is_at_or_under(f"{moving}/panel", moving)
    assert not _path_is_at_or_under("/World/env_0/room/mechanism/frame/panel", moving)
    assert not _path_is_at_or_under("/World/env_0/room/mechanism/doorway", moving)


def test_triangle_mesh_canonicalization_preserves_winding_and_holes() -> None:
    counts = (4, 3)
    indices = (0, 1, 2, 3, 0, 3, 4)
    assert _triangulate_faces(counts, indices, vertex_count=5, orientation="rightHanded") == (
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
    )
    assert _triangulate_faces(counts, indices, vertex_count=5, orientation="leftHanded") == (
        (0, 2, 1),
        (0, 3, 2),
        (0, 4, 3),
    )
    assert _triangulate_faces(
        counts,
        indices,
        vertex_count=5,
        orientation="rightHanded",
        hole_faces=frozenset({0}),
    ) == ((0, 3, 4),)


@pytest.mark.parametrize(
    ("counts", "indices", "vertex_count", "orientation"),
    (
        ((2,), (0, 1), 2, "rightHanded"),
        ((3,), (0, 1), 2, "rightHanded"),
        ((3,), (0, 1, 2), 3, "unknown"),
    ),
)
def test_triangle_mesh_canonicalization_rejects_incomplete_authored_topology(
    counts: tuple[int, ...],
    indices: tuple[int, ...],
    vertex_count: int,
    orientation: str,
) -> None:
    with pytest.raises(NativePlanningError, match="collision_cooking_failed"):
        _triangulate_faces(counts, indices, vertex_count=vertex_count, orientation=orientation)


def test_bounding_cube_center_offset_respects_carrier_scale_and_rotation() -> None:
    parent = PlanningGeometryLocalPose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    result = _compose_scaled_local_pose(parent, (2.0, 3.0, 4.0), (0.5, -0.5, 0.25))
    assert result.position_m == (2.0, 0.5, 4.0)
    assert result.orientation_xyzw == parent.orientation_xyzw


def _triangle_mesh_input() -> _MeshInput:
    return _MeshInput(
        ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (0.0, 1.0, 0.0)),
        (3,),
        (0, 1, 2),
        ((0, 1, 2),),
        (),
        "none",
        "rightHanded",
        "a" * 64,
    )


def test_mirrored_collision_mesh_is_baked_without_changing_topology() -> None:
    source = _triangle_mesh_input()
    reflected = _reflect_mesh_input(source, (-1, 1, 1))
    assert reflected.vertices[1] == (-1.0, 2.0, 3.0)
    assert reflected.face_indices == source.face_indices
    assert reflected.triangles == source.triangles
    assert reflected.source_sha256 != source.source_sha256


def test_sheared_collision_mesh_is_baked_into_authoritative_vertices() -> None:
    source = _triangle_mesh_input()
    baked = _bake_mesh_linear_transform(
        source,
        ((1.0, 0.0, 0.0), (0.5, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    assert baked.vertices[1] == (2.0, 2.0, 3.0)
    assert baked.face_indices == source.face_indices
    assert baked.source_sha256 != source.source_sha256


@pytest.mark.parametrize(
    ("axis", "expected_scale", "expected_axis"),
    (
        ("X", (4.0, 3.0, 2.0), (1.0, 0.0, 0.0)),
        ("Y", (2.0, 4.0, 3.0), (0.0, 1.0, 0.0)),
        ("Z", (2.0, 3.0, 4.0), (0.0, 0.0, 1.0)),
    ),
)
def test_usd_cylinder_axis_is_canonicalized_to_public_z_axis(
    axis: str,
    expected_scale: tuple[float, float, float],
    expected_axis: tuple[float, float, float],
) -> None:
    pose = PlanningGeometryLocalPose()
    result, scale = _cylinder_pose_scale(pose, (2.0, 3.0, 4.0), axis)
    assert scale == expected_scale
    assert _rotate((0.0, 0.0, 1.0), result.orientation_xyzw) == pytest.approx(expected_axis)


def test_usd_cylinder_unknown_axis_fails_closed() -> None:
    with pytest.raises(NativePlanningError, match="collision_geometry_unsupported"):
        _cylinder_pose_scale(PlanningGeometryLocalPose(), (1.0, 1.0, 1.0), "UNKNOWN")


def _filter_interacts(
    encoding: dict[str, tuple[int, int]],
    left: str,
    right: str,
) -> bool:
    left_group, left_mask = encoding.get(left, (1, 2**32 - 1))
    right_group, right_mask = encoding.get(right, (1, 2**32 - 1))
    return bool(left_group & right_mask) and bool(right_group & left_mask)


def test_filtered_pair_encoding_is_exact_deterministic_and_world_permissive() -> None:
    bodies = ("/robot/c", "/robot/a", "/robot/e", "/robot/d", "/robot/b")
    filtered = frozenset(
        {
            ("/robot/a", "/robot/b"),
            ("/robot/a", "/robot/c"),
            ("/robot/b", "/robot/c"),
            ("/robot/b", "/robot/d"),
        }
    )
    encoding, next_bit, classes = _exact_filtered_pair_encoding(
        bodies,
        filtered,
        first_class_bit=1,
        self_filtered_bodies=frozenset({"/robot/a"}),
    )
    repeated = _exact_filtered_pair_encoding(
        tuple(reversed(bodies)),
        filtered,
        first_class_bit=1,
        self_filtered_bodies=frozenset({"/robot/a"}),
    )
    assert repeated == (encoding, next_bit, classes)
    assert next_bit == 3
    for left_index, left in enumerate(sorted(bodies)):
        assert _filter_interacts(encoding, left, left) is (left == "/robot/e")
        for right in sorted(bodies)[left_index + 1 :]:
            assert _filter_interacts(encoding, left, right) is ((left, right) not in filtered)
        assert _filter_interacts(encoding, left, "/world")


def test_filtered_pair_encoding_rejects_unowned_pair_and_bit_exhaustion() -> None:
    with pytest.raises(NativePlanningError, match="collision_filter_unsupported"):
        _exact_filtered_pair_encoding(
            ("/robot/a",),
            frozenset({("/other/a", "/robot/a")}),
            first_class_bit=1,
        )
    bodies = tuple(f"/robot/link_{index:02d}" for index in range(32))
    with pytest.raises(NativePlanningError, match="collision_filter_unsupported"):
        _exact_filtered_pair_encoding(
            bodies,
            frozenset(),
            first_class_bit=1,
            self_filtered_bodies=frozenset(bodies),
        )


def test_embedded_joint_descriptor_uses_component_authored_alias() -> None:
    class _Attribute:
        def __init__(self, value: object) -> None:
            self._value = value

        def Get(self) -> object:
            return self._value

    class _JointSchema:
        def GetAxisAttr(self) -> _Attribute:
            return _Attribute("Z")

        def GetLowerLimitAttr(self) -> _Attribute:
            return _Attribute(-90.0)

        def GetUpperLimitAttr(self) -> _Attribute:
            return _Attribute(90.0)

    class _UsdPhysics:
        @staticmethod
        def RevoluteJoint(_prim: object) -> _JointSchema:
            return _JointSchema()

    class _Modules:
        UsdPhysics = _UsdPhysics()

    class _Prim:
        @staticmethod
        def GetName() -> str:
            return "RevoluteJoint"

    admission = object.__new__(_PlanningAdmission)
    admission._m = _Modules()  # type: ignore[attr-defined]
    descriptor = admission._joint_descriptor(
        _Prim(),
        "joint.door",
        "entity.door",
        "link.frame",
        "link.door",
        "frame.hinge",
        PlanningJointType.REVOLUTE,
        authored_name="hinge",
    )
    assert descriptor.authored_name == "hinge"
    assert descriptor.position_unit == "rad"
    assert descriptor.lower == pytest.approx(-math.pi / 2.0)
    assert descriptor.upper == pytest.approx(math.pi / 2.0)

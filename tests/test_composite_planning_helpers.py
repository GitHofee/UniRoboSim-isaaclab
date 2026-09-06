from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from unirobosim import (
    COMPOSITE_WORLD_SCHEMA_VERSION,
    PHYSICAL_WORLD_SCHEMA_VERSION,
    ArrayValue,
    CapabilityId,
    CapabilityRequirement,
    EntityKind,
    EntityPath,
    EntitySpec,
    ParticleFluidSpec,
    PlanningGeometryLocalPose,
    PlanningJointType,
    PlanningSceneIncompleteError,
    WorldSpec,
)

from unirobosim_isaaclab.native_planning import (
    _bake_mesh_linear_transform,
    _compose_scaled_local_pose,
    _cylinder_pose_scale,
    _effective_collision_relative_transform,
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


def test_isaac_sim_6_sphere_fill_collision_schema_is_admitted_but_unknown_schemas_fail_closed() -> None:
    class _Attribute:
        @staticmethod
        def Get() -> bool:
            return True

    class _CollisionAPI:
        @staticmethod
        def GetCollisionEnabledAttr() -> _Attribute:
            return _Attribute()

    modules = SimpleNamespace(
        UsdPhysics=SimpleNamespace(
            CollisionAPI=lambda _prim: _CollisionAPI(),
            FilteredPairsAPI=lambda _prim: None,
        ),
        PhysxSchema=SimpleNamespace(PhysxCollisionAPI=lambda _prim: None),
    )
    admission = object.__new__(_PlanningAdmission)
    admission._m = modules
    admission._accounted_filtered_pair_sources = set()

    admitted = SimpleNamespace(
        GetAppliedSchemas=lambda: ("PhysicsCollisionAPI", "PhysxSphereFillCollisionAPI"),
    )
    admission._validate_collision_common(admitted)

    rejected = SimpleNamespace(
        GetAppliedSchemas=lambda: ("PhysicsCollisionAPI", "PhysxUnknownCollisionAPI"),
    )
    with pytest.raises(NativePlanningError, match="collision_geometry_unsupported"):
        admission._validate_collision_common(rejected)


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


def test_particle_fluid_is_not_published_as_planning_geometry() -> None:
    fluid = EntitySpec(
        EntityPath("/water"),
        EntityKind.PARTICLE_FLUID,
        particle_fluid=ParticleFluidSpec(ArrayValue.from_nested(((0.0, 0.0, 1.0),))),
    )
    admission = object.__new__(_PlanningAdmission)
    admission._world = SimpleNamespace(_spec=SimpleNamespace(entities=(fluid,)))
    admission._frames = {}
    admission._entities = {}
    admission._geometries = {}
    admission._ground_geometry = lambda: None
    admission._verify_complete_native_world = lambda: None

    catalog = admission._build_catalog()

    assert catalog.entities == ()
    assert catalog.links == ()
    assert catalog.joints == ()
    assert catalog.geometries == ()
    assert tuple(frame.frame_id for frame in catalog.frames) == ("frame.world",)


def test_embedded_moving_subtree_matching_does_not_claim_anchor_siblings() -> None:
    moving = "/World/env_0/room/mechanism/door"
    assert _path_is_at_or_under(moving, moving)
    assert _path_is_at_or_under(f"{moving}/panel", moving)
    assert not _path_is_at_or_under("/World/env_0/room/mechanism/frame/panel", moving)
    assert not _path_is_at_or_under("/World/env_0/room/mechanism/doorway", moving)


def test_effective_collision_transform_keeps_shared_ancestor_scale_and_isolates_queries() -> None:
    calls: list[object] = []
    caches: list[_Cache] = []

    class _Matrix:
        def __init__(self, name: str) -> None:
            self.name = name

        def RemoveScaleShear(self) -> _Matrix:
            calls.append(("remove_scale_shear", self.name))
            return _Matrix(f"pose({self.name})")

        def GetInverse(self) -> _Matrix:
            calls.append(("inverse", self.name))
            return _Matrix(f"inverse({self.name})")

        def __mul__(self, other: object) -> _Matrix:
            assert isinstance(other, _Matrix)
            calls.append(("multiply", self.name, other.name))
            return _Matrix(f"{self.name}*{other.name}")

    class _Cache:
        def __init__(self) -> None:
            self.relative_called = False
            caches.append(self)

        def ComputeRelativeTransform(self, prim: object, owner: object) -> tuple[_Matrix, bool]:
            assert not self.relative_called, "relative queries must not share a cache"
            self.relative_called = True
            calls.append(("relative_reset_check", prim, owner))
            return _Matrix("scale-cancelling-relative"), False

        def GetLocalToWorldTransform(self, value: object) -> _Matrix:
            assert self.relative_called
            calls.append(("world", value))
            return _Matrix("collision-world" if value == "collision" else "owner-world-with-scale")

    modules = SimpleNamespace(UsdGeom=SimpleNamespace(XformCache=_Cache))
    result = _effective_collision_relative_transform(modules, "collision", "owner")
    repeated = _effective_collision_relative_transform(modules, "collision", "owner")

    assert result.name == "collision-world*inverse(pose(owner-world-with-scale))"
    assert repeated.name == result.name
    assert len(caches) == 2
    assert caches[0] is not caches[1]
    assert ("multiply", "collision-world", "inverse(pose(owner-world-with-scale))") in calls


def test_effective_collision_transform_rejects_reset_stack() -> None:
    cache = SimpleNamespace(ComputeRelativeTransform=lambda _prim, _owner: (object(), True))
    modules = SimpleNamespace(UsdGeom=SimpleNamespace(XformCache=lambda: cache))
    with pytest.raises(NativePlanningError, match="collision_geometry_unsupported"):
        _effective_collision_relative_transform(modules, object(), object())


def test_self_relative_transform_bypasses_usd_for_equal_prim_handles() -> None:
    from unirobosim_isaaclab import native_planning

    prim = SimpleNamespace(path="/Body")
    same_prim_handle = SimpleNamespace(path="/Body")
    assert prim is not same_prim_handle and prim == same_prim_handle

    def unsafe_query(*args: object) -> None:
        pytest.fail("a same-Prim query must never enter the USD wrapper")

    modules = SimpleNamespace(Gf=SimpleNamespace(Matrix4d=lambda diagonal: ("identity", diagonal)))
    cache = SimpleNamespace(ComputeRelativeTransform=unsafe_query)
    assert native_planning._compute_relative_transform(modules, cache, prim, same_prim_handle) == (
        ("identity", 1.0),
        False,
    )


@pytest.mark.parametrize("resets", (False, True))
def test_distinct_prim_relative_transform_preserves_usd_matrix_and_reset(resets: bool) -> None:
    from unirobosim_isaaclab import native_planning

    prim, ancestor, matrix = object(), object(), object()
    calls = []

    def relative(actual_prim: object, actual_ancestor: object) -> tuple[object, bool]:
        calls.append((actual_prim, actual_ancestor))
        return matrix, resets

    cache = SimpleNamespace(ComputeRelativeTransform=relative)
    assert native_planning._compute_relative_transform(object(), cache, prim, ancestor) == (matrix, resets)
    assert calls == [(prim, ancestor)]


@pytest.mark.parametrize("caller", ("_collision_geometry", "_collision_clone_signature"))
def test_collision_admission_and_clone_pass_modules_to_relative_helper(
    monkeypatch: pytest.MonkeyPatch, caller: str
) -> None:
    from unirobosim_isaaclab import native_planning

    modules = object()
    prim = object()
    owner = object()
    admission = object.__new__(_PlanningAdmission)
    admission._m = modules
    admission._xform_cache = object()
    admission._validate_collision_common = lambda _prim: None
    calls: list[tuple[object, object, object]] = []

    class _ReachedHelper(Exception):
        pass

    def relative(actual_modules: object, actual_prim: object, actual_owner: object) -> object:
        calls.append((actual_modules, actual_prim, actual_owner))
        raise _ReachedHelper

    monkeypatch.setattr(native_planning, "_effective_collision_relative_transform", relative)
    with pytest.raises(_ReachedHelper):
        if caller == "_collision_geometry":
            admission._collision_geometry(object(), "entity", prim, owner, None, "frame")
        else:
            admission._collision_clone_signature(prim, owner, "/root")
    assert calls == [(modules, prim, owner)]


@pytest.mark.parametrize("resets", (False, True), ids=("local-transform", "reset-rejected"))
def test_mesh_input_uses_query_local_cache_and_keeps_carrier_transform(
    monkeypatch: pytest.MonkeyPatch, resets: bool
) -> None:
    from unirobosim_isaaclab import native_planning

    carrier = object()
    mesh_prim = object()
    caches: list[_Cache] = []

    class _Cache:
        def __init__(self) -> None:
            self.queried = False
            caches.append(self)

        def ComputeRelativeTransform(self, actual_mesh: object, actual_carrier: object) -> tuple[object, bool]:
            assert (actual_mesh, actual_carrier) == (mesh_prim, carrier)
            assert not self.queried
            self.queried = True
            return SimpleNamespace(Transform=lambda point: (point[0] + 2.0, point[1], point[2])), resets

    def attribute(value: object) -> SimpleNamespace:
        return SimpleNamespace(Get=lambda: value)

    mesh = SimpleNamespace(
        GetPointsAttr=lambda: attribute(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        GetFaceVertexCountsAttr=lambda: attribute((3,)),
        GetFaceVertexIndicesAttr=lambda: attribute((0, 1, 2)),
        GetOrientationAttr=lambda: attribute("rightHanded"),
        GetHoleIndicesAttr=lambda: attribute(()),
        GetSubdivisionSchemeAttr=lambda: attribute("none"),
    )
    admission = object.__new__(_PlanningAdmission)
    admission._m = SimpleNamespace(UsdGeom=SimpleNamespace(XformCache=_Cache, Mesh=lambda _prim: mesh))
    admission._xform_cache = object()  # Any access to the shared cache must fail.
    admission._walk = lambda _carrier: (carrier, mesh_prim)
    monkeypatch.setattr(native_planning, "single_exact_convex_mesh", lambda *_args: mesh_prim)

    if resets:
        with pytest.raises(NativePlanningError, match="collision_cooking_failed"):
            admission._mesh_input(carrier)
        assert len(caches) == 1
    else:
        first = admission._mesh_input(carrier)
        second = admission._mesh_input(carrier)
        assert first.vertices == ((2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0))
        assert second == first
        assert len(caches) == 2
        assert caches[0] is not caches[1]


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


def test_triangle_mesh_canonicalization_skips_degenerate_faces_like_physx() -> None:
    assert _triangulate_faces(
        (0, 3, 2, 4, 0),
        (0, 1, 2, 3, 4, 0, 2, 3, 4),
        vertex_count=5,
        orientation="rightHanded",
    ) == (
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
    )


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

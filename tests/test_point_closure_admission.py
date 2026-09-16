from __future__ import annotations

from types import SimpleNamespace

import pytest
from unirobosim import (
    EntityKind,
    EntityPath,
    PlanningFrameDescriptor,
    PlanningFrameKind,
    PlanningJointType,
    PlanningSceneCatalog,
)

from unirobosim_isaaclab.native_planning import _PlanningAdmission
from unirobosim_isaaclab.native_protocols import NativePlanningError

Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
Gf = pytest.importorskip("pxr.Gf")
Sdf = pytest.importorskip("pxr.Sdf")


def _fixture(*, clones=1, scale=(1.0, 1.0, 1.0), meters=1.0):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    root = UsdGeom.Xform.Define(stage, "/World/env_0/Robot")
    root.AddScaleOp().Set(scale)
    for name in ("A", "B", "C"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"{root.GetPath()}/{name}").GetPrim())
    hinge = UsdPhysics.RevoluteJoint.Define(stage, f"{root.GetPath()}/hinge")
    hinge.CreateBody0Rel().SetTargets([f"{root.GetPath()}/A"])
    hinge.CreateBody1Rel().SetTargets([f"{root.GetPath()}/B"])
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{root.GetPath()}/bridge")
    fixed.CreateBody0Rel().SetTargets([f"{root.GetPath()}/B"])
    fixed.CreateBody1Rel().SetTargets([f"{root.GetPath()}/C"])
    anchor = UsdPhysics.FixedJoint.Define(stage, f"{root.GetPath()}/root_anchor")
    anchor.CreateBody0Rel().SetTargets([root.GetPath()])
    anchor.CreateBody1Rel().SetTargets([f"{root.GetPath()}/A"])
    closure = UsdPhysics.SphericalJoint.Define(stage, f"{root.GetPath()}/point")
    closure.CreateBody0Rel().SetTargets([f"{root.GetPath()}/A"])
    closure.CreateBody1Rel().SetTargets([f"{root.GetPath()}/C"])
    closure.CreateExcludeFromArticulationAttr().Set(True)
    closure.CreateLocalPos0Attr().Set((0.1, 0.2, 0.3))
    closure.CreateLocalPos1Attr().Set((-0.1, 0.0, 0.0))
    for environment in range(1, clones):
        clone = UsdGeom.Xform.Define(stage, f"/World/env_{environment}/Robot").GetPrim()
        clone.GetReferences().AddInternalReference(root.GetPath())
    spec = SimpleNamespace(
        path=EntityPath("/Robot"),
        kind=EntityKind.ARTICULATION,
        joint_names=("hinge",),
        metadata={},
        embedded_binding=None,
    )
    admission = object.__new__(_PlanningAdmission)
    admission._m = SimpleNamespace(Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, Gf=Gf)
    admission._stage = stage
    admission._world = SimpleNamespace(
        _articulations={spec.path: SimpleNamespace(joint_names=("hinge",))},
        _spec=SimpleNamespace(entities=(spec,), environments=SimpleNamespace(count=clones)),
    )
    admission._walk_cache = {}
    admission._xform_cache = UsdGeom.XformCache()
    admission._point_closures = []
    admission._point_closure_filtered_pairs = set()
    admission._accounted_constraints = set()
    admission._accounted_bodies = set()
    admission._accounted_colliders = set()
    admission._accounted_filtered_pair_sources = set()
    admission._frames = {}
    admission._geometries = {}
    admission._entity_root = lambda _spec, index: stage.GetPrimAtPath(f"/World/env_{index}/Robot")
    admission._articulation_collision_filter_projection = lambda *args: ({}, None, None, 0, 0, True)
    return stage, admission, spec, closure, anchor


def test_excluded_sphere_is_separate_from_tree_and_exports_exact_nonfixed_coupling():
    _, admission, spec, _, _ = _fixture()
    entity, links, joints, frames, geometry, _ = admission._entity_catalog(spec)
    assert len(joints) == 2
    assert {item.joint_type for item in joints} == {PlanningJointType.REVOLUTE, PlanningJointType.FIXED}
    (closure,) = admission._point_closures
    assert closure.coupled_joint_ids == tuple(
        item.joint_id for item in joints if item.joint_type is PlanningJointType.REVOLUTE
    )
    assert closure.link_a_id != closure.link_b_id
    assert closure.anchor_a_m == pytest.approx((0.1, 0.2, 0.3))
    assert len(closure.constraint_sha256) == len(closure.content_sha256) == 64
    catalog = PlanningSceneCatalog.build(
        "isaac.test",
        "world",
        1,
        0,
        1,
        1,
        (entity,),
        links,
        tuple(sorted(joints, key=lambda item: item.joint_id)),
        tuple(
            sorted(
                (*frames, PlanningFrameDescriptor("frame.world", PlanningFrameKind.WORLD, None, None, None)),
                key=lambda item: item.frame_id,
            )
        ),
        geometry,
        point_closures=(closure,),
    )
    assert catalog.schema_version == "unirobosim.planning-scene/v3"
    assert len(catalog.point_closures) == 1
    assert len(admission._point_closure_filtered_pairs) == 1


def test_point_anchors_preserve_scale_and_stage_si_units():
    _, admission, spec, _, _ = _fixture(scale=(2.0, 3.0, 4.0), meters=0.01)
    admission._entity_catalog(spec)
    (closure,) = admission._point_closures
    assert closure.anchor_a_m == pytest.approx((0.002, 0.006, 0.012))
    assert closure.anchor_b_m == pytest.approx((-0.002, 0.0, 0.0))


@pytest.mark.parametrize(
    "change",
    [
        "disabled",
        "not-excluded",
        "revolute",
        "fixed",
        "limit",
        "break-force",
        "break-torque",
        "drive",
        "mimic",
        "animated",
        "same-body",
        "world",
        "outside",
        "nonrigid-endpoint",
    ],
)
def test_unsupported_constraints_fail_explicitly(change):
    stage, admission, spec, closure, _ = _fixture()
    prim = closure.GetPrim()
    if change == "disabled":
        closure.CreateJointEnabledAttr().Set(False)
    elif change == "not-excluded":
        closure.CreateExcludeFromArticulationAttr().Set(False)
    elif change in {"revolute", "fixed"}:
        prim.SetTypeName("PhysicsRevoluteJoint" if change == "revolute" else "PhysicsFixedJoint")
    elif change == "limit":
        closure.CreateConeAngle0LimitAttr().Set(0.0)
    elif change == "break-force":
        closure.CreateBreakForceAttr().Set(1.0)
    elif change == "break-torque":
        closure.CreateBreakTorqueAttr().Set(1.0)
    elif change in {"drive", "mimic"}:
        prim.CreateAttribute(f"test:{change}:stiffness", Sdf.ValueTypeNames.Float).Set(1.0)
    elif change == "animated":
        closure.CreateLocalPos0Attr().Set((0.0, 0.0, 0.0), 1.0)
    elif change == "same-body":
        closure.CreateBody1Rel().SetTargets(closure.GetBody0Rel().GetTargets())
    elif change == "world":
        closure.CreateBody1Rel().ClearTargets(True)
    elif change == "outside":
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Other/body").GetPrim())
        closure.CreateBody1Rel().SetTargets(["/Other/body"])
    else:
        closure.CreateBody1Rel().SetTargets(["/World/env_0/Robot"])
    with pytest.raises(NativePlanningError) as error:
        admission._entity_catalog(spec)
    assert error.value.code == "constraint_unsupported"


@pytest.mark.parametrize("anchor_kind", ["outer", "world", "reverse-world", "reverse-outer"])
def test_root_anchor_accepts_world_or_outer_frame_in_either_order(anchor_kind):
    _, admission, spec, _, anchor = _fixture()
    if "world" in anchor_kind:
        anchor.CreateBody0Rel().ClearTargets(True)
    if anchor_kind.startswith("reverse"):
        first, second = anchor.GetBody0Rel().GetTargets(), anchor.GetBody1Rel().GetTargets()
        anchor.CreateBody0Rel().SetTargets(second)
        anchor.CreateBody1Rel().SetTargets(first)
    result = admission._entity_catalog(spec)
    assert result[-1].root_link_name == "A"
    assert len(result[2]) == 2


@pytest.mark.parametrize("change", ["multiple", "nonroot", "inner-frame", "disabled"])
def test_fixed_anchor_is_unique_enabled_and_must_target_actual_root(change):
    stage, admission, spec, _, anchor = _fixture()
    if change == "multiple":
        extra = UsdPhysics.FixedJoint.Define(stage, "/World/env_0/Robot/root_anchor2")
        extra.CreateBody1Rel().SetTargets(["/World/env_0/Robot/A"])
    elif change == "nonroot":
        anchor.CreateBody1Rel().SetTargets(["/World/env_0/Robot/B"])
    elif change == "inner-frame":
        UsdGeom.Xform.Define(stage, "/World/env_0/Robot/Inner")
        anchor.CreateBody0Rel().SetTargets(["/World/env_0/Robot/Inner"])
    else:
        anchor.CreateJointEnabledAttr().Set(False)
    with pytest.raises(NativePlanningError) as error:
        admission._entity_catalog(spec)
    assert error.value.code == "constraint_unsupported"


@pytest.mark.parametrize("change", [None, "type", "endpoint", "anchor", "enabled", "collision", "limit"])
def test_clones_compare_constraint_type_endpoints_anchors_and_properties(change):
    stage, admission, spec, _, _ = _fixture(clones=2)
    admission._entity_catalog(spec)
    clone = UsdPhysics.SphericalJoint(stage.GetPrimAtPath("/World/env_1/Robot/point"))
    if change == "type":
        clone.GetPrim().SetTypeName("PhysicsFixedJoint")
    elif change == "endpoint":
        clone.CreateBody1Rel().SetTargets(["/World/env_1/Robot/B"])
    elif change == "anchor":
        clone.CreateLocalPos0Attr().Set((0.2, 0.0, 0.0))
    elif change == "enabled":
        clone.CreateJointEnabledAttr().Set(False)
    elif change == "collision":
        clone.CreateCollisionEnabledAttr().Set(True)
    elif change == "limit":
        clone.CreateConeAngle1LimitAttr().Set(30.0)
    if change is None:
        admission._verify_cloned_environments()
    else:
        with pytest.raises(NativePlanningError) as error:
            admission._verify_cloned_environments()
        assert error.value.code == "constraint_unsupported"


def test_clone_parent_scale_cannot_change_effective_anchors_without_colliders():
    stage, admission, spec, _, _ = _fixture(clones=2)
    admission._entity_catalog(spec)
    root = UsdGeom.Xform(stage.GetPrimAtPath("/World/env_1/Robot"))
    root.GetOrderedXformOps()[0].Set((2.0, 2.0, 2.0))
    with pytest.raises(NativePlanningError) as error:
        admission._verify_cloned_environments()
    assert error.value.code == "constraint_unsupported"


@pytest.mark.parametrize(
    "schema,accepted",
    [
        ("NewtonCollisionAPI", True),
        ("NewtonMeshCollisionAPI", True),
        ("NewtonOtherCollisionAPI", False),
        ("NewtonUnknownAPI", False),
        ("OtherCollisionAPI", False),
    ],
)
def test_only_two_inert_newton_markers_are_allowed_even_when_unregistered(schema, accepted):
    stage, admission, _, _, _ = _fixture()
    mesh = UsdGeom.Mesh.Define(stage, "/World/env_0/Robot/A/collision").GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh)
    collision = UsdPhysics.MeshCollisionAPI.Apply(mesh)
    collision.CreateApproximationAttr().Set("convexHull")
    mesh.AddAppliedSchema(schema)
    admission._m.PhysxSchema = SimpleNamespace(PhysxCollisionAPI=lambda _: None)
    before = stage.GetRootLayer().ExportToString()
    if accepted:
        admission._validate_collision_common(mesh)
        assert collision.GetApproximationAttr().Get() == "convexHull"
    else:
        with pytest.raises(NativePlanningError) as error:
            admission._validate_collision_common(mesh)
        assert error.value.code == "collision_geometry_unsupported"
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("collision_enabled", [False, True])
def test_closure_collision_suppression_is_scoped_to_its_two_endpoints(collision_enabled):
    _, admission, spec, closure, _ = _fixture()
    closure.CreateCollisionEnabledAttr().Set(collision_enabled)
    admission._entity_catalog(spec)
    root = admission._entity_root(spec, 0)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    admission._m.PhysxSchema = SimpleNamespace(
        PhysxArticulationAPI=lambda _: SimpleNamespace(
            GetEnabledSelfCollisionsAttr=lambda: SimpleNamespace(Get=lambda: True)
        )
    )
    admission._next_collision_filter_bit = 1
    bodies = {str(prim.GetPath()): prim for prim in admission._walk(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    names = {path: prim.GetName() for path, prim in bodies.items()}
    encoding, *_ = _PlanningAdmission._articulation_collision_filter_projection(
        admission, root, bodies, names, tuple(bodies)
    )
    by_name = {name: encoding.get(path, (1, 2**32 - 1)) for path, name in names.items()}

    def collides(left, right):
        group_a, mask_a = by_name[left]
        group_b, mask_b = by_name[right]
        return bool((group_a & mask_b) and (group_b & mask_a))

    assert collides("A", "C") is collision_enabled
    assert collides("A", "B")
    assert collides("B", "C")

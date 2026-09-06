"""Real USD contracts, optional when the lightweight test environment has no pxr.

Composition cases preserve scale/reset contracts. Same-Prim cases bypass the
OpenUSD 25.11 uninitialized reset output without depending on its undefined value.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from unirobosim import PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION, EntityPath, FrozenMap, PlanningGeometryMotionClass

from unirobosim_isaaclab import native_planning
from unirobosim_isaaclab.native_planning import (
    _effective_collision_relative_transform,
    _PlanningAdmission,
)
from unirobosim_isaaclab.native_protocols import NativePlanningError


@pytest.fixture(scope="module", autouse=True)
def usd_modules() -> None:
    # Do not load native SDKs during collection of lightweight import tests.
    global Usd, UsdGeom, UsdPhysics, Gf, MODULES
    Usd = pytest.importorskip("pxr.Usd", reason="real USD transform tests require pxr")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    Gf = pytest.importorskip("pxr.Gf")
    MODULES = SimpleNamespace(Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, Gf=Gf)


def _composed_stage(*, instanceable: bool, owner_reset: bool) -> tuple[object, object]:
    source = Usd.Stage.CreateInMemory()
    asset = UsdGeom.Xform.Define(source, "/Asset")
    source.SetDefaultPrim(asset.GetPrim())
    asset.AddTranslateOp().Set((100.0, 0.0, 0.0))
    owner = UsdGeom.Xform.Define(source, "/Asset/Body")
    owner.AddTranslateOp().Set((10.0, 0.0, 0.0))
    owner.AddScaleOp().Set((2.0, 3.0, 4.0))
    owner.SetResetXformStack(owner_reset)
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    carrier = UsdGeom.Xform.Define(source, "/Asset/Body/Carrier")
    carrier.AddTranslateOp().Set((1.0, 2.0, 3.0))
    UsdPhysics.CollisionAPI.Apply(carrier.GetPrim())
    mesh = UsdGeom.Mesh.Define(source, "/Asset/Body/Carrier/Mesh")
    mesh.AddTranslateOp().Set((1.0, 1.0, 1.0))
    mesh.CreatePointsAttr(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr("none")
    cube = UsdGeom.Cube.Define(source, "/Asset/Body/Cube")
    cube.AddTranslateOp().Set((2.0, 3.0, 4.0))
    cube.CreateSizeAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    composed = Usd.Stage.CreateInMemory()
    for name in ("reference", "clone"):
        prim = composed.DefinePrim(f"/{name}")
        prim.GetReferences().AddReference(source.GetRootLayer().identifier)
        prim.SetInstanceable(instanceable)
    return source, composed


@pytest.mark.parametrize("instanceable", (False, True), ids=("reference", "instance-proxy"))
@pytest.mark.parametrize("owner_reset", (False, True), ids=("owner-inherits", "owner-resets"))
def test_composed_owner_relative_queries_preserve_scale_and_mesh_coordinates(
    instanceable: bool, owner_reset: bool
) -> None:
    source, stage = _composed_stage(instanceable=instanceable, owner_reset=owner_reset)
    assert source
    shared = UsdGeom.XformCache()
    admission = object.__new__(_PlanningAdmission)
    admission._m = MODULES
    admission._xform_cache = shared
    admission._walk = lambda prim: tuple(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    admission._validate_collision_common = lambda prim: None
    admission._source_sha256 = lambda spec: "a" * 64
    admission._prim_signature = lambda prim, root: prim.GetName()

    signatures = []
    # Interleave world, owner/collider and mesh/carrier queries, then repeat in
    # reverse order. Owner reset must not reject valid descendants.
    for name in ("reference", "clone", "clone", "reference"):
        owner = stage.GetPrimAtPath(f"/{name}/Body")
        carrier = stage.GetPrimAtPath(f"/{name}/Body/Carrier")
        mesh = stage.GetPrimAtPath(f"/{name}/Body/Carrier/Mesh")
        cube = stage.GetPrimAtPath(f"/{name}/Body/Cube")
        assert mesh.IsInstanceProxy() is instanceable
        shared.GetLocalToWorldTransform(mesh)
        shared.GetLocalToWorldTransform(owner)
        relative, resets = shared.ComputeRelativeTransform(mesh, owner)
        fresh_relative, fresh_resets = UsdGeom.XformCache().ComputeRelativeTransform(mesh, owner)
        assert not resets and not fresh_resets
        assert Gf.IsClose(relative, fresh_relative, 1.0e-10)

        effective = _effective_collision_relative_transform(MODULES, mesh, owner)
        assert tuple(effective.Transform(Gf.Vec3d(0.0))) == pytest.approx((4.0, 9.0, 16.0))
        assert tuple(effective.TransformDir(Gf.Vec3d(1.0, 1.0, 1.0))) == pytest.approx((2.0, 3.0, 4.0))
        mesh_input = admission._mesh_input(carrier)
        assert mesh_input.vertices == ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0))

        (geometry,) = admission._collision_geometry(
            SimpleNamespace(path=EntityPath("/asset")),
            "entity.asset",
            cube,
            owner,
            None,
            "frame.asset",
            motion=PlanningGeometryMotionClass.STATIC,
        )
        assert geometry.parent_frame_T_geometry.position_m == pytest.approx((4.0, 9.0, 16.0))
        assert geometry.scale == pytest.approx((2.0, 3.0, 4.0))
        signatures.append(
            admission._collision_clone_signature(cube, owner, f"/{name}", motion=PlanningGeometryMotionClass.STATIC)
        )
    assert all(signature == signatures[0] for signature in signatures)


def test_composed_collision_reset_between_owner_and_carrier_is_rejected() -> None:
    source, stage = _composed_stage(instanceable=True, owner_reset=True)
    UsdGeom.Xformable(source.GetPrimAtPath("/Asset/Body/Carrier")).SetResetXformStack(True)
    owner = stage.GetPrimAtPath("/reference/Body")
    carrier = stage.GetPrimAtPath("/reference/Body/Carrier")
    with pytest.raises(NativePlanningError, match="collision_geometry_unsupported"):
        _effective_collision_relative_transform(MODULES, carrier, owner)


def test_composed_mesh_reset_below_carrier_is_rejected() -> None:
    source, stage = _composed_stage(instanceable=True, owner_reset=True)
    UsdGeom.Xformable(source.GetPrimAtPath("/Asset/Body/Carrier/Mesh")).SetResetXformStack(True)
    admission = object.__new__(_PlanningAdmission)
    admission._m = MODULES
    admission._walk = lambda prim: tuple(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    with pytest.raises(NativePlanningError, match="collision_cooking_failed"):
        admission._mesh_input(stage.GetPrimAtPath("/reference/Body/Carrier"))


@pytest.fixture
def guarded_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make bypass verification deterministic even on builds that happen to return False."""
    native_cache = UsdGeom.XformCache

    class GuardedCache:
        def __init__(self) -> None:
            self.native = native_cache()

        def ComputeRelativeTransform(self, prim: object, ancestor: object) -> tuple[object, bool]:
            assert prim != ancestor, "same-Prim calls must not reach the unsafe USD wrapper"
            return self.native.ComputeRelativeTransform(prim, ancestor)

        def GetLocalToWorldTransform(self, prim: object) -> object:
            return self.native.GetLocalToWorldTransform(prim)

    monkeypatch.setattr(UsdGeom, "XformCache", GuardedCache)


@pytest.mark.parametrize("owner_reset", (False, True))
def test_self_relative_after_real_reset_is_identity_without_reset(owner_reset: bool) -> None:
    stage = Usd.Stage.CreateInMemory()
    owner = UsdGeom.Xform.Define(stage, "/Body")
    owner.AddScaleOp().Set((2.0, 3.0, 4.0))
    owner.SetResetXformStack(owner_reset)
    child = UsdGeom.Xform.Define(stage, "/Body/Reset")
    child.AddTranslateOp().Set((1.0, 2.0, 3.0))
    child.SetResetXformStack(True)
    shared = UsdGeom.XformCache()
    for _ in range(100):
        _, resets = shared.ComputeRelativeTransform(child.GetPrim(), owner.GetPrim())
        assert resets
        for cache in (shared, UsdGeom.XformCache()):
            # A separately retrieved handle still denotes the same USD Prim.
            matrix, resets = native_planning._compute_relative_transform(
                MODULES, cache, owner.GetPrim(), stage.GetPrimAtPath("/Body")
            )
            assert matrix == Gf.Matrix4d(1.0)
            assert resets is False


def test_equal_paths_in_different_stages_are_not_same_prim() -> None:
    stages = [Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()]
    prims = [UsdGeom.Xform.Define(stage, "/Body") for stage in stages]
    prims[0].AddTranslateOp().Set((1.0, 2.0, 3.0))
    matrix, resets = native_planning._compute_relative_transform(
        MODULES, UsdGeom.XformCache(), prims[0].GetPrim(), prims[1].GetPrim()
    )
    assert tuple(matrix.ExtractTranslation()) == (1.0, 2.0, 3.0)
    assert not resets


@pytest.mark.usefixtures("guarded_caches")
@pytest.mark.parametrize("instanceable", (False, True))
@pytest.mark.parametrize("owner_reset", (False, True))
def test_same_prim_collision_and_clone_keep_body_and_ancestor_scale(instanceable: bool, owner_reset: bool) -> None:
    source, stage = _composed_stage(instanceable=instanceable, owner_reset=owner_reset)
    UsdGeom.Xformable(source.GetPrimAtPath("/Asset")).AddScaleOp().Set((0.5, 2.0, 3.0))
    body = UsdGeom.Cube.Define(source, "/Asset/Body")
    body.CreateSizeAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    admission = object.__new__(_PlanningAdmission)
    admission._m = MODULES
    admission._validate_collision_common = lambda prim: None
    admission._source_sha256 = lambda spec: "a" * 64
    admission._prim_signature = lambda prim, root: prim.GetName()
    expected_scale = (2.0, 3.0, 4.0) if owner_reset else (1.0, 6.0, 12.0)
    signatures = []
    for name in ("reference", "clone"):
        prim = stage.GetPrimAtPath(f"/{name}/Body")
        assert prim.IsInstanceProxy() is instanceable
        (geometry,) = admission._collision_geometry(
            SimpleNamespace(path=EntityPath("/asset")),
            "entity.asset",
            prim,
            stage.GetPrimAtPath(f"/{name}/Body"),
            None,
            "frame.asset",
            motion=PlanningGeometryMotionClass.STATIC,
        )
        assert geometry.parent_frame_T_geometry.position_m == pytest.approx((0.0, 0.0, 0.0))
        assert geometry.scale == pytest.approx(expected_scale)
        signatures.append(
            admission._collision_clone_signature(prim, prim, f"/{name}", motion=PlanningGeometryMotionClass.STATIC)
        )
    assert signatures[0] == signatures[1]


@pytest.mark.usefixtures("guarded_caches")
@pytest.mark.parametrize("instanceable", (False, True))
def test_mesh_as_its_own_carrier_keeps_original_vertices(instanceable: bool) -> None:
    source, stage = _composed_stage(instanceable=instanceable, owner_reset=True)
    assert source
    admission = object.__new__(_PlanningAdmission)
    admission._m = MODULES
    admission._walk = lambda prim: tuple(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    for name in ("reference", "clone"):
        mesh = stage.GetPrimAtPath(f"/{name}/Body/Carrier/Mesh")
        mesh_input = admission._mesh_input(mesh)
        assert mesh_input.vertices == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert mesh_input.triangles == ((0, 1, 2),)


@pytest.mark.usefixtures("guarded_caches")
@pytest.mark.parametrize("instanceable", (False, True))
@pytest.mark.parametrize("explicit_owner", (False, True))
@pytest.mark.parametrize("source_name", ("Body", "Carrier"), ids=("self-admitted", "child-reset-rejected"))
def test_native_named_frames_and_clones_handle_self_but_reject_intermediate_reset(
    instanceable: bool, explicit_owner: bool, source_name: str
) -> None:
    source, stage = _composed_stage(instanceable=instanceable, owner_reset=True)
    UsdGeom.Xformable(source.GetPrimAtPath("/Asset/Body/Carrier")).SetResetXformStack(True)
    entry = {
        "name": "native_frame",
        "owner_link": "Body" if explicit_owner else None,
        "source": {"kind": "native_named", "name": source_name},
    }
    spec = SimpleNamespace(
        path=EntityPath("/asset"),
        metadata=FrozenMap(
            {
                "planning_frame_declarations": {
                    "schema": PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION,
                    "component_sha256": "a" * 64,
                    "entries": (entry,),
                }
            }
        ),
    )
    admission = object.__new__(_PlanningAdmission)
    admission._m = MODULES
    admission._xform_cache = UsdGeom.XformCache()
    admission._walk = lambda prim: tuple(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
    admission._source_sha256 = lambda spec: "a" * 64
    admission._prim_signature = lambda prim, root: prim.GetName()
    admission._frames = {}
    admission._entities = {spec.path: SimpleNamespace(root_link_name="Body")}
    descriptors = []
    reference, clone = stage.GetPrimAtPath("/reference"), stage.GetPrimAtPath("/clone")

    def declare() -> None:
        admission._declared_frames(
            spec,
            reference,
            "entity.asset",
            "frame.asset",
            {"Body": "link.body"},
            {"Body": "frame.body"},
            [],
            descriptors,
            "Body",
        )

    if source_name == "Carrier":
        with pytest.raises(NativePlanningError, match="frame_ambiguous"):
            declare()
        with pytest.raises(NativePlanningError, match="frame_ambiguous"):
            admission._verify_declared_clone_frames(spec, reference, clone)
    else:
        declare()
        assert len(descriptors) == 1
        (binding,) = admission._frames.values()
        assert binding.local_pose.position_m == (0.0, 0.0, 0.0)
        assert binding.local_pose.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
        admission._verify_declared_clone_frames(spec, reference, clone)

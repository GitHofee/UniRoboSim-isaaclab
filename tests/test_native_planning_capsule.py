import subprocess
import sys
from types import SimpleNamespace

import pytest
from unirobosim import EntityPath, PlanningGeometryMotionClass

from unirobosim_isaaclab.native_planning import _PlanningAdmission, _rotate


@pytest.mark.parametrize("axis, expected", [("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))])
@pytest.mark.parametrize("rotated", [False, True])
def test_authored_capsule_reaches_planning_without_replacing_native_collider(axis, expected, rotated):
    code = (
        "from tests.test_native_planning_capsule import _check_capsule; "
        f"_check_capsule({axis!r}, {expected!r}, {rotated!r})"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _check_capsule(axis, expected, rotated):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    owner = UsdGeom.Xform.Define(stage, "/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(owner)
    capsule = UsdGeom.Capsule.Define(stage, "/body/collision")
    capsule.CreateRadiusAttr(0.00165)
    capsule.CreateHeightAttr(0.009998)
    capsule.CreateAxisAttr(axis)
    if rotated:
        capsule.AddOrientOp().Set(Gf.Quatf(0.7071067690849304, Gf.Vec3f(0.7059485912322998, -0.04045521840453148, 0.0)))
        matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(capsule.GetPrim(), owner)
        expected = tuple(matrix.TransformDir(Gf.Vec3d(*expected)).GetNormalized())
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    admission = object.__new__(_PlanningAdmission)
    admission._contact_prim_paths = {}
    admission._m = SimpleNamespace(
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        PhysxSchema=SimpleNamespace(PhysxCollisionAPI=lambda _prim: None),
        Gf=Gf,
    )
    admission._stage = stage
    admission._accounted_filtered_pair_sources = set()
    admission._source_sha256 = lambda _spec: "a" * 64
    spec = SimpleNamespace(path=EntityPath("/body"))
    before = stage.GetRootLayer().ExportToString()
    (geometry,) = admission._collision_geometry(
        spec, "body", capsule.GetPrim(), owner, "link", "frame", motion=PlanningGeometryMotionClass.DYNAMIC
    )
    assert geometry.representation.value == "capsule"
    assert geometry.inline.dimensions_m == pytest.approx((0.00165, 0.009998))
    assert geometry.scale[0] == geometry.scale[1]
    assert _rotate((0.0, 0.0, 1.0), geometry.parent_frame_T_geometry.orientation_xyzw) == pytest.approx(expected)
    assert stage.GetRootLayer().ExportToString() == before


def test_capsule_actual_elliptical_scale_still_fails_closed():
    code = "from tests.test_native_planning_capsule import _check_elliptical; _check_elliptical()"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _check_elliptical():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from unirobosim_isaaclab.native_protocols import NativePlanningError

    stage = Usd.Stage.CreateInMemory()
    owner = UsdGeom.Xform.Define(stage, "/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(owner)
    capsule = UsdGeom.Capsule.Define(stage, "/body/collision")
    capsule.AddScaleOp().Set((1.0, 1.01, 1.0))
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    admission = object.__new__(_PlanningAdmission)
    admission._contact_prim_paths = {}
    admission._m = SimpleNamespace(
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        PhysxSchema=SimpleNamespace(PhysxCollisionAPI=lambda _prim: None),
        Gf=Gf,
    )
    admission._stage = stage
    admission._accounted_filtered_pair_sources = set()
    admission._source_sha256 = lambda _spec: "a" * 64
    with pytest.raises(NativePlanningError, match="collision_geometry_unsupported"):
        admission._collision_geometry(
            SimpleNamespace(path=EntityPath("/body")),
            "body",
            capsule.GetPrim(),
            owner,
            "link",
            "frame",
            motion=PlanningGeometryMotionClass.DYNAMIC,
        )

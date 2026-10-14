import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from unirobosim_isaaclab.primary_ray_partition import PARTITION_SETTING, PrimaryRayPartitionScope


class Settings:
    def get(self, key):
        return key == PARTITION_SETTING


def fixture():
    s = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(s, "/World/link")
    parent.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    for name in ("left", "right"):
        mesh = UsdGeom.Mesh.Define(s, "/World/link/" + name)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mat = UsdShade.Material.Define(s, "/World/material_" + name)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    cams = {}
    for name, hidden in [("head", ()), ("left", ("/World/link/left",)), ("right", ("/World/link/right",))]:
        path = "/World/cam_" + name
        UsdGeom.Camera.Define(s, path)
        cams[path] = hidden
    return s, cams, parent


def scope(s, cams):
    return PrimaryRayPartitionScope(
        s, cams, usd=Usd, sdf=Sdf, usd_geom=UsdGeom, usd_physics=UsdPhysics, usd_shade=UsdShade, settings=Settings()
    )


def test_policy_motion_material_and_cleanup():
    s, cams, parent = fixture()
    baseline = s.GetRootLayer().ExportToString()
    session = s.GetSessionLayer().ExportToString()
    obj = scope(s, cams)
    assert len(obj.receipt["copies"]) == 4
    for item in obj.receipt["copies"]:
        cp = s.GetPrimAtPath(item["copy"])
        orig = s.GetPrimAtPath(item["source"])
        assert cp.GetAttribute("primvars:hideForCamera").Get() is False
        assert cp.GetAttribute("primvars:invisibleToSecondaryRays").Get() is True
        assert orig.GetAttribute("primvars:invisibleToSecondaryRays").Get() is None
        assert (
            UsdShade.MaterialBindingAPI(cp).ComputeBoundMaterial()[0].GetPath()
            == UsdShade.MaterialBindingAPI(orig).ComputeBoundMaterial()[0].GetPath()
        )
    parent.GetOrderedXformOps()[0].Set(Gf.Vec3d(4, 5, 6))
    xc = UsdGeom.XformCache()
    for item in obj.receipt["copies"]:
        assert xc.GetLocalToWorldTransform(s.GetPrimAtPath(item["source"])) == xc.GetLocalToWorldTransform(
            s.GetPrimAtPath(item["copy"])
        )
    parent.GetOrderedXformOps()[0].Set(Gf.Vec3d(1, 2, 3))
    frozen = obj.layer.ExportToString()
    for _ in range(120):
        for path in cams:
            assert s.GetPrimAtPath(path).GetAttribute("omni:scenePartition").Get()
    assert obj.layer.ExportToString() == frozen
    obj.close()
    obj.close()
    assert s.GetRootLayer().ExportToString() == baseline
    assert s.GetSessionLayer().ExportToString() == session
    assert not s.GetPrimAtPath(obj.receipt["copies"][0]["copy"])


def test_collision_rejected_without_mutation():
    s, cams, _ = fixture()
    UsdPhysics.CollisionAPI.Apply(s.GetPrimAtPath("/World/link/left"))
    before = s.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError, match="visual only"):
        scope(s, cams)
    assert s.GetSessionLayer().ExportToString() == before


def test_preexisting_partition_rejected():
    s, cams, _ = fixture()
    s.GetPrimAtPath("/World/cam_head").CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("existing")
    with pytest.raises(ValueError, match="already has"):
        scope(s, cams)


def test_stronger_opinion_failure_removes_owned_layer():
    s, cams, _ = fixture()
    with Usd.EditContext(s, s.GetSessionLayer()):
        UsdGeom.PrimvarsAPI(s.GetPrimAtPath("/World/link/left")).CreatePrimvar(
            "hideForCamera", Sdf.ValueTypeNames.Bool
        ).Set(False)
    before = s.GetSessionLayer().ExportToString()
    with pytest.raises(RuntimeError, match="stronger opinion"):
        scope(s, cams)
    assert s.GetSessionLayer().ExportToString() == before


def test_instance_proxy_layer_is_reversible():
    s, cams, _ = fixture()
    UsdGeom.Xform.Define(s, "/World/template")
    mesh = UsdGeom.Mesh.Define(s, "/World/template/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mat = UsdShade.Material.Define(s, "/World/material_instance")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    ref = s.DefinePrim("/World/instance")
    ref.GetReferences().AddInternalReference("/World/template")
    ref.SetInstanceable(True)
    target = "/World/instance/mesh"
    assert s.GetPrimAtPath(target).IsInstanceProxy()
    cams["/World/cam_left"] = (target,)
    cams["/World/cam_right"] = ()
    before = s.GetRootLayer().ExportToString()
    obj = scope(s, cams)
    assert not s.GetPrimAtPath(target).IsInstanceProxy()
    assert len(obj.receipt["copies"]) == 2
    obj.close()
    assert s.GetPrimAtPath(target).IsInstanceProxy()
    assert s.GetRootLayer().ExportToString() == before

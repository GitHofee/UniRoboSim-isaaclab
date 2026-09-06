"""Real USD affine clone math, separate from native cooking/schema acceptance."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unirobosim_isaaclab.native_planning import (
    _collision_pose_scale_bake,
    _effective_collision_relative_transform,
)
from unirobosim_isaaclab.native_protocols import NativePlanningError


@pytest.fixture(scope="module")
def usd():
    return SimpleNamespace(
        Gf=pytest.importorskip("pxr.Gf"),
        Usd=pytest.importorskip("pxr.Usd"),
        UsdGeom=pytest.importorskip("pxr.UsdGeom"),
    )


def _stage(usd, *, shift, child, scale, owner_reset=False, instanceable=False):
    source = usd.Usd.Stage.CreateInMemory()
    asset = usd.UsdGeom.Xform.Define(source, "/Asset")
    source.SetDefaultPrim(asset.GetPrim())
    ancestor = usd.UsdGeom.Xform.Define(source, "/Asset/Ancestor")
    ancestor.AddScaleOp().Set(scale)
    owner = usd.UsdGeom.Mesh.Define(source, "/Asset/Ancestor/Body")
    owner.AddTranslateOp().Set((0.2, -0.1, 0.3))
    owner.AddRotateYOp().Set(17.0)
    owner.SetResetXformStack(owner_reset)
    if child:
        shape = usd.UsdGeom.Mesh.Define(source, "/Asset/Ancestor/Body/Shape")
        shape.AddTranslateOp().Set((0.1, 0.2, 0.3))
        shape.AddRotateZOp().Set(11.0)
    composed = usd.Usd.Stage.CreateInMemory()
    for name, offset in (("reference", 0.0), ("clone", shift)):
        root = usd.UsdGeom.Xform.Define(composed, f"/{name}")
        root.GetPrim().GetReferences().AddReference(source.GetRootLayer().identifier)
        root.AddTranslateOp().Set((offset, 0.0, 10.0))
        root.AddRotateZOp().Set(34.37746770784939)
        root.GetPrim().SetInstanceable(instanceable)
    owners = [composed.GetPrimAtPath(f"/{name}/Ancestor/Body") for name in ("reference", "clone")]
    carriers = [composed.GetPrimAtPath(str(owner.GetPath()) + "/Shape") for owner in owners] if child else owners
    return source, composed, owners, carriers


@pytest.mark.parametrize("shift", (0.0, 20.0, 1e3, 1e5))
@pytest.mark.parametrize("child", (False, True), ids=("self", "child"))
@pytest.mark.parametrize(
    "scale", ((1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (-2.0, 3.0, 4.0)), ids=("unit", "nonuniform", "authored-mirror")
)
@pytest.mark.parametrize("owner_reset", (False, True))
@pytest.mark.parametrize("instanceable", (False, True), ids=("reference", "instance-proxy"))
def test_equal_affine_clone_signatures_ignore_only_global_translation(
    usd, shift, child, scale, owner_reset, instanceable
):
    source, stage, owners, carriers = _stage(
        usd, shift=shift, child=child, scale=scale, owner_reset=owner_reset, instanceable=instanceable
    )
    assert source and stage
    assert all(prim.IsInstanceProxy() == instanceable for prim in (*owners, *carriers))
    matrices = [
        _effective_collision_relative_transform(usd, carrier, owner)
        for carrier, owner in zip(carriers, owners, strict=True)
    ]
    # Exact equality is intentional: the production clone guard is strict.
    signatures = [_collision_pose_scale_bake(usd, matrix, mesh_capable=True) for matrix in matrices]
    assert signatures[0] == signatures[1]
    # Independently retain the old world-space mathematical definition within
    # double-precision cancellation error. No physical scale/shear is dropped.
    cache = usd.UsdGeom.XformCache()
    for matrix, carrier, owner in zip(matrices, carriers, owners, strict=True):
        expected = (
            cache.GetLocalToWorldTransform(carrier)
            * cache.GetLocalToWorldTransform(owner).RemoveScaleShear().GetInverse()
        )
        assert usd.Gf.IsClose(matrix, expected, 1e-9)


@pytest.mark.parametrize("change", ("translation", "scale", "ancestor_scale"))
def test_real_clone_affine_differences_remain_distinct(usd, change):
    source, stage, owners, carriers = _stage(usd, shift=20.0, child=True, scale=(2.0, 3.0, 4.0))
    assert source
    if change == "translation":
        stage.GetPrimAtPath("/clone/Ancestor/Body/Shape").GetAttribute("xformOp:translate").Set((0.12, 0.2, 0.3))
    elif change == "scale":
        usd.UsdGeom.Xformable(carriers[1]).AddScaleOp().Set((1.1, 1.0, 1.0))
    else:
        stage.GetPrimAtPath("/clone/Ancestor").GetAttribute("xformOp:scale").Set((2.1, 3.0, 4.0))
    signatures = [
        _collision_pose_scale_bake(usd, _effective_collision_relative_transform(usd, carrier, owner), mesh_capable=True)
        for carrier, owner in zip(carriers, owners, strict=True)
    ]
    assert signatures[0] != signatures[1]


def test_real_intermediate_reset_still_rejects(usd):
    source, stage, owners, carriers = _stage(usd, shift=20.0, child=True, scale=(2.0, 3.0, 4.0))
    assert source and stage
    usd.UsdGeom.Xformable(carriers[1]).SetResetXformStack(True)
    with pytest.raises(NativePlanningError, match="collision_geometry_unsupported"):
        _effective_collision_relative_transform(usd, carriers[1], owners[1])


@pytest.mark.parametrize("child", (False, True), ids=("self", "child"))
def test_translation_removal_does_not_mutate_cache_matrix_or_stage(usd, child):
    source, stage, owners, carriers = _stage(usd, shift=1e5, child=child, scale=(2.0, 3.0, 4.0))
    before = (source.GetRootLayer().ExportToString(), stage.GetRootLayer().ExportToString())
    real_cache = usd.UsdGeom.XformCache()
    cached_owner_matrix = real_cache.GetLocalToWorldTransform(owners[1])
    matrix_before = usd.Gf.Matrix4d(cached_owner_matrix)
    assert cached_owner_matrix.ExtractTranslation().GetLength() > 1e4
    # Return the same real matrix object, so mutating the object returned by the
    # cache is observable even if a particular USD binding normally copies it.
    cache = SimpleNamespace(
        ComputeRelativeTransform=real_cache.ComputeRelativeTransform,
        GetLocalToWorldTransform=lambda prim: (
            cached_owner_matrix if prim == owners[1] else real_cache.GetLocalToWorldTransform(prim)
        ),
    )
    modules = SimpleNamespace(Gf=usd.Gf, UsdGeom=SimpleNamespace(XformCache=lambda: cache))
    first = _effective_collision_relative_transform(modules, carriers[1], owners[1])
    second = _effective_collision_relative_transform(modules, carriers[1], owners[1])
    assert first == second
    assert cached_owner_matrix == matrix_before
    assert real_cache.GetLocalToWorldTransform(owners[1]) == matrix_before
    assert (source.GetRootLayer().ExportToString(), stage.GetRootLayer().ExportToString()) == before


@pytest.mark.parametrize("schema_name", ("Cube", "Sphere", "Cylinder"))
@pytest.mark.parametrize("singular", (False, True), ids=("shear", "singular"))
def test_primitive_unrepresentable_affine_still_rejects(usd, schema_name, singular):
    source, stage, owners, carriers = _stage(usd, shift=20.0, child=True, scale=(0.0 if singular else 2.0, 3.0, 4.0))
    assert source and stage
    carriers[0].SetTypeName(schema_name)
    assert carriers[0].IsA(getattr(usd.UsdGeom, schema_name))
    matrix = _effective_collision_relative_transform(usd, carriers[0], owners[0])
    # This is the exact non-mesh representation boundary used by all three
    # primitive admission paths, not a native PhysX primitive cooking test.
    with pytest.raises(NativePlanningError, match="collision_geometry_unsupported"):
        _collision_pose_scale_bake(usd, matrix, mesh_capable=False)
    if not singular:
        assert _collision_pose_scale_bake(usd, matrix, mesh_capable=True)[3] is not None

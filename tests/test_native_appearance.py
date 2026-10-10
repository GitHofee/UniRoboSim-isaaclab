"""CPU OpenUSD contract checks; these do not require a simulator process."""

import hashlib
from types import SimpleNamespace

import pytest
from pxr import Sdf, Usd, UsdGeom
from unirobosim import (
    AppearanceBinding,
    AppearanceEnvironment,
    AppearanceMaterial,
    AppearanceRenderer,
    AppearanceSnapshot,
    AppearanceTexture,
    EntityKind,
    EntityPath,
    EntitySpec,
    WorldSpec,
    appearance_topology_sha256,
)

from unirobosim_isaaclab.native import _native_name
from unirobosim_isaaclab.native_appearance import apply, capture


def make_world():
    stage = Usd.Stage.CreateInMemory()
    path = "/World/env_0/" + _native_name(EntityPath("/panel"))
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["IsaacLinkAPI"]))
    mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
    mesh.GetPrim().SetCustomDataByKey("unirobosim:visual_key", "body/visual:0")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateFaceVertexCountsAttr([3])
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    pv.Set([(0, 0), (1, 0), (0, 1)])
    pv.SetIndices([2, 1, 0])
    values = {}
    settings = SimpleNamespace(set=lambda k, v: values.__setitem__(k, v), get=values.get)
    world = SimpleNamespace(
        _spec=WorldSpec(
            "test",
            (EntitySpec(EntityPath("/panel"), EntityKind.STATIC_SCENE, asset_uri="file:///not-loaded.usd"),),
            schema_version="unirobosim.world/v0alpha6",
            build_resource_manifest_sha256="0" * 64,
        ),
        _m=SimpleNamespace(
            sim_utils=SimpleNamespace(get_current_stage=lambda: stage),
            carb=SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)),
        ),
        _invalidate_render=lambda: None,
    )
    return world, stage, mesh


def snapshot(mesh, texture=None):
    return AppearanceSnapshot(
        (AppearanceMaterial("mat", (0.2, 0.5, 0.8, 1), 0.8, 0.2, textures=() if texture is None else (texture,)),),
        (
            AppearanceBinding(
                "/panel",
                "body/visual:0",
                "mat",
                uv_coordinates=((0, 0), (1, 0), (0, 1)) if texture else None,
                uv_topology_sha256=appearance_topology_sha256(mesh.GetPointsAttr().Get(), [0, 1, 2], [3])
                if texture
                else None,
            ),
        ),
        (),
        AppearanceEnvironment((0.04, 0.08, 0.12), (0.1, 0.1, 0.1)),
        AppearanceRenderer("genesis", "rasterizer"),
    )


def test_texture_replaces_indexed_uv_and_roundtrips(tmp_path):
    from PIL import Image

    world, stage, mesh = make_world()
    file = tmp_path / "texture.png"
    Image.new("RGBA", (2, 2), (128, 64, 32, 255)).save(file)
    texture = AppearanceTexture(
        "base_color", file.as_uri(), "srgb", sha256=hashlib.sha256(file.read_bytes()).hexdigest()
    )
    restored = apply(world, snapshot(mesh, texture))
    assert not restored.limitations
    assert restored.conversion_notes
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    assert not pv.IsIndexed()
    assert pv.GetInterpolation() == "vertex"
    result = capture(world)
    assert result.materials[0].textures[0].sha256 == texture.sha256
    assert result.bindings[0].uv_indices is None
    assert result.materials[0].base_color_linear_rgba == pytest.approx((0.2, 0.5, 0.8, 1))
    # Unresolved native texture interpretation preserves geometry evidence but
    # explicitly marks the appearance incomplete; callers must not record it as complete.
    for prim in stage.Traverse():
        attribute = prim.GetAttribute("inputs:sourceColorSpace")
        if attribute:
            attribute.Set("auto")
    incomplete = capture(world)
    assert any("unresolved" in limitation for limitation in incomplete.limitations)
    assert not incomplete.materials[0].textures
    assert incomplete.bindings[0].uv_topology_sha256 == result.bindings[0].uv_topology_sha256
    # Switching to constant must disconnect old shader graph inputs.
    apply(world, snapshot(mesh))
    captured_constant = capture(world)
    assert not captured_constant.materials[0].textures
    assert captured_constant.bindings[0].uv_topology_sha256 == result.bindings[0].uv_topology_sha256


def test_validation_does_not_mutate_stage(tmp_path):
    from dataclasses import replace

    world, stage, mesh = make_world()
    value = snapshot(mesh)
    invalid = replace(value, bindings=(replace(value.bindings[0], mesh_key="missing"),))
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="does not resolve"):
        apply(world, invalid)
    assert stage.GetRootLayer().ExportToString() == before


def test_unvalidated_png_alpha_is_rejected_before_mutation(tmp_path):
    from PIL import Image

    world, stage, mesh = make_world()
    file = tmp_path / "alpha.png"
    Image.new("RGBA", (2, 2), (128, 64, 32, 128)).save(file)
    texture = AppearanceTexture("base_color", file.as_uri(), "srgb")
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="PNG alpha"):
        apply(world, snapshot(mesh, texture))
    assert stage.GetRootLayer().ExportToString() == before


def test_particle_apply_updates_visible_brdf_before_any_render():
    from pxr import Gf, UsdShade, Vt
    from unirobosim import ArrayValue, ParticleFluidSpec

    from unirobosim_isaaclab.native import IsaacLabNativeWorld

    w = object.__new__(IsaacLabNativeWorld)
    stage = Usd.Stage.CreateInMemory()
    entity = EntitySpec(
        EntityPath("/water"),
        EntityKind.PARTICLE_FLUID,
        particle_fluid=ParticleFluidSpec(ArrayValue.from_nested([[0.0, 0.0, 0.0]]), particle_radius_m=0.015),
    )
    w._spec = WorldSpec("test", (entity,), schema_version="unirobosim.world/v0alpha6")
    path = "/World/env_0/" + _native_name(entity.path)
    points = UsdGeom.Points.Define(stage, path + "/particles")
    points.CreatePointsAttr([(0, 0, 0)])
    points.CreateWidthsAttr([0.03])
    values = {}
    w._m = SimpleNamespace(
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
        Sdf=Sdf,
        Gf=Gf,
        Vt=Vt,
        sim_utils=SimpleNamespace(get_current_stage=lambda: stage),
        carb=SimpleNamespace(
            settings=SimpleNamespace(
                get_settings=lambda: SimpleNamespace(set=lambda k, v: values.__setitem__(k, v), get=values.get)
            )
        ),
    )
    w._write_particle_colors(points, [(1.0, 0.0, 0.0, 1.0)])
    w._sync_particle_visual(points, 0.015)
    snap = AppearanceSnapshot(
        (AppearanceMaterial("water", (1, 1, 1, 1), 0.8, 0.2, color_source="particle_colors"),),
        (AppearanceBinding("/water", "particles", "water"),),
        (),
        AppearanceEnvironment((0, 0, 0), (0.1, 0.1, 0.1)),
        AppearanceRenderer("genesis", "rasterizer"),
    )
    apply(w, snap)
    actual = capture(w).materials[0]
    assert actual.roughness == pytest.approx(0.8)
    assert actual.metallic == pytest.approx(0.2)
    assert "/particle_visual/prototypes/" in actual.source_parameters["native_shader"]

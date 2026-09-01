from __future__ import annotations

from types import SimpleNamespace

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from unirobosim import (
    ArrayValue,
    DebugLifetime,
    DebugMeshResource,
    DebugMeshStyle,
    DebugPrimitive,
    DebugPrimitiveKind,
)

from unirobosim_isaaclab.native import IsaacLabNativeWorld


def _resource() -> DebugMeshResource:
    return DebugMeshResource(
        "mesh.unit_triangle",
        ArrayValue.from_nested(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype="float32",
        ),
        ArrayValue.from_nested([[0, 1, 2]], dtype="int32"),
    )


def _primitive(position: tuple[float, float, float]) -> DebugPrimitive:
    return DebugPrimitive(
        primitive_id="triangle",
        layer="planning",
        group="world",
        source="test",
        kind=DebugPrimitiveKind.MESH_INSTANCE,
        geometry_m=ArrayValue.from_nested(
            [[[*position, 0.0, 0.0, 0.0, 1.0, 0.5, 0.75, 1.25]]]
        ),
        environment_indices=(0,),
        color_rgba=(0.2, 0.7, 1.0, 0.5),
        size=0.002,
        lifetime=DebugLifetime.persistent(),
        mesh_resource_id="mesh.unit_triangle",
        mesh_style=DebugMeshStyle.SOLID_WITH_EDGES,
    )


def test_mesh_instance_authors_cached_render_only_usd_and_updates_only_trs() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(  # type: ignore[attr-defined]
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        Vt=Vt,
        sim_utils=SimpleNamespace(get_current_stage=lambda: stage),
    )
    world._origins_cpu = ((10.0, 20.0, 30.0),)  # type: ignore[attr-defined]
    resource = _resource()
    world._debug_mesh_resources = {resource.resource_id: resource}  # type: ignore[attr-defined]
    world._debug_mesh_paths = {}  # type: ignore[attr-defined]
    world._debug_mesh_resource_ids = {}  # type: ignore[attr-defined]
    world._debug_mesh_signatures = {}  # type: ignore[attr-defined]

    primitive = _primitive((1.0, 2.0, 3.0))
    world._upsert_debug_mesh(primitive)
    root_path = world._debug_mesh_paths[primitive.key]  # type: ignore[attr-defined]
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(root_path))
    assert instancer.GetPositionsAttr().Get() == Vt.Vec3fArray([(11.0, 22.0, 33.0)])
    assert instancer.GetScalesAttr().Get() == Vt.Vec3fArray([(0.5, 0.75, 1.25)])
    mesh_prim = stage.GetPrimAtPath(f"{root_path}/Prototypes/mesh/surface")
    assert UsdGeom.Mesh(mesh_prim).GetFaceVertexIndicesAttr().Get() == Vt.IntArray([0, 1, 2])
    assert not mesh_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI)

    world._upsert_debug_mesh(_primitive((4.0, 5.0, 6.0)))
    assert instancer.GetPositionsAttr().Get() == Vt.Vec3fArray([(14.0, 25.0, 36.0)])
    assert UsdGeom.Mesh(mesh_prim).GetFaceVertexIndicesAttr().Get() == Vt.IntArray([0, 1, 2])

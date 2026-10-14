"""Native elastic materials and permanent soft-pad structural mounts."""

from typing import Any

from unirobosim import DeformableMaterialSpec


def _filter_backing(modules: Any, stage: Any, root: str, paths: tuple[Any, ...]) -> None:
    prim = stage.GetPrimAtPath(root)
    had_api = prim.HasAPI(modules.UsdPhysics.FilteredPairsAPI)
    api = modules.UsdPhysics.FilteredPairsAPI.Apply(prim)
    if not api:
        raise RuntimeError("native backing collision filter API is unavailable")
    relation = api.CreateFilteredPairsRel()
    original = relation.GetTargets()
    try:
        for path in paths:
            if not relation.AddTarget(path):
                raise RuntimeError("failed to author backing collision filter")
    except Exception:
        if had_api:
            relation.SetTargets(original)
        else:
            prim.RemoveAPI(modules.UsdPhysics.FilteredPairsAPI)
        raise


def author_robot_self_collision_filter(modules: Any, stage: Any, root: str, robot_root: str) -> None:
    """Inherit only an explicitly disabled parent articulation self-collision policy."""
    prims = tuple(modules.Usd.PrimRange(stage.GetPrimAtPath(robot_root), modules.Usd.TraverseInstanceProxies()))
    articulations = tuple(p for p in prims if p.HasAPI(modules.UsdPhysics.ArticulationRootAPI))
    if len(articulations) != 1:
        raise ValueError("mounted pad parent must have exactly one articulation root")
    attribute = articulations[0].GetAttribute("physxArticulation:enabledSelfCollisions")
    if attribute and attribute.Get() is False:
        _filter_backing(
            modules, stage, root, tuple(p.GetPath() for p in prims if p.HasAPI(modules.UsdPhysics.RigidBodyAPI))
        )


def author_elastic_material(modules: Any, stage: Any, root: str, material: DeformableMaterialSpec) -> None:
    """Author native material values verbatim; damping is not linear drag per second."""
    path = f"{root}/elastic_material"
    if stage.GetPrimAtPath(path):
        raise RuntimeError("elastic material target already exists")
    try:
        shade = modules.UsdShade.Material.Define(stage, path)
        prim = shade.GetPrim()
        if not prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI") or not prim.ApplyAPI("PhysxDeformableMaterialAPI"):
            raise RuntimeError("native deformable material schemas are unavailable")
        for name, value in {
            "omniphysics:youngsModulus": material.youngs_modulus_pa,
            "omniphysics:poissonsRatio": material.poissons_ratio,
            "omniphysics:staticFriction": material.static_friction,
            "omniphysics:dynamicFriction": material.dynamic_friction,
            "physxDeformableMaterial:elasticityDamping": material.elasticity_damping,
        }.items():
            attribute = prim.GetAttribute(name)
            if not attribute or not attribute.Set(value):
                raise RuntimeError(f"native deformable material attribute is unavailable: {name}")
        binding = modules.UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root))
        if not binding or not binding.Bind(shade, materialPurpose="physics"):
            raise RuntimeError("failed to bind native deformable material")
    except Exception:
        stage.RemovePrim(path)
        raise


def author_rigid_mount(modules: Any, stage: Any, root: str, parent: str, indices: tuple[int, ...], points: Any) -> None:
    """Native solver coupling; reject failed authoring and remove partial attachment."""
    path = f"{root}/structural_mount"
    if stage.GetPrimAtPath(path):
        raise RuntimeError("structural mount target already exists")
    try:
        prim = stage.DefinePrim(path, "OmniPhysicsVtxXformAttachment")
        cache = modules.UsdGeom.XformCache()
        transform = (
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))
            * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent)).GetInverse()
        )
        anchors = tuple(tuple(float(v) for v in transform.Transform(modules.Gf.Vec3d(*points[i]))) for i in indices)
        for name, targets in (
            ("omniphysics:src0", [modules.Sdf.Path(f"{root}/sim_mesh")]),
            ("omniphysics:src1", [modules.Sdf.Path(parent)]),
        ):
            relation = prim.GetRelationship(name)
            if not relation or not relation.SetTargets(targets):
                raise RuntimeError(f"failed to author native structural mount relationship: {name}")
        for name, value in (
            ("omniphysics:vtxIndicesSrc0", modules.Vt.IntArray(indices)),
            ("omniphysics:localPositionsSrc1", modules.Vt.Vec3fArray(anchors)),
        ):
            attribute = prim.GetAttribute(name)
            if not attribute or not attribute.Set(value):
                raise RuntimeError(f"failed to author native structural mount attribute: {name}")
        _filter_backing(modules, stage, root, (modules.Sdf.Path(parent),))
    except Exception:
        stage.RemovePrim(path)
        raise

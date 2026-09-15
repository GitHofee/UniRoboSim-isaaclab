from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from unirobosim import CapabilityId, ContactComplianceSpec

from unirobosim_isaaclab import CAPABILITIES
from unirobosim_isaaclab.contact_compliance import author_contact_compliance


class _MaterialAPI:
    """PhysX attribute writer substitute; USD composition itself is real."""

    def __init__(self, prim: Any) -> None:
        self.prim = prim

    @classmethod
    def Apply(cls, prim: Any) -> _MaterialAPI:
        return cls(prim)

    def CreateCompliantContactStiffnessAttr(self) -> Any:
        return self.prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float)

    def CreateCompliantContactDampingAttr(self) -> Any:
        return self.prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float)

    def CreateCompliantContactAccelerationSpringAttr(self) -> Any:
        return self.prim.CreateAttribute("physxMaterial:compliantContactAccelerationSpring", Sdf.ValueTypeNames.Bool)

    def CreateRestitutionCombineModeAttr(self) -> Any:
        return self.prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token)

    def CreateDampingCombineModeAttr(self) -> Any:
        return self.prim.CreateAttribute("physxMaterial:dampingCombineMode", Sdf.ValueTypeNames.Token)


_MODULES = SimpleNamespace(
    Usd=Usd,
    UsdGeom=UsdGeom,
    UsdPhysics=UsdPhysics,
    UsdShade=UsdShade,
    PhysxSchema=SimpleNamespace(PhysxMaterialAPI=_MaterialAPI),
)


def _collider(stage: Any, path: str) -> Any:
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def _material(stage: Any, path: str, friction: float = 0.8) -> Any:
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(friction)
    api.CreateDynamicFrictionAttr().Set(0.6)
    api.CreateRestitutionAttr().Set(0.15)
    material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("multiply")
    return material


def _bind(prim: Any, material: Any, purpose: str = "physics", **kwargs: Any) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose=purpose, **kwargs)


def _resolved(prim: Any, purpose: str = "physics") -> Any:
    return UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose=purpose)[0]


def test_default_off_never_touches_modules_or_stage() -> None:
    assert author_contact_compliance(object(), object(), ("/not/loaded",), None) == ()


def test_private_material_preserves_shared_friction_visual_and_collision_settings() -> None:
    stage = Usd.Stage.CreateInMemory()
    source = _material(stage, "/Shared/material")
    visual = UsdShade.Material.Define(stage, "/Shared/visual")
    selected = _collider(stage, "/World/env_0/target/collision")
    untouched = _collider(stage, "/World/env_0/other/collision")
    for prim in (selected, untouched):
        _bind(prim, source)
        _bind(prim, visual, purpose="")
    selected.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.002)
    selected.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    source_before = source.GetPrim().GetPrimStack()[0].GetAsText()

    paths = author_contact_compliance(
        _MODULES, stage, ("/World/env_0/target",), ContactComplianceSpec(1200.0, 7.0, "min", "max")
    )

    assert len(paths) == 1
    result = _resolved(selected)
    assert str(result.GetPath()) == paths[0]
    prim = result.GetPrim()
    assert UsdPhysics.MaterialAPI(prim).GetStaticFrictionAttr().Get() == pytest.approx(0.8)
    assert UsdPhysics.MaterialAPI(prim).GetDynamicFrictionAttr().Get() == pytest.approx(0.6)
    assert UsdPhysics.MaterialAPI(prim).GetRestitutionAttr().Get() == pytest.approx(0.15)
    assert prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == "multiply"
    assert prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == 1200.0
    assert prim.GetAttribute("physxMaterial:compliantContactDamping").Get() == 7.0
    assert prim.GetAttribute("physxMaterial:compliantContactAccelerationSpring").Get() is False
    assert prim.GetAttribute("physxMaterial:restitutionCombineMode").Get() == "min"
    assert prim.GetAttribute("physxMaterial:dampingCombineMode").Get() == "max"
    assert source.GetPrim().GetPrimStack()[0].GetAsText() == source_before
    assert _resolved(untouched).GetPath() == source.GetPath()
    assert _resolved(selected, "").GetPath() == visual.GetPath()
    assert selected.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.002)
    assert selected.GetAttribute("physxCollision:restOffset").Get() == 0.0


def test_all_colliders_and_environments_get_scoped_materials_with_no_aliasing() -> None:
    stage = Usd.Stage.CreateInMemory()
    source = _material(stage, "/Shared/common")
    roots = ("/World/env_0/target", "/World/env_1/target")
    colliders = []
    for root in roots:
        for suffix in ("left", "right"):
            prim = _collider(stage, f"{root}/{suffix}")
            _bind(prim, source)
            colliders.append(prim)
    paths = author_contact_compliance(_MODULES, stage, roots, ContactComplianceSpec(200.0, 0.0))
    assert len(paths) == 2
    assert _resolved(colliders[0]).GetPath() == _resolved(colliders[1]).GetPath()
    assert _resolved(colliders[2]).GetPath() == _resolved(colliders[3]).GetPath()
    assert _resolved(colliders[0]).GetPath() != _resolved(colliders[2]).GetPath()
    assert not source.GetPrim().GetAttribute("physxMaterial:compliantContactStiffness")


def test_distinct_materials_remain_distinct_and_disabled_colliders_are_untouched() -> None:
    stage = Usd.Stage.CreateInMemory()
    left = _collider(stage, "/Object/left")
    right = _collider(stage, "/Object/right")
    disabled = _collider(stage, "/Object/disabled")
    UsdPhysics.CollisionAPI(disabled).CreateCollisionEnabledAttr().Set(False)
    _bind(left, _material(stage, "/Materials/left", 0.1))
    _bind(right, _material(stage, "/Materials/right", 0.9))
    paths = author_contact_compliance(_MODULES, stage, ("/Object",), ContactComplianceSpec(1000.0, 2.0))
    assert len(paths) == 2
    assert UsdPhysics.MaterialAPI(_resolved(left).GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(0.1)
    assert UsdPhysics.MaterialAPI(_resolved(right).GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(0.9)
    assert not _resolved(disabled)


def test_mass_units_are_converted() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
    collider = _collider(stage, "/Object")
    _bind(collider, _material(stage, "/Mat"))
    author_contact_compliance(_MODULES, stage, ("/Object",), ContactComplianceSpec(100.0, 2.0))
    prim = _resolved(collider).GetPrim()
    assert UsdPhysics.MaterialAPI(prim).GetStaticFrictionAttr().Get() == pytest.approx(0.8)
    assert prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == pytest.approx(100000.0)
    assert prim.GetAttribute("physxMaterial:compliantContactDamping").Get() == pytest.approx(2000.0)


@pytest.mark.parametrize("case", ["empty", "unbound", "reserved", "strong-ancestor", "subset", "instance"])
def test_unsupported_binding_structures_fail_before_authoring(case: str) -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
    if case != "empty":
        collider = _collider(stage, "/Object/collision")
    if case == "reserved":
        UsdGeom.Scope.Define(stage, "/Object/__unirobosim_contact_compliance")
    elif case == "strong-ancestor":
        _bind(root, _material(stage, "/Mat"), bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    elif case == "subset":
        subset = UsdGeom.Subset.Define(stage, "/Object/collision/faces")
        subset.CreateFamilyNameAttr().Set("materialBind")
    elif case == "instance":
        _collider(stage, "/Prototype/collision")
        stage.RemovePrim(collider.GetPath())
        root.GetReferences().AddInternalReference("/Prototype")
        root.SetInstanceable(True)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="contact_compliance"):
        author_contact_compliance(_MODULES, stage, ("/Object",), ContactComplianceSpec(1000.0, 2.0))
    assert stage.GetRootLayer().ExportToString() == before


def test_weak_inherited_material_is_preserved_in_private_override() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
    collider = _collider(stage, "/Object/collision")
    source = _material(stage, "/Mat", 0.3)
    _bind(root, source)
    author_contact_compliance(_MODULES, stage, ("/Object",), ContactComplianceSpec(900.0, 3.0))
    assert UsdPhysics.MaterialAPI(_resolved(collider).GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(0.3)
    assert _resolved(root).GetPath() == source.GetPath()


def test_capability_declares_force_law_scope_and_no_mesh_deformation() -> None:
    capability = CAPABILITIES.get(CapabilityId("physics.contact.compliant@1"))
    assert capability is not None
    assert capability.properties["contact_law"] == "implicit-force-spring-damper"
    assert capability.properties["mesh_deformation"] is False


def test_referenced_environment_clones_keep_local_overrides_and_survive_round_trip() -> None:
    stage = Usd.Stage.CreateInMemory()
    roots = ("/World/env_0/target", "/World/env_1/target")
    source = _material(stage, f"{roots[0]}/original_material", 0.7)
    _bind(_collider(stage, f"{roots[0]}/collision"), source)
    clone = UsdGeom.Xform.Define(stage, roots[1]).GetPrim()
    clone.GetReferences().AddInternalReference(roots[0])
    paths = author_contact_compliance(_MODULES, stage, roots, ContactComplianceSpec(1200.0, 8.0))
    assert len(paths) == 2
    first = _resolved(stage.GetPrimAtPath(f"{roots[0]}/collision"))
    second = _resolved(stage.GetPrimAtPath(f"{roots[1]}/collision"))
    assert str(first.GetPath()) == paths[0]
    assert str(second.GetPath()) == paths[1]
    second.GetPrim().GetAttribute("physxMaterial:compliantContactStiffness").Set(600.0)
    assert first.GetPrim().GetAttribute("physxMaterial:compliantContactStiffness").Get() == 1200.0
    assert not source.GetPrim().GetAttribute("physxMaterial:compliantContactStiffness")
    restored = Usd.Stage.Open(stage.Flatten())
    restored_material = _resolved(restored.GetPrimAtPath(f"{roots[1]}/collision"))
    assert restored_material.GetPrim().GetAttribute("physxMaterial:compliantContactStiffness").Get() == 600.0
    assert UsdPhysics.MaterialAPI(restored_material.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(0.7)


def test_collection_binding_is_rejected_without_modifying_stage() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
    collider = _collider(stage, "/Object/collision")
    material = _material(stage, "/Mat")
    collection = Usd.CollectionAPI.Apply(root, "all_collisions")
    collection.CreateIncludesRel().AddTarget(collider.GetPath())
    UsdShade.MaterialBindingAPI.Apply(root).Bind(collection, material, "contact", materialPurpose="physics")
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="collection material bindings"):
        author_contact_compliance(_MODULES, stage, ("/Object",), ContactComplianceSpec(1000.0, 2.0))
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("stiffness", [1e300, 1e-300])
def test_unrepresentable_native_values_fail_before_authoring(stiffness: float) -> None:
    stage = Usd.Stage.CreateInMemory()
    _bind(_collider(stage, "/Object"), _material(stage, "/Mat"))
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="float32 range"):
        author_contact_compliance(_MODULES, stage, ("/Object",), ContactComplianceSpec(stiffness, 0.0))
    assert stage.GetRootLayer().ExportToString() == before

"""Build-time, object-scoped USD compliant contact material overrides.

No simulator/USD imports occur at module import time. The native world supplies
its already initialized USD modules. Source materials and visual bindings remain
untouched; a private material references each resolved source material.
"""

from __future__ import annotations

import math
import struct
from typing import Any

from unirobosim import ContactComplianceSpec


def author_contact_compliance(
    modules: Any,
    stage: Any,
    roots: tuple[str, ...],
    compliance: ContactComplianceSpec | None,
) -> tuple[str, ...]:
    """Override every enabled collider, before physics initialization.

    Returns private material paths for diagnostic readback. Unsupported USD
    binding arrangements fail explicitly rather than silently losing materials.
    """
    if compliance is None:
        return ()

    usd, physics, shade = modules.Usd, modules.UsdPhysics, modules.UsdShade
    # USD stiffness/damping units depend on stage mass, not stage length.
    mass_unit = physics.GetStageKilogramsPerUnit(stage)
    if not math.isfinite(mass_unit) or mass_unit <= 0:
        raise ValueError("contact_compliance requires finite positive stage kilogramsPerUnit")
    stiffness = _native_float(compliance.stiffness_n_m / mass_unit)
    damping = _native_float(compliance.damping_n_s_m / mass_unit)
    plans: list[tuple[str, list[tuple[Any, Any]]]] = []
    # Validate all environments before authoring any overrides.
    for root in roots:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid():
            raise ValueError(f"contact_compliance object root does not exist: {root}")
        scope = f"{root}/__unirobosim_contact_compliance"
        if stage.GetPrimAtPath(scope).IsValid():
            raise ValueError(f"contact_compliance reserved material scope already exists: {scope}")
        colliders: list[tuple[Any, Any]] = []
        for prim in usd.PrimRange(root_prim, usd.TraverseInstanceProxies()):
            if not prim.HasAPI(physics.CollisionAPI):
                continue
            if physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            if prim.IsInstanceProxy() or prim.IsInstance():
                raise ValueError(f"contact_compliance does not support instance colliders: {prim.GetPath()}")
            for child in prim.GetChildren():
                if child.IsA(modules.UsdGeom.Subset):
                    subset = modules.UsdGeom.Subset(child)
                    if subset.GetFamilyNameAttr().Get() == "materialBind":
                        raise ValueError(
                            f"contact_compliance does not support material subsets: {child.GetPath()}"
                        )
            binding = shade.MaterialBindingAPI(prim)
            material, relationship = binding.ComputeBoundMaterial(materialPurpose="physics")
            if relationship:
                if ":collection:" in relationship.GetName():
                    raise ValueError(
                        f"contact_compliance does not support collection material bindings: {relationship.GetPath()}"
                    )
                if (
                    relationship.GetPrim() != prim
                    and shade.MaterialBindingAPI.GetMaterialBindingStrength(relationship)
                    == shade.Tokens.strongerThanDescendants
                ):
                    raise ValueError(
                        "contact_compliance cannot override stronger ancestor material binding: "
                        f"{relationship.GetPath()}"
                    )
                if not material:
                    raise ValueError(f"contact_compliance found unresolved material binding: {relationship.GetPath()}")
            if not material or not material.GetPrim().HasAPI(physics.MaterialAPI):
                raise ValueError(
                    "contact_compliance requires a bound UsdPhysics.MaterialAPI material to preserve "
                    f"existing friction and restitution: {prim.GetPath()}"
                )
            colliders.append((prim, material))
        if not colliders:
            raise ValueError(f"contact_compliance requires at least one enabled collider: {root}")
        plans.append((scope, colliders))

    authored: list[str] = []
    for scope, colliders in plans:
        materials: dict[str, Any] = {}
        for collider, source in colliders:
            source_path = str(source.GetPath()) if source else ""
            if source_path not in materials:
                path = f"{scope}/material_{len(materials)}"
                material = shade.Material.Define(stage, path)
                if source:
                    material.GetPrim().GetReferences().AddInternalReference(source.GetPath())
                physics.MaterialAPI.Apply(material.GetPrim())
                api = modules.PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
                api.CreateCompliantContactStiffnessAttr().Set(stiffness)
                api.CreateCompliantContactDampingAttr().Set(damping)
                api.CreateCompliantContactAccelerationSpringAttr().Set(False)
                # PhysX reuses restitution combination for compliant stiffness.
                api.CreateRestitutionCombineModeAttr().Set(compliance.stiffness_combine_mode)
                api.CreateDampingCombineModeAttr().Set(compliance.damping_combine_mode)
                materials[source_path] = material
                authored.append(path)
            material = materials[source_path]
            binding = shade.MaterialBindingAPI.Apply(collider)
            binding.Bind(material, bindingStrength=shade.Tokens.strongerThanDescendants, materialPurpose="physics")
            resolved, _ = binding.ComputeBoundMaterial(materialPurpose="physics")
            if not resolved or resolved.GetPath() != material.GetPath():
                raise ValueError(f"contact_compliance physics material override did not resolve: {collider.GetPath()}")
    return tuple(authored)


def _native_float(value: float) -> float:
    """Reject float32 overflow/underflow instead of disabling the contact law."""
    try:
        native = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("contact_compliance value is outside the native float32 range") from error
    if not math.isfinite(native) or (value != 0 and native == 0):
        raise ValueError("contact_compliance value is outside the native float32 range")
    return float(native)

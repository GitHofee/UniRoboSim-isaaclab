"""Explicit linear-light appearance profile using public USD/RTX APIs."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IsaacLabAppearanceConfig:
    background_rgb: tuple[float, float, float] = (0.04, 0.08, 0.12)
    ambient_rgb: tuple[float, float, float] = (0.1, 0.1, 0.1)
    light_direction: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    light_color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_intensity: float = 5.0
    # Explicit renderer unit conversion, not a change of target illumination.
    light_unit_scale: float = 1.0
    exposure_time_s: float = 0.02
    exposure_iso: float = 100.0
    exposure_f_stop: float = 5.0
    roughness: float = 1.0
    metallic: float = 0.0
    # Full native prim-path substring -> canonical linear RGBA; last match wins.
    material_overrides: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()

    def __post_init__(self):
        for value in (self.background_rgb, self.ambient_rgb, self.light_color_rgb):
            if len(value) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in value):
                raise ValueError("appearance RGB requires three finite linear values in [0,1]")
        if (
            len(self.light_direction) != 3
            or not all(math.isfinite(v) for v in self.light_direction)
            or not any(self.light_direction)
        ):
            raise ValueError("appearance light direction must be finite and nonzero")
        for v in (self.light_intensity, self.light_unit_scale, self.roughness, self.metallic):
            if not math.isfinite(v) or v < 0:
                raise ValueError("appearance scalars must be finite and nonnegative")
        for match, rgba in self.material_overrides:
            if not match or len(rgba) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in rgba):
                raise ValueError("appearance override requires a path substring and linear RGBA")


def apply_appearance(stage, settings, config):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    values = {
        "/rtx/post/tonemap/op": 1,
        "/rtx/post/tonemap/enableSrgbToGamma": True,
        "/rtx/post/tonemap/colorMode": 0,
        "/rtx/post/tonemap/filmIso": config.exposure_iso,
        "/rtx/post/tonemap/exposureTime": config.exposure_time_s,
        "/rtx/post/tonemap/fNumber": config.exposure_f_stop,
        "/rtx/post/tonemap/responsivity": 1.0,
        "/rtx/post/tonemap/ocio/enabled": False,
        "/rtx/post/histogram/enabled": False,
        "/rtx/background/source/type": 2,
        "/rtx/background/source/color": config.background_rgb,
    }
    for key, value in values.items():
        settings.set(key, value)
    light = UsdLux.DistantLight.Define(stage, "/World/unirobosimDirectionalLight")
    light.CreateIntensityAttr(config.light_intensity * config.light_unit_scale)
    light.CreateColorAttr(Gf.Vec3f(*config.light_color_rgb))
    light.CreateAngleAttr(0.53)
    light.CreateNormalizeAttr(True)
    direction = Gf.Vec3d(*config.light_direction).GetNormalized()
    UsdGeom.Xformable(light).AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, -1), direction).GetQuat()))
    UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(True)
    # A Lambertian surface under uniform hemisphere radiance L has outgoing L*albedo.
    dome = UsdLux.DomeLight.Define(stage, "/World/unirobosimAmbientLight")
    dome.CreateIntensityAttr(1.0)
    dome.CreateColorAttr(Gf.Vec3f(*config.ambient_rgb))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            for schema in ("OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"):
                prim.AddAppliedSchema(schema)
            for name, value in {
                "exposure": 0.0,
                "exposure:fStop": config.exposure_f_stop,
                "exposure:iso": config.exposure_iso,
                "exposure:responsivity": 1.0,
                "exposure:time": config.exposure_time_s,
            }.items():
                prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
            prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        if not prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Points):
            continue
        rgba = None
        for match, color in config.material_overrides:
            if match in str(prim.GetPath()):
                rgba = color
        if rgba is None:
            continue
        path = str(prim.GetPath()) + "/linearMaterial"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        for name, value, kind in (
            ("diffuseColor", Gf.Vec3f(*rgba[:3]), Sdf.ValueTypeNames.Color3f),
            ("opacity", rgba[3], Sdf.ValueTypeNames.Float),
            ("roughness", config.roughness, Sdf.ValueTypeNames.Float),
            ("metallic", config.metallic, Sdf.ValueTypeNames.Float),
        ):
            shader.CreateInput(name, kind).Set(value)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return tuple(values)


def read_appearance(stage, settings, keys):
    from pxr import UsdGeom, UsdLux, UsdShade

    def plain(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        try:
            return [plain(x) for x in v]
        except TypeError:
            return str(v)

    result = {
        "settings": {k: plain(settings.get(k)) for k in keys},
        "lights": [],
        "shaders": [],
        "cameras": [],
        "bindings": [],
        "particle_primvars": [],
        "particle_instances": [],
    }
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.PointInstancer):
            inst = UsdGeom.PointInstancer(prim)
            result["particle_instances"].append(
                {
                    "path": path,
                    "positions": plain(inst.GetPositionsAttr().Get()),
                    "indices": plain(inst.GetProtoIndicesAttr().Get()),
                    "prototypes": plain(inst.GetPrototypesRel().GetTargets()),
                }
            )
        if prim.IsA(UsdGeom.Gprim):
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            result["bindings"].append({"path": path, "material": str(bound.GetPath()) if bound else None})
        if prim.IsA(UsdGeom.Points):
            points = UsdGeom.Points(prim)
            result["particle_primvars"].append(
                {
                    "path": path,
                    "colors": plain(points.GetDisplayColorPrimvar().ComputeFlattened()),
                    "opacity": plain(points.GetDisplayOpacityPrimvar().ComputeFlattened()),
                    "color_interpolation": str(points.GetDisplayColorPrimvar().GetInterpolation()),
                    "opacity_interpolation": str(points.GetDisplayOpacityPrimvar().GetInterpolation()),
                }
            )
        if prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdGeom.Camera):
            group = "lights" if prim.HasAPI(UsdLux.LightAPI) else "shaders" if prim.IsA(UsdShade.Shader) else "cameras"
            result[group].append(
                {
                    "path": path,
                    "attributes": {
                        a.GetName(): plain(a.Get()) for a in prim.GetAttributes() if a.HasAuthoredValueOpinion()
                    },
                    "connections": {
                        a.GetName(): [str(x) for x in a.GetConnections()]
                        for a in prim.GetAttributes()
                        if a.GetConnections()
                    },
                }
            )
    return result

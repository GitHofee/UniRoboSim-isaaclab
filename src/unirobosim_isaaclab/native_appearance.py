"""Native appearance capture/restore using semantic mesh identities and USD APIs."""

import hashlib
import math
from pathlib import Path
from urllib.parse import unquote, urlparse

from unirobosim import (
    AppearanceApplyResult,
    AppearanceBinding,
    AppearanceEnvironment,
    AppearanceLight,
    AppearanceMaterial,
    AppearanceRenderer,
    AppearanceSnapshot,
    AppearanceTexture,
    EntityKind,
    FrozenMap,
    appearance_topology_sha256,
)


def visual_targets(world):
    from pxr import Usd, UsdGeom

    from .native import _native_name

    stage = world._m.sim_utils.get_current_stage()
    targets = {}
    for entity in world._spec.entities:
        if entity.kind is EntityKind.CAMERA_SENSOR:
            continue
        for environment in range(world._spec.environments.count):
            root = stage.GetPrimAtPath(f"/World/env_{environment}/{_native_name(entity.path)}")
            if entity.kind is EntityKind.PARTICLE_FLUID:
                targets[(entity.path.value, "particles", environment)] = stage.GetPrimAtPath(
                    str(root.GetPath()) + "/particles"
                )
                continue
            groups = {}
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                imageable = UsdGeom.Imageable(prim)
                if imageable.ComputePurpose() in ("guide", "proxy") or imageable.ComputeVisibility() == "invisible":
                    continue
                explicit = prim.GetCustomDataByKey("unirobosim:visual_key")
                if explicit:
                    key = explicit
                else:
                    parent = prim.GetParent()
                    link = None
                    while parent and parent != root.GetParent():
                        authored = parent.GetMetadata("apiSchemas")
                        if "IsaacLinkAPI" in parent.GetAppliedSchemas() or (
                            authored and "IsaacLinkAPI" in authored.GetAppliedItems()
                        ):
                            link = parent.GetName()
                            break
                        parent = parent.GetParent()
                    key = f"link:{link}/visual:0" if link else "body/visual:0"
                groups.setdefault(key, []).append(prim)
            for key, prims in groups.items():
                if len(prims) != 1:
                    raise ValueError(
                        f"appearance identity ambiguous: {entity.path.value} {key}; author explicit visual keys"
                    )
                targets[(entity.path.value, key, environment)] = prims[0]
    return targets


def _input(shader, name, default=None):
    value = shader.GetInput(name)
    visited = set()
    for _ in range(16):
        if not value:
            return default
        attr = value.GetAttr()
        path = str(attr.GetPath())
        if path in visited:
            raise ValueError("cyclic material connection")
        visited.add(path)
        connections = value.GetConnectedSource()
        if not connections:
            return value.Get() if value.Get() is not None else default
        source, key, kind = connections
        from pxr import UsdShade

        if str(kind) == "Input":
            value = source.GetInput(key)
        else:
            src = UsdShade.Shader(source.GetPrim())
            if src and src.GetIdAttr().Get() == "UsdUVTexture":
                return src
            raise ValueError(f"unsupported material graph at {path}")
    raise ValueError("material graph depth exceeded")


def _local_texture(texture):
    parsed = urlparse(texture.uri)
    if parsed.scheme not in ("", "file"):
        raise ValueError("appearance texture must be resolved to local resource")
    path = Path(unquote(parsed.path) if parsed.scheme else texture.uri)
    if not path.is_file() or path.suffix.lower() != ".png":
        raise ValueError("appearance base-color texture requires existing PNG")
    if texture.sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != texture.sha256:
        raise ValueError("appearance texture digest mismatch")
    from PIL import Image

    with Image.open(path) as image:
        if image.convert("RGBA").getchannel("A").getextrema() != (255, 255):
            raise ValueError("non-opaque PNG alpha textures are not supported by the validated RTX appearance path")
    return path


def apply(world, snapshot):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

    if not isinstance(snapshot, AppearanceSnapshot):
        raise TypeError("appearance requires AppearanceSnapshot")
    if snapshot.limitations:
        raise ValueError("cannot apply incomplete appearance snapshot: " + str(snapshot.limitations))
    if (
        snapshot.renderer.color_space != "linear-srgb"
        or snapshot.renderer.output_transfer not in ("gamma22", "srgb")
        or snapshot.renderer.tone_mapping not in ("clamp", "linear")
    ):
        raise ValueError("unsupported renderer transfer/tone mapping")
    if snapshot.environment.ambient_semantics != "lambertian_albedo_multiplier":
        raise ValueError("unsupported ambient semantics")
    stage = world._m.sim_utils.get_current_stage()
    settings = world._m.carb.settings.get_settings()
    targets = visual_targets(world)
    materials = {x.material_id: x for x in snapshot.materials}
    staged = []
    used_targets = set()
    for binding in snapshot.bindings:
        mat = materials[binding.material_id]
        environments = (
            range(world._spec.environments.count)
            if binding.environment_indices is None
            else binding.environment_indices
        )
        textures = []
        for texture in mat.textures:
            if texture.channel != "base_color" or texture.uv_set != "st":
                raise ValueError("only base-color PNG with st UV is supported")
            textures.append((texture, _local_texture(texture)))
        if len(textures) > 1:
            raise ValueError("duplicate base-color textures")
        for env in environments:
            key = (binding.entity_path, binding.mesh_key, env)
            if key not in targets:
                raise ValueError(f"appearance target does not resolve: {key}")
            if key in used_targets:
                raise ValueError("overlapping appearance binding targets")
            used_targets.add(key)
            prim = targets[key]
            if textures:
                if binding.uv_coordinates is None:
                    raise ValueError("texture binding is missing explicit UV coordinates")
                mesh = UsdGeom.Mesh(prim)
                if not mesh:
                    raise ValueError("texture UV restore requires mesh topology")
                if not binding.uv_topology_sha256 or binding.uv_topology_sha256 != appearance_topology_sha256(
                    mesh.GetPointsAttr().Get(),
                    mesh.GetFaceVertexIndicesAttr().Get(),
                    mesh.GetFaceVertexCountsAttr().Get(),
                ):
                    raise ValueError("UV topology digest mismatch; refusing vertex-order guess")
                count = (
                    len(mesh.GetPointsAttr().Get())
                    if binding.uv_interpolation == "vertex"
                    else len(mesh.GetFaceVertexIndicesAttr().Get())
                )
                if len(binding.uv_indices if binding.uv_indices is not None else binding.uv_coordinates) != count:
                    raise ValueError("texture UV count differs from target topology")
            staged.append((binding, mat, prim, textures))
    for light in snapshot.lights:
        allowed = ("linear_rgb_irradiance", "lux") if light.kind == "directional" else ("linear_rgb_radiance", "nit")
        if light.intensity_unit not in allowed:
            raise ValueError(f"unsupported {light.kind} intensity unit: {light.intensity_unit}")
    # Official Replicator synchronous material loading: no placeholder frames.
    settings.set("/rtx/materialDb/syncLoads", True)
    settings.set("/rtx/hydra/materialSyncLoads", True)
    settings.set("/app/asyncRendering", False)
    # All resource and semantic validation above precedes mutation.
    ambient = snapshot.environment.ambient_linear_rgb
    world._particle_appearance = {}
    for binding, mat, prim, textures in staged:
        native_path = prim.GetPath()
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.IsInstance():
                ancestor.SetInstanceable(False)
            ancestor = ancestor.GetParent()
        prim = stage.GetPrimAtPath(native_path)
        path = str(prim.GetPath()) + "/recordedMaterial"
        material = UsdShade.Material.Define(stage, path)
        if not stage.GetPrimAtPath(path + "/surface"):
            try:
                from omni.usd.commands import CreateShaderPrimFromSdrCommand
            except ImportError:
                pass
            else:
                CreateShaderPrimFromSdrCommand(
                    parent_path=path, identifier="UsdPreviewSurface", stage_or_context=stage, prim_name="surface"
                ).do()
        shader = UsdShade.Shader.Define(stage, path + "/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        shader.GetPrim().SetCustomDataByKey("unirobosim:material_id", mat.material_id)
        rgba = mat.base_color_linear_rgba
        for name, value, kind in [
            ("diffuseColor", Gf.Vec3f(*rgba[:3]), Sdf.ValueTypeNames.Color3f),
            ("opacity", rgba[3], Sdf.ValueTypeNames.Float),
            ("roughness", mat.roughness, Sdf.ValueTypeNames.Float),
            ("metallic", mat.metallic, Sdf.ValueTypeNames.Float),
            ("emissiveColor", Gf.Vec3f(*(rgba[i] * ambient[i] / 1133.6 for i in range(3))), Sdf.ValueTypeNames.Color3f),
        ]:
            shader.CreateInput(name, kind).DisconnectSource()
            shader.CreateInput(name, kind).Set(value)
        if textures:
            texture, file = textures[0]
            uv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, binding.uv_interpolation
            )
            uv.Set(Vt.Vec2fArray([tuple(x) for x in binding.uv_coordinates]))
            if binding.uv_indices is not None:
                uv.SetIndices(Vt.IntArray(binding.uv_indices))
            else:
                uv.BlockIndices()
            reader = UsdShade.Shader.Define(stage, path + "/uv")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(stage, path + "/baseColor")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                "sRGB" if texture.color_space == "srgb" else "raw"
            )
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*rgba))
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            emission = UsdShade.Shader.Define(stage, path + "/ambientTexture")
            emission.CreateIdAttr("UsdUVTexture")
            emission.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            for name in ("file", "sourceColorSpace"):
                emission.CreateInput(name, tex.GetInput(name).GetTypeName()).Set(tex.GetInput(name).Get())
            emission.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            emission.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(*(rgba[i] * ambient[i] / 1133.6 for i in range(3)), 1)
            )
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                emission.ConnectableAPI(), "rgb"
            )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        if binding.mesh_key == "particles":
            world._particle_appearance[str(prim.GetPath())] = (mat.roughness, mat.metallic, ambient)
    # Particle prototypes are the actual rendered materials. Synchronize now,
    # so immediate capture after apply observes restored BRDF before first frame.
    for binding, _mat, prim, _textures in staged:
        if binding.mesh_key == "particles":
            points = UsdGeom.Points(stage.GetPrimAtPath(prim.GetPath()))
            world._sync_particle_visual(points, float(points.GetWidthsAttr().Get()[0]) / 2.)
    for prim in list(stage.Traverse()):
        if prim.HasAPI(UsdLux.LightAPI):
            stage.RemovePrim(prim.GetPath())
    for light in snapshot.lights:
        path = "/World/recordedLights/l_" + hashlib.sha256(light.light_id.encode()).hexdigest()[:16]
        lamp = (
            UsdLux.DistantLight.Define(stage, path)
            if light.kind == "directional"
            else UsdLux.DomeLight.Define(stage, path)
        )
        lamp.CreateIntensityAttr(light.intensity)
        lamp.CreateColorAttr(Gf.Vec3f(*light.color_linear_rgb))
        lamp.CreateExposureAttr(0.0)
        lamp.GetPrim().CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(2505)
        lamp.GetPrim().SetCustomDataByKey("unirobosim:light_id", light.light_id)
        lamp.GetPrim().SetCustomDataByKey("unirobosim:source_unit", light.intensity_unit)
        if light.kind == "directional":
            lamp.CreateNormalizeAttr(True)
            lamp.CreateAngleAttr(light.angular_diameter_degrees)
            q = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*light.direction_world)).GetQuat()
            UsdGeom.Xformable(lamp).AddOrientOp().Set(Gf.Quatf(q))
        UsdLux.ShadowAPI.Apply(lamp.GetPrim()).CreateShadowEnableAttr(light.shadows)
    values = {
        "/rtx/sceneDb/ambientLightIntensity": 0.0,
        "/rtx/useViewLightingMode": False,
        "/rtx/post/tonemap/op": 1,
        "/rtx/post/tonemap/enableSrgbToGamma": True,
        "/rtx/post/tonemap/ocio/enabled": False,
        "/rtx/post/tonemap/colorMode": 0,
        "/rtx/post/histogram/enabled": False,
        "/rtx/indirectDiffuse/enabled": False,
        "/rtx/ambientOcclusion/enabled": False,
        "/rtx/background/source/type": 2,
        "/rtx/background/source/color": snapshot.environment.background_linear_rgb,
    }
    for key, value in values.items():
        settings.set(key, value)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            for schema in ("OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"):
                prim.AddAppliedSchema(schema)
            for name, value in {
                "exposure": math.log2(snapshot.renderer.exposure_multiplier),
                "exposure:iso": 100.0,
                "exposure:time": 1.0,
                "exposure:fStop": 1.0,
                "exposure:responsivity": 0.8821367311933349,
            }.items():
                prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
            prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)
    # These two values are authored native metadata, then re-read by capture.
    layer_data = dict(stage.GetRootLayer().customLayerData)
    layer_data["unirobosimAmbient"] = list(ambient)
    stage.GetRootLayer().customLayerData = layer_data
    world._appearance_keys = tuple(values)
    world._invalidate_render()
    return AppearanceApplyResult(
        len(materials),
        len(staged),
        len(snapshot.lights),
        (),
        ("Source gamma22 is rendered with native RTX sRGB output transfer",)
        if snapshot.renderer.output_transfer == "gamma22"
        else (),
    )


def capture(world):
    from pxr import Gf, UsdGeom, UsdLux, UsdShade
    from unirobosim import appearance_topology_sha256

    stage = world._m.sim_utils.get_current_stage()
    settings = world._m.carb.settings.get_settings()
    targets = visual_targets(world)
    materials = []
    bindings = []
    lights = []
    limitations = []
    if settings.get("/rtx/sceneDb/ambientLightIntensity"):
        limitations.append("Unrecorded RTX sceneDb global ambient lighting is active")
    if settings.get("/rtx/useViewLightingMode"):
        limitations.append("RTX viewport lighting cannot be captured as scene lights")
    ambient = tuple(stage.GetRootLayer().customLayerData.get("unirobosimAmbient") or (0.0, 0.0, 0.0))
    for (entity, key, env), prim in targets.items():
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            raise ValueError(f"appearance material not bound: {prim.GetPath()}")
        shader, _, _ = material.ComputeSurfaceSource()
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            raise ValueError("only actual UsdPreviewSurface capture is supported")
        material_id = f"{entity}:{key}:env{env}"
        if key == "particles":
            inst = UsdGeom.PointInstancer.Get(stage, prim.GetPath().GetParentPath().AppendChild("particle_visual"))
            prototypes = inst.GetPrototypesRel().GetTargets()
            if not prototypes:
                raise ValueError("particle renderer has no material prototypes")
            signatures = []
            for prototype in prototypes:
                material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prototype)).ComputeBoundMaterial()
                shader, _, _ = material.ComputeSurfaceSource()
                signatures.append((float(_input(shader, "roughness", 0.5)), float(_input(shader, "metallic", 0.0))))
            if len(set(signatures)) != 1:
                raise ValueError("nonuniform particle BRDF is not representable in v1")
        color = _input(shader, "diffuseColor", (0.18, 0.18, 0.18))
        textures = []
        uv = None
        uv_indices = None
        uv_interp = "vertex"
        topology = None
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            topology = appearance_topology_sha256(
                mesh.GetPointsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
            )
        opacity_value = _input(shader, "opacity", 1.0)
        if isinstance(opacity_value, UsdShade.Shader):
            if not isinstance(color, UsdShade.Shader) or opacity_value.GetPath() != color.GetPath():
                raise ValueError("independent opacity texture is unsupported")
            opacity = 1.0
        else:
            opacity = float(opacity_value)
        if isinstance(color, UsdShade.Shader):
            tex = color
            file = _input(tex, "file")
            uri = file.resolvedPath or file.path
            colorspace = _input(tex, "sourceColorSpace", "auto")
            if colorspace not in ("sRGB", "raw"):
                limitations.append(
                    f"Texture color space {colorspace!r} is unresolved at {prim.GetPath()}; "
                    "texture appearance is incomplete, geometry binding remains available"
                )
            else:
                path = Path(uri)
                texture = AppearanceTexture(
                    "base_color",
                    path.as_uri(),
                    "srgb" if colorspace == "sRGB" else "linear",
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                textures.append(texture)
            scale = tuple(_input(tex, "scale", (1.0, 1.0, 1.0, 1.0)))
            color = scale[:3]
            if isinstance(opacity_value, UsdShade.Shader):
                opacity = scale[3]
            mesh = UsdGeom.Mesh(prim)
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if not mesh or not pv:
                raise ValueError("textured mesh lacks st UV")
            uv = tuple(tuple(float(v) for v in x) for x in pv.Get())
            uv_interp = str(pv.GetInterpolation())
            uv_indices = tuple(int(x) for x in pv.GetIndices()) if pv.IsIndexed() else None
            topology = appearance_topology_sha256(
                mesh.GetPointsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
            )
        color_source = "particle_colors" if key == "particles" else "constant"
        if key == "particles":
            # Visual sphere prototypes are the renderer's actual material source.
            inst = UsdGeom.PointInstancer.Get(stage, prim.GetPath().GetParentPath().AppendChild("particle_visual"))
            prototype = inst.GetPrototypesRel().GetTargets()[0]
            material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prototype)).ComputeBoundMaterial()
            shader, _, _ = material.ComputeSurfaceSource()
            color = (1.0, 1.0, 1.0)
            opacity = 1.0
        rgba = tuple(float(v) for v in color) + (opacity,)
        roughness = float(_input(shader, "roughness", 0.5))
        metallic = float(_input(shader, "metallic", 0.0))
        emission = _input(shader, "emissiveColor", (0.0, 0.0, 0.0))
        if not isinstance(emission, UsdShade.Shader) and key != "particles":
            expected = tuple(rgba[i] * ambient[i] / 1133.6 for i in range(3))
            if any(abs(float(emission[i]) - expected[i]) > 1e-6 for i in range(3)):
                raise ValueError("original emissive material cannot be represented by v1 ambient contract")
        materials.append(
            AppearanceMaterial(
                material_id,
                rgba,
                roughness,
                metallic,
                color_source,
                tuple(textures),
                FrozenMap({"native_shader": str(shader.GetPath())}),
            )
        )
        bindings.append(
            AppearanceBinding(
                entity, key, material_id, (env,), str(prim.GetPath()), uv, uv_interp, uv_indices, topology
            )
        )
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        light = UsdLux.LightAPI(prim)
        intensity = float(light.GetIntensityAttr().Get()) * 2 ** float(light.GetExposureAttr().Get())
        color = tuple(float(x) for x in light.GetColorAttr().Get())
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        direction = None
        angle = 0.0
        if prim.IsA(UsdLux.DistantLight):
            kind = "directional"
            angle = float(UsdLux.DistantLight(prim).GetAngleAttr().Get())
            direction = tuple(matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized())
            if not light.GetNormalizeAttr().Get():
                raise ValueError("capture unnormalized distant light requires explicit solid-angle conversion")
            unit = "lux"
        elif prim.IsA(UsdLux.DomeLight):
            kind = "dome"
            unit = "nit"
        else:
            raise ValueError(f"unsupported native light type: {prim.GetTypeName()}")
        light_id = prim.GetCustomDataByKey("unirobosim:light_id") or str(prim.GetPath())
        shadow = UsdLux.ShadowAPI(prim).GetShadowEnableAttr().Get()
        lights.append(
            AppearanceLight(
                light_id,
                kind,
                color,
                intensity,
                unit,
                direction,
                tuple(matrix.ExtractTranslation()),
                shadows=True if shadow is None else bool(shadow),
                angular_diameter_degrees=angle,
                source_parameters=FrozenMap(
                    {
                        "native_path": str(prim.GetPath()),
                        "usdlux_version": prim.GetAttribute("omni:rtx:usdluxVersion").Get(),
                    }
                ),
            )
        )
    camera = next(
        (
            UsdGeom.Camera(x)
            for x in stage.Traverse()
            if x.IsA(UsdGeom.Camera) and str(x.GetPath()).startswith("/World/")
        ),
        None,
    )
    exposure = 1.0
    if camera:
        prim = camera.GetPrim()

        def get(k, default):
            value = prim.GetAttribute(k).Get()
            return value if value is not None else default

        exposure = (
            float(get("exposure:time", 1.0))
            * (float(get("exposure:iso", 100.0)) / 100)
            * 2 ** float(get("exposure", 0.0))
            / float(get("exposure:fStop", 1.0)) ** 2
        )
    op = settings.get("/rtx/post/tonemap/op")
    return AppearanceSnapshot(
        tuple(materials),
        tuple(bindings),
        tuple(lights),
        AppearanceEnvironment(tuple(settings.get("/rtx/background/source/color") or (0.0, 0.0, 0.0)), ambient),
        AppearanceRenderer(
            "isaaclab",
            "rtx",
            output_transfer="srgb",
            tone_mapping="linear" if op == 1 else "clamp" if op == 0 else f"unsupported-native-{op}",
            exposure_multiplier=exposure,
            source_parameters=FrozenMap(
                {
                    "tonemap_operator": op,
                    "rtx_responsivity_correction": 0.8821367311933349,
                    "native_global_ambient_intensity": settings.get("/rtx/sceneDb/ambientLightIntensity") or 0.0,
                    "native_view_lighting": bool(settings.get("/rtx/useViewLightingMode")),
                    "preview_emission_scale": 1133.6,
                }
            ),
        ),
        tuple(limitations),
    )

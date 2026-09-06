"""Anonymous clone admission, effective native geometry and strict-difference gates.

Diagnostic evidence, not whole HG03/HG04 acceptance. No source assets or SDK
methods are modified. The native-only process uses supported fast shutdown.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import struct
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

from unirobosim import (
    CapabilityId,
    CapabilityRequirement,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    PhysicsSpec,
    Pose,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider
from unirobosim_isaaclab.native import IsaacLabNativeRuntime, _native_name
from unirobosim_isaaclab.native_planning import _PlanningAdmission, _stable_id
from unirobosim_isaaclab.native_protocols import NativePlanningError

MESH = """
point3f[] points = [(-0.1,-0.1,-0.1),(0.1,-0.1,-0.1),(0.1,0.1,-0.1),(-0.1,0.1,-0.1),
(-0.1,-0.1,0.1),(0.1,-0.1,0.1),(0.1,0.1,0.1),(-0.1,0.1,0.1)]
int[] faceVertexCounts = [4,4,4,4,4,4]
int[] faceVertexIndices = [0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
uniform token subdivisionScheme = "none"
uniform token physics:approximation = "convexHull"
"""


def make_usd(child, *, nested=False, instance_proxy=False):
    geometry = (
        (
            'def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {\n'
            "double3 xformOp:translate = (0.2,-0.1,0.3)\n"
            'uniform token[] xformOpOrder = ["xformOp:translate"]\nfloat physics:mass = 1\n'
            'def Mesh "Shape" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {\n'
            "double3 xformOp:translate = (0.1,0.2,0.3)\n"
            'uniform token[] xformOpOrder = ["xformOp:translate"]\n' + MESH + "\n}\n}"
        )
        if child
        else (
            'def Mesh "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", '
            '"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {\n'
            "double3 xformOp:translate = (0.2,-0.1,0.3)\n"
            'uniform token[] xformOpOrder = ["xformOp:translate"]\nfloat physics:mass = 1\n' + MESH + "\n}"
        )
    )
    if instance_proxy:
        geometry = (
            'def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {\n'
            "float physics:mass = 1\n"
            'def Xform "Carrier" (prepend references = @mesh.usda@; instanceable = true) {\n'
            "double3 xformOp:translate = (0.1,0.2,0.3)\n"
            'uniform token[] xformOpOrder = ["xformOp:translate"]\n}\n}'
        )
    if nested:
        geometry = (
            'def Xform "Ancestor" {\ndouble3 xformOp:scale = (1.5,1.2,0.8)\n'
            'uniform token[] xformOpOrder = ["xformOp:scale"]\n' + geometry + "\n}"
        )
    return '#usda 1.0\n(defaultPrim="Asset"; metersPerUnit=1; upAxis="Z")\ndef Xform "Asset" {\n' + geometry + "\n}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--environments", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(60)
    result = {
        "error": None,
        "checks": {},
        "process_profile": "native helper; fast_shutdown=True",
    }

    class DiagnosticRuntime(IsaacLabNativeRuntime):
        def build_world(self, spec):
            try:
                return super().build_world(spec)
            except Exception as error:  # noqa: BLE001 -- preserve cause before public sanitization
                result["native_build_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                raise

    with tempfile.TemporaryDirectory(prefix="isaac-effective-collision-") as directory:
        mesh_asset = Path(directory) / "mesh.usda"
        mesh_asset.write_text(
            '#usda 1.0\n(defaultPrim="Geometry"; metersPerUnit=1; upAxis="Z")\n'
            'def Xform "Geometry" {\n'
            'def Mesh "Shape" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {\n'
            + MESH
            + "\n}\n}\n",
            encoding="utf-8",
        )
        assets = []
        cases = (
            ("self_mesh", False, False, False),
            ("child_mesh", True, False, False),
            ("nested_self", False, True, False),
            ("nested_child", True, True, False),
            ("instance_child", True, False, True),
        )
        paths = tuple(EntityPath("/" + name) for name, *_ in cases)
        for name, child, nested, instance_proxy in cases:
            asset = Path(directory) / (name + ".usda")
            asset.write_text(make_usd(child, nested=nested, instance_proxy=instance_proxy), encoding="utf-8")
            assets.append(asset)
        spec = WorldSpec(
            "effective-mesh",
            tuple(
                EntitySpec(
                    path,
                    EntityKind.RIGID_BODY,
                    asset_uri=str(asset),
                    pose=Pose((index * 4.0, 0, 10), (0, 0, math.sin(0.3), math.cos(0.3))),
                )
                for index, (path, asset) in enumerate(zip(paths, assets, strict=True))
            ),
            environments=EnvironmentSpec(args.environments),
            physics=PhysicsSpec(gravity_m_s2=(0, 0, 0)),
            requirements=(CapabilityRequirement(CapabilityId("planning.scene@2")),),
        )
        provider = IsaacLabProvider(
            IsaacLabAdapterConfig(environment_spacing_m=20),
            runtime_factory=lambda config: DiagnosticRuntime(config, process_isolated=True),
        )
        print("DIAGNOSTIC opening native runtime", flush=True)
        with provider.open() as session:
            faulthandler.cancel_dump_traceback_later()
            print("DIAGNOSTIC native runtime ready", flush=True)
            try:
                world = session.build(spec)
                native = world._native
                modules = native._m
                result["usd"] = {
                    "path": modules.Usd.__file__,
                    "version": modules.Usd.GetVersion(),
                }
                stage = modules.sim_utils.get_current_stage()
                result["catalog"] = asdict(world.planning_scene_catalog())
                result["native_colliders"] = []
                catalog = world.planning_scene_catalog()
                geometries = {value.geometry_id: value for value in catalog.geometries}
                for env in range(args.environments):
                    state = world.planning_scene_state(env)
                    transforms = {value.geometry_id: value.world_pose for value in state.geometry_transforms}
                    for index, path in enumerate(paths):
                        root = stage.GetPrimAtPath(f"/World/env_{env}/{_native_name(path)}")
                        colliders = [
                            prim
                            for prim in modules.Usd.PrimRange(root, modules.Usd.TraverseInstanceProxies())
                            if prim.HasAPI(modules.UsdPhysics.CollisionAPI)
                        ]
                        result["checks"][f"{path.value}.env{env}.one_effective_collider"] = len(colliders) == 1
                        for prim in colliders:
                            zero_path = str(prim.GetPath()).replace(f"/env_{env}/", "/env_0/")
                            geometry = geometries[_stable_id("geometry", path.value, zero_path)]
                            lease = world.resolve_planning_geometry(geometry.geometry_id, environment_index=env)
                            try:
                                content = lease.read()
                            finally:
                                lease.close()
                            count = geometry.resource_layout.vertex_shape[0]
                            vertices = list(struct.iter_unpack("<fff", content[: count * 12]))
                            pose = transforms[geometry.geometry_id]
                            rotation = modules.Gf.Quatd(
                                pose.orientation_xyzw[3],
                                modules.Gf.Vec3d(*pose.orientation_xyzw[:3]),
                            )
                            published = [
                                rotation.Transform(
                                    modules.Gf.Vec3d(*(point[axis] * geometry.scale[axis] for axis in range(3)))
                                )
                                + modules.Gf.Vec3d(*pose.position_m)
                                for point in vertices
                            ]
                            matrix = modules.UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                            source_points = modules.UsdGeom.Mesh(prim).GetPointsAttr().Get()
                            effective = [
                                matrix.Transform(point) - modules.Gf.Vec3d(*native._origins_cpu[env])
                                for point in source_points
                            ]
                            max_distance = max(
                                max(min((point - other).GetLength() for other in published) for point in effective),
                                max(min((point - other).GetLength() for other in effective) for point in published),
                            )
                            result["checks"][f"{path.value}.env{env}.cooked_vertices_effective_world"] = (
                                len(vertices) == len(source_points) == 8 and max_distance < 2e-5
                            )
                            result["checks"][f"{path.value}.env{env}.instance_proxy"] = (
                                prim.IsInstanceProxy() == cases[index][3]
                            )
                            expected_scale = (1.5, 1.2, 0.8) if cases[index][2] else (1.0, 1.0, 1.0)
                            result["checks"][f"{path.value}.env{env}.ancestor_scale_preserved"] = all(
                                abs(actual - expected) < 1e-6
                                for actual, expected in zip(geometry.scale, expected_scale, strict=True)
                            )
                            result["native_colliders"].append(
                                {
                                    "path": str(prim.GetPath()),
                                    "schemas": prim.GetAppliedSchemas(),
                                    "instance_proxy": prim.IsInstanceProxy(),
                                    "scale": geometry.scale,
                                    "max_vertex_distance_m": max_distance,
                                    "vertex_count": count,
                                    "planning_pose": asdict(pose),
                                }
                            )
                result["checks"]["clone_geometry_catalog_equal"] = all(
                    world.planning_scene_catalog(0).geometries == world.planning_scene_catalog(env).geometries
                    for env in range(args.environments)
                )
                if args.environments == 2:
                    # Only anonymous in-memory clone opinions are changed below,
                    # after the public positive gate. No step/forward follows.
                    # Re-run the actual strict verifier, with a fresh admission
                    # object for each case so no stale query cache hides changes.
                    result["negative_control_profile"] = "native stage edits; private strict admission; no physics step"
                    child_root = f"/World/env_1/{_native_name(paths[1])}"
                    nested_root = f"/World/env_1/{_native_name(paths[3])}"
                    controls = (
                        (
                            "translation",
                            child_root + "/Body/Shape",
                            "xformOp:translate",
                            modules.Gf.Vec3d(0.12, 0.2, 0.3),
                        ),
                        ("mesh_points", child_root + "/Body/Shape", "points", None),
                        ("ancestor_scale", nested_root + "/Ancestor", "xformOp:scale", modules.Gf.Vec3d(1.6, 1.2, 0.8)),
                    )
                    for name, prim_path, attribute_name, changed in controls:
                        prim = stage.GetPrimAtPath(prim_path)
                        attribute = prim.GetAttribute(attribute_name)
                        original = attribute.Get()
                        if name == "mesh_points":
                            changed = list(original)
                            changed[0] = modules.Gf.Vec3f(-0.12, -0.1, -0.1)
                        attribute.Set(changed)
                        try:
                            try:
                                _PlanningAdmission(native)
                            except NativePlanningError as error:
                                frames = traceback.extract_tb(error.__traceback__)
                                result.setdefault("rejections", {})[name] = {
                                    "code": str(error),
                                    "frames": [
                                        {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                        for frame in frames
                                    ],
                                }
                                result["checks"][f"different_clone.{name}.rejected"] = (
                                    str(error) == "collision_geometry_unsupported"
                                    and frames[-1].name == "_verify_cloned_environments"
                                )
                            else:
                                result["checks"][f"different_clone.{name}.rejected"] = False
                        finally:
                            attribute.Clear()
                        result["checks"][f"different_clone.{name}.restored"] = attribute.Get() == original
                    # Restoration must preserve the exact positive guard, too.
                    _PlanningAdmission(native)
                    result["checks"]["restored_clones_readmit"] = True
            except Exception:  # noqa: BLE001 -- persist diagnostic failure before native fast shutdown
                result["error"] = traceback.format_exc()
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps({"error": result["error"], "checks": result["checks"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()

"""Native contact-response readback and rest-distance evidence; no asset edits."""

from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path

from unirobosim import CapabilityId, CapabilityRequirement, EntityKind, EntityPath, EntitySpec, Pose, WorldSpec

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider
from unirobosim_isaaclab.native import IsaacLabNativeRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"passed": False, "cases": []}
    try:
        with tempfile.TemporaryDirectory(prefix="contact-offset-native-") as td:
            d = Path(td)
            floor = d / "floor.usda"
            floor.write_text("""#usda 1.0
(defaultPrim="Floor"; metersPerUnit=1; upAxis="Z")
def Xform "Floor" (prepend apiSchemas=["PhysicsRigidBodyAPI"]) {
 bool physics:kinematicEnabled=true
 def Cube "Collider" (prepend apiSchemas=["PhysicsCollisionAPI","PhysxCollisionAPI"]) {
 double size=1
 double3 xformOp:scale=(20,20,0.1)
 double3 xformOp:translate=(0,0,-0.05)
 uniform token[] xformOpOrder=["xformOp:translate","xformOp:scale"]
 float physxCollision:contactOffset=0.02
 float physxCollision:restOffset=0
}
}
""")
            specs = [EntitySpec(EntityPath("/floor"), EntityKind.RIGID_BODY, asset_uri=str(floor))]
            settings = [
                ("zero", 0.0005, 0.0),
                ("legacy_zero", 0.0, 0.0),
                ("positive", 0.02, 0.005),
                ("negative", 0.02, -0.005),
                ("automatic", float("-inf"), float("-inf")),
            ]
            for i, (name, contact, rest) in enumerate(settings):
                f = d / (name + ".usda")
                f.write_text(
                    """#usda 1.0
(defaultPrim="Body"; metersPerUnit=1; upAxis="Z")
def Xform "Body" (prepend apiSchemas=["PhysicsRigidBodyAPI","PhysicsMassAPI"]) {
 float physics:mass=1
 def Cube "Collider" (prepend apiSchemas=["PhysicsCollisionAPI","PhysxCollisionAPI"]) {
 double size=0.1
 float physxCollision:contactOffset="""
                    + str(contact)
                    + """
 float physxCollision:restOffset="""
                    + str(rest)
                    + """\n}\n}\n"""
                )
                specs.append(
                    EntitySpec(
                        EntityPath("/" + name), EntityKind.RIGID_BODY, asset_uri=str(f), pose=Pose((i * 1.0, 0, 0.3))
                    )
                )
            provider = IsaacLabProvider(
                IsaacLabAdapterConfig(), runtime_factory=lambda c: IsaacLabNativeRuntime(c, process_isolated=True)
            )
            with provider.open() as session:
                try:
                    world = session.build(
                        WorldSpec(
                            "offset-native",
                            tuple(specs),
                            requirements=(CapabilityRequirement(CapabilityId("planning.scene@2")),),
                        )
                    )
                    native = world._native
                    catalog = world.planning_scene_catalog()
                    result["geometry_count"] = len(catalog.geometries)
                    result["stage_meters_per_unit"] = native._m.UsdGeom.GetStageMetersPerUnit(
                        native._m.sim_utils.get_current_stage()
                    )
                    for _ in range(240):
                        world.step(1)
                    for name, contact, rest in settings:
                        asset = native._rigids[EntityPath("/" + name)]
                        view = asset.root_physx_view

                        def values(x):
                            if hasattr(x, "cpu"):
                                x = x.cpu()
                            if hasattr(x, "numpy"):
                                x = x.numpy()
                            return x.tolist()

                        result["cases"].append(
                            {
                                "name": name,
                                "authored_contact": contact if contact != float("-inf") else "auto",
                                "authored_rest": rest if rest != float("-inf") else "auto",
                                "physx_contact": values(view.get_contact_offsets()),
                                "physx_rest": values(view.get_rest_offsets()),
                                "root_pose": values(asset.data.root_link_pose_w.torch),
                            }
                        )
                    for row in result["cases"]:
                        if row["name"] != "automatic":
                            if row["name"] == "legacy_zero":
                                assert row["physx_contact"][0][0] > 0.0
                            else:
                                assert abs(row["physx_contact"][0][0] - row["authored_contact"]) < 1e-7
                            assert abs(row["physx_rest"][0][0] - row["authored_rest"]) < 1e-7
                            assert abs(row["root_pose"][0][2] - (0.05 + row["authored_rest"])) < 0.002
                    result["passed"] = True
                except Exception as e:
                    result["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
                finally:
                    Path(args.output).write_text(json.dumps(result, indent=2))
                    print("OFFSET_RESULT " + json.dumps(result), flush=True)
    except Exception as e:
        result["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("OFFSET_RESULT " + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

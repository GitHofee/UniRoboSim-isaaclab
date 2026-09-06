"""Public isolated-worker articulation same-tick state regression.

Run with an Isaac Lab interpreter and the adapter/Core sources on PYTHONPATH:
  python scripts/articulation_state_conformance.py --output /tmp/state-result.json
Temporary anonymous USD only; no SDK monkeypatch or source-asset edits.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

from unirobosim import (
    ArrayValue,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    KinematicTarget,
    PhysicsSpec,
    Pose,
    RenderArticulationState,
    RenderStateFrame,
    SceneCommand,
    SceneCommandKind,
    SceneCommandStatus,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider

USD = """#usda 1.0
(defaultPrim = "Robot"; metersPerUnit = 1; upAxis = "Z")
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Cube "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"])
    {
        double size = 0.2
        double3 xformOp:translate = (0.2, -0.1, 0.3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 1
        point3f physics:centerOfMass = (0.03, 0.02, 0.01)
    }
    def Cube "child" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"])
    {
        double size = 0.2
        double3 xformOp:translate = (0.2, -0.1, 0.7)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 1
    }
    def PhysicsRevoluteJoint "hinge" (prepend apiSchemas = ["PhysicsDriveAPI:angular"])
    {
        rel physics:body0 = </Robot/base>
        rel physics:body1 = </Robot/child>
        uniform token physics:axis = "Y"
        point3f physics:localPos0 = (0, 0, 0.2)
        point3f physics:localPos1 = (0, 0, -0.2)
        float physics:lowerLimit = -90
        float physics:upperLimit = 90
        float drive:angular:physics:stiffness = 100
        float drive:angular:physics:damping = 10
        float drive:angular:physics:maxForce = 100
    }
}
"""


def vectors_close(actual, expected, tolerance=2e-5):
    return len(actual) == len(expected) and all(abs(a - b) <= tolerance for a, b in zip(actual, expected, strict=True))


def run(world, gravity):
    path = EntityPath("/robot")
    targets = (
        KinematicTarget("root", path),
        KinematicTarget("base", path, "base"),
        KinematicTarget("child", path, "child"),
    )

    def sample():
        return [[asdict(value) for value in world.read_selected_kinematics(targets, env)] for env in (0, 1)]

    result = {"checks": {}, "spawn": sample()}
    world.step(12)
    result["set_pose_before"] = sample()
    tick = world.tick
    outcome = world.apply_scene_command(
        SceneCommand(
            "pose",
            "audit",
            "audit",
            world.generation,
            SceneCommandKind.SET_POSE,
            path,
            environment_index=0,
            target_pose=Pose((0, 0, 15), (0, 0, math.sqrt(0.5), math.sqrt(0.5))),
        )
    )
    after = result["set_pose_after"] = sample()
    result["checks"]["set_pose_applied_without_step"] = (
        outcome.status is SceneCommandStatus.APPLIED and world.tick == tick
    )
    result["checks"]["set_pose_three_link_velocities"] = all(
        vectors_close(row["linear_velocity_m_s"], (0, 0, 0)) and vectors_close(row["angular_velocity_rad_s"], (0, 0, 0))
        for row in after[0]
    )
    result["checks"]["set_pose_repeat"] = sample() == after
    result["checks"]["set_pose_unselected"] = after[1] == result["set_pose_before"][1]
    world.step()
    result["set_pose_next"] = sample()
    result["checks"]["set_pose_physical_next"] = vectors_close(
        result["set_pose_next"][0][0]["linear_velocity_m_s"], (0, 0, gravity / 120)
    )

    # Existing render-state input goes through the SDK COM writer. Check the
    # public link velocity using v_link = v_com - omega x R * local_COM.
    before = result["render_before"] = sample()
    tick = world.tick
    world.apply_render_state(
        RenderStateFrame(
            articulations=(
                RenderArticulationState(
                    world.resolve(path),
                    ArrayValue.from_rows(((0.0,),)),
                    ArrayValue.from_rows(((0.0,),)),
                    environment_indices=(0,),
                    root_positions_m=ArrayValue.from_rows(((0.2, -0.1, 15.3),)),
                    root_orientations_xyzw=ArrayValue.from_rows(((0, 0, math.sqrt(0.5), math.sqrt(0.5)),)),
                    root_linear_velocities_m_s=ArrayValue.from_rows(((0.4, -0.3, 0.2),)),
                    root_angular_velocities_rad_s=ArrayValue.from_rows(((0.2, -0.1, 0.7),)),
                ),
            )
        )
    )
    after = result["render_after"] = sample()
    expected = ((0.422, -0.284, 0.196), (0.422, -0.284, 0.196), (0.382, -0.364, 0.196))
    result["checks"]["render_three_rotated_com_link_velocities"] = all(
        vectors_close(row["linear_velocity_m_s"], value)
        and vectors_close(row["angular_velocity_rad_s"], (0.2, -0.1, 0.7))
        for row, value in zip(after[0], expected, strict=True)
    )
    result["checks"]["render_no_step"] = world.tick == tick
    result["checks"]["render_repeat"] = sample() == after
    result["checks"]["render_unselected"] = after[1] == before[1]

    world.step(3)
    result["reset_before"] = sample()
    tick = world.tick
    world.reset((0,))
    after = result["reset_after"] = sample()
    result["checks"]["reset_three_link_velocities"] = all(
        vectors_close(row["linear_velocity_m_s"], (0, 0, 0)) and vectors_close(row["angular_velocity_rad_s"], (0, 0, 0))
        for row in after[0]
    )
    result["checks"]["reset_repeat"] = sample() == after
    result["checks"]["reset_no_step"] = world.tick == tick
    result["checks"]["reset_unselected"] = after[1] == result["reset_before"][1]
    world.step()
    result["reset_next"] = sample()
    result["checks"]["reset_physical_next"] = vectors_close(
        result["reset_next"][0][0]["linear_velocity_m_s"], (0, 0, gravity / 120)
    )
    # Angular-zero checkpoint isolates timestamp coherence from the independently
    # tracked SDK link-writer/COM physical-buffer defect at nonzero angular speed.
    world.apply_render_state(
        RenderStateFrame(
            articulations=(
                RenderArticulationState(
                    world.resolve(path),
                    ArrayValue.from_rows(((0.0,),)),
                    ArrayValue.from_rows(((0.0,),)),
                    environment_indices=(0,),
                    root_positions_m=ArrayValue.from_rows(((0.2, -0.1, 15.3),)),
                    root_orientations_xyzw=ArrayValue.from_rows(((0, 0, math.sqrt(0.5), math.sqrt(0.5)),)),
                    root_linear_velocities_m_s=ArrayValue.from_rows(((0.4, -0.3, 0.2),)),
                    root_angular_velocities_rad_s=ArrayValue.from_rows(((0.0, 0.0, 0.0),)),
                ),
            )
        )
    )
    # Positive step makes the baseline capture unambiguously fresh too.
    world.step()
    result["checkpoint_before"] = sample()
    checkpoint = world.create_checkpoint()
    world.step()
    result["checkpoint_reference_next"] = sample()
    world.apply_scene_command(
        SceneCommand(
            "checkpoint-mutation",
            "audit",
            "audit",
            world.generation,
            SceneCommandKind.SET_POSE,
            path,
            environment_index=0,
            target_pose=Pose((2, 1, 12), (0, math.sqrt(0.5), 0, math.sqrt(0.5))),
        )
    )
    result["checkpoint_mutated"] = sample()
    world.restore_checkpoint(checkpoint)
    after = result["checkpoint_after"] = sample()
    fields = ("linear_velocity_m_s", "angular_velocity_rad_s")
    result["checks"]["checkpoint_three_link_velocities"] = all(
        vectors_close(row[field], saved[field])
        for rows, saved_rows in zip(after, result["checkpoint_before"], strict=True)
        for row, saved in zip(rows, saved_rows, strict=True)
        for field in fields
    )
    result["checks"]["checkpoint_three_link_poses"] = all(
        vectors_close(row["pose"][field], saved["pose"][field])
        for rows, saved_rows in zip(after, result["checkpoint_before"], strict=True)
        for row, saved in zip(rows, saved_rows, strict=True)
        for field in ("position", "orientation_xyzw")
    )
    result["checks"]["checkpoint_repeat"] = sample() == after
    world.step()
    result["checkpoint_replay_next"] = sample()
    result["checks"]["checkpoint_physical_replay"] = all(
        vectors_close(row[field], saved[field])
        for rows, saved_rows in zip(result["checkpoint_replay_next"], result["checkpoint_reference_next"], strict=True)
        for row, saved in zip(rows, saved_rows, strict=True)
        for field in fields
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    result = {"error": None, "gravity": args.gravity}
    try:
        with tempfile.TemporaryDirectory(prefix="isaac-state-conformance-") as directory:
            asset = Path(directory) / "robot.usda"
            asset.write_text(USD, encoding="utf-8")
            spec = WorldSpec(
                "articulation-state",
                (
                    EntitySpec(
                        EntityPath("/robot"),
                        EntityKind.ARTICULATION,
                        asset_uri=str(asset),
                        pose=Pose((0, 0, 10)),
                        joint_names=("hinge",),
                        initial_joint_positions=(0.0,),
                    ),
                ),
                environments=EnvironmentSpec(2),
                physics=PhysicsSpec(time_step_seconds=1 / 120, gravity_m_s2=(0, 0, args.gravity)),
            )
            with IsaacLabProvider(IsaacLabAdapterConfig(environment_spacing_m=20)).open() as session:
                result.update(run(session.build(spec), args.gravity))
        if args.cache_only:
            result["checks"] = {
                name: value for name, value in result["checks"].items() if not name.startswith("reset_")
            }
    except Exception:
        result["error"] = traceback.format_exc()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["error"] or not all(result.get("checks", {}).values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Public-worker checkpoint replay with native-effective COM frame controls.

Run in an Isaac Lab environment, adapter/Core on PYTHONPATH. Temporary anonymous
USD only. Existing checkpoint link-velocity semantics must survive restore.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

from articulation_state_conformance import USD, vectors_close
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
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider


def run(world, cases):
    states = []
    for path, kind, translation, index in cases:
        angular = ((0.2, -0.1, 0.7), (-0.3, 0.4, -0.6)) if kind != "offset_linear" else ((0, 0, 0),) * 2
        states.append(
            RenderArticulationState(
                world.resolve(path),
                ArrayValue.from_rows(((0.12,), (-0.18,)) if kind != "offset_linear" else ((0.0,),) * 2),
                ArrayValue.from_rows(((0.2,), (-0.15,)) if kind != "offset_linear" else ((0.0,),) * 2),
                root_positions_m=ArrayValue.from_rows(((translation + index * 2 + 0.2, -0.1, 10.3),) * 2),
                root_orientations_xyzw=ArrayValue.from_rows(
                    ((0, 0, math.sin(0.3), math.cos(0.3)), (math.sin(0.2), 0, 0, math.cos(0.2)))
                ),
                root_linear_velocities_m_s=ArrayValue.from_rows(((0.4, -0.3, 0.2), (-0.2, 0.1, -0.4))),
                root_angular_velocities_rad_s=ArrayValue.from_rows(angular),
            )
        )
    world.apply_render_state(RenderStateFrame(articulations=tuple(states)))

    def sample():
        values = {}
        for path, _kind, _translation, _index in cases:
            articulation = world.read_articulation(world.resolve(path))
            values[path.value] = {
                "joint_positions": articulation.joint_positions.rows(),
                "joint_velocities": articulation.joint_velocities.rows(),
                "links": [
                    [
                        asdict(value)
                        for value in world.read_selected_kinematics(
                            (
                                KinematicTarget("root", path),
                                KinematicTarget("base", path, "base"),
                                KinematicTarget("child", path, "child"),
                            ),
                            env,
                        )
                    ]
                    for env in (0, 1)
                ],
            }
        return values

    result = {"checks": {}}

    def compare(label, actual, expected):
        for path, _kind, translation, _index in cases:
            for env in (0, 1):
                prefix = f"{label}.{path.value}.env{env}"
                for field in ("joint_positions", "joint_velocities"):
                    result["checks"][f"{prefix}.{field}"] = vectors_close(
                        actual[path.value][field][env], expected[path.value][field][env], tolerance=1e-4
                    )
                for link, reference in zip(
                    actual[path.value]["links"][env], expected[path.value]["links"][env], strict=True
                ):
                    name = link["target_id"]
                    # A large global translation has float32 position spacing;
                    # velocity/orientation tolerances remain unchanged.
                    result["checks"][f"{prefix}.{name}.position"] = vectors_close(
                        link["pose"]["position"],
                        reference["pose"]["position"],
                        tolerance=max(1e-4, abs(translation) * 2e-7),
                    )
                    result["checks"][f"{prefix}.{name}.orientation"] = vectors_close(
                        link["pose"]["orientation_xyzw"],
                        reference["pose"]["orientation_xyzw"],
                        tolerance=1e-4,
                    )
                    for field in ("linear_velocity_m_s", "angular_velocity_rad_s"):
                        result["checks"][f"{prefix}.{name}.{field}"] = vectors_close(
                            link[field], reference[field], tolerance=1e-4
                        )

    world.step(3)
    result["saved"] = sample()
    checkpoint = world.create_checkpoint()
    world.step()
    result["reference_next"] = sample()
    for repeat in range(2):
        world.step(5)
        # Repeat0 explicitly primes every read before restore. Repeat1 leaves
        # positive-step caches fresh/unread until restore's own preflight.
        if repeat == 0:
            result["mutated"] = sample()
            result["checks"]["nontrivial_mutation"] = any(
                not vectors_close(
                    result["mutated"][path.value]["joint_positions"][0],
                    result["saved"][path.value]["joint_positions"][0],
                    tolerance=1e-5,
                )
                for path, *_ in cases
            )
        tick = world.tick
        world.restore_checkpoint(checkpoint)
        result["checks"][f"repeat{repeat}.clock_preserved"] = world.tick == tick
        same_tick = result[f"restored{repeat}"] = sample()
        compare(f"restore{repeat}", same_tick, result["saved"])
        result["checks"][f"repeat{repeat}.cached_repeat"] = sample() == same_tick
        world.step()
        replay = result[f"replay_next{repeat}"] = sample()
        compare(f"next{repeat}", replay, result["reference_next"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--translations", default="0,1000,100000")
    args = parser.parse_args()
    result = {"error": None, "checks": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="isaac-checkpoint-frame-") as directory:
            offset = Path(directory) / "offset.usda"
            identity = Path(directory) / "identity.usda"
            offset.write_text(USD, encoding="utf-8")
            identity.write_text(USD.replace("(0.03, 0.02, 0.01)", "(0, 0, 0)"), encoding="utf-8")
            cases = tuple(
                (EntityPath(f"/t{group}_{kind}"), kind, float(translation), index)
                for group, translation in enumerate(args.translations.split(","))
                for index, kind in enumerate(("offset_angular", "identity_angular", "offset_linear"))
            )
            spec = WorldSpec(
                "checkpoint-inertia",
                tuple(
                    EntitySpec(
                        path,
                        EntityKind.ARTICULATION,
                        asset_uri=str(identity if kind == "identity_angular" else offset),
                        pose=Pose((translation + index * 2, 0, 10)),
                        joint_names=("hinge",),
                        initial_joint_positions=(0.0,),
                    )
                    for path, kind, translation, index in cases
                ),
                environments=EnvironmentSpec(2),
                physics=PhysicsSpec(time_step_seconds=1 / 120, gravity_m_s2=(0, 0, 0)),
            )
            with IsaacLabProvider(IsaacLabAdapterConfig(environment_spacing_m=20)).open() as session:
                result.update(run(session.build(spec), cases))
    except Exception:
        result["error"] = traceback.format_exc()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "error": result["error"],
                "checks": len(result["checks"]),
                "failed": [key for key, value in result["checks"].items() if not value],
            },
            indent=2,
        )
    )
    if result["error"] or not result["checks"] or not all(result["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Native SDK getter closure under public writes, in an owned fast-shutdown process.

Unlike the default public-worker regression, this deliberately reads SDK buffers
and changes in-memory SDK defaults for native controls. It never edits SDK code.
Kit fast shutdown ends this dedicated process: JSON is flushed before close,
and acceptance requires both exit0 and every JSON check true (not exit0 alone).
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from pathlib import Path

from articulation_state_conformance import USD, vectors_close
from unirobosim import (
    ArrayValue,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    PhysicsSpec,
    Pose,
    RenderArticulationState,
    RenderStateFrame,
    SceneCommand,
    SceneCommandKind,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider
from unirobosim_isaaclab.native import IsaacLabNativeRuntime


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rotate(q, p):
    t = [2 * value for value in cross(q[:3], p)]
    c = cross(q[:3], t)
    return [p[i] + q[3] * t[i] + c[i] for i in range(3)]


def compose_pose(link, local):
    rotated = rotate(link[3:], local[:3])
    a, b = link[3:], local[3:]
    c = cross(a[:3], b[:3])
    q = [a[3] * b[i] + b[3] * a[i] + c[i] for i in range(3)]
    q.append(a[3] * b[3] - sum(a[i] * b[i] for i in range(3)))
    return [link[i] + rotated[i] for i in range(3)] + q


def link_velocity(pose, local, com_velocity):
    offset = rotate(pose[3:], local[:3])
    shifted = cross(com_velocity[3:], offset)
    return [com_velocity[i] - shifted[i] for i in range(3)] + com_velocity[3:]


def flatten(value):
    for item in value:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def sample(asset):
    data = asset.data
    # Public getters are deliberately read before independent native truth;
    # checking truth must not accidentally refresh the SDK caches for the test.
    names = (
        "root_link_pose_w",
        "root_com_pose_w",
        "root_com_vel_w",
        "root_link_vel_w",
        "body_link_pose_w",
        "body_com_pose_w",
        "body_com_vel_w",
        "body_link_vel_w",
        "root_state_w",
        "root_link_state_w",
        "root_com_state_w",
        "body_state_w",
        "body_link_state_w",
        "body_com_state_w",
        "projected_gravity_b",
        "heading_w",
        "root_link_lin_vel_b",
        "root_link_ang_vel_b",
        "root_com_lin_vel_b",
        "root_com_ang_vel_b",
    )
    actual = {name: getattr(data, name).torch.detach().cpu().tolist() for name in names}
    root_pose = asset.root_view.get_root_transforms().numpy().tolist()
    root_com_vel = asset.root_view.get_root_velocities().numpy().tolist()
    body_pose = asset.root_view.get_link_transforms().numpy().tolist()
    body_com_vel = asset.root_view.get_link_velocities().numpy().tolist()
    local = data.body_com_pose_b.torch.detach().cpu().tolist()
    expected = {name: [] for name in names}
    for env in range(len(root_pose)):
        rp, rc, local_row = root_pose[env], root_com_vel[env], local[env]
        root_link_vel = link_velocity(rp, local_row[0], rc)
        root_com_pose = compose_pose(rp, local_row[0])
        body_link_vel = [
            link_velocity(p, c, v) for p, c, v in zip(body_pose[env], local_row, body_com_vel[env], strict=True)
        ]
        body_com_pose = [compose_pose(p, c) for p, c in zip(body_pose[env], local_row, strict=True)]
        inverse_q = [-rp[3], -rp[4], -rp[5], rp[6]]
        forward = rotate(rp[3:], (1, 0, 0))
        values = {
            "root_link_pose_w": rp,
            "root_com_pose_w": root_com_pose,
            "root_com_vel_w": rc,
            "root_link_vel_w": root_link_vel,
            "body_link_pose_w": body_pose[env],
            "body_com_pose_w": body_com_pose,
            "body_com_vel_w": body_com_vel[env],
            "body_link_vel_w": body_link_vel,
            "root_state_w": rp + rc,
            "root_link_state_w": rp + root_link_vel,
            "root_com_state_w": root_com_pose + rc,
            "body_state_w": [p + v for p, v in zip(body_pose[env], body_com_vel[env], strict=True)],
            "body_link_state_w": [p + v for p, v in zip(body_pose[env], body_link_vel, strict=True)],
            "body_com_state_w": [p + v for p, v in zip(body_com_pose, body_com_vel[env], strict=True)],
            "projected_gravity_b": rotate(inverse_q, (0, 0, -1)),
            "heading_w": math.atan2(forward[1], forward[0]),
            "root_link_lin_vel_b": rotate(inverse_q, root_link_vel[:3]),
            "root_link_ang_vel_b": rotate(inverse_q, root_link_vel[3:]),
            "root_com_lin_vel_b": rotate(inverse_q, rc[:3]),
            "root_com_ang_vel_b": rotate(inverse_q, rc[3:]),
        }
        for name, value in values.items():
            expected[name].append(value)
    return {
        "actual": actual,
        "expected": expected,
        "timestamp": data._sim_timestamp,
        "checks": {name: vectors_close(list(flatten(actual[name])), list(flatten(expected[name]))) for name in names},
    }


def run(world, result):
    path = EntityPath("/robot")
    native = world._native
    asset = native._articulations[path]
    result["initial"] = sample(asset)
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
    result["render"] = sample(asset)
    world.apply_scene_command(
        SceneCommand(
            "pose",
            "audit",
            "audit",
            world.generation,
            SceneCommandKind.SET_POSE,
            path,
            environment_index=0,
            # Keep the transformed forward axis away from vertical: yaw of a
            # vertical axis is undefined and float32 rounding dominates atan2.
            target_pose=Pose((0, 0, 15), (math.sin(math.pi / 8), 0, 0, math.cos(math.pi / 8))),
        )
    )
    result["set_pose"] = sample(asset)
    defaults = [[0.15, -0.25, 0.35, 0.45, -0.55, 0.65], [-0.1, 0.2, -0.3, -0.4, 0.5, -0.6]]
    asset.data.default_root_vel.torch[:] = native._m.torch.tensor(defaults, device=native._sim.device)
    # Explicitly native external-state fixture before a public selective reset.
    external = [[0.4, -0.3, 0.2, 0.1, 0.2, 0.3], [-0.3, 0.2, -0.1, -0.2, -0.3, -0.4]]
    asset.write_root_com_velocity_to_sim_index(
        root_velocity=native._m.torch.tensor(external, device=native._sim.device)
    )
    world.step(3)
    result["before_reset"] = sample(asset)
    world.reset((0,))
    result["reset"] = sample(asset)
    result["default_controls"] = {
        "selected_nonzero_default": vectors_close(result["reset"]["actual"]["root_com_vel_w"][0], defaults[0]),
        "unselected_unchanged": vectors_close(
            result["reset"]["actual"]["root_com_vel_w"][1], result["before_reset"]["actual"]["root_com_vel_w"][1]
        ),
    }
    world.step()
    result["nonzero_default_next"] = sample(asset)
    result["default_controls"]["nonzero_default_survives_physical_step"] = (
        sum(value * value for value in result["nonzero_default_next"]["actual"]["root_com_vel_w"][0]) > 0.1
    )
    world.reset((1, 0))
    result["reset_reversed"] = sample(asset)
    result["default_controls"]["reversed_all_environment_defaults"] = all(
        vectors_close(actual, expected)
        for actual, expected in zip(result["reset_reversed"]["actual"]["root_com_vel_w"], defaults, strict=True)
    )
    result["checks"] = {
        f"{stage}.{name}": value
        for stage in ("initial", "render", "set_pose", "reset", "reset_reversed")
        for name, value in result[stage]["checks"].items()
    }
    result["checks"]["same_tick_no_fake_time"] = (
        result["initial"]["timestamp"] == result["render"]["timestamp"] == result["set_pose"]["timestamp"]
        and result["before_reset"]["timestamp"] == result["reset"]["timestamp"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"error": None, "process_profile": "dedicated native helper, adapter fast_shutdown=True"}
    with tempfile.TemporaryDirectory(prefix="isaac-native-data-") as directory:
        asset = Path(directory) / "robot.usda"
        asset.write_text(USD, encoding="utf-8")
        spec = WorldSpec(
            "native-state-getters",
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
            physics=PhysicsSpec(time_step_seconds=1 / 120),
        )
        provider = IsaacLabProvider(
            IsaacLabAdapterConfig(environment_spacing_m=20),
            runtime_factory=lambda config: IsaacLabNativeRuntime(config, process_isolated=True),
        )
        with provider.open() as session:
            try:
                run(session.build(spec), result)
            except Exception:
                result["error"] = traceback.format_exc()
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "error": result["error"],
                        "checks": result.get("checks"),
                        "default_controls": result.get("default_controls"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

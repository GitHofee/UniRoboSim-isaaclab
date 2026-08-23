"""Bounded native DROID smoke for the Core 0.9 Isaac Lab adapter contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from unirobosim import (
    PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION,
    ArrayValue,
    ArticulationCommand,
    BoxGeometrySpec,
    CameraModality,
    CameraSpec,
    CapabilityId,
    CapabilityRequirement,
    CommandMode,
    EntityKind,
    EntityPath,
    EntitySpec,
    FrozenMap,
    PhysicsSpec,
    Pose,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, IsaacLabProvider

ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
GRIPPER_JOINTS = (
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
)
JOINTS = ARM_JOINTS + GRIPPER_JOINTS
INITIAL = (0.0, -0.2, 0.0, -1.8, 0.0, 1.6, 0.7) + (0.0,) * 6


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values)


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _look_at_xyzw(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    forward = _normalize(tuple(target[index] - eye[index] for index in range(3)))
    right = _normalize(_cross(forward, (0.0, 0.0, 1.0)))
    up = _cross(right, forward)
    back = tuple(-value for value in forward)
    m00, m10, m20 = right
    m01, m11, m21 = up
    m02, m12, m22 = back
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        result = (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        result = ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        result = ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)
    return _normalize(result)  # type: ignore[return-value]


def run(asset: Path, *, planning: bool, width: int = 320, height: int = 180) -> dict[str, object]:
    asset_digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    requirements = (CapabilityRequirement(CapabilityId("planning.scene@2")),) if planning else ()
    metadata = FrozenMap(
        {
            "planning_entity_kind": "robot",
            "planning_frame_declarations": {
                "schema": PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION,
                "component_sha256": asset_digest,
                "entries": (
                    {
                        "name": "gripper_center",
                        "owner_link": "gripper_center",
                        "source": {"kind": "link", "name": "gripper_center"},
                    },
                ),
            },
        }
    )
    eye = (2.2, -2.2, 1.6)
    spec = WorldSpec(
        "droid-core-090-native-smoke",
        (
            EntitySpec(
                EntityPath("/droid"),
                EntityKind.ARTICULATION,
                joint_names=JOINTS,
                initial_joint_positions=INITIAL,
                asset_uri=str(asset),
                metadata=metadata,
            ),
            EntitySpec(
                EntityPath("/red-cube"),
                EntityKind.RIGID_BODY,
                pose=Pose((0.62, 0.12, 0.04)),
                box=BoxGeometrySpec(
                    dimensions_m=(0.08, 0.08, 0.08),
                    mass_kg=0.1,
                    color_rgba=(1.0, 0.0, 0.0, 1.0),
                ),
            ),
            EntitySpec(
                EntityPath("/camera"),
                EntityKind.CAMERA_SENSOR,
                pose=Pose(eye, _look_at_xyzw(eye, (0.0, 0.0, 0.65))),
                camera=CameraSpec(
                    width_px=width,
                    height_px=height,
                    modalities=(CameraModality.RGB,),
                    horizontal_fov_degrees=60.0,
                    near_plane_m=0.05,
                    far_plane_m=20.0,
                ),
            ),
        ),
        physics=PhysicsSpec(1.0 / 240.0),
        requirements=requirements,
    )
    provider = IsaacLabProvider(IsaacLabAdapterConfig(enable_cameras=True, render=True))
    with provider.open() as session:
        with session.build(spec) as world:
            handle = world.resolve(EntityPath("/droid"))
            initial = world.read_articulation(handle)
            world.apply_articulation_command(
                ArticulationCommand(
                    handle,
                    CommandMode.POSITION,
                    ArrayValue.from_rows(((0.1, -0.4, 0.0, -2.2, 0.0, 2.5, 0.55),)),
                    degree_of_freedom_indices=tuple(range(7)),
                )
            )
            world.apply_articulation_command(
                ArticulationCommand(
                    handle,
                    CommandMode.POSITION,
                    ArrayValue.from_rows(((0.7, -0.7, 0.7, -0.7, -0.7, 0.7),)),
                    degree_of_freedom_indices=tuple(range(7, 13)),
                )
            )
            world.step(8)
            current = world.read_articulation(handle)
            rgb = world.read_sensor(world.resolve(EntityPath("/camera"))).channel(CameraModality.RGB)
            result: dict[str, object] = {
                "provider": provider.descriptor.provider_id,
                "provider_version": provider.descriptor.version,
                "schemas": provider.descriptor.supported_world_schema_versions,
                "joint_names": current.joint_names,
                "joint_position_units": current.joint_position_units,
                "initial": initial.joint_positions.rows()[0],
                "current": current.joint_positions.rows()[0],
                "tick": current.tick.step_index,
                "camera_shape": rgb.shape,
                "camera_sha256": hashlib.sha256(bytes(rgb.values)).hexdigest(),
            }
            if planning:
                catalog = world.planning_scene_catalog()  # type: ignore[attr-defined]
                state = world.planning_scene_state()  # type: ignore[attr-defined]
                frame = next(item for item in catalog.frames if item.name == "gripper_center")
                frame_state = next(item for item in state.frames if item.frame_id == frame.frame_id)
                resource_geometries = tuple(item for item in catalog.geometries if item.resource_id is not None)
                resource_sha256 = None
                if resource_geometries:
                    lease = world.resolve_planning_geometry(resource_geometries[0].geometry_id)  # type: ignore[attr-defined]
                    try:
                        resource_sha256 = hashlib.sha256(lease.read()).hexdigest()
                    finally:
                        lease.close()
                result.update(
                    {
                        "planning_entities": len(catalog.entities),
                        "planning_frames": len(catalog.frames),
                        "planning_geometries": len(catalog.geometries),
                        "planning_resource_sha256": resource_sha256,
                        "gripper_center_position_m": frame_state.world_pose.position_m,
                        "gripper_center_quaternion_xyzw": frame_state.world_pose.orientation_xyzw,
                    }
                )
            return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        type=Path,
        default=Path("/home/ubuntu/projects/gen_data/data/robots/droid/droid.usd"),
    )
    parser.add_argument("--without-planning", action="store_true")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.asset.resolve(),
                planning=not args.without_planning,
                width=args.width,
                height=args.height,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

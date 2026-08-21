"""Real Isaac Lab acceptance for one-way articulated-container fluid pouring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    CameraModality,
    CameraSpec,
    CommandMode,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    ParticleFluidSpec,
    PhysicsSpec,
    Pose,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider

POURING_RIG = EntityPath("/articulation/pouring_rig")
WATER = EntityPath("/fluid/water")
CAMERA = EntityPath("/sensor/camera")
JOINT_NAME = "cup_hinge"
TARGET_ANGLE_RAD = math.radians(105.0)
DEFAULT_PARTICLE_SPACING_M = 0.014
WATER_GRID_MIN_M = (-0.755, -0.091, 0.54)
WATER_GRID_INTERVAL_EXTENTS_M = (0.378, 0.182, 0.294)


def _particle_grid(
    spacing_m: float,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[int, int, int]]:
    """Discretize the same water volume at a requested physical resolution."""

    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("particle spacing must be finite and positive")
    counts = tuple(max(2, round(extent_m / spacing_m) + 1) for extent_m in WATER_GRID_INTERVAL_EXTENTS_M)
    particles = tuple(
        (
            WATER_GRID_MIN_M[0] + spacing_m * x_index,
            WATER_GRID_MIN_M[1] + spacing_m * y_index,
            WATER_GRID_MIN_M[2] + spacing_m * z_index,
        )
        for z_index in range(counts[2])
        for y_index in range(counts[1])
        for x_index in range(counts[0])
    )
    return particles, counts  # type: ignore[return-value]


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    return tuple(value / magnitude for value in vector)  # type: ignore[return-value]


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _matrix_to_xyzw(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        )
    diagonal = max(range(3), key=lambda index: matrix[index][index])
    if diagonal == 0:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return (
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[2][1] - matrix[1][2]) / scale,
        )
    if diagonal == 1:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return (
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
        )
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return (
        (matrix[0][2] + matrix[2][0]) / scale,
        (matrix[1][2] + matrix[2][1]) / scale,
        0.25 * scale,
        (matrix[1][0] - matrix[0][1]) / scale,
    )


def _look_at_xyzw(
    eye: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    forward = _normalize(tuple(target[axis] - eye[axis] for axis in range(3)))  # type: ignore[arg-type]
    right = _normalize(_cross(forward, (0.0, 0.0, 1.0)))
    up = _cross(right, forward)
    back = tuple(-value for value in forward)
    return _matrix_to_xyzw(
        (
            (right[0], up[0], back[0]),
            (right[1], up[1], back[1]),
            (right[2], up[2], back[2]),
        )
    )


def _world_spec(
    asset: Path,
    *,
    initial_particles: tuple[tuple[float, float, float], ...],
    particle_spacing_m: float,
    width: int | None,
    height: int | None,
) -> WorldSpec:
    entities: list[EntitySpec] = [
        EntitySpec(
            POURING_RIG,
            EntityKind.ARTICULATION,
            joint_names=(JOINT_NAME,),
            initial_joint_positions=(0.0,),
            joint_effort_limits=(800.0,),
            asset_uri=str(asset),
        ),
        EntitySpec(
            WATER,
            EntityKind.PARTICLE_FLUID,
            particle_fluid=ParticleFluidSpec(
                ArrayValue.from_nested(initial_particles),
                particle_radius_m=particle_spacing_m * (6.0 / 7.0),
                particle_mass_kg=1000.0 * particle_spacing_m**3,
                dynamic_viscosity_pa_s=0.001,
                surface_tension_n_m=0.0074,
            ),
        ),
    ]
    if width is not None and height is not None:
        eye = (0.28, -3.65, 1.58)
        entities.append(
            EntitySpec(
                CAMERA,
                EntityKind.CAMERA_SENSOR,
                pose=Pose(
                    position=eye,
                    orientation_xyzw=_look_at_xyzw(eye, (0.20, 0.0, 0.54)),
                ),
                camera=CameraSpec(
                    width_px=width,
                    height_px=height,
                    horizontal_fov_degrees=58.0,
                    near_plane_m=0.05,
                    far_plane_m=20.0,
                    modalities=(CameraModality.RGB,),
                ),
            )
        )
    return WorldSpec(
        "articulated-fluid-pour-acceptance",
        tuple(entities),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(
            time_step_seconds=1.0 / 120.0,
            substeps=2,
            gravity_m_s2=(0.0, 0.0, -9.81),
        ),
    )


class _Encoder:
    def __init__(self, path: Path, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._process = subprocess.Popen(
            (
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                "30",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: bytes) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self._process.stdin.write(frame)

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = b"" if self._process.stderr is None else self._process.stderr.read()
        code = self._process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed with code {code}: {stderr.decode(errors='replace')}")


def _points(state: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in point) for point in state.particle_positions_m.nested()[0])


def _fraction(points: tuple[tuple[float, float, float], ...], predicate: Any) -> float:
    return sum(1 for point in points if predicate(point)) / len(points)


def _inside_upright_cup(point: tuple[float, float, float]) -> bool:
    return -0.79 <= point[0] <= -0.31 and -0.20 <= point[1] <= 0.20 and 0.50 <= point[2] <= 0.98


def _inside_receiver(point: tuple[float, float, float]) -> bool:
    return -0.39 <= point[0] <= 1.49 and -0.58 <= point[1] <= 0.58 and 0.08 <= point[2] <= 0.66


def _inside_scene(point: tuple[float, float, float]) -> bool:
    return -3.5 <= point[0] <= 3.5 and -2.5 <= point[1] <= 2.5 and -0.10 <= point[2] <= 2.5


def _capture(world: Any, camera: Any, encoder: _Encoder, width: int, height: int) -> tuple[int, int]:
    tick_before = world.tick
    channel = world.read_sensor(camera).channel(CameraModality.RGB)
    if world.tick != tick_before:
        raise RuntimeError("camera read advanced physics")
    if channel.shape != (1, height, width, 3) or channel.dtype != "uint8":
        raise RuntimeError(f"unexpected camera output: shape={channel.shape}, dtype={channel.dtype}")
    frame = bytes(channel.values)
    encoder.write(frame)
    return min(frame), max(frame)


def _smoothstep(progress: float) -> float:
    clamped = max(0.0, min(1.0, progress))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def run(
    asset: Path,
    output_dir: Path,
    *,
    width: int | None,
    height: int | None,
    settle_frames: int,
    tilt_frames: int,
    pour_frames: int,
    particle_spacing_m: float,
) -> dict[str, Any]:
    render = width is not None and height is not None
    initial_particles, particle_grid_shape = _particle_grid(particle_spacing_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    provider = create_provider(
        IsaacLabAdapterConfig(
            headless=True,
            device="cuda:0",
            enable_cameras=render,
            render=render,
            anti_aliasing="fxaa",
            texture_streaming=False,
            render_on_step=False,
            fluid_render_mode="isosurface" if render else "particles",
        )
    )
    session = provider.open()
    world: Any | None = None
    encoder: _Encoder | None = None
    trace: list[dict[str, Any]] = []
    try:
        world = session.build(
            _world_spec(
                asset,
                initial_particles=initial_particles,
                particle_spacing_m=particle_spacing_m,
                width=width,
                height=height,
            )
        )
        articulation = world.resolve(POURING_RIG)
        fluid = world.resolve(WATER)
        camera = world.resolve(CAMERA) if render else None
        initial_state = world.read_particle_fluid(fluid)
        initial_points = _points(initial_state)
        initial_joint = world.read_articulation(articulation)
        video_path: Path | None = None
        rgb_range = [255, 0]
        if render:
            assert width is not None and height is not None and camera is not None
            video_path = output_dir / f"articulated-fluid-pour-{width}x{height}.mp4"
            encoder = _Encoder(video_path, width, height)

        total_frames = settle_frames + tilt_frames + pour_frames
        settled_points = initial_points
        maximum_joint_error = 0.0
        for frame_index in range(total_frames):
            if frame_index < settle_frames:
                phase = "settle"
                target = 0.0
            elif frame_index < settle_frames + tilt_frames:
                phase = "tilt"
                progress = (frame_index - settle_frames + 1) / max(1, tilt_frames)
                target = TARGET_ANGLE_RAD * _smoothstep(progress)
            else:
                phase = "pour"
                target = TARGET_ANGLE_RAD
            world.apply_articulation_command(
                ArticulationCommand(
                    articulation,
                    CommandMode.POSITION,
                    ArrayValue.from_nested(((target,),)),
                )
            )
            world.step(4)
            joint_state = world.read_articulation(articulation)
            particle_state = world.read_particle_fluid(fluid)
            points = _points(particle_state)
            actual_joint = float(joint_state.joint_positions.values[0])
            maximum_joint_error = max(maximum_joint_error, abs(actual_joint - target))
            if frame_index == settle_frames - 1:
                settled_points = points
            trace.append(
                {
                    "frame": frame_index,
                    "phase": phase,
                    "tick": world.tick.step_index,
                    "joint_target_rad": target,
                    "joint_position_rad": actual_joint,
                    "upright_cup_fraction": _fraction(points, _inside_upright_cup),
                    "receiver_fraction": _fraction(points, _inside_receiver),
                    "scene_fraction": _fraction(points, _inside_scene),
                }
            )
            if encoder is not None:
                low, high = _capture(world, camera, encoder, width, height)
                rgb_range[0] = min(rgb_range[0], low)
                rgb_range[1] = max(rgb_range[1], high)

        if encoder is not None:
            encoder.close()
            encoder = None
        final_state = world.read_particle_fluid(fluid)
        final_points = _points(final_state)
        final_joint = world.read_articulation(articulation)
        settled_retained_fraction = _fraction(settled_points, _inside_upright_cup)
        receiver_fraction = _fraction(final_points, _inside_receiver)
        scene_fraction = _fraction(final_points, _inside_scene)
        mean_particle_displacement = sum(
            math.dist(before, after) for before, after in zip(initial_points, final_points, strict=True)
        ) / len(initial_points)

        reset_errors: list[float] = []
        for _ in range(5):
            world.reset((0,))
            reset_joint = world.read_articulation(articulation)
            reset_fluid = world.read_particle_fluid(fluid)
            joint_error = max(
                abs(float(value)) for value in reset_joint.joint_positions.values + reset_joint.joint_velocities.values
            )
            particle_error = max(
                abs(float(after) - float(before))
                for after, before in zip(
                    reset_fluid.particle_positions_m.values,
                    initial_state.particle_positions_m.values,
                    strict=True,
                )
            )
            reset_errors.append(max(joint_error, particle_error))

        final_joint_position = float(final_joint.joint_positions.values[0])
        checks = {
            "initial_state_finite_and_complete": len(initial_points) == len(initial_particles)
            and all(math.isfinite(value) for point in initial_points for value in point),
            "upright_cup_retains_particles": settled_retained_fraction >= 0.90,
            "joint_reaches_pour_angle": abs(final_joint_position - TARGET_ANGLE_RAD) <= 0.12,
            "fluid_transfers_to_receiver": receiver_fraction >= 0.65,
            "particles_remain_in_scene": scene_fraction >= 0.98,
            "fluid_motion_is_nontrivial": mean_particle_displacement >= 0.35,
            "reset_repeatability": max(reset_errors) < 1.0e-4,
            "camera_contrast": not render or rgb_range[1] > rgb_range[0],
        }
        videos = [] if video_path is None else [str(video_path.resolve())]
        result: dict[str, Any] = {
            "status": "passed" if all(checks.values()) else "failed",
            "coupling_contract": {
                "direction": "articulation-to-particle-fluid",
                "particle_reaction_loads_on_articulation": False,
            },
            "checks": checks,
            "physics": {
                "engine": "Isaac Sim 6.0.1 PhysX PBD",
                "device": "cuda:0",
                "time_step_seconds": 1.0 / 120.0,
                "substeps": 2,
                "gravity_m_s2": [0.0, 0.0, -9.81],
            },
            "particle_count": len(initial_points),
            "particle_grid_shape_xyz": list(particle_grid_shape),
            "particle_spacing_m": particle_spacing_m,
            "particle_radius_m": particle_spacing_m * (6.0 / 7.0),
            "particle_mass_kg": 1000.0 * particle_spacing_m**3,
            "settled_retained_fraction": settled_retained_fraction,
            "receiver_fraction": receiver_fraction,
            "scene_fraction": scene_fraction,
            "mean_particle_displacement_m": mean_particle_displacement,
            "initial_joint_rad": float(initial_joint.joint_positions.values[0]),
            "final_joint_rad": final_joint_position,
            "target_joint_rad": TARGET_ANGLE_RAD,
            "max_command_tracking_error_rad": maximum_joint_error,
            "reset_count": len(reset_errors),
            "max_reset_error": max(reset_errors),
            "ticks_executed": world.tick.step_index,
            "frames_captured": total_frames if render else 0,
            "camera_rendered": render,
            "live_capture": render,
            "camera_rgb_range": rgb_range if render else None,
            "videos": videos,
            "video_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in videos},
            "wall_seconds": time.time() - started,
        }
        (output_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if result["status"] != "passed":
            raise RuntimeError(f"articulated pouring acceptance failed: {checks}")
        return result
    finally:
        if encoder is not None:
            encoder.close()
        if world is not None:
            world.close()
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--settle-frames", type=int, default=45)
    parser.add_argument("--tilt-frames", type=int, default=60)
    parser.add_argument("--pour-frames", type=int, default=75)
    parser.add_argument("--particle-spacing-m", type=float, default=DEFAULT_PARTICLE_SPACING_M)
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    result = run(
        args.asset.resolve(),
        args.output_dir.resolve(),
        width=args.width,
        height=args.height,
        settle_frames=args.settle_frames,
        tilt_frames=args.tilt_frames,
        pour_frames=args.pour_frames,
        particle_spacing_m=args.particle_spacing_m,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

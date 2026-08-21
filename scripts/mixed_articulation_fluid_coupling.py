"""Real PhysX acceptance for bidirectional articulation/PBD-fluid coupling."""

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
    ParticleFluidCommand,
    ParticleFluidSpec,
    PhysicsSpec,
    PointCommandMode,
    Pose,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider

ACTIVE_ORIGIN = (-0.75, 0.0, 0.0)
CONTROL_ORIGIN = (0.35, 0.0, 0.0)
ACTIVE_ARTICULATION = EntityPath("/articulation/active")
CONTROL_ARTICULATION = EntityPath("/articulation/control")
ACTIVE_FLUID = EntityPath("/fluid/active")
CONTROL_FLUID = EntityPath("/fluid/control")
CAMERA = EntityPath("/sensor/camera")
JOINT_NAME = "paddle_hinge"


def _particle_grid(*, y_values: tuple[float, float]) -> tuple[tuple[float, float, float], ...]:
    return tuple((x, y, z) for z in (0.38, 0.5, 0.62) for y in y_values for x in (0.24, 0.35, 0.46, 0.57))


PUSH_PARTICLES = _particle_grid(y_values=(0.055, 0.11))
IMPACT_PARTICLES = _particle_grid(y_values=(-0.50, -0.445))


def _add_origin(
    points: tuple[tuple[float, float, float], ...], origin: tuple[float, float, float]
) -> tuple[tuple[float, float, float], ...]:
    return tuple((point[0] + origin[0], point[1] + origin[1], point[2] + origin[2]) for point in points)


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


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
    matrix = (
        (right[0], up[0], back[0]),
        (right[1], up[1], back[1]),
        (right[2], up[2], back[2]),
    )
    return _matrix_to_xyzw(matrix)


def _world_spec(asset: Path, *, width: int | None, height: int | None) -> WorldSpec:
    entities: list[EntitySpec] = [
        EntitySpec(
            ACTIVE_ARTICULATION,
            EntityKind.ARTICULATION,
            pose=Pose(position=ACTIVE_ORIGIN),
            joint_names=(JOINT_NAME,),
            initial_joint_positions=(0.0,),
            joint_effort_limits=(500.0,),
            asset_uri=str(asset),
        ),
        EntitySpec(
            CONTROL_ARTICULATION,
            EntityKind.ARTICULATION,
            pose=Pose(position=CONTROL_ORIGIN),
            joint_names=(JOINT_NAME,),
            initial_joint_positions=(0.0,),
            joint_effort_limits=(500.0,),
            asset_uri=str(asset),
        ),
        EntitySpec(
            ACTIVE_FLUID,
            EntityKind.PARTICLE_FLUID,
            pose=Pose(position=ACTIVE_ORIGIN),
            particle_fluid=ParticleFluidSpec(
                ArrayValue.from_nested(PUSH_PARTICLES),
                particle_radius_m=0.025,
                particle_mass_kg=0.08,
                dynamic_viscosity_pa_s=0.01,
                surface_tension_n_m=0.05,
            ),
        ),
        EntitySpec(
            CONTROL_FLUID,
            EntityKind.PARTICLE_FLUID,
            pose=Pose(position=CONTROL_ORIGIN),
            particle_fluid=ParticleFluidSpec(
                ArrayValue.from_nested(PUSH_PARTICLES),
                particle_radius_m=0.025,
                particle_mass_kg=0.08,
                dynamic_viscosity_pa_s=0.01,
                surface_tension_n_m=0.05,
            ),
        ),
    ]
    if width is not None and height is not None:
        eye = (0.15, -2.35, 1.35)
        entities.append(
            EntitySpec(
                CAMERA,
                EntityKind.CAMERA_SENSOR,
                pose=Pose(position=eye, orientation_xyzw=_look_at_xyzw(eye, (0.1, 0.05, 0.48))),
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
        "mixed-articulation-fluid-coupling",
        tuple(entities),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(
            time_step_seconds=1.0 / 120.0,
            substeps=2,
            gravity_m_s2=(0.0, 0.0, 0.0),
        ),
    )


def _points(state: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in point) for point in state.particle_positions_m.nested()[0])


def _velocities(state: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in point) for point in state.particle_velocities_m_s.nested()[0])


def _mean_displacement(
    before: tuple[tuple[float, float, float], ...], after: tuple[tuple[float, float, float], ...]
) -> float:
    return sum(math.dist(left, right) for left, right in zip(before, after, strict=True)) / len(before)


def _center(points: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    return tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))  # type: ignore[return-value]


def _mean_y_velocity(velocities: tuple[tuple[float, float, float], ...]) -> float:
    return sum(value[1] for value in velocities) / len(velocities)


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


def _capture(world: Any, camera: Any, encoder: _Encoder, width: int, height: int) -> tuple[int, int]:
    before = world.tick
    channel = world.read_sensor(camera).channel(CameraModality.RGB)
    if world.tick != before:
        raise RuntimeError("camera read advanced physics")
    if channel.shape != (1, height, width, 3) or channel.dtype != "uint8":
        raise RuntimeError(f"unexpected camera output: shape={channel.shape}, dtype={channel.dtype}")
    frame = bytes(channel.values)
    encoder.write(frame)
    return min(frame), max(frame)


def _particle_command(
    handle: Any,
    mode: PointCommandMode,
    values: tuple[tuple[float, float, float], ...],
) -> ParticleFluidCommand:
    return ParticleFluidCommand(
        handle,
        mode,
        ArrayValue.from_nested((values,)),
        environment_indices=(0,),
        particle_indices=tuple(range(len(values))),
    )


def run(
    asset: Path,
    output_dir: Path,
    *,
    width: int | None,
    height: int | None,
    frames_per_phase: int,
) -> dict[str, Any]:
    render = width is not None and height is not None
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
        )
    )
    session = provider.open()
    encoders: list[_Encoder] = []
    world: Any | None = None
    try:
        world = session.build(_world_spec(asset, width=width, height=height))
        active_articulation = world.resolve(ACTIVE_ARTICULATION)
        control_articulation = world.resolve(CONTROL_ARTICULATION)
        active_fluid = world.resolve(ACTIVE_FLUID)
        control_fluid = world.resolve(CONTROL_FLUID)
        camera = world.resolve(CAMERA) if render else None
        initial_active = _points(world.read_particle_fluid(active_fluid))
        initial_control = _points(world.read_particle_fluid(control_fluid))
        push_trace: list[dict[str, Any]] = []
        push_video: Path | None = None
        push_encoder: _Encoder | None = None
        frame_range = [255, 0]
        if render:
            assert width is not None and height is not None and camera is not None
            push_video = output_dir / "articulation-to-fluid-1920x1080.mp4"
            push_encoder = _Encoder(push_video, width, height)
            encoders.append(push_encoder)
        for frame_index in range(frames_per_phase):
            progress = min(1.0, (frame_index + 1) / max(1, frames_per_phase * 0.6))
            smooth = progress * progress * (3.0 - 2.0 * progress)
            world.apply_articulation_command(
                ArticulationCommand(
                    active_articulation,
                    CommandMode.POSITION,
                    ArrayValue.from_nested(((0.70 * smooth,),)),
                )
            )
            world.step(4)
            active_joint = world.read_articulation(active_articulation)
            control_joint = world.read_articulation(control_articulation)
            active_state = world.read_particle_fluid(active_fluid)
            control_state = world.read_particle_fluid(control_fluid)
            push_trace.append(
                {
                    "tick": world.tick.step_index,
                    "active_joint_rad": float(active_joint.joint_positions.values[0]),
                    "control_joint_rad": float(control_joint.joint_positions.values[0]),
                    "active_fluid_com_m": _center(_points(active_state)),
                    "control_fluid_com_m": _center(_points(control_state)),
                }
            )
            if push_encoder is not None:
                low, high = _capture(world, camera, push_encoder, width, height)
                frame_range[0] = min(frame_range[0], low)
                frame_range[1] = max(frame_range[1], high)
        if push_encoder is not None:
            push_encoder.close()
            encoders.remove(push_encoder)
        pushed_active_state = world.read_particle_fluid(active_fluid)
        pushed_control_state = world.read_particle_fluid(control_fluid)
        pushed_joint_state = world.read_articulation(active_articulation)
        active_push_displacement = _mean_displacement(initial_active, _points(pushed_active_state))
        control_push_displacement = _mean_displacement(initial_control, _points(pushed_control_state))
        push_extra_displacement = active_push_displacement - control_push_displacement
        push_joint_travel = abs(float(pushed_joint_state.joint_positions.values[0]))

        world.reset((0,))
        world.apply_articulation_command(
            ArticulationCommand(active_articulation, CommandMode.EFFORT, ArrayValue.from_nested(((1.0,),)))
        )
        world.step(30)
        effort_probe_state = world.read_articulation(active_articulation)
        effort_probe_joint_travel = abs(float(effort_probe_state.joint_positions.values[0]))
        effort_probe_velocity = abs(float(effort_probe_state.joint_velocities.values[0]))

        world.reset((0,))
        impact_active_points = _add_origin(IMPACT_PARTICLES, ACTIVE_ORIGIN)
        impact_control_points = _add_origin(IMPACT_PARTICLES, CONTROL_ORIGIN)
        impact_active_velocity = tuple((0.0, 2.2, 0.0) for _ in IMPACT_PARTICLES)
        impact_control_velocity = tuple((0.0, 0.0, 0.0) for _ in IMPACT_PARTICLES)
        world.apply_particle_fluid_command(
            _particle_command(active_fluid, PointCommandMode.POSITION, impact_active_points)
        )
        world.apply_particle_fluid_command(
            _particle_command(control_fluid, PointCommandMode.POSITION, impact_control_points)
        )
        world.apply_particle_fluid_command(
            _particle_command(active_fluid, PointCommandMode.VELOCITY, impact_active_velocity)
        )
        world.apply_particle_fluid_command(
            _particle_command(control_fluid, PointCommandMode.VELOCITY, impact_control_velocity)
        )
        for articulation in (active_articulation, control_articulation):
            world.apply_articulation_command(
                ArticulationCommand(articulation, CommandMode.EFFORT, ArrayValue.from_nested(((0.0,),)))
            )
        impact_initial_state = world.read_particle_fluid(active_fluid)
        impact_initial_y_velocity = _mean_y_velocity(_velocities(impact_initial_state))
        impact_trace: list[dict[str, Any]] = []
        impact_video: Path | None = None
        impact_encoder: _Encoder | None = None
        if render:
            assert width is not None and height is not None and camera is not None
            impact_video = output_dir / "fluid-to-articulation-1920x1080.mp4"
            impact_encoder = _Encoder(impact_video, width, height)
            encoders.append(impact_encoder)
        active_joint_peak = 0.0
        active_velocity_peak = 0.0
        control_joint_peak = 0.0
        control_velocity_peak = 0.0
        for _ in range(frames_per_phase):
            world.step(4)
            active_state = world.read_articulation(active_articulation)
            control_state = world.read_articulation(control_articulation)
            active_position = float(active_state.joint_positions.values[0])
            active_velocity = float(active_state.joint_velocities.values[0])
            control_position = float(control_state.joint_positions.values[0])
            control_velocity = float(control_state.joint_velocities.values[0])
            active_joint_peak = max(active_joint_peak, abs(active_position))
            active_velocity_peak = max(active_velocity_peak, abs(active_velocity))
            control_joint_peak = max(control_joint_peak, abs(control_position))
            control_velocity_peak = max(control_velocity_peak, abs(control_velocity))
            impact_trace.append(
                {
                    "tick": world.tick.step_index,
                    "active_joint_rad": active_position,
                    "active_joint_velocity_rad_s": active_velocity,
                    "control_joint_rad": control_position,
                    "control_joint_velocity_rad_s": control_velocity,
                    "active_fluid_mean_y_velocity_m_s": _mean_y_velocity(
                        _velocities(world.read_particle_fluid(active_fluid))
                    ),
                }
            )
            if impact_encoder is not None:
                low, high = _capture(world, camera, impact_encoder, width, height)
                frame_range[0] = min(frame_range[0], low)
                frame_range[1] = max(frame_range[1], high)
        if impact_encoder is not None:
            impact_encoder.close()
            encoders.remove(impact_encoder)
        impact_final_state = world.read_particle_fluid(active_fluid)
        impact_final_y_velocity = _mean_y_velocity(_velocities(impact_final_state))

        reset_errors: list[float] = []
        for _ in range(10):
            world.reset((0,))
            reset_joint = world.read_articulation(active_articulation)
            reset_fluid = world.read_particle_fluid(active_fluid)
            joint_error = max(
                abs(float(value)) for value in reset_joint.joint_positions.values + reset_joint.joint_velocities.values
            )
            fluid_error = max(
                abs(float(after) - float(before))
                for after, before in zip(
                    reset_fluid.particle_positions_m.values,
                    ArrayValue.from_nested((_add_origin(PUSH_PARTICLES, ACTIVE_ORIGIN),)).values,
                    strict=True,
                )
            )
            reset_errors.append(max(joint_error, fluid_error))

        checks = {
            "articulation_to_fluid_joint_travel": push_joint_travel >= 0.35,
            "articulation_to_fluid_extra_mean_displacement": push_extra_displacement >= 0.015,
            "passive_articulation_mobility": effort_probe_joint_travel >= 0.02,
            "fluid_to_articulation_joint_response": active_joint_peak >= max(0.02, control_joint_peak * 10.0),
            "fluid_to_articulation_velocity_response": active_velocity_peak >= max(0.10, control_velocity_peak * 10.0),
            "fluid_momentum_reduced": impact_final_y_velocity <= impact_initial_y_velocity - 0.05,
            "reset_repeatability": max(reset_errors) < 1.0e-4,
            "camera_contrast": not render or frame_range[1] > frame_range[0],
        }
        result: dict[str, Any] = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "physics": {
                "engine": "PhysX PBD + articulation",
                "device": "cuda:0",
                "time_step_seconds": 1.0 / 120.0,
                "substeps": 2,
                "gravity_m_s2": [0.0, 0.0, 0.0],
            },
            "particle_count_per_group": len(PUSH_PARTICLES),
            "articulation_to_fluid": {
                "joint_travel_rad": push_joint_travel,
                "active_mean_displacement_m": active_push_displacement,
                "control_mean_displacement_m": control_push_displacement,
                "extra_mean_displacement_m": push_extra_displacement,
            },
            "fluid_to_articulation": {
                "effort_probe_joint_travel_rad": effort_probe_joint_travel,
                "effort_probe_joint_velocity_rad_s": effort_probe_velocity,
                "active_joint_peak_rad": active_joint_peak,
                "control_joint_peak_rad": control_joint_peak,
                "active_joint_velocity_peak_rad_s": active_velocity_peak,
                "control_joint_velocity_peak_rad_s": control_velocity_peak,
                "fluid_initial_mean_y_velocity_m_s": impact_initial_y_velocity,
                "fluid_final_mean_y_velocity_m_s": impact_final_y_velocity,
            },
            "max_reset_error": max(reset_errors),
            "reset_count": len(reset_errors),
            "ticks_executed": world.tick.step_index,
            "camera_rendered": render,
            "live_capture": render,
            "camera_rgb_range": frame_range if render else None,
            "videos": [str(path.resolve()) for path in (push_video, impact_video) if path is not None],
            "wall_seconds": time.time() - started,
        }
        (output_dir / "push-trace.json").write_text(json.dumps(push_trace, indent=2) + "\n", encoding="utf-8")
        (output_dir / "impact-trace.json").write_text(json.dumps(impact_trace, indent=2) + "\n", encoding="utf-8")
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if result["status"] != "passed":
            raise RuntimeError(f"bidirectional coupling acceptance failed: {checks}")
        return result
    finally:
        for encoder in encoders:
            encoder.close()
        if world is not None:
            world.close()
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames-per-phase", type=int, default=90)
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    result = run(
        args.asset.resolve(),
        args.output_dir.resolve(),
        width=args.width,
        height=args.height,
        frames_per_phase=args.frames_per_phase,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["videos"]:
        result["video_sha256"] = {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in result["videos"]
        }
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

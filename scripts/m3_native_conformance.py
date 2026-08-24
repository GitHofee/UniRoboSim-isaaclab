"""Real M3 fluid, RGB/depth camera, native-debug, and video conformance runner."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from unirobosim import (
    ArrayValue,
    CameraModality,
    CameraSpec,
    DebugBatch,
    DebugBus,
    DebugLifetime,
    DebugPrimitive,
    DebugPrimitiveKind,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    NativeWorldDebugSink,
    ParticleFluidCommand,
    ParticleFluidSpec,
    PhysicsSpec,
    PointCommandMode,
    Pose,
    RigidBodyCommand,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider


def _versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "unirobosim",
            "unirobosim-isaaclab",
            "isaaclab",
            "isaaclab_physx",
            "isaacsim",
            "torch",
            "torchvision",
            "torchaudio",
        )
    }


def _create_scene_asset(path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Scene")
    cube = UsdGeom.Cube.Define(stage, "/Scene/Cube")
    cube.CreateSizeAttr(0.5)
    cube.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.9, 0.15, 0.05)])
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
    floor = UsdGeom.Cube.Define(stage, "/Scene/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddScaleOp().Set(Gf.Vec3f(8.0, 8.0, 0.1))
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.4))
    floor.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.15, 0.2, 0.25)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def _particle_grid() -> tuple[tuple[float, float, float], ...]:
    return tuple((x * 0.065, y * 0.065, z * 0.065) for z in range(3) for y in range(3) for x in range(3))


def _world(asset: Path, *, include_rigid: bool = True) -> WorldSpec:
    entities = [
        EntitySpec(
            EntityPath("/matter/water"),
            EntityKind.PARTICLE_FLUID,
            pose=Pose(position=(1.0, -0.1, 0.8)),
            particle_fluid=ParticleFluidSpec(
                ArrayValue.from_nested(_particle_grid()),
                particle_radius_m=0.03,
                rest_density_kg_m3=1000.0,
                dynamic_viscosity_pa_s=0.01,
                surface_tension_n_m=0.05,
            ),
        ),
        EntitySpec(
            EntityPath("/sensors/front"),
            EntityKind.CAMERA_SENSOR,
            pose=Pose(position=(-1.5, 0.0, 1.0)),
            camera=CameraSpec(
                width_px=320,
                height_px=180,
                horizontal_fov_degrees=70.0,
                near_plane_m=0.05,
                far_plane_m=20.0,
            ),
        ),
    ]
    if include_rigid:
        entities.insert(
            0,
            EntitySpec(
                EntityPath("/scene/obstacle"),
                EntityKind.RIGID_BODY,
                pose=Pose(position=(2.0, 0.0, 0.35)),
                asset_uri=str(asset),
            ),
        )
    return WorldSpec(
        "native-m3-fluid-camera-debug",
        tuple(entities),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(time_step_seconds=1.0 / 60.0, substeps=2),
    )


def _debug_batch() -> DebugBatch:
    axes = DebugPrimitive(
        "camera-axis",
        "m3-conformance",
        DebugPrimitiveKind.LINE_LIST,
        ArrayValue.from_nested(
            [
                [
                    [[0.5, 0.0, 0.75], [1.5, 0.0, 0.75]],
                    [[1.0, -0.5, 0.75], [1.0, 0.5, 0.75]],
                    [[1.0, 0.0, 0.25], [1.0, 0.0, 1.25]],
                ]
            ]
        ),
        (0,),
        color_rgba=(0.1, 1.0, 0.15, 1.0),
        size=0.025,
    )
    points = DebugPrimitive(
        "targets",
        "m3-conformance",
        DebugPrimitiveKind.POINT_SET,
        ArrayValue.from_nested([[[1.0, 0.0, 0.75], [2.0, 0.0, 0.35]]]),
        (0,),
        color_rgba=(1.0, 0.95, 0.05, 1.0),
        size=0.12,
        lifetime=DebugLifetime.steps(30),
    )
    return DebugBatch((axes, points))


def _finite(values: tuple[float | int | bool, ...]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def run(video_path: Path | None, *, frames: int = 60, include_rigid: bool = True) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    started = time.time()
    result: dict[str, Any] = {"status": "running", "versions": _versions(), "checks": []}
    provider = create_provider(
        IsaacLabAdapterConfig(
            headless=True,
            device="cuda:0",
            enable_cameras=True,
            render=True,
        )
    )
    probe = provider.probe()
    result["probe"] = {"available": probe.available, "reason": probe.reason, "details": probe.details.to_dict()}
    if not probe.available:
        raise RuntimeError(f"native profile unavailable: {probe.reason}")

    session = provider.open()
    writer: Any | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="unirobosim-m3-") as directory:
            asset = Path(directory) / "scene.usda"
            _create_scene_asset(asset)
            world = session.build(_world(asset, include_rigid=include_rigid))
            fluid_handle = world.resolve(EntityPath("/matter/water"))
            camera_handle = world.resolve(EntityPath("/sensors/front"))
            rigid_handle = world.resolve(EntityPath("/scene/obstacle")) if include_rigid else None

            initial = world.read_particle_fluid(fluid_handle)
            initial_rigid = world.read_rigid_body(rigid_handle) if rigid_handle is not None else None
            assert initial.particle_positions_m.shape == (1, 27, 3)
            assert _finite(initial.particle_positions_m.values)
            position_target = (1.25, -0.05, 1.1)
            world.apply_particle_fluid_command(
                ParticleFluidCommand(
                    fluid_handle,
                    PointCommandMode.POSITION,
                    ArrayValue.from_nested([[position_target]]),
                    environment_indices=(0,),
                    particle_indices=(0,),
                )
            )
            positioned = world.read_particle_fluid(fluid_handle)
            position_command_error = max(
                abs(float(actual) - expected)
                for actual, expected in zip(
                    positioned.particle_positions_m.nested()[0][0],
                    position_target,
                    strict=True,
                )
            )
            assert position_command_error < 1.0e-4
            world.step()
            positioned_after_step = world.read_particle_fluid(fluid_handle)
            position_command_step_error = max(
                abs(float(actual) - expected)
                for actual, expected in zip(
                    positioned_after_step.particle_positions_m.nested()[0][0],
                    position_target,
                    strict=True,
                )
            )
            assert position_command_step_error < 0.2
            world.reset((0,))
            initial = world.read_particle_fluid(fluid_handle)
            world.apply_particle_fluid_command(
                ParticleFluidCommand(
                    fluid_handle,
                    PointCommandMode.VELOCITY,
                    ArrayValue.from_nested([[[0.4, 0.0, 0.2]]]),
                    environment_indices=(0,),
                    particle_indices=(0,),
                )
            )
            if rigid_handle is not None:
                world.apply_rigid_body_command(
                    RigidBodyCommand(
                        rigid_handle,
                        ArrayValue.from_nested([[2.0, 0.0, 0.0]]),
                        ArrayValue.from_nested([[0.0, 0.0, 0.0]]),
                    )
                )

            debug_bus = DebugBus((NativeWorldDebugSink(world),), max_active_primitives=16)
            debug_report = debug_bus.publish(_debug_batch())
            assert debug_report.accepted_count == 2

            if video_path is not None:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (320, 180),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {video_path}")

            rgb_min = 255
            rgb_max = 0
            nonzero_depth_min = float("inf")
            nonzero_depth_max = 0.0
            for _ in range(frames):
                before = world.tick
                sample = world.read_sensor(camera_handle)
                assert world.tick == before
                rgb = sample.channel(CameraModality.RGB)
                depth = sample.channel(CameraModality.DEPTH)
                assert rgb.shape == (1, 180, 320, 3) and rgb.dtype == "uint8"
                assert depth.shape == (1, 180, 320) and depth.dtype == "float32"
                frame = np.asarray(rgb.values, dtype=np.uint8).reshape(rgb.shape)[0]
                depth_image = np.asarray(depth.values, dtype=np.float32).reshape(depth.shape)[0]
                rgb_min = min(rgb_min, int(frame.min()))
                rgb_max = max(rgb_max, int(frame.max()))
                hits = depth_image[depth_image > 0.0]
                if hits.size:
                    nonzero_depth_min = min(nonzero_depth_min, float(hits.min()))
                    nonzero_depth_max = max(nonzero_depth_max, float(hits.max()))
                if writer is not None:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                world.step()
                debug_bus.advance()

            if writer is not None:
                writer.release()
                writer = None
            assert rgb_max > rgb_min
            assert nonzero_depth_max > nonzero_depth_min > 0.0
            moved = world.read_particle_fluid(fluid_handle)
            assert _finite(moved.particle_positions_m.values)
            position_delta = max(
                abs(float(after) - float(before))
                for after, before in zip(
                    moved.particle_positions_m.values,
                    initial.particle_positions_m.values,
                    strict=True,
                )
            )
            velocity_magnitude = max(abs(float(value)) for value in moved.particle_velocities_m_s.values)
            print(
                json.dumps(
                    {
                        "fluid_position_delta_m": position_delta,
                        "fluid_velocity_max_abs_m_s": velocity_magnitude,
                        "initial_first": initial.particle_positions_m.nested()[0][0],
                        "moved_first": moved.particle_positions_m.nested()[0][0],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            assert position_delta > 1.0e-4
            rigid_position_delta = 0.0
            if rigid_handle is not None and initial_rigid is not None:
                moved_rigid = world.read_rigid_body(rigid_handle)
                rigid_position_delta = max(
                    abs(float(after) - float(before))
                    for after, before in zip(
                        moved_rigid.positions_m.values,
                        initial_rigid.positions_m.values,
                        strict=True,
                    )
                )
                assert rigid_position_delta > 1.0e-4
            world.reset((0,))
            reset = world.read_particle_fluid(fluid_handle)
            reset_error = max(
                abs(float(after) - float(before))
                for after, before in zip(
                    reset.particle_positions_m.values,
                    initial.particle_positions_m.values,
                    strict=True,
                )
            )
            assert reset_error < 1.0e-4
            rigid_reset_error = 0.0
            if rigid_handle is not None and initial_rigid is not None:
                reset_rigid = world.read_rigid_body(rigid_handle)
                rigid_reset_error = max(
                    abs(float(after) - float(before))
                    for after, before in zip(
                        reset_rigid.positions_m.values,
                        initial_rigid.positions_m.values,
                        strict=True,
                    )
                )
                assert rigid_reset_error < 1.0e-4
            debug_bus.close()
            world.close()

            result["checks"].extend(
                (
                    {"name": "native_physx_pbd_fluid_state_command_step_reset", "passed": True},
                    {"name": "native_rgb_depth_shape_dtype_tick_and_content", "passed": True},
                    {"name": "native_debug_stable_overlay_and_lifetime", "passed": True},
                )
            )
            if rigid_handle is not None:
                result["checks"].append({"name": "native_usd_rigid_bridge_state_wrench_step_reset", "passed": True})
            result.update(
                {
                    "fluid_position_delta_m": position_delta,
                    "fluid_position_command_max_error_m": position_command_error,
                    "fluid_position_command_after_step_max_error_m": position_command_step_error,
                    "fluid_reset_max_error_m": reset_error,
                    "rigid_position_delta_m": rigid_position_delta,
                    "rigid_reset_max_error_m": rigid_reset_error,
                    "rgb_range": [rgb_min, rgb_max],
                    "depth_nonzero_range_m": [nonzero_depth_min, nonzero_depth_max],
                    "video": None if video_path is None else str(video_path),
                }
            )
    finally:
        if writer is not None:
            writer.release()
        session.close()
    result["status"] = "passed"
    result["elapsed_seconds"] = time.time() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fluid-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.video, frames=args.frames, include_rigid=not args.fluid_only)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "versions": _versions(),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Native Isaac Sim 6 scene-control conformance with retained video evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from unirobosim import (
    CameraModality,
    EntityPath,
    Pose,
    SceneCommand,
    SceneCommandKind,
    SceneCommandStatus,
    SceneDragMode,
    Sim,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider


def _versions() -> dict[str, str]:
    names = (
        "unirobosim",
        "unirobosim-isaaclab",
        "isaaclab",
        "isaacsim",
        "torch",
        "opencv-python-headless",
    )
    return {name: importlib.metadata.version(name) for name in names}


def _create_asset(path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Scene")
    cube = UsdGeom.Cube.Define(stage, "/Scene/Cube")
    cube.CreateSizeAttr(0.5)
    cube.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.1, 0.55, 1.0)])
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
    marker = UsdGeom.Sphere.Define(stage, "/Scene/Cube/Marker")
    marker.CreateRadiusAttr(0.09)
    marker.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.32))
    marker.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(1.0, 0.3, 0.05)])
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()


def _drag_command(
    world: Any,
    command_id: str,
    kind: SceneCommandKind,
    *,
    target: Pose | None = None,
) -> SceneCommand:
    return SceneCommand(
        command_id,
        "native-acceptance",
        "native-lease",
        world.generation,
        kind,
        EntityPath("/box"),
        0,
        target,
        "native-drag",
        SceneDragMode.KINEMATIC if kind is SceneCommandKind.DRAG_BEGIN else None,
        (0.0, -0.65, 0.35) if kind is SceneCommandKind.DRAG_BEGIN else None,
    )


def run(output: Path, *, frames: int = 90) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "isaac-scene-drag.mp4"
    result: dict[str, Any] = {
        "status": "running",
        "started_at_unix": time.time(),
        "versions": _versions(),
        "checks": [],
        "video": str(video_path),
    }
    provider = create_provider(IsaacLabAdapterConfig(headless=True, device="cuda:0", enable_cameras=True, render=True))
    probe = provider.probe()
    result["probe"] = {"available": probe.available, "reason": probe.reason, "details": probe.details.to_dict()}
    if not probe.available:
        raise RuntimeError(f"native profile unavailable: {probe.reason}")

    sim = Sim(
        provider=provider,
        world_id="isaac-m5-scene-control",
        time_step_seconds=1.0 / 30.0,
        gravity_m_s2=(0.0, 0.0, 0.0),
    )
    writer: Any | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="unirobosim-scene-m5-") as directory:
            asset = Path(directory) / "box.usda"
            _create_asset(asset)
            sim.add_rigid_body("box", asset_uri=str(asset), position_m=(0.0, -0.65, 0.35))
            camera = sim.add_camera(
                "camera",
                resolution=(640, 360),
                outputs=(CameraModality.RGB,),
                position_m=(-2.4, 0.0, 1.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
            sim.start()
            world = sim.world
            initial = world.scene_snapshot()
            initial_box = next(item for item in initial.entities if item.path == EntityPath("/box"))
            expected_initial = (0.0, -0.65, 0.35)
            initial_error = max(
                abs(actual - expected)
                for actual, expected in zip(initial_box.pose.position, expected_initial, strict=True)
            )
            result["initial_pose"] = list(initial_box.pose.position)
            result["initial_pose_max_error_m"] = initial_error
            assert initial_box.draggable and initial_error < 1.0e-4
            assert (
                world.apply_scene_command(_drag_command(world, "begin", SceneCommandKind.DRAG_BEGIN)).status
                is SceneCommandStatus.APPLIED
            )

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (640, 360),
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV video writer did not open")
            final_target = Pose((0.0, 0.85, 0.75))
            frame_means: list[float] = []
            frame_stds: list[float] = []
            frame_non_black: list[float] = []
            for frame in range(frames):
                progress = (frame + 1) / frames if frames <= 30 else min(1.0, max(0.0, (frame - 15) / (frames - 30)))
                target = Pose(
                    (
                        0.0,
                        -0.65 + 1.5 * progress,
                        0.35 + 0.4 * progress,
                    )
                )
                command = _drag_command(world, f"update-{frame}", SceneCommandKind.DRAG_UPDATE, target=target)
                assert world.apply_scene_command(command).status is SceneCommandStatus.APPLIED
                world.step()
                rgb = camera.read(CameraModality.RGB)
                image = np.asarray(rgb.values, dtype=np.uint8).reshape(rgb.shape)[0]
                frame_means.append(float(image.mean()))
                frame_stds.append(float(image.std()))
                frame_non_black.append(float(np.count_nonzero(image > 8) / image.size))
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            assert (
                world.apply_scene_command(_drag_command(world, "end", SceneCommandKind.DRAG_END)).status
                is SceneCommandStatus.APPLIED
            )
            final_box = next(item for item in world.scene_snapshot().entities if item.path == EntityPath("/box"))
            error = max(
                abs(actual - expected)
                for actual, expected in zip(final_box.pose.position, final_target.position, strict=True)
            )
            assert error < 1.0e-4
            result["rgb_mean_range"] = [min(frame_means), max(frame_means)]
            result["rgb_std_range"] = [min(frame_stds), max(frame_stds)]
            result["rgb_non_black_fraction_range"] = [min(frame_non_black), max(frame_non_black)]
            assert max(frame_means) > 1.0
            # A black or flat-color frame can satisfy a mean-only check. Require
            # visible spatial contrast so retained evidence really contains the scene.
            assert min(frame_stds) > 2.0
            assert max(frame_non_black) > 0.005
            duplicate = world.apply_scene_command(_drag_command(world, "end", SceneCommandKind.DRAG_END))
            assert duplicate.status is SceneCommandStatus.DUPLICATE
            delta = world.scene_delta(initial.sequence)
            assert delta.sequence > initial.sequence and len(delta.upserts) == 2
            result["checks"] = [
                "easyapi_native_rigid_and_camera_build",
                "native_scene_snapshot",
                "native_scene_delta",
                "kinematic_drag_begin_update_end",
                "native_rigid_pose_write_and_readback",
                "duplicate_command_idempotency",
                "native_rgb_video",
            ]
            result["final_pose"] = list(final_box.pose.position)
            result["final_pose_max_error_m"] = error
            result["scene_sequence"] = delta.sequence
            result["status"] = "passed"
    finally:
        if writer is not None:
            writer.release()
        sim.close()
        result["finished_at_unix"] = time.time()
        (output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    result = run(args.output.resolve(), frames=args.frames)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

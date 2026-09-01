"""Real Isaac M4 native overlay, trace and video conformance."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from m3_native_conformance import _create_scene_asset, _versions, _world
from unirobosim import (
    ArrayValue,
    CameraModality,
    DebugBatch,
    DebugBudget,
    DebugBus,
    DebugLifetime,
    DebugMeshResource,
    DebugMeshStyle,
    DebugPrimitive,
    DebugPrimitiveKind,
    DebugTraceReader,
    EntityPath,
    NativeWorldDebugSink,
    ParticleFluidCommand,
    PointCommandMode,
    TestDebugSink,
    TraceDebugSink,
    build_portable_viewer,
    replay_debug_trace,
)

from unirobosim_isaaclab import IsaacLabAdapterConfig, create_provider

_MESH_RESOURCE = DebugMeshResource(
    "m4.unit_tetrahedron",
    ArrayValue.from_nested(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype="float32",
    ),
    ArrayValue.from_nested(
        ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        dtype="int32",
    ),
)


def _primitive(kind: DebugPrimitiveKind, frame: int) -> DebugPrimitive:
    phase = frame / 89.0
    common = {
        "primitive_id": kind.value,
        "layer": "acceptance.native",
        "group": "m4-primitives",
        "source": "isaaclab.m4-conformance",
        "kind": kind,
        "environment_indices": (0,),
        "color_rgba": (0.1, 0.95, 1.0, 1.0),
        "size": 0.025,
        "lifetime": DebugLifetime.persistent(),
    }
    if kind is DebugPrimitiveKind.POINT_SET:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[0.7 + phase * 0.8, -0.55, 0.85]]]),
            **(common | {"size": 0.12, "color_rgba": (1.0, 0.95, 0.05, 1.0)}),
        )
    if kind is DebugPrimitiveKind.LINE_LIST:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[[0.5, -0.65, 0.25], [2.5, -0.65, 0.25]]]]),
            **(common | {"color_rgba": (0.1, 1.0, 0.25, 1.0)}),
        )
    if kind is DebugPrimitiveKind.COORDINATE_AXES:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[1.2, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0]]]),
            **(common | {"size": 0.5}),
        )
    if kind is DebugPrimitiveKind.TEXT:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[0.75, -0.55, 1.2]]]),
            text=((f"M4 DEBUG {frame:02d}",),),
            **(common | {"size": 0.18, "color_rgba": (1.0, 0.6, 0.1, 1.0)}),
        )
    if kind is DebugPrimitiveKind.BOUNDING_BOX:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[2.0, 0.0, 0.35, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 1.0]]]),
            **(common | {"color_rgba": (1.0, 0.1, 0.2, 1.0)}),
        )
    if kind is DebugPrimitiveKind.MESH_INSTANCE:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested(
                [[[1.6, -0.25, 0.55, 0.0, 0.0, 0.0, 1.0, 0.45, 0.45, 0.45]]]
            ),
            mesh_resource_id=_MESH_RESOURCE.resource_id,
            mesh_style=DebugMeshStyle.SOLID_WITH_EDGES,
            **(common | {"color_rgba": (0.15, 0.45, 1.0, 0.7)}),
        )
    trajectory = tuple(
        (
            0.55 + index * 0.34,
            0.55,
            0.45 + 0.25 * math.sin(index * 0.8 + phase * math.pi * 2.0),
        )
        for index in range(6)
    )
    return DebugPrimitive(
        geometry_m=ArrayValue.from_nested([trajectory]),
        sample_times_s=ArrayValue.from_nested([[index * 0.2 for index in range(6)]]),
        **(common | {"color_rgba": (0.85, 0.2, 1.0, 1.0)}),
    )


def _batch(frame: int) -> DebugBatch:
    return DebugBatch(
        tuple(_primitive(kind, frame) for kind in DebugPrimitiveKind),
        step_index=frame,
        sim_time_s=frame / 30.0,
        world_generation=1,
        event_id=f"native-frame-{frame:04d}",
        mesh_resources=(_MESH_RESOURCE,),
    )


def run(output_dir: Path, *, frames: int = 90) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "m4-native-debug.mp4"
    trace_path = output_dir / "m4-native-debug.urs-debug.jsonl"
    viewer_path = output_dir / "m4-native-portable-viewer.html"
    started = time.time()
    result: dict[str, Any] = {"status": "running", "versions": _versions(), "checks": []}
    provider = create_provider(IsaacLabAdapterConfig(headless=True, device="cuda:0", enable_cameras=True, render=True))
    probe = provider.probe()
    result["probe"] = {
        "available": probe.available,
        "reason": probe.reason,
        "details": probe.details.to_dict(),
    }
    if not probe.available:
        raise RuntimeError(f"native profile unavailable: {probe.reason}")

    session = provider.open()
    writer: Any | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="unirobosim-m4-") as directory:
            asset = Path(directory) / "scene.usda"
            _create_scene_asset(asset)
            world = session.build(_world(asset, include_rigid=True))
            camera = world.resolve(EntityPath("/sensors/front"))
            fluid = world.resolve(EntityPath("/matter/water"))
            world.apply_particle_fluid_command(
                ParticleFluidCommand(
                    fluid,
                    PointCommandMode.VELOCITY,
                    ArrayValue.from_nested([[[0.25, 0.0, 0.15]]]),
                    environment_indices=(0,),
                    particle_indices=(0,),
                )
            )
            trace_sink = TraceDebugSink(trace_path, run_id="m4-native-acceptance")
            debug_bus = DebugBus(
                (NativeWorldDebugSink(world), trace_sink),
                budget=DebugBudget(
                    max_active_primitives=16,
                    max_primitives_per_publish=16,
                    max_vertices_per_publish=50_000,
                    max_events_per_second=10_000,
                    max_payload_bytes_per_second=128 * 1024 * 1024,
                    max_publish_duration_ms=1000.0,
                ),
            )
            temporary = DebugPrimitive(
                "temporary",
                "acceptance.native",
                DebugPrimitiveKind.POINT_SET,
                ArrayValue.from_nested([[[1.0, 0.0, 1.4]]]),
                (0,),
                group="lifetime-check",
                color_rgba=(1.0, 1.0, 1.0, 1.0),
                size=0.1,
                lifetime=DebugLifetime.steps(2),
            )
            assert debug_bus.publish(DebugBatch((temporary,))).accepted_count == 1

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
            maximum_active = 0
            group_clear_count = 0
            for frame_index in range(frames):
                report = debug_bus.publish(_batch(frame_index))
                assert report.accepted_count == len(DebugPrimitiveKind)
                maximum_active = max(maximum_active, report.active_count)
                before = world.tick
                rgb = world.read_sensor(camera).channel(CameraModality.RGB)
                assert world.tick == before and rgb.shape == (1, 180, 320, 3)
                image = np.asarray(rgb.values, dtype=np.uint8).reshape(rgb.shape)[0]
                rgb_min = min(rgb_min, int(image.min()))
                rgb_max = max(rgb_max, int(image.max()))
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                world.step()
                debug_bus.advance()
                if frame_index == frames // 2:
                    group_clear_count = debug_bus.clear(group="m4-primitives")
                    assert group_clear_count == len(DebugPrimitiveKind)
            writer.release()
            writer = None
            assert rgb_max > rgb_min
            assert debug_bus.active_count == len(DebugPrimitiveKind)
            debug_bus.close()
            trace = DebugTraceReader().read(trace_path)
            replay_sink = TestDebugSink()
            replay = replay_debug_trace(trace, replay_sink)
            viewer = build_portable_viewer(trace, viewer_path, title="M4 Isaac Native Trace")
            assert replay.final_active_count == len(DebugPrimitiveKind)
            assert {item.kind for item in replay_sink.primitives} == set(DebugPrimitiveKind)
            world.close()

            result["checks"] = [
                {"name": "native_all_primitives_visible_and_updated", "passed": True},
                {"name": "native_stable_key_group_clear_and_step_lifetime", "passed": True},
                {"name": "native_trace_replay_matches_final_active_state", "passed": True},
                {"name": "native_rgb_capture_nonconstant", "passed": True},
            ]
            result.update(
                {
                    "frames": frames,
                    "rgb_range": [rgb_min, rgb_max],
                    "maximum_active_primitives": maximum_active,
                    "group_clear_count": group_clear_count,
                    "trace_events": trace.manifest.event_count,
                    "trace_primitives": trace.manifest.primitive_count,
                    "viewer_frames": viewer.frame_count,
                    "video": str(video_path),
                    "trace": str(trace_path),
                    "viewer": str(viewer_path),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    try:
        result = run(args.output_dir, frames=args.frames)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "versions": _versions(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

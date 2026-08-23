"""DROID acceptance hook for FastSim's shared three-backend runner.

The hook owns only Isaac-specific world construction and authority-thread
telemetry capture.  FastSim remains the sole owner of chunk scheduling,
tickets, lifecycle events, traces, and video encoding.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar, NoReturn, cast

from unirobosim import (
    PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION,
    BoxGeometrySpec,
    CameraModality,
    CameraSpec,
    CapabilityId,
    CapabilityRequirement,
    EntityKind,
    EntityPath,
    EntitySpec,
    FrozenMap,
    Pose,
    WorldSpec,
)

from ._version import DISTRIBUTION_VERSION
from .config import IsaacLabAdapterConfig
from .provider import IsaacLabProvider

_ASSET = Path("/home/ubuntu/projects/gen_data/data/robots/droid/droid.usd")
_ASSET_SHA256 = "50265df8344bca0677dc76ffe3d08fe427bd69ab588213328c2973d86351d71c"
_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
_GRIPPER_JOINTS = (
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
)
_JOINTS = _ARM_JOINTS + _GRIPPER_JOINTS
_CAMERA_PATH = EntityPath("/camera")
_EXPECTED_SPEC_SCHEMA = "fastsim-droid-three-backend-equivalence/4"
_RUN_KINDS = frozenset({"rulebased_blocking", "model_servo_preempt"})
_XWININFO_WINDOW = re.compile(r'^\s*(0x[0-9a-fA-F]+)\s+(?:"([^"]*)"|\(has no name\)):', re.MULTILINE)
_ISAAC_WINDOW_MARKERS = ("isaac", "omniverse", "kit")


def _xwininfo(display: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the fixed X11 observer without involving a shell."""

    try:
        return subprocess.run(
            ("xwininfo", "-display", display, *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as caught:
        raise RuntimeError("xwininfo is unavailable for visible Isaac window evidence") from caught


def _x11_window_snapshot(display: str) -> dict[str, str]:
    result = _xwininfo(display, "-root", "-tree")
    if result.returncode != 0:
        raise RuntimeError(f"xwininfo could not inspect DISPLAY={display!r}: {result.stderr.strip()}")
    return {window_id.lower(): (title or "") for window_id, title in _XWININFO_WINDOW.findall(result.stdout)}


def _x11_window_details(display: str, window_id: str) -> tuple[bool, str]:
    result = _xwininfo(display, "-id", window_id)
    details = result.stdout + result.stderr
    return result.returncode == 0 and "Map State: IsViewable" in result.stdout, details


def _wait_for_native_window(
    display: str,
    baseline_window_ids: frozenset[str],
    *,
    timeout_s: float = 15.0,
) -> tuple[str, str, str]:
    deadline = time.monotonic() + timeout_s
    last_candidates: list[str] = []
    while True:
        snapshot = _x11_window_snapshot(display)
        candidates: list[tuple[str, str, str]] = []
        last_candidates.clear()
        for window_id, title in snapshot.items():
            if window_id in baseline_window_ids:
                continue
            viewable, details = _x11_window_details(display, window_id)
            if not viewable:
                continue
            last_candidates.append(f"{window_id}={title!r}")
            if title and any(marker in title.casefold() for marker in _ISAAC_WINDOW_MARKERS):
                candidates.append((window_id, title, details))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Prefer the largest native application surface over helper windows.
            def area(candidate: tuple[str, str, str]) -> int:
                match = re.search(r"Width:\s*(\d+).*?Height:\s*(\d+)", candidate[2], re.DOTALL)
                return 0 if match is None else int(match[1]) * int(match[2])

            return max(candidates, key=area)
        if time.monotonic() >= deadline:
            observed = ", ".join(last_candidates) if last_candidates else "none"
            raise RuntimeError(
                f"no new viewable Isaac native window appeared on DISPLAY={display}; observed={observed}"
            )
        time.sleep(0.1)


def _wait_for_window_close(display: str, window_id: str, *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        result = _xwininfo(display, "-id", window_id)
        if result.returncode != 0:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _number_tuple(value: object, count: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{label} must contain finite numbers")
        result.append(float(item))
    return tuple(result)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    length = math.sqrt(sum(value * value for value in values))
    if length <= 0.0:
        raise ValueError("camera look-at direction is degenerate")
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


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _rotate_vector_xyzw(
    orientation: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    axis = orientation[:3]
    cross = _cross(axis, vector)
    doubled = cast(tuple[float, float, float], tuple(2.0 * value for value in cross))
    correction = _cross(axis, doubled)
    return cast(
        tuple[float, float, float],
        tuple(vector[index] + orientation[3] * doubled[index] + correction[index] for index in range(3)),
    )


def _effective_camera_calibration(
    native: object,
    *,
    focus_distance_m: float,
    up_reference: tuple[float, float, float],
) -> Mapping[str, object]:
    calibration = cast(Any, native)
    resolution = tuple(calibration.resolution_px)
    intrinsic = tuple(float(value) for value in calibration.intrinsic_matrix)
    if len(resolution) != 2 or any(type(value) is not int or value <= 0 for value in resolution):
        raise RuntimeError("Isaac effective camera resolution is invalid")
    if len(intrinsic) != 9 or any(not math.isfinite(value) for value in intrinsic):
        raise RuntimeError("Isaac effective camera intrinsic matrix is invalid")
    width, height = resolution
    fx, fy = intrinsic[0], intrinsic[4]
    if fx <= 0.0 or fy <= 0.0:
        raise RuntimeError("Isaac effective camera focal lengths must be positive")
    if str(calibration.projection) != "perspective":
        raise RuntimeError(f"Isaac effective camera projection is not pinhole: {calibration.projection!r}")
    position = cast(tuple[float, float, float], tuple(float(value) for value in calibration.position_m))
    orientation = cast(
        tuple[float, float, float, float],
        _normalize(tuple(float(value) for value in calibration.orientation_opengl_xyzw)),
    )
    forward = cast(
        tuple[float, float, float],
        _normalize(_rotate_vector_xyzw(orientation, (0.0, 0.0, -1.0))),
    )
    optical_up = cast(
        tuple[float, float, float],
        _normalize(_rotate_vector_xyzw(orientation, (0.0, 1.0, 0.0))),
    )
    reference = cast(tuple[float, float, float], _normalize(up_reference))
    projected_reference = cast(
        tuple[float, float, float],
        _normalize(tuple(reference[index] - _dot(reference, forward) * forward[index] for index in range(3))),
    )
    if _dot(projected_reference, optical_up) < 1.0 - 1.0e-6:
        raise RuntimeError("Isaac effective camera roll differs from the frozen world-up reference")
    look_at = tuple(position[index] + focus_distance_m * forward[index] for index in range(3))
    clipping = tuple(float(value) for value in calibration.clipping_range_m)
    if len(clipping) != 2 or clipping[0] <= 0.0 or clipping[1] <= clipping[0]:
        raise RuntimeError("Isaac effective camera clipping range is invalid")
    return MappingProxyType(
        {
            "schema_version": "unirobosim-effective-camera-calibration/1",
            "resolution_px": resolution,
            "model": "pinhole",
            "K_row_major": intrinsic,
            "projection": MappingProxyType(
                {
                    "horizontal_fov_deg": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
                    "vertical_fov_deg": math.degrees(2.0 * math.atan(height / (2.0 * fy))),
                    "near_m": clipping[0],
                    "far_m": clipping[1],
                }
            ),
            "extrinsics": MappingProxyType(
                {
                    "eye_m": position,
                    "look_at_m": look_at,
                    "up": reference,
                }
            ),
            "evidence": MappingProxyType(
                {
                    "intrinsics_source": "IsaacLab Camera.data.intrinsic_matrices read after reset",
                    "projection_source": (
                        "effective USD Camera projection/focal/aperture/clipping plus render resolution"
                    ),
                    "extrinsics_source": "IsaacLab native world pose/quaternion with verified world-up roll reference",
                }
            ),
        }
    )


def _look_at_xyzw(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    forward = cast(
        tuple[float, float, float],
        _normalize(tuple(target[index] - eye[index] for index in range(3))),
    )
    right = cast(tuple[float, float, float], _normalize(_cross(forward, up_axis)))
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
    return cast(tuple[float, float, float, float], _normalize(result))


def _compile_plan(spec: Mapping[str, object], output_dir: Path) -> object:
    config = import_module("fastsim.config")
    assets = import_module("fastsim.assets")
    robot = _mapping(spec.get("robot"), "acceptance robot")
    initial = _number_tuple(robot.get("initial_joint_position"), 7, "initial arm position") + (0.0,) * 6
    registry = config.ComponentRegistry()
    registry.register(
        {
            "schema": "fastsim-component/1",
            "id": "robot://droid",
            "version": "1.0.0",
            "kind": "robot",
            "semantics": {
                "joints": list(_JOINTS),
                "groups": {"arm": list(_ARM_JOINTS), "gripper": list(_GRIPPER_JOINTS)},
                "joint_units": {joint: "rad" for joint in _JOINTS},
            },
            "variants": {
                "isaaclab": {
                    "resources": {
                        "model": {
                            "uri": str(_ASSET),
                            "sha256": _ASSET_SHA256,
                            "role": "simulation",
                            "format": "model/vnd.usd",
                        }
                    }
                }
            },
        },
        stable=True,
        source="droid_acceptance.py",
        provider="droid-acceptance",
    )
    cache_dir = output_dir / ".fastsim-resource-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolver = assets.ResourceResolver(
        trusted_roots=[_ASSET.parent],
        cache_dir=cache_dir,
        offline=True,
    )
    compiled = config.ConfigCompiler(registry, resolver).compile_mapping(
        {
            "schema": "fastsim/2",
            "name": "droid-three-backend-equivalence-isaaclab",
            "backend": "isaaclab",
            "runtime": {
                "physics_hz": 240.0,
                "control_hz": 240.0,
                "sensor_hz": {},
                "rate_policy": "exact",
                "seed": 17,
            },
            "scenario": {
                "scene": {
                    "robots": {
                        "droid": {
                            "use": "robot://droid",
                            "pose": {
                                "xyz_m": list(
                                    _number_tuple(
                                        _mapping(robot.get("base_pose_world"), "robot base pose").get("position_m"),
                                        3,
                                        "robot base position",
                                    )
                                ),
                                "quat_xyzw": list(
                                    _number_tuple(
                                        _mapping(robot.get("base_pose_world"), "robot base pose").get(
                                            "quaternion_xyzw"
                                        ),
                                        4,
                                        "robot base orientation",
                                    )
                                ),
                            },
                            "scale": [1.0, 1.0, 1.0],
                            "initial_state": {
                                "joints": {
                                    joint: {"position": position, "velocity": 0.0}
                                    for joint, position in zip(_JOINTS, initial, strict=True)
                                }
                            },
                        }
                    }
                }
            },
            "control": {"default_robot": "droid", "precedence": []},
        }
    )
    return compiled.execution_plan


def _enriched_projection(plan: object, spec: Mapping[str, object]) -> Any:
    projection_module = import_module("fastsim.integrations.unirobosim.projection")
    projection = projection_module.project_execution_plan(plan)
    if len(projection.articulations) != 1 or len(projection.world_spec.entities) != 1:
        raise RuntimeError("DROID acceptance projection must contain exactly one articulation")
    articulation = replace(projection.articulations[0], entity_id="droid")
    physical = replace(projection.entities[0], entity_id="droid")
    robot_entity = projection.world_spec.entities[0]
    robot_metadata = robot_entity.metadata.to_dict()
    robot_metadata.update(
        {
            "fastsim_entity_id": "droid",
            "planning_entity_kind": "robot",
            "planning_frame_declarations": {
                "schema": PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION,
                "component_sha256": _ASSET_SHA256,
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
    robot_entity = replace(robot_entity, metadata=FrozenMap(robot_metadata))

    probe = _mapping(spec.get("scene_probe"), "scene probe")
    probe_pose = _mapping(probe.get("pose_world"), "scene probe pose")
    simulation = _mapping(spec.get("simulation"), "simulation")
    gravity = cast(
        tuple[float, float, float],
        _number_tuple(simulation.get("gravity_m_s2"), 3, "simulation gravity"),
    )
    camera = _mapping(spec.get("camera"), "camera")
    eye = cast(tuple[float, float, float], _number_tuple(camera.get("eye_m"), 3, "camera eye"))
    look_at = cast(tuple[float, float, float], _number_tuple(camera.get("look_at_m"), 3, "camera look-at"))
    up = cast(tuple[float, float, float], _number_tuple(camera.get("up"), 3, "camera up"))
    cube = EntitySpec(
        EntityPath("/red-cube"),
        EntityKind.RIGID_BODY,
        pose=Pose(
            cast(
                tuple[float, float, float],
                _number_tuple(probe_pose.get("position_m"), 3, "scene probe position"),
            ),
            cast(
                tuple[float, float, float, float],
                _number_tuple(probe_pose.get("quaternion_xyzw"), 4, "scene probe orientation"),
            ),
        ),
        box=BoxGeometrySpec(
            dimensions_m=cast(
                tuple[float, float, float],
                _number_tuple(probe.get("size_m"), 3, "scene probe size"),
            ),
            mass_kg=0.1,
            color_rgba=cast(
                tuple[float, float, float, float],
                _number_tuple(probe.get("rgba"), 4, "scene probe rgba"),
            ),
        ),
    )
    camera_entity = EntitySpec(
        _CAMERA_PATH,
        EntityKind.CAMERA_SENSOR,
        pose=Pose(eye, _look_at_xyzw(eye, look_at, up)),
        camera=CameraSpec(
            width_px=_integer(camera["width"], "camera width"),
            height_px=_integer(camera["height"], "camera height"),
            modalities=(CameraModality.RGB,),
            horizontal_fov_degrees=_number(camera["horizontal_fov_deg"], "camera horizontal FOV"),
            near_plane_m=_number(camera["near_m"], "camera near plane"),
            far_plane_m=_number(camera["far_m"], "camera far plane"),
        ),
    )
    requirements_by_id = {
        requirement.capability.value: requirement for requirement in projection.world_spec.requirements
    }
    for capability in ("planning.scene@2", "sensor.camera@1", "sensor.camera.rgb@1"):
        requirements_by_id[capability] = CapabilityRequirement(CapabilityId(capability))
    world_spec = _acceptance_world_spec(
        projection.world_spec,
        entities=(robot_entity, cube, camera_entity),
        requirements=tuple(requirements_by_id[key] for key in sorted(requirements_by_id)),
        gravity_m_s2=gravity,
    )
    return replace(
        projection,
        world_spec=world_spec,
        entities=(physical,),
        articulations=(articulation,),
        default_entity_id="droid",
        planning_reads_demanded=True,
    )


def _acceptance_world_spec(
    base: WorldSpec,
    *,
    entities: tuple[EntitySpec, ...],
    requirements: tuple[CapabilityRequirement, ...],
    gravity_m_s2: tuple[float, float, float],
) -> WorldSpec:
    """Apply acceptance-only scene contents and the cross-backend zero-gravity contract."""

    return replace(
        base,
        entities=entities,
        requirements=requirements,
        physics=replace(base.physics, gravity_m_s2=gravity_m_s2),
    )


def _entry_point(provider: IsaacLabProvider) -> object:
    return SimpleNamespace(
        name="isaaclab",
        group="unirobosim.backends",
        value="unirobosim_isaaclab:create_easy_provider",
        dist=SimpleNamespace(name="unirobosim-isaaclab", version=DISTRIBUTION_VERSION),
        load=lambda: lambda: provider,
    )


@dataclass(frozen=True, slots=True)
class _DroidTelemetryRequest:
    """Acceptance-only read executed by FastSim's simulation authority."""

    operation: ClassVar[str] = "isaaclab.droid.telemetry"
    probe: _DroidTelemetryProbe
    expected_generation: int
    expected_tick: int

    def execute(self, backend: object, context: object) -> Mapping[str, object]:
        return self.probe._capture_on_authority(
            backend,
            context,
            expected_generation=self.expected_generation,
            expected_tick=self.expected_tick,
        )


def _raise_telemetry_read_failure(kind: str, message: str) -> NoReturn:
    """Reject a stale diagnostic request without failing FastSim Runtime."""

    authority_reads = import_module("fastsim.runtime._authority_reads")
    kinds = {
        "integrity": authority_reads._AuthorityReadFailureKind.INTEGRITY,
        "stale_generation": authority_reads._AuthorityReadFailureKind.STALE_GENERATION,
    }
    failure = authority_reads._AuthorityReadFailure(kinds[kind], message)
    raise authority_reads._ExpectedAuthorityReadFailure(failure) from None


class _DroidTelemetryProbe:
    """Read DROID state and RGB without changing the formal adapter type.

    Planning resource requests deliberately accept only FastSim's exact
    ``_UniRoboSimAdapter``.  The acceptance recorder therefore submits a
    generation-pinned authority read instead of subclassing or wrapping that
    adapter.  Production lifecycle and planning-resource ownership remain
    identical to a normal FastSim run.
    """

    def __init__(
        self,
        adapter: object,
        kernel: Any,
        sample_stride: int,
        focus_distance_m: float,
        up_reference: tuple[float, float, float],
        width_px: int,
        height_px: int,
    ) -> None:
        self._adapter = adapter
        self._kernel = kernel
        self._sample_stride = sample_stride
        self._focus_distance_m = focus_distance_m
        self._up_reference = up_reference
        self._width_px = width_px
        self._height_px = height_px
        self._lock = threading.Lock()
        self._generation = 0
        self._camera_handle: object | None = None
        self._camera_calibration: tuple[int, Mapping[str, object]] | None = None
        self._ee_frame_id: str | None = None
        self._last_sample: tuple[int, int, Mapping[str, object]] | None = None

    def sample(self, tick: int) -> Mapping[str, object]:
        if type(tick) is not int or tick < 0 or tick % self._sample_stride != 0:
            raise ValueError("Isaac sample tick is outside the frozen 30 Hz schedule")
        snapshot = self._kernel.snapshot
        generation = getattr(snapshot, "generation", None)
        current_tick = getattr(snapshot, "tick", None)
        if type(generation) is not int or generation <= 0 or current_tick != tick:
            raise RuntimeError("Isaac sample request differs from the current runtime state")
        with self._lock:
            if self._last_sample is not None and self._last_sample[:2] == (generation, tick):
                return self._last_sample[2]
        sample = self._submit(generation, tick)
        with self._lock:
            self._last_sample = (generation, tick, sample)
        return sample

    def camera_calibration(self) -> Mapping[str, object]:
        snapshot = self._kernel.snapshot
        generation = getattr(snapshot, "generation", None)
        tick = getattr(snapshot, "tick", None)
        if type(generation) is not int or generation <= 0 or type(tick) is not int or tick < 0:
            raise RuntimeError("Isaac runtime state is unavailable for camera calibration")
        with self._lock:
            calibration = self._camera_calibration
        if calibration is not None and calibration[0] == generation:
            return calibration[1]
        self._submit(generation, tick)
        with self._lock:
            calibration = self._camera_calibration
        if calibration is None or calibration[0] != generation:
            raise RuntimeError("Isaac effective camera calibration is unavailable before reset")
        return calibration[1]

    def close(self) -> None:
        with self._lock:
            self._camera_handle = None
            self._camera_calibration = None
            self._ee_frame_id = None
            self._last_sample = None

    def _submit(self, expected_generation: int, expected_tick: int) -> Mapping[str, object]:
        ticket = self._kernel.submit_authority_read(_DroidTelemetryRequest(self, expected_generation, expected_tick))
        outcome = ticket.result(30.0)
        resolved = outcome.resolve(self._kernel.snapshot)
        sample = getattr(resolved, "value", None)
        if not isinstance(sample, Mapping):
            failure = getattr(resolved, "message", "authority telemetry read did not return a sample")
            raise RuntimeError(str(failure))
        return cast(Mapping[str, object], sample)

    def _capture_on_authority(
        self,
        backend: object,
        context: object,
        *,
        expected_generation: int,
        expected_tick: int,
    ) -> Mapping[str, object]:
        if backend is not self._adapter:
            _raise_telemetry_read_failure("integrity", "Isaac telemetry request reached a foreign adapter")
        generation = getattr(context, "generation", None)
        context_tick = getattr(context, "tick", None)
        if generation != expected_generation or context_tick != expected_tick:
            _raise_telemetry_read_failure(
                "stale_generation",
                "Isaac telemetry request belongs to a stale runtime state",
            )
        with self._lock:
            if generation != self._generation:
                self._generation = generation
                self._camera_handle = None
                self._camera_calibration = None
                self._ee_frame_id = None
                self._last_sample = None

        adapter = cast(Any, backend)
        if adapter._local_tick != expected_tick:
            _raise_telemetry_read_failure(
                "stale_generation",
                "Isaac adapter tick differs from the FastSim runtime tick",
            )
        world = adapter._required_world()
        articulation = adapter._projection.articulations[0]
        joint_state = world.read_articulation(adapter._live[articulation.entity_id].handle)
        with self._lock:
            camera_handle = self._camera_handle
        if camera_handle is None:
            camera_handle = world.resolve(_CAMERA_PATH)
            with self._lock:
                self._camera_handle = camera_handle
        rgb = world.read_sensor(camera_handle).channel(CameraModality.RGB)
        expected_shape = (1, self._height_px, self._width_px, 3)
        if rgb.shape != expected_shape:
            raise RuntimeError(f"Isaac RGB shape differs from frozen acceptance shape: {rgb.shape}")
        with self._lock:
            cached_calibration = self._camera_calibration
        if cached_calibration is None:
            effective_calibration = _effective_camera_calibration(
                world.read_camera_calibration(camera_handle),
                focus_distance_m=self._focus_distance_m,
                up_reference=self._up_reference,
            )
            with self._lock:
                self._camera_calibration = (generation, effective_calibration)

        planning = world.planning_scene_state(0)
        with self._lock:
            ee_frame_id = self._ee_frame_id
        if ee_frame_id is None:
            catalog = world.planning_scene_catalog(0)
            matches = tuple(frame.frame_id for frame in catalog.frames if frame.name == "gripper_center")
            if len(matches) != 1:
                raise RuntimeError("planning scene must expose exactly one gripper_center frame")
            ee_frame_id = matches[0]
            with self._lock:
                self._ee_frame_id = ee_frame_id
        frame_matches = tuple(frame for frame in planning.frames if frame.frame_id == ee_frame_id)
        if len(frame_matches) != 1:
            raise RuntimeError("planning state is missing the gripper_center frame")
        pose = frame_matches[0].world_pose
        positions = tuple(float(value) for value in joint_state.joint_positions.rows()[0])
        if joint_state.joint_names != _JOINTS or len(positions) != len(_JOINTS):
            raise RuntimeError("Isaac articulation state differs from the frozen DROID axis order")
        rgb_bytes = rgb.to_bytes()
        return MappingProxyType(
            {
                "simulation_tick": expected_tick,
                "simulation_time_s": float(cast(Any, context).sim_time_seconds),
                "arm": MappingProxyType({"joint_ids": _ARM_JOINTS, "position_rad": positions[: len(_ARM_JOINTS)]}),
                "gripper": MappingProxyType(
                    {"joint_ids": _GRIPPER_JOINTS, "position_rad": positions[len(_ARM_JOINTS) :]}
                ),
                "end_effector": MappingProxyType(
                    {
                        "frame_id": "gripper_center",
                        "position_m": tuple(float(value) for value in pose.position_m),
                        "quaternion_xyzw": tuple(float(value) for value in pose.orientation_xyzw),
                    }
                ),
                "rgb": MappingProxyType(
                    {
                        "data": rgb_bytes,
                        "width": self._width_px,
                        "height": self._height_px,
                        "format": "rgb8",
                    }
                ),
            }
        )


def _compose(
    plan: object,
    projection: Any,
    provider: IsaacLabProvider,
    sample_stride: int,
    focus_distance_m: float,
    up_reference: tuple[float, float, float],
) -> tuple[object, Any]:
    control = import_module("fastsim.control")
    runtime = import_module("fastsim.runtime")
    adapter_module = import_module("fastsim.integrations.unirobosim.adapter")
    composition = import_module("fastsim.integrations.unirobosim.composition")
    projection_module = import_module("fastsim.integrations.unirobosim.projection")
    services = import_module("fastsim.integrations.unirobosim.services")
    planning_module = import_module("fastsim.integrations.unirobosim._planning_raw")

    adapter = adapter_module._UniRoboSimAdapter(
        projection,
        entry_points=lambda: (_entry_point(provider),),
    )
    rate_policy = control.ScheduleRatePolicy(projection.rate_policy)
    executor = control.ControlChunkExecutor(
        "droid-equivalence-isaaclab",
        controller_registry=composition._controller_registry(),
        dependency_provider=services.PlanWorldDependencies(projection, generation=lambda: adapter.generation),
        capability_provider=services.PlanControlCapabilities(projection),
        options=control.ControlChunkExecutorOptions(rate_policy=rate_policy),
    )
    driver = control.ControlAuthorityDriver(executor, adapter)
    planning_raw = planning_module._RawPlanningRoot(
        run_id="droid-equivalence-isaaclab",
        plan_digest=projection.plan_digest,
    )
    planning_bridge = planning_module._RawPlanningBridge(
        adapter._planning_capture,
        adapter._planning_delta,
        adapter._planning_revoke_resources,
        planning_raw,
        provider_id=projection.backend.provider_id,
        world_id=projection.world_spec.world_id,
    )
    kernel = runtime.RuntimeKernel(
        plan,
        adapter,
        run_id="droid-equivalence-isaaclab",
        seed=17,
        options=runtime.RuntimeOptions(step_pacing_seconds=0.0),
        authority_participants=(driver.participant_spec(), planning_bridge.participant_spec()),
    )
    camera_entities = tuple(entity for entity in projection.world_spec.entities if entity.path == _CAMERA_PATH)
    if len(camera_entities) != 1 or camera_entities[0].camera is None:
        raise RuntimeError("DROID acceptance projection must contain exactly one scene camera")
    camera_spec = camera_entities[0].camera
    adapter._droid_telemetry_probe = _DroidTelemetryProbe(
        adapter,
        kernel,
        sample_stride,
        focus_distance_m,
        up_reference,
        camera_spec.width_px,
        camera_spec.height_px,
    )
    planning_raw._bind_authority_reads(kernel.submit_authority_read, lambda: kernel.snapshot)
    executor.bind_authority_submitter(kernel.submit_authority)
    for observation_provider in projection_module.observation_providers(projection):
        kernel.observations.register(observation_provider)
    bundle = composition.UniRoboSimRuntimeBundle(
        runtime=kernel,
        control=control.ControlService(executor),
        projection=projection,
        adapter=adapter,
        executor=executor,
        planning_raw=planning_raw,
    )
    return bundle, adapter


@dataclass(slots=True)
class DroidAcceptanceBackendRun:
    """Unprepared real FastSim bundle plus detached Isaac telemetry."""

    bundle: object
    _adapter: Any = field(repr=False)
    _gravity_m_s2: tuple[float, float, float] = field(repr=False)
    _visible_window: bool = field(repr=False)
    _display: str | None = field(repr=False)
    _baseline_window_ids: frozenset[str] = field(repr=False)
    _output_dir: Path = field(repr=False)
    _run_kind: str = field(repr=False)
    _closed: bool = field(default=False, repr=False)
    _window_evidence: Mapping[str, object] | None = field(default=None, repr=False)
    _window_id: str | None = field(default=None, repr=False)

    def sample(self, tick: int) -> Mapping[str, object]:
        probe = getattr(self._adapter, "_droid_telemetry_probe", None)
        if type(probe) is not _DroidTelemetryProbe:
            raise RuntimeError("Isaac simulator-camera telemetry is unavailable")
        return probe.sample(tick)

    @property
    def camera_calibration(self) -> Mapping[str, object]:
        probe = getattr(self._adapter, "_droid_telemetry_probe", None)
        if type(probe) is not _DroidTelemetryProbe:
            raise RuntimeError("Isaac simulator-camera calibration is unavailable")
        return probe.camera_calibration()

    @property
    def physics_diagnostics(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "gravity_m_s2": self._gravity_m_s2,
                "source": "effective UniRoboSim WorldSpec used for native build",
            }
        )

    @property
    def window_evidence(self) -> Mapping[str, object]:
        if self._window_evidence is not None:
            return self._window_evidence
        if not self._visible_window:
            self._window_evidence = MappingProxyType(
                {
                    "schema_version": "fastsim-visible-window-evidence/1",
                    "requested": False,
                    "headless": True,
                    "observed": False,
                    "display": "not requested",
                    "native_window_id": "not requested",
                    "window_title": "not requested",
                    "source": "visible_window=False; xwininfo was not invoked",
                }
            )
            return self._window_evidence
        if self._display is None:
            raise RuntimeError("visible Isaac window was requested without DISPLAY")
        window_id, title, details = _wait_for_native_window(self._display, self._baseline_window_ids)
        self._window_id = window_id
        command = f"xwininfo -display {self._display} -id {window_id}"
        evidence_path = self._output_dir / f"droid-{self._run_kind}-isaaclab.window-xwininfo.txt"
        evidence_path.write_text(f"$ {command}\n{details}", encoding="utf-8")
        self._window_evidence = MappingProxyType(
            {
                "schema_version": "fastsim-visible-window-evidence/1",
                "requested": True,
                "headless": False,
                "observed": True,
                "display": self._display,
                "native_window_id": window_id,
                "window_title": title,
                "source": f"{command}: Map State: IsViewable",
            }
        )
        return self._window_evidence

    def close(self) -> None:
        if self._closed:
            return
        probe = getattr(self._adapter, "_droid_telemetry_probe", None)
        if type(probe) is _DroidTelemetryProbe:
            probe.close()
        if self._visible_window and self._display is not None and self._window_id is not None:
            observed_closed = _wait_for_window_close(self._display, self._window_id)
            close_path = self._output_dir / f"droid-{self._run_kind}-isaaclab.window-close.json"
            close_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fastsim-visible-window-close-evidence/1",
                        "display": self._display,
                        "native_window_id": self._window_id,
                        "observed_closed": observed_closed,
                        "source": "xwininfo returned nonzero after native bundle close",
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if not observed_closed:
                raise RuntimeError(f"Isaac native window {self._window_id} remained after bundle close")
        self._closed = True


def create_backend_run(
    spec: Mapping[str, object],
    run_kind: str,
    output_dir: str | Path,
    *,
    visible_window: bool = False,
) -> DroidAcceptanceBackendRun:
    """Create the unprepared Isaac/FastSim bundle required by the shared runner."""

    canonical = _mapping(spec, "acceptance spec")
    if canonical.get("schema_version") != _EXPECTED_SPEC_SCHEMA:
        raise ValueError("unsupported DROID acceptance spec schema")
    if run_kind not in _RUN_KINDS:
        raise ValueError("unsupported DROID acceptance run kind")
    if type(visible_window) is not bool:
        raise TypeError("visible_window must be a bool")
    simulation = _mapping(canonical.get("simulation"), "acceptance simulation")
    physics_hz = _integer(simulation.get("physics_hz", 0), "physics_hz")
    camera = _mapping(canonical.get("camera"), "acceptance camera")
    sample_hz = _integer(camera.get("fps", 0), "camera fps")
    if physics_hz != 240 or sample_hz != 30 or physics_hz % sample_hz:
        raise ValueError("Isaac hook requires the frozen 240 Hz / 30 Hz schedule")
    if not _ASSET.is_file() or hashlib.sha256(_ASSET.read_bytes()).hexdigest() != _ASSET_SHA256:
        raise RuntimeError("pinned DROID Isaac USD is missing or changed")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    display = os.environ.get("DISPLAY") if visible_window else None
    if visible_window and (display is None or not display.strip()):
        raise RuntimeError("visible Isaac window requires a non-empty DISPLAY")
    baseline_window_ids = frozenset(_x11_window_snapshot(display)) if display is not None else frozenset()
    plan = _compile_plan(canonical, destination)
    projection = _enriched_projection(plan, canonical)
    provider = IsaacLabProvider(
        IsaacLabAdapterConfig(
            headless=not visible_window,
            enable_cameras=True,
            render=True,
            render_on_step=False,
            anti_aliasing="fxaa",
            texture_streaming=False,
        )
    )
    eye = cast(tuple[float, float, float], _number_tuple(camera.get("eye_m"), 3, "camera eye"))
    look_at = cast(tuple[float, float, float], _number_tuple(camera.get("look_at_m"), 3, "camera look-at"))
    up_reference = cast(tuple[float, float, float], _number_tuple(camera.get("up"), 3, "camera up"))
    focus_distance_m = math.sqrt(sum((look_at[index] - eye[index]) ** 2 for index in range(3)))
    bundle, adapter = _compose(
        plan,
        projection,
        provider,
        physics_hz // sample_hz,
        focus_distance_m,
        up_reference,
    )
    return DroidAcceptanceBackendRun(
        bundle,
        adapter,
        projection.world_spec.physics.gravity_m_s2,
        visible_window,
        display,
        baseline_window_ids,
        destination,
        run_kind,
    )


__all__ = ("DroidAcceptanceBackendRun", "create_backend_run")

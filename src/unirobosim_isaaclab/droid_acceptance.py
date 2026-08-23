"""DROID acceptance hook for FastSim's shared three-backend runner.

The hook owns only Isaac-specific world construction and authority-thread
telemetry capture.  FastSim remains the sole owner of chunk scheduling,
tickets, lifecycle events, traces, and video encoding.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

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
_EXPECTED_SPEC_SCHEMA = "fastsim-droid-three-backend-equivalence/3"
_RUN_KINDS = frozenset({"rulebased_blocking", "model_servo_preempt"})


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
        value="unirobosim_isaaclab:create_easy_provider",
        dist=SimpleNamespace(name="unirobosim-isaaclab", version=DISTRIBUTION_VERSION),
        load=lambda: lambda: provider,
    )


def _telemetry_adapter_type(base: type[Any]) -> type[Any]:
    class DroidTelemetryAdapter(base):  # type: ignore[misc]
        def __init__(
            self,
            projection: object,
            provider: IsaacLabProvider,
            sample_stride: int,
            focus_distance_m: float,
            up_reference: tuple[float, float, float],
        ) -> None:
            super().__init__(projection, entry_points=lambda: (_entry_point(provider),))
            self._droid_sample_stride = sample_stride
            self._droid_focus_distance_m = focus_distance_m
            self._droid_up_reference = up_reference
            self._droid_sample_lock = threading.Lock()
            self._droid_samples: dict[int, Mapping[str, object]] = {}
            self._droid_last_sample: tuple[int, Mapping[str, object]] | None = None
            self._droid_camera_handle: object | None = None
            self._droid_camera_calibration: Mapping[str, object] | None = None
            self._droid_ee_frame_id: str | None = None

        def prepare(self, plan: object, *, seed: int) -> object:
            tick = super().prepare(plan, seed=seed)
            self._droid_capture()
            return tick

        def reset(self, *, seed: int) -> object:
            tick = super().reset(seed=seed)
            with self._droid_sample_lock:
                self._droid_samples.clear()
                self._droid_last_sample = None
            self._droid_camera_handle = None
            self._droid_camera_calibration = None
            self._droid_ee_frame_id = None
            self._droid_capture()
            return tick

        def step(self) -> object:
            tick = super().step()
            if self._local_tick % self._droid_sample_stride == 0:
                self._droid_capture()
            return tick

        def _droid_capture(self) -> None:
            world = self._required_world()
            articulation = self._projection.articulations[0]
            joint_state = world.read_articulation(self._live[articulation.entity_id].handle)
            if self._droid_camera_handle is None:
                self._droid_camera_handle = world.resolve(_CAMERA_PATH)
            rgb = world.read_sensor(self._droid_camera_handle).channel(CameraModality.RGB)
            if rgb.shape != (1, 1080, 1920, 3):
                raise RuntimeError(f"Isaac RGB shape differs from frozen acceptance shape: {rgb.shape}")
            if self._droid_camera_calibration is None:
                self._droid_camera_calibration = _effective_camera_calibration(
                    world.read_camera_calibration(self._droid_camera_handle),
                    focus_distance_m=self._droid_focus_distance_m,
                    up_reference=self._droid_up_reference,
                )
            planning = world.planning_scene_state(0)
            if self._droid_ee_frame_id is None:
                catalog = world.planning_scene_catalog(0)
                matches = tuple(frame.frame_id for frame in catalog.frames if frame.name == "gripper_center")
                if len(matches) != 1:
                    raise RuntimeError("planning scene must expose exactly one gripper_center frame")
                self._droid_ee_frame_id = matches[0]
            frame_matches = tuple(frame for frame in planning.frames if frame.frame_id == self._droid_ee_frame_id)
            if len(frame_matches) != 1:
                raise RuntimeError("planning state is missing the gripper_center frame")
            pose = frame_matches[0].world_pose
            positions = tuple(float(value) for value in joint_state.joint_positions.rows()[0])
            if joint_state.joint_names != _JOINTS or len(positions) != len(_JOINTS):
                raise RuntimeError("Isaac articulation state differs from the frozen DROID axis order")
            tick = int(self._local_tick)
            rgb_bytes = bytes(cast(tuple[int, ...], rgb.values))
            sample: Mapping[str, object] = MappingProxyType(
                {
                    "simulation_tick": tick,
                    "simulation_time_s": tick / 240.0,
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
                            "width": 1920,
                            "height": 1080,
                            "format": "rgb8",
                        }
                    ),
                }
            )
            with self._droid_sample_lock:
                if len(self._droid_samples) >= 4:
                    raise RuntimeError("runner did not consume bounded Isaac telemetry in time")
                self._droid_samples[tick] = sample

        def droid_sample(self, tick: int) -> Mapping[str, object]:
            if type(tick) is not int or tick < 0 or tick % self._droid_sample_stride != 0:
                raise ValueError("Isaac sample tick is outside the frozen 30 Hz schedule")
            with self._droid_sample_lock:
                if self._droid_last_sample is not None and self._droid_last_sample[0] == tick:
                    return self._droid_last_sample[1]
                try:
                    sample = self._droid_samples.pop(tick)
                except KeyError:
                    raise RuntimeError(f"Isaac authority sample for tick {tick} is unavailable") from None
                self._droid_last_sample = (tick, sample)
                return sample

        def droid_camera_calibration(self) -> Mapping[str, object]:
            if self._droid_camera_calibration is None:
                raise RuntimeError("Isaac effective camera calibration is unavailable before reset")
            return self._droid_camera_calibration

        def close(self) -> None:
            try:
                super().close()
            finally:
                with self._droid_sample_lock:
                    self._droid_samples.clear()
                    self._droid_last_sample = None

    return DroidTelemetryAdapter


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

    adapter_type = _telemetry_adapter_type(adapter_module._UniRoboSimAdapter)
    adapter = adapter_type(projection, provider, sample_stride, focus_distance_m, up_reference)
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
    _closed: bool = field(default=False, repr=False)

    def sample(self, tick: int) -> Mapping[str, object]:
        return cast(Mapping[str, object], self._adapter.droid_sample(tick))

    @property
    def camera_calibration(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self._adapter.droid_camera_calibration())

    @property
    def physics_diagnostics(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "gravity_m_s2": self._gravity_m_s2,
                "source": "effective UniRoboSim WorldSpec used for native build",
            }
        )

    def close(self) -> None:
        self._closed = True


def create_backend_run(
    spec: Mapping[str, object],
    run_kind: str,
    output_dir: str | Path,
) -> DroidAcceptanceBackendRun:
    """Create the unprepared Isaac/FastSim bundle required by the shared runner."""

    canonical = _mapping(spec, "acceptance spec")
    if canonical.get("schema_version") != _EXPECTED_SPEC_SCHEMA:
        raise ValueError("unsupported DROID acceptance spec schema")
    if run_kind not in _RUN_KINDS:
        raise ValueError("unsupported DROID acceptance run kind")
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
    plan = _compile_plan(canonical, destination)
    projection = _enriched_projection(plan, canonical)
    provider = IsaacLabProvider(
        IsaacLabAdapterConfig(
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
    return DroidAcceptanceBackendRun(bundle, adapter, projection.world_spec.physics.gravity_m_s2)


__all__ = ("DroidAcceptanceBackendRun", "create_backend_run")

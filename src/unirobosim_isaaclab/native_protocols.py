"""Narrow SDK-independent seam between public runtime and Isaac-native implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from unirobosim import (
    CameraModality,
    CommandMode,
    DebugBatch,
    EntityPath,
    KinematicTarget,
    PlanningArticulationState,
    PlanningAttachment,
    PlanningEntityDescriptor,
    PlanningEntityState,
    PlanningFrameDescriptor,
    PlanningFrameState,
    PlanningGeometryDescriptor,
    PlanningGeometryRepresentation,
    PlanningGeometryTransform,
    PlanningJointDescriptor,
    PlanningLinkDescriptor,
    PlanningLinkState,
    PointCommandMode,
    Pose,
    WorldSpec,
)

Matrix = tuple[tuple[float, ...], ...]
Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
PointBatch = tuple[tuple[Vector3, ...], ...]
NativeSensorValues = tuple[float | int, ...] | bytes
NativeSensorChannel = tuple[CameraModality, tuple[int, ...], NativeSensorValues]
NativeSensorSample = tuple[NativeSensorChannel, ...]
NativeSensorBatch = tuple[NativeSensorSample, ...]
NativeDebugReport = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class NativePhysicsDiagnostics:
    """Effective native timing read back from an initialized simulator."""

    native_step_dt_seconds: float
    substeps: int
    world_step_dt_seconds: float
    source: str


@dataclass(frozen=True, slots=True)
class NativeCameraCalibration:
    """Effective calibration read back from one initialized native camera."""

    resolution_px: tuple[int, int]
    intrinsic_matrix: tuple[float, ...]
    projection: str
    focal_length: float
    horizontal_aperture: float
    clipping_range_m: tuple[float, float]
    position_m: Vector3
    orientation_opengl_xyzw: Quaternion


@dataclass(frozen=True, slots=True)
class NativeKinematicState:
    """One selected environment-local body state returned by the native SDK."""

    target_id: str
    entity_path: EntityPath
    link_name: str | None
    position_m: Vector3
    orientation_xyzw: Quaternion
    linear_velocity_m_s: Vector3
    angular_velocity_rad_s: Vector3


class NativePlanningError(RuntimeError):
    """Bounded private failure transported across the native worker seam."""

    _CODES = frozenset(
        {
            "catalog_invalid",
            "attachment_entity_missing",
            "attachment_frame_missing",
            "attachment_geometry_missing",
            "attachment_link_missing",
            "attachment_pose_invalid",
            "attachment_value_invalid",
            "attachment_pose_missing",
            "collision_cooking_failed",
            "collision_filter_unsupported",
            "collision_geometry_unsupported",
            "constraint_unsupported",
            "frame_ambiguous",
            "frame_missing",
            "generation_stale",
            "hash_mismatch",
            "native_failure",
            "resource_missing",
            "soft_matter_unsupported",
            "topology_unsupported",
        }
    )

    def __init__(self, code: str) -> None:
        canonical = code if type(code) is str and code in self._CODES else "native_failure"
        super().__init__(canonical)
        self.code = canonical


@dataclass(frozen=True, slots=True)
class NativePlanningCatalog:
    """Portable, generation-neutral catalog produced by native admission."""

    entities: tuple[PlanningEntityDescriptor, ...]
    links: tuple[PlanningLinkDescriptor, ...]
    joints: tuple[PlanningJointDescriptor, ...]
    frames: tuple[PlanningFrameDescriptor, ...]
    geometries: tuple[PlanningGeometryDescriptor, ...]


@dataclass(frozen=True, slots=True)
class NativePlanningState:
    """One coherent environment-local state captured from a committed tick."""

    step_index: int
    entities: tuple[PlanningEntityState, ...]
    links: tuple[PlanningLinkState, ...]
    frames: tuple[PlanningFrameState, ...]
    articulations: tuple[PlanningArticulationState, ...]
    geometry_transforms: tuple[PlanningGeometryTransform, ...]
    attachments: tuple[PlanningAttachment, ...]


@dataclass(frozen=True, slots=True)
class NativePlanningResource:
    """One lazily materialized canonical geometry payload."""

    geometry_id: str
    representation: PlanningGeometryRepresentation
    content: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class NativeArticulationCommand:
    """One fully validated articulation setter queued for a native physics tick."""

    path: EntityPath
    mode: CommandMode
    targets: Matrix
    environment_indices: tuple[int, ...]
    degree_of_freedom_indices: tuple[int, ...]


class NativeWorldDriver(Protocol):
    def physics_diagnostics(self) -> NativePhysicsDiagnostics: ...

    def reset(self, environment_indices: tuple[int, ...]) -> None: ...

    def apply_articulation(
        self,
        path: EntityPath,
        mode: CommandMode,
        targets: Matrix,
        environment_indices: tuple[int, ...],
        degree_of_freedom_indices: tuple[int, ...],
    ) -> None: ...

    def read_articulation(self, path: EntityPath) -> tuple[Matrix, Matrix]: ...

    def apply_articulation_commands_and_step(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
    ) -> None: ...

    def apply_articulation_commands_step_and_read(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[Matrix, Matrix], ...]: ...

    def apply_articulation_commands_step_and_read_sensors(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
        sensor_paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[tuple[Matrix, Matrix], ...], NativeSensorBatch]: ...

    def apply_rigid_body_wrench(
        self,
        path: EntityPath,
        forces_n: Matrix,
        torques_n_m: Matrix,
        environment_indices: tuple[int, ...],
    ) -> None: ...

    def read_rigid_body(self, path: EntityPath) -> tuple[Matrix, Matrix, Matrix, Matrix]: ...

    def set_rigid_body_pose(
        self,
        path: EntityPath,
        position_m: Vector3,
        orientation_xyzw: Quaternion,
        environment_index: int,
    ) -> None: ...

    def attach_rigid_body(
        self,
        attachment_id: str,
        parent_path: EntityPath,
        parent_link_name: str | None,
        child_path: EntityPath,
        child_link_name: str | None,
        environment_index: int,
        parent_T_child: Pose | None,
    ) -> Pose: ...

    def detach_rigid_body(
        self,
        attachment_id: str,
        child_path: EntityPath,
        environment_index: int,
    ) -> None: ...

    def read_contact(self, path: EntityPath) -> Matrix: ...

    def apply_deformable_position(
        self,
        path: EntityPath,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        point_indices: tuple[int, ...],
    ) -> None: ...

    def read_deformable(self, path: EntityPath) -> tuple[PointBatch, PointBatch]: ...

    def apply_particle_fluid(
        self,
        path: EntityPath,
        mode: PointCommandMode,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        particle_indices: tuple[int, ...],
    ) -> None: ...

    def read_particle_fluid(self, path: EntityPath) -> tuple[PointBatch, PointBatch]: ...

    def read_sensor(self, path: EntityPath) -> NativeSensorSample: ...

    def read_sensors(self, paths: tuple[EntityPath, ...]) -> NativeSensorBatch: ...

    def camera_calibration(self, path: EntityPath) -> NativeCameraCalibration: ...

    def read_selected_kinematics(
        self,
        targets: tuple[KinematicTarget, ...],
        environment_index: int = 0,
    ) -> tuple[NativeKinematicState, ...]: ...

    def publish_debug(self, batch: DebugBatch) -> NativeDebugReport: ...

    def clear_debug(self, layer: str | None, group: str | None, primitive_id: str | None) -> int: ...

    def step(self, count: int) -> None: ...

    def close(self) -> None: ...


class NativePlanningWorldDriver(NativeWorldDriver, Protocol):
    def planning_catalog(self, environment_index: int = 0) -> NativePlanningCatalog: ...

    def planning_state(self, environment_index: int = 0) -> NativePlanningState: ...

    def planning_resource(self, geometry_id: str, environment_index: int = 0) -> NativePlanningResource: ...


class NativeRuntime(Protocol):
    def build_world(self, spec: WorldSpec) -> NativeWorldDriver: ...

    def close(self) -> None: ...

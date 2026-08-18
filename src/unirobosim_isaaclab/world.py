"""Strict public world facade over the narrow native driver."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import unquote, urlparse

from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    ArticulationState,
    BuildFingerprint,
    BuildReport,
    CameraModality,
    CommandError,
    ContactState,
    DebugBatch,
    DebugPublishReport,
    DeformableCommand,
    DeformableState,
    EntityHandle,
    EntityKind,
    EntityNotFoundError,
    EntityPath,
    EntitySpec,
    FrozenMap,
    LifecycleError,
    ParticleFluidCommand,
    ParticleFluidState,
    PointCommandMode,
    Pose,
    ResetResult,
    RigidBodyCommand,
    RigidBodyState,
    SceneCommand,
    SceneCommandKind,
    SceneCommandResult,
    SceneCommandStatus,
    SceneDelta,
    SceneDragMode,
    SceneEntityState,
    SceneSnapshot,
    SceneVisual,
    SceneVisualKind,
    SensorChannel,
    SensorSample,
    StaleHandleError,
    Tick,
    UniRoboSimError,
    UnsupportedCapabilityError,
    ValidationError,
    WorldBuildError,
    WorldSpec,
    WorldState,
)

from .native_protocols import NativeWorldDriver, PointBatch

if TYPE_CHECKING:
    from .provider import IsaacLabSession

_T = TypeVar("_T")


def _local_asset_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    return None


class IsaacLabWorld:
    def __init__(
        self,
        session: IsaacLabSession,
        spec: WorldSpec,
        generation: int,
        native: NativeWorldDriver,
    ) -> None:
        self._session = session
        self._spec = spec
        self._generation = generation
        self._native = native
        self._state = WorldState.READY
        self._step_index = 0
        self._reset_count = 0
        self._scene_sequence = 0
        self._scene_results: dict[str, SceneCommandResult] = {}
        self._drags: dict[str, tuple[EntityPath, int, Pose]] = {}
        self._entities = {entity.path: entity for entity in spec.entities}
        self._build_report = BuildReport(
            fingerprint=BuildFingerprint(
                provider_id=session.descriptor.provider_id,
                provider_version=session.descriptor.version,
                contract_version=session.descriptor.contract_version,
                world_digest=spec.digest,
                capability_digest=session.descriptor.capabilities.digest,
            ),
            world_id=spec.world_id,
            generation=generation,
            environment_count=spec.environments.count,
            entity_count=len(spec.entities),
        )

    @staticmethod
    def validate_build_spec(spec: WorldSpec, *, backend_id: str) -> None:
        for entity in spec.entities:
            if entity.kind is EntityKind.ARTICULATION:
                if entity.asset_uri is None:
                    raise WorldBuildError(
                        "Isaac Lab articulations require a local USD asset_uri",
                        operation="session.build.preflight",
                        backend_id=backend_id,
                        world_id=spec.world_id,
                        entity_path=entity.path.value,
                    )
                asset = _local_asset_path(entity.asset_uri)
                if asset is None or asset.suffix.lower() not in {".usd", ".usda", ".usdc"} or not asset.is_file():
                    raise WorldBuildError(
                        "articulation asset_uri must resolve to an existing local USD file",
                        operation="session.build.preflight",
                        backend_id=backend_id,
                        world_id=spec.world_id,
                        entity_path=entity.path.value,
                        details={"asset_uri": entity.asset_uri},
                    )
            if entity.kind is EntityKind.RIGID_BODY:
                if entity.asset_uri is None and entity.box is None:
                    raise WorldBuildError(
                        "Isaac Lab rigid bodies require a portable box or local USD asset_uri",
                        operation="session.build.preflight",
                        backend_id=backend_id,
                        world_id=spec.world_id,
                        entity_path=entity.path.value,
                    )
                asset = None if entity.asset_uri is None else _local_asset_path(entity.asset_uri)
                if entity.asset_uri is not None and (
                    asset is None or asset.suffix.lower() not in {".usd", ".usda", ".usdc"} or not asset.is_file()
                ):
                    raise WorldBuildError(
                        "rigid asset_uri must resolve to an existing local USD file",
                        operation="session.build.preflight",
                        backend_id=backend_id,
                        world_id=spec.world_id,
                        entity_path=entity.path.value,
                        details={"asset_uri": entity.asset_uri},
                    )
            if entity.kind is EntityKind.SURFACE_DEFORMABLE and entity.deformable is not None:
                if entity.deformable.kinematic_node_indices:
                    raise UnsupportedCapabilityError(
                        "Isaac Lab 3.0 surface deformables have no kinematic target buffer",
                        operation="session.build.preflight",
                        backend_id=backend_id,
                        world_id=spec.world_id,
                        entity_path=entity.path.value,
                    )

    @property
    def world_id(self) -> str:
        return self._spec.world_id

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def state(self) -> WorldState:
        return self._state

    @property
    def tick(self) -> Tick:
        return Tick(self._step_index, self._step_index * self._spec.physics.time_step_seconds)

    @property
    def build_report(self) -> BuildReport:
        return self._build_report

    def _ensure_ready(self, operation: str) -> None:
        if self._state is not WorldState.READY:
            raise LifecycleError(
                "world is closed",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                details={"state": self._state.value},
            )

    def _native_call(
        self,
        operation: str,
        function: Callable[[], _T],
        *,
        entity_path: str | None = None,
    ) -> _T:
        try:
            return function()
        except UniRoboSimError:
            raise
        except Exception as exc:
            raise UniRoboSimError(
                "Isaac Lab native operation failed",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=entity_path,
                cause=exc,
            ) from exc

    @staticmethod
    def _indices(values: Iterable[int] | None, size: int, name: str, *, operation: str) -> tuple[int, ...]:
        if values is None:
            return tuple(range(size))
        result = tuple(values)
        if (
            not result
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= size for index in result
            )
            or len(result) != len(set(result))
        ):
            raise ValidationError(
                f"{name} must be a non-empty unique in-range selection",
                operation=operation,
                details={"selection": list(result), "size": size},
            )
        return result

    def _handle_token(self, path: EntityPath) -> str:
        raw = f"{self._session.session_id}|{self.world_id}|{self.generation}|{path.value}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def resolve(self, path: EntityPath) -> EntityHandle:
        self._ensure_ready("world.resolve")
        if not isinstance(path, EntityPath):
            raise ValidationError("resolve requires an EntityPath", operation="world.resolve")
        entity = self._entities.get(path)
        if entity is None:
            raise EntityNotFoundError(
                "logical entity path does not exist",
                operation="world.resolve",
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=path.value,
            )
        return EntityHandle(
            provider_id=self._session.descriptor.provider_id,
            session_id=self._session.session_id,
            world_id=self.world_id,
            generation=self.generation,
            path=path,
            entity_kind=entity.kind,
            token=self._handle_token(path),
        )

    def _validate_handle(self, handle: EntityHandle, operation: str) -> EntitySpec:
        if not isinstance(handle, EntityHandle):
            raise StaleHandleError("operation requires an EntityHandle", operation=operation, world_id=self.world_id)
        expected = (
            self._session.descriptor.provider_id,
            self._session.session_id,
            self.world_id,
            self.generation,
            self._handle_token(handle.path),
        )
        actual = (handle.provider_id, handle.session_id, handle.world_id, handle.generation, handle.token)
        entity = self._entities.get(handle.path)
        if actual != expected or entity is None or handle.entity_kind is not entity.kind:
            raise StaleHandleError(
                "entity handle does not belong to this live world generation",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=handle.path.value,
                details={"expected_generation": self.generation, "actual_generation": handle.generation},
            )
        return entity

    def reset(self, environment_indices: Iterable[int] | None = None) -> ResetResult:
        operation = "world.reset"
        self._ensure_ready(operation)
        environments = self._indices(
            environment_indices, self._spec.environments.count, "environment_indices", operation=operation
        )
        self._native_call(operation, lambda: self._native.reset(environments))
        self._reset_count += 1
        self._scene_sequence += 1
        self._drags.clear()
        return ResetResult(environments, self._reset_count, self.tick)

    def apply_articulation_command(self, command: ArticulationCommand) -> None:
        operation = "world.apply_articulation_command"
        self._ensure_ready(operation)
        if not isinstance(command, ArticulationCommand):
            raise CommandError("operation requires an ArticulationCommand", operation=operation)
        entity = self._validate_handle(command.handle, operation)
        if entity.kind is not EntityKind.ARTICULATION:
            raise CommandError("entity is not an articulation", operation=operation, entity_path=entity.path.value)
        environments = self._indices(
            command.environment_indices, self._spec.environments.count, "environment_indices", operation=operation
        )
        degrees = self._indices(
            command.degree_of_freedom_indices, len(entity.joint_names), "degree_of_freedom_indices", operation=operation
        )
        expected = (len(environments), len(degrees))
        if command.targets.shape != expected:
            raise CommandError(
                "target shape must exactly match selected environments and degrees of freedom",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=entity.path.value,
                details={"expected_shape": list(expected), "actual_shape": list(command.targets.shape)},
            )
        targets = tuple(tuple(float(value) for value in row) for row in command.targets.rows())
        self._native_call(
            operation,
            lambda: self._native.apply_articulation(entity.path, command.mode, targets, environments, degrees),
            entity_path=entity.path.value,
        )

    def read_articulation(self, handle: EntityHandle) -> ArticulationState:
        operation = "world.read_articulation"
        self._ensure_ready(operation)
        entity = self._validate_handle(handle, operation)
        if entity.kind is not EntityKind.ARTICULATION:
            raise CommandError("entity is not an articulation", operation=operation, entity_path=entity.path.value)
        position, velocity = self._native_call(
            operation, lambda: self._native.read_articulation(entity.path), entity_path=entity.path.value
        )
        return ArticulationState(ArrayValue.from_rows(position), ArrayValue.from_rows(velocity), self.tick)

    def apply_rigid_body_command(self, command: RigidBodyCommand) -> None:
        operation = "world.apply_rigid_body_command"
        self._ensure_ready(operation)
        if not isinstance(command, RigidBodyCommand):
            raise CommandError("operation requires a RigidBodyCommand", operation=operation)
        entity = self._validate_handle(command.handle, operation)
        if entity.kind is not EntityKind.RIGID_BODY:
            raise CommandError("entity is not a rigid body", operation=operation, entity_path=entity.path.value)
        environments = self._indices(
            command.environment_indices, self._spec.environments.count, "environment_indices", operation=operation
        )
        expected = (len(environments), 3)
        if command.forces_n.shape != expected or command.torques_n_m.shape != expected:
            raise CommandError(
                "force and torque shapes must exactly match selected environments and xyz",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=entity.path.value,
                details={
                    "expected_shape": list(expected),
                    "force_shape": list(command.forces_n.shape),
                    "torque_shape": list(command.torques_n_m.shape),
                },
            )
        forces = tuple(tuple(float(value) for value in row) for row in command.forces_n.rows())
        torques = tuple(tuple(float(value) for value in row) for row in command.torques_n_m.rows())
        self._native_call(
            operation,
            lambda: self._native.apply_rigid_body_wrench(entity.path, forces, torques, environments),
            entity_path=entity.path.value,
        )

    def read_rigid_body(self, handle: EntityHandle) -> RigidBodyState:
        operation = "world.read_rigid_body"
        self._ensure_ready(operation)
        entity = self._validate_handle(handle, operation)
        if entity.kind is not EntityKind.RIGID_BODY:
            raise CommandError("entity is not a rigid body", operation=operation, entity_path=entity.path.value)
        positions, orientations, linear_velocities, angular_velocities = self._native_call(
            operation,
            lambda: self._native.read_rigid_body(entity.path),
            entity_path=entity.path.value,
        )
        return RigidBodyState(
            positions_m=ArrayValue.from_rows(positions),
            orientations_xyzw=ArrayValue.from_rows(orientations),
            linear_velocities_m_s=ArrayValue.from_rows(linear_velocities),
            angular_velocities_rad_s=ArrayValue.from_rows(angular_velocities),
            tick=self.tick,
        )

    def read_contact(self, handle: EntityHandle, force_threshold_n: float = 1.0e-6) -> ContactState:
        operation = "world.read_contact"
        self._ensure_ready(operation)
        entity = self._validate_handle(handle, operation)
        if entity.kind is not EntityKind.RIGID_BODY:
            raise CommandError("entity is not a rigid body", operation=operation, entity_path=entity.path.value)
        if (
            isinstance(force_threshold_n, bool)
            or not isinstance(force_threshold_n, (int, float))
            or not math.isfinite(float(force_threshold_n))
            or force_threshold_n < 0.0
        ):
            raise ValidationError("force_threshold_n must be a finite non-negative number", operation=operation)
        forces = self._native_call(
            operation,
            lambda: self._native.read_contact(entity.path),
            entity_path=entity.path.value,
        )
        threshold_squared = float(force_threshold_n) ** 2
        in_contact = tuple(sum(float(value) ** 2 for value in row) > threshold_squared for row in forces)
        return ContactState(
            net_normal_forces_n=ArrayValue.from_rows(forces),
            in_contact=ArrayValue((len(in_contact),), in_contact, dtype="bool"),
            tick=self.tick,
        )

    def apply_deformable_command(self, command: DeformableCommand) -> None:
        operation = "world.apply_deformable_command"
        self._ensure_ready(operation)
        if not isinstance(command, DeformableCommand):
            raise CommandError("operation requires a DeformableCommand", operation=operation)
        entity = self._validate_handle(command.handle, operation)
        if (
            entity.kind not in {EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE}
            or entity.deformable is None
        ):
            raise CommandError("entity is not a deformable", operation=operation, entity_path=entity.path.value)
        if command.mode is not PointCommandMode.POSITION or entity.kind is not EntityKind.VOLUME_DEFORMABLE:
            raise UnsupportedCapabilityError(
                "this adapter only supports position commands on volume-deformable kinematic nodes",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=entity.path.value,
                details={"mode": command.mode.value, "topology": entity.deformable.topology.value},
            )
        environments = self._indices(
            command.environment_indices, self._spec.environments.count, "environment_indices", operation=operation
        )
        points = self._indices(command.node_indices, entity.deformable.node_count, "node_indices", operation=operation)
        if not set(points).issubset(entity.deformable.kinematic_node_indices):
            raise CommandError(
                "position commands may select only nodes declared kinematic at build time",
                operation=operation,
                entity_path=entity.path.value,
                details={"selected": list(points), "kinematic": list(entity.deformable.kinematic_node_indices)},
            )
        expected = (len(environments), len(points), 3)
        if command.targets.shape != expected:
            raise CommandError(
                "target shape must exactly match selected environments and nodes",
                operation=operation,
                entity_path=entity.path.value,
                details={"expected_shape": list(expected), "actual_shape": list(command.targets.shape)},
            )
        nested = command.targets.nested()
        targets: PointBatch = tuple(
            tuple((float(vector[0]), float(vector[1]), float(vector[2])) for vector in environment)
            for environment in nested
        )
        self._native_call(
            operation,
            lambda: self._native.apply_deformable_position(entity.path, targets, environments, points),
            entity_path=entity.path.value,
        )

    def read_deformable(self, handle: EntityHandle) -> DeformableState:
        operation = "world.read_deformable"
        self._ensure_ready(operation)
        entity = self._validate_handle(handle, operation)
        if entity.kind not in {EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE}:
            raise CommandError("entity is not a deformable", operation=operation, entity_path=entity.path.value)
        position, velocity = self._native_call(
            operation, lambda: self._native.read_deformable(entity.path), entity_path=entity.path.value
        )
        return DeformableState(ArrayValue.from_nested(position), ArrayValue.from_nested(velocity), self.tick)

    def apply_particle_fluid_command(self, command: ParticleFluidCommand) -> None:
        operation = "world.apply_particle_fluid_command"
        self._ensure_ready(operation)
        if not isinstance(command, ParticleFluidCommand):
            raise CommandError("operation requires a ParticleFluidCommand", operation=operation)
        entity = self._validate_handle(command.handle, operation)
        if entity.kind is not EntityKind.PARTICLE_FLUID or entity.particle_fluid is None:
            raise CommandError("entity is not a particle fluid", operation=operation, entity_path=entity.path.value)
        if command.mode is PointCommandMode.FORCE:
            raise UnsupportedCapabilityError(
                "this adapter supports particle position and velocity commands, not force",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=entity.path.value,
            )
        environments = self._indices(
            command.environment_indices,
            self._spec.environments.count,
            "environment_indices",
            operation=operation,
        )
        particles = self._indices(
            command.particle_indices,
            entity.particle_fluid.particle_count,
            "particle_indices",
            operation=operation,
        )
        expected = (len(environments), len(particles), 3)
        if command.targets.shape != expected:
            raise CommandError(
                "target shape must exactly match selected environments and particles",
                operation=operation,
                entity_path=entity.path.value,
                details={"expected_shape": list(expected), "actual_shape": list(command.targets.shape)},
            )
        nested = command.targets.nested()
        targets: PointBatch = tuple(
            tuple((float(vector[0]), float(vector[1]), float(vector[2])) for vector in environment)
            for environment in nested
        )
        self._native_call(
            operation,
            lambda: self._native.apply_particle_fluid(
                entity.path,
                command.mode,
                targets,
                environments,
                particles,
            ),
            entity_path=entity.path.value,
        )

    def read_particle_fluid(self, handle: EntityHandle) -> ParticleFluidState:
        operation = "world.read_particle_fluid"
        self._ensure_ready(operation)
        entity = self._validate_handle(handle, operation)
        if entity.kind is not EntityKind.PARTICLE_FLUID:
            raise CommandError("entity is not a particle fluid", operation=operation, entity_path=entity.path.value)
        positions, velocities = self._native_call(
            operation,
            lambda: self._native.read_particle_fluid(entity.path),
            entity_path=entity.path.value,
        )
        return ParticleFluidState(ArrayValue.from_nested(positions), ArrayValue.from_nested(velocities), self.tick)

    def read_sensor(self, handle: EntityHandle) -> SensorSample:
        operation = "world.read_sensor"
        self._ensure_ready(operation)
        entity = self._validate_handle(handle, operation)
        if entity.kind is not EntityKind.CAMERA_SENSOR or entity.camera is None:
            raise CommandError("entity is not a camera sensor", operation=operation, entity_path=entity.path.value)
        native_channels = self._native_call(
            operation,
            lambda: self._native.read_sensor(entity.path),
            entity_path=entity.path.value,
        )
        if tuple(item[0] for item in native_channels) != entity.camera.modalities:
            raise UniRoboSimError(
                "Isaac Lab returned camera modalities in an invalid order",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=entity.path.value,
            )
        channels = tuple(
            SensorChannel(
                modality,
                ArrayValue(shape, values, dtype="uint8" if modality is CameraModality.RGB else "float32"),
            )
            for modality, shape, values in native_channels
        )
        return SensorSample(handle, channels, self.tick)

    def publish_debug(self, batch: DebugBatch) -> DebugPublishReport:
        operation = "world.publish_debug"
        self._ensure_ready(operation)
        if not isinstance(batch, DebugBatch):
            raise ValidationError("publish requires a DebugBatch", operation=operation)
        if any(
            environment >= self._spec.environments.count
            for primitive in batch.primitives
            for environment in primitive.environment_indices
        ):
            raise ValidationError("debug batch contains an out-of-range environment", operation=operation)
        accepted, dropped, active = self._native_call(operation, lambda: self._native.publish_debug(batch))
        return DebugPublishReport(accepted, dropped, active)

    def clear_debug(
        self,
        *,
        layer: str | None = None,
        group: str | None = None,
        primitive_id: str | None = None,
    ) -> int:
        operation = "world.clear_debug"
        self._ensure_ready(operation)
        for name, value in (("layer", layer), ("group", group), ("primitive_id", primitive_id)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValidationError(f"debug {name} must be a non-empty string", operation=operation)
        return self._native_call(operation, lambda: self._native.clear_debug(layer, group, primitive_id))

    def step(self, count: int = 1) -> Tick:
        self._ensure_ready("world.step")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValidationError("step count must be a positive integer", operation="world.step")
        self._native_call("world.step", lambda: self._native.step(count))
        self._step_index += count
        self._scene_sequence += count
        return self.tick

    @staticmethod
    def _proxy_visual(entity: EntitySpec) -> SceneVisual:
        if entity.kind is EntityKind.RIGID_BODY:
            dimensions = (0.5, 0.5, 0.5) if entity.box is None else entity.box.dimensions_m
            color = (0.15, 0.7, 0.95, 1.0) if entity.box is None else entity.box.color_rgba
        elif entity.kind is EntityKind.ARTICULATION:
            dimensions = (0.55, 0.45, 0.7)
            color = (0.92, 0.49, 0.16, 1.0)
        elif entity.kind is EntityKind.PARTICLE_FLUID:
            dimensions = (0.45, 0.45, 0.45)
            color = (0.1, 0.45, 1.0, 0.72)
        elif entity.kind in {EntityKind.SURFACE_DEFORMABLE, EntityKind.VOLUME_DEFORMABLE}:
            dimensions = (0.7, 0.7, 0.15)
            color = (0.55, 0.35, 0.9, 0.82)
        else:
            dimensions = (0.18, 0.18, 0.28)
            color = (0.2, 0.85, 0.45, 0.9)
        return SceneVisual("body", SceneVisualKind.BOX, dimensions_m=dimensions, color_rgba=color)

    def _rigid_pose(self, path: EntityPath, environment: int) -> Pose:
        state = self.read_rigid_body(self.resolve(path))
        return Pose(
            tuple(float(value) for value in state.positions_m.rows()[environment]),  # type: ignore[arg-type]
            tuple(float(value) for value in state.orientations_xyzw.rows()[environment]),  # type: ignore[arg-type]
        )

    def _scene_entities(self) -> tuple[SceneEntityState, ...]:
        result: list[SceneEntityState] = []
        for entity in self._spec.entities:
            rigid_state = (
                self.read_rigid_body(self.resolve(entity.path)) if entity.kind is EntityKind.RIGID_BODY else None
            )
            articulation_state = (
                self.read_articulation(self.resolve(entity.path)) if entity.kind is EntityKind.ARTICULATION else None
            )
            for environment in range(self._spec.environments.count):
                if rigid_state is not None:
                    pose = Pose(
                        tuple(float(value) for value in rigid_state.positions_m.rows()[environment]),  # type: ignore[arg-type]
                        tuple(float(value) for value in rigid_state.orientations_xyzw.rows()[environment]),  # type: ignore[arg-type]
                    )
                    linear = tuple(float(value) for value in rigid_state.linear_velocities_m_s.rows()[environment])
                    angular = tuple(float(value) for value in rigid_state.angular_velocities_rad_s.rows()[environment])
                    joints: tuple[float, ...] = ()
                else:
                    pose = entity.pose
                    linear = angular = (0.0, 0.0, 0.0)
                    joints = (
                        ()
                        if articulation_state is None
                        else tuple(float(value) for value in articulation_state.joint_positions.rows()[environment])
                    )
                result.append(
                    SceneEntityState(
                        entity.path,
                        entity.kind,
                        environment,
                        pose,
                        linear,  # type: ignore[arg-type]
                        angular,  # type: ignore[arg-type]
                        entity.joint_names,
                        joints,
                        (self._proxy_visual(entity),),
                        draggable=entity.kind is EntityKind.RIGID_BODY,
                        metadata=FrozenMap(
                            {
                                "native_backend": "isaaclab",
                                "visual_fidelity": "portable_proxy",
                                "asset_uri": entity.asset_uri,
                            }
                        ),
                    )
                )
        return tuple(result)

    def scene_snapshot(self) -> SceneSnapshot:
        self._ensure_ready("world.scene_snapshot")
        return SceneSnapshot(
            self._session.descriptor.provider_id,
            self.world_id,
            self.generation,
            self._scene_sequence,
            self.tick,
            self._scene_entities(),
        )

    def scene_delta(self, base_sequence: int) -> SceneDelta:
        self._ensure_ready("world.scene_delta")
        if (
            not isinstance(base_sequence, int)
            or isinstance(base_sequence, bool)
            or not 0 <= base_sequence <= self._scene_sequence
        ):
            raise ValidationError("base sequence is invalid", operation="world.scene_delta")
        return SceneDelta(
            self.world_id,
            self.generation,
            base_sequence,
            self._scene_sequence,
            self.tick,
            () if base_sequence == self._scene_sequence else self._scene_entities(),
        )

    def _scene_result(
        self,
        command: SceneCommand,
        status: SceneCommandStatus,
        code: str | None = None,
        message: str | None = None,
    ) -> SceneCommandResult:
        result = SceneCommandResult(
            command.command_id,
            status,
            self.generation,
            self._scene_sequence,
            self.tick,
            code,
            message,
        )
        self._scene_results[command.command_id] = result
        if len(self._scene_results) > self._session.config.max_cached_scene_commands:
            del self._scene_results[next(iter(self._scene_results))]
        return result

    def _set_rigid_pose(self, path: EntityPath, environment: int, pose: Pose) -> None:
        self._native_call(
            "world.apply_scene_command",
            lambda: self._native.set_rigid_body_pose(
                path,
                pose.position,
                pose.orientation_xyzw,
                environment,
            ),
            entity_path=path.value,
        )

    def apply_scene_command(self, command: SceneCommand) -> SceneCommandResult:
        self._ensure_ready("world.apply_scene_command")
        if not isinstance(command, SceneCommand):
            raise ValidationError("operation requires SceneCommand", operation="world.apply_scene_command")
        previous = self._scene_results.get(command.command_id)
        if previous is not None:
            return SceneCommandResult(
                command.command_id,
                SceneCommandStatus.DUPLICATE,
                previous.generation,
                previous.scene_sequence,
                previous.tick,
                message="command was already processed",
            )
        if command.expected_generation != self.generation:
            return self._scene_result(command, SceneCommandStatus.REJECTED, "stale_generation", "generation mismatch")
        entity = self._entities.get(command.entity_path)
        if entity is None or command.environment_index >= self._spec.environments.count:
            return self._scene_result(command, SceneCommandStatus.REJECTED, "target_not_found", "target does not exist")
        if entity.kind is not EntityKind.RIGID_BODY:
            return self._scene_result(
                command,
                SceneCommandStatus.REJECTED,
                "unsupported_entity_kind",
                "only free rigid bodies are draggable",
            )
        environment = command.environment_index
        if command.kind is SceneCommandKind.SET_POSE:
            assert command.target_pose is not None
            self._set_rigid_pose(entity.path, environment, command.target_pose)
        elif command.kind is SceneCommandKind.DRAG_BEGIN:
            assert command.drag_id is not None
            if command.drag_mode is not SceneDragMode.KINEMATIC:
                return self._scene_result(
                    command, SceneCommandStatus.REJECTED, "unsupported_drag_mode", "use kinematic"
                )
            if command.drag_id in self._drags:
                return self._scene_result(command, SceneCommandStatus.REJECTED, "drag_exists", "drag already exists")
            self._drags[command.drag_id] = (entity.path, environment, self._rigid_pose(entity.path, environment))
        else:
            assert command.drag_id is not None
            active = self._drags.get(command.drag_id)
            if active is None or active[:2] != (entity.path, environment):
                return self._scene_result(command, SceneCommandStatus.REJECTED, "drag_not_active", "drag is not active")
            if command.kind is SceneCommandKind.DRAG_UPDATE:
                assert command.target_pose is not None
                self._set_rigid_pose(entity.path, environment, command.target_pose)
            elif command.kind is SceneCommandKind.DRAG_CANCEL:
                self._set_rigid_pose(entity.path, environment, active[2])
                del self._drags[command.drag_id]
            else:
                del self._drags[command.drag_id]
        self._scene_sequence += 1
        return self._scene_result(command, SceneCommandStatus.APPLIED)

    def _close(self, *, notify_session: bool) -> None:
        if self._state is WorldState.CLOSED:
            return
        self._state = WorldState.CLOSED
        self._scene_results.clear()
        self._drags.clear()
        self._entities.clear()
        try:
            self._native_call("world.close", self._native.close)
        finally:
            if notify_session:
                self._session._world_closed(self)

    def close(self) -> None:
        self._close(notify_session=True)

    def __enter__(self) -> IsaacLabWorld:
        self._ensure_ready("world.enter")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

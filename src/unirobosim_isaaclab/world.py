"""Strict public world facade over the narrow native driver."""

from __future__ import annotations

import hashlib
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
    CommandError,
    DeformableCommand,
    DeformableState,
    EntityHandle,
    EntityKind,
    EntityNotFoundError,
    EntityPath,
    EntitySpec,
    LifecycleError,
    ParticleFluidCommand,
    ParticleFluidState,
    PointCommandMode,
    ResetResult,
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
                if entity.asset_uri is None:
                    raise WorldBuildError(
                        "Isaac Lab rigid bodies require a local USD asset_uri",
                        operation="session.build.preflight",
                        backend_id=backend_id,
                        world_id=spec.world_id,
                        entity_path=entity.path.value,
                    )
                asset = _local_asset_path(entity.asset_uri)
                if asset is None or asset.suffix.lower() not in {".usd", ".usda", ".usdc"} or not asset.is_file():
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
            if entity.kind is EntityKind.PARTICLE_FLUID:
                raise UnsupportedCapabilityError(
                    "particle-fluid state is not advertised by this adapter build",
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
        self._ensure_ready("world.apply_particle_fluid_command")
        if not isinstance(command, ParticleFluidCommand):
            raise CommandError(
                "operation requires a ParticleFluidCommand", operation="world.apply_particle_fluid_command"
            )
        self._validate_handle(command.handle, "world.apply_particle_fluid_command")
        raise UnsupportedCapabilityError(
            "particle-fluid commands are not advertised by this adapter build",
            operation="world.apply_particle_fluid_command",
            backend_id=self._session.descriptor.provider_id,
            world_id=self.world_id,
            entity_path=command.handle.path.value,
        )

    def read_particle_fluid(self, handle: EntityHandle) -> ParticleFluidState:
        self._ensure_ready("world.read_particle_fluid")
        self._validate_handle(handle, "world.read_particle_fluid")
        raise UnsupportedCapabilityError(
            "particle-fluid state is not advertised by this adapter build",
            operation="world.read_particle_fluid",
            backend_id=self._session.descriptor.provider_id,
            world_id=self.world_id,
            entity_path=handle.path.value,
        )

    def step(self, count: int = 1) -> Tick:
        self._ensure_ready("world.step")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValidationError("step count must be a positive integer", operation="world.step")
        self._native_call("world.step", lambda: self._native.step(count))
        self._step_index += count
        return self.tick

    def _close(self, *, notify_session: bool) -> None:
        if self._state is WorldState.CLOSED:
            return
        self._state = WorldState.CLOSED
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

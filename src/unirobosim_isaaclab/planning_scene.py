"""Demand-only public ``planning.scene@2`` world for the Isaac Lab adapter."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar, cast

from unirobosim import (
    CapabilityId,
    EntityKind,
    PlanningGeometryAxisConvention,
    PlanningGeometryDescriptor,
    PlanningGeometryLease,
    PlanningGeometryRepresentation,
    PlanningGeometryResourceDescriptor,
    PlanningGeometryResourceRevokedError,
    PlanningGeometryStorageKind,
    PlanningSceneCatalog,
    PlanningSceneContractError,
    PlanningSceneDelta,
    PlanningSceneDeltaContinuityError,
    PlanningSceneDeltaKind,
    PlanningSceneHashMismatchError,
    PlanningSceneIncompleteError,
    PlanningSceneNotFoundError,
    PlanningSceneRepresentationError,
    PlanningSceneStaleGenerationError,
    PlanningSceneState,
    ResetResult,
    Tick,
    WorldSpec,
    parse_planning_frame_declarations,
)

from .native_protocols import (
    NativePlanningCatalog,
    NativePlanningError,
    NativePlanningResource,
    NativePlanningState,
    NativePlanningWorldDriver,
)
from .world import IsaacLabWorld

if TYPE_CHECKING:
    from .provider import IsaacLabSession

_T = TypeVar("_T")
_HISTORY_LIMIT = 128
_MAX_COUNTER = 2**63 - 1
_MAX_GEOMETRY_RESOURCE_BYTES = 64 * 1024 * 1024
_PLANNING_CAPABILITY = CapabilityId("planning.scene@2")
_SOFT_KINDS = frozenset(
    {
        EntityKind.SURFACE_DEFORMABLE,
        EntityKind.VOLUME_DEFORMABLE,
        EntityKind.PARTICLE_FLUID,
    }
)


def planning_scene_demanded(spec: WorldSpec) -> bool:
    """Return whether the locked WorldSpec explicitly demands planning reads."""

    return any(requirement.capability == _PLANNING_CAPABILITY for requirement in spec.requirements)


def validate_planning_build_spec(spec: WorldSpec, *, backend_id: str) -> None:
    """Bounded static admission before any native planning traversal is allowed."""

    for entity in spec.entities:
        if entity.kind is EntityKind.STATIC_SCENE:
            raise PlanningSceneIncompleteError(
                "the Isaac Lab planning profile cannot yet publish a complete static-scene collider forest",
                operation="planning_scene.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                entity_path=entity.path.value,
            ) from None
        if entity.kind is EntityKind.COMPOSITE_SCENE:
            raise PlanningSceneIncompleteError(
                "the Isaac Lab planning profile cannot yet publish a complete composite-scene collider forest",
                operation="planning_scene.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                entity_path=entity.path.value,
            ) from None
        if entity.kind in _SOFT_KINDS:
            raise PlanningSceneIncompleteError(
                "the Isaac Lab planning profile does not admit soft-matter collision participants",
                operation="planning_scene.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                entity_path=entity.path.value,
            ) from None
        try:
            frame_declarations = parse_planning_frame_declarations(entity.metadata.get("planning_frame_declarations"))
        except PlanningSceneIncompleteError:
            raise
        if entity.kind is EntityKind.CAMERA_SENSOR and frame_declarations is not None:
            raise PlanningSceneIncompleteError(
                "nonphysical camera entities cannot own planning frames",
                operation="planning_scene.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                entity_path=entity.path.value,
            ) from None
        planning_kind = entity.metadata.get("planning_entity_kind")
        motion_class = entity.metadata.get("planning_motion_class")
        if entity.kind is EntityKind.CAMERA_SENSOR and (planning_kind is not None or motion_class is not None):
            raise PlanningSceneIncompleteError(
                "nonphysical camera entities cannot declare planning entity or motion classes",
                operation="planning_scene.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                entity_path=entity.path.value,
            ) from None
        if planning_kind is not None and planning_kind not in {"robot", "articulation", "rigid_object", "other"}:
            raise PlanningSceneIncompleteError(
                "planning entity kind is unsupported",
                operation="planning_scene.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                entity_path=entity.path.value,
            ) from None
        if motion_class is not None and motion_class not in {"static", "kinematic", "dynamic"}:
            raise PlanningSceneIncompleteError(
                "planning motion class is unsupported",
                operation="planning_scene.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                entity_path=entity.path.value,
            ) from None


@dataclass(slots=True)
class _LeaseEpoch:
    live: bool = True
    lock: threading.RLock = field(default_factory=threading.RLock)


class _PlanningLease:
    __slots__ = ("_closed", "_content", "_descriptor", "_epoch")

    def __init__(
        self,
        descriptor: PlanningGeometryResourceDescriptor,
        content: bytes,
        epoch: _LeaseEpoch,
    ) -> None:
        self._descriptor = descriptor
        self._content = content
        self._epoch = epoch
        self._closed = False

    def _ensure_live(self, operation: str) -> None:
        if self._closed or not self._epoch.live:
            raise PlanningGeometryResourceRevokedError(
                "planning geometry resource is no longer live",
                operation=operation,
            ) from None

    @property
    def descriptor(self) -> PlanningGeometryResourceDescriptor:
        with self._epoch.lock:
            self._ensure_live("planning_geometry.descriptor")
            return self._descriptor

    @property
    def closed(self) -> bool:
        with self._epoch.lock:
            return self._closed or not self._epoch.live

    def read(self, offset: int = 0, length: int | None = None) -> bytes:
        with self._epoch.lock:
            self._ensure_live("planning_geometry.read")
            start, count = self._descriptor.read_span(offset, length)
            return self._content[start : start + count]

    def close(self) -> None:
        with self._epoch.lock:
            self._closed = True


@dataclass(slots=True)
class _PlanningEnvironment:
    generation: int
    catalog: PlanningSceneCatalog
    state: PlanningSceneState
    history: OrderedDict[int, PlanningSceneState]
    force_resync: bool = False
    lease_epoch: _LeaseEpoch = field(default_factory=_LeaseEpoch)
    lease_serial: int = 0
    storage_cache: dict[tuple[str, PlanningGeometryRepresentation, str], bytes] = field(default_factory=dict)


def _revoke(environment: _PlanningEnvironment) -> None:
    with environment.lease_epoch.lock:
        environment.lease_epoch.live = False
        environment.storage_cache.clear()


class IsaacLabPlanningWorld(IsaacLabWorld):
    """The only Isaac Lab public World subtype that exposes planning reads."""

    def __init__(
        self,
        session: IsaacLabSession,
        spec: WorldSpec,
        generation: int,
        native: NativePlanningWorldDriver,
    ) -> None:
        self._planning_authority_thread = threading.get_ident()
        self._planning_environments: dict[int, _PlanningEnvironment] = {}
        self._planning_native = native
        super().__init__(session, spec, generation, native)
        try:
            for environment_index in range(spec.environments.count):
                self._planning_environments[environment_index] = self._build_environment(
                    environment_index,
                    generation,
                    force_resync=False,
                )
        except BaseException:
            for environment in self._planning_environments.values():
                _revoke(environment)
            self._planning_environments.clear()
            super()._close(notify_session=False)
            raise

    def _planning_require_authority(self, operation: str) -> None:
        if threading.get_ident() != self._planning_authority_thread:
            raise PlanningSceneContractError(
                "planning-scene calls require the World authority thread",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        self._ensure_ready(operation)

    def _planning_environment_index(self, value: object, operation: str) -> int:
        if type(value) is not int or not 0 <= value < self._spec.environments.count:
            raise PlanningSceneContractError(
                "planning environment index is out of range",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        return value

    def _native_planning_call(self, operation: str, function: Callable[[], _T], *, preflight: bool = False) -> _T:
        failure_code: str | None = None
        try:
            return function()
        except NativePlanningError as caught:
            failure_code = caught.code
            caught.__traceback__ = None
            caught.__cause__ = None
            caught.__context__ = None
        except BaseException as caught:
            try:
                caught.__traceback__ = None
                caught.__cause__ = None
                caught.__context__ = None
            except BaseException:
                pass
            failure_code = "native_failure"
        if preflight:
            raise PlanningSceneIncompleteError(
                f"Isaac Lab planning admission failed ({failure_code})",
                operation="planning_scene.preflight",
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        raise PlanningSceneContractError(
            f"Isaac Lab planning operation failed ({failure_code})",
            operation=operation,
            backend_id=self._session.descriptor.provider_id,
            world_id=self.world_id,
        ) from None

    @staticmethod
    def _validate_native_catalog(value: object) -> NativePlanningCatalog:
        if type(value) is not NativePlanningCatalog:
            raise NativePlanningError("catalog_invalid")
        return value

    @staticmethod
    def _validate_native_state(value: object) -> NativePlanningState:
        if type(value) is not NativePlanningState or type(value.step_index) is not int or value.step_index < 0:
            raise NativePlanningError("catalog_invalid")
        return value

    def _validated_resource_content(
        self,
        value: object,
        geometry: PlanningGeometryDescriptor,
        operation: str,
    ) -> bytes:
        valid = False
        try:
            if type(value) is NativePlanningResource:
                native = value
                valid = (
                    type(native.geometry_id) is str
                    and native.geometry_id == geometry.geometry_id
                    and type(native.representation) is PlanningGeometryRepresentation
                    and native.representation is geometry.representation
                    and type(native.content) is bytes
                    and type(native.sha256) is str
                    and native.sha256 == geometry.sha256
                    and geometry.resource_layout is not None
                    and len(native.content) == geometry.resource_layout.decoded_byte_size
                    and len(native.content) <= _MAX_GEOMETRY_RESOURCE_BYTES
                    and hashlib.sha256(native.content).hexdigest() == native.sha256
                )
        except BaseException as error:
            try:
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
            except BaseException:
                pass
            valid = False
        if not valid:
            raise PlanningSceneHashMismatchError(
                "native planning geometry payload does not match the admitted catalog",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        return bytes(cast(NativePlanningResource, value).content)

    def _public_catalog(
        self,
        native: NativePlanningCatalog,
        environment_index: int,
        generation: int,
    ) -> PlanningSceneCatalog:
        catalog = PlanningSceneCatalog.build(
            self._session.descriptor.provider_id,
            self.world_id,
            generation,
            environment_index,
            1,
            1,
            native.entities,
            native.links,
            native.joints,
            native.frames,
            native.geometries,
        )
        if any(
            geometry.resource_layout is not None
            and geometry.resource_layout.decoded_byte_size > _MAX_GEOMETRY_RESOURCE_BYTES
            for geometry in catalog.geometries
        ):
            raise NativePlanningError("catalog_invalid")
        return catalog

    def _public_state(
        self,
        native: NativePlanningState,
        catalog: PlanningSceneCatalog,
        *,
        sequence: int,
        world_revision: int,
        transform_revision: int,
        attachment_revision: int,
    ) -> PlanningSceneState:
        if native.step_index != self._step_index:
            raise NativePlanningError("generation_stale")
        state = PlanningSceneState(
            self._session.descriptor.provider_id,
            self.world_id,
            catalog.generation,
            catalog.environment_index,
            Tick(native.step_index, native.step_index * self._spec.physics.time_step_seconds),
            sequence,
            world_revision,
            catalog.catalog_revision,
            catalog.geometry_revision,
            catalog.content_sha256,
            transform_revision,
            attachment_revision,
            catalog.world_frame_id,
            native.entities,
            native.links,
            native.frames,
            native.articulations,
            native.geometry_transforms,
            native.attachments,
        )
        state.validate_against(catalog)
        return state

    def _build_environment(
        self,
        environment_index: int,
        generation: int,
        *,
        force_resync: bool,
    ) -> _PlanningEnvironment:
        native_catalog = self._native_planning_call(
            "planning_scene.preflight",
            lambda: self._validate_native_catalog(self._planning_native.planning_catalog(environment_index)),
            preflight=True,
        )
        catalog = self._native_planning_call(
            "planning_scene.preflight",
            lambda: self._public_catalog(native_catalog, environment_index, generation),
            preflight=True,
        )
        native_state = self._native_planning_call(
            "planning_scene.preflight",
            lambda: self._validate_native_state(self._planning_native.planning_state(environment_index)),
            preflight=True,
        )
        state = self._native_planning_call(
            "planning_scene.preflight",
            lambda: self._public_state(
                native_state,
                catalog,
                sequence=1,
                world_revision=1,
                transform_revision=1,
                attachment_revision=1,
            ),
            preflight=True,
        )
        return _PlanningEnvironment(
            generation,
            catalog,
            state,
            OrderedDict(((state.sequence, state),)),
            force_resync=force_resync,
        )

    @staticmethod
    def _dynamic_payload(native: NativePlanningState) -> tuple[object, ...]:
        return (
            native.step_index,
            native.entities,
            native.links,
            native.frames,
            native.articulations,
            native.geometry_transforms,
        )

    @staticmethod
    def _state_dynamic_payload(state: PlanningSceneState) -> tuple[object, ...]:
        return (
            state.tick.step_index,
            state.entities,
            state.links,
            state.frames,
            state.articulations,
            state.geometry_transforms,
        )

    def _refresh_state(self, environment_index: int) -> PlanningSceneState:
        environment = self._planning_environments[environment_index]
        native = self._native_planning_call(
            "world.planning_scene_state",
            lambda: self._validate_native_state(self._planning_native.planning_state(environment_index)),
        )
        dynamic_changed = self._dynamic_payload(native) != self._state_dynamic_payload(environment.state)
        attachment_changed = native.attachments != environment.state.attachments
        if not dynamic_changed and not attachment_changed:
            return environment.state
        if environment.state.sequence >= _MAX_COUNTER or environment.state.world_revision >= _MAX_COUNTER:
            raise PlanningSceneContractError(
                "planning state counter is exhausted; reset is required",
                operation="world.planning_scene_state",
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        transform_revision = environment.state.transform_revision + int(dynamic_changed)
        attachment_revision = environment.state.attachment_revision + int(attachment_changed)
        if transform_revision > _MAX_COUNTER or attachment_revision > _MAX_COUNTER:
            raise PlanningSceneContractError(
                "planning revision counter is exhausted; reset is required",
                operation="world.planning_scene_state",
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        state = self._public_state(
            native,
            environment.catalog,
            sequence=environment.state.sequence + 1,
            world_revision=environment.state.world_revision + 1,
            transform_revision=transform_revision,
            attachment_revision=attachment_revision,
        )
        if dynamic_changed and attachment_changed:
            environment.force_resync = True
            environment.history = OrderedDict(((state.sequence, state),))
        else:
            environment.history[state.sequence] = state
            while len(environment.history) > _HISTORY_LIMIT:
                environment.history.popitem(last=False)
        environment.state = state
        return state

    def planning_scene_catalog(self, environment_index: int = 0) -> PlanningSceneCatalog:
        operation = "world.planning_scene_catalog"
        self._planning_require_authority(operation)
        environment = self._planning_environment_index(environment_index, operation)
        return self._planning_environments[environment].catalog

    def planning_scene_state(self, environment_index: int = 0) -> PlanningSceneState:
        operation = "world.planning_scene_state"
        self._planning_require_authority(operation)
        environment = self._planning_environment_index(environment_index, operation)
        return self._refresh_state(environment)

    def planning_scene_delta(self, base_sequence: int, environment_index: int = 0) -> PlanningSceneDelta:
        operation = "world.planning_scene_delta"
        self._planning_require_authority(operation)
        environment_index = self._planning_environment_index(environment_index, operation)
        if type(base_sequence) is not int or not 1 <= base_sequence <= _MAX_COUNTER:
            raise PlanningSceneContractError(
                "planning delta base_sequence must be a positive bounded integer",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        runtime = self._planning_environments[environment_index]
        current = self._refresh_state(environment_index)
        previous = runtime.history.get(base_sequence)
        if runtime.force_resync or previous is None:
            runtime.force_resync = False
            return PlanningSceneDelta(
                current.provider_id,
                current.world_id,
                current.generation,
                current.environment_index,
                current.tick,
                base_sequence,
                current.sequence,
                current.world_revision,
                current.world_revision,
                current.catalog_revision,
                current.catalog_revision,
                None,
                None,
                current.geometry_revision,
                current.geometry_revision,
                current.transform_revision,
                current.transform_revision,
                current.attachment_revision,
                current.attachment_revision,
                PlanningSceneDeltaKind.RESYNC,
                resync_required=True,
            )
        if previous.sequence == current.sequence:
            raise PlanningSceneDeltaContinuityError(
                "no committed planning delta exists after base_sequence",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        attachment_changed = previous.attachment_revision != current.attachment_revision
        transform_changed = previous.transform_revision != current.transform_revision
        if attachment_changed and transform_changed:
            return PlanningSceneDelta(
                current.provider_id,
                current.world_id,
                current.generation,
                current.environment_index,
                current.tick,
                base_sequence,
                current.sequence,
                current.world_revision,
                current.world_revision,
                current.catalog_revision,
                current.catalog_revision,
                None,
                None,
                current.geometry_revision,
                current.geometry_revision,
                current.transform_revision,
                current.transform_revision,
                current.attachment_revision,
                current.attachment_revision,
                PlanningSceneDeltaKind.RESYNC,
                resync_required=True,
            )
        kind = PlanningSceneDeltaKind.ATTACHMENT if attachment_changed else PlanningSceneDeltaKind.STATE
        return PlanningSceneDelta(
            current.provider_id,
            current.world_id,
            current.generation,
            current.environment_index,
            current.tick,
            base_sequence,
            current.sequence,
            previous.world_revision,
            current.world_revision,
            previous.catalog_revision,
            current.catalog_revision,
            previous.catalog_content_sha256,
            current.catalog_content_sha256,
            previous.geometry_revision,
            current.geometry_revision,
            previous.transform_revision,
            current.transform_revision,
            previous.attachment_revision,
            current.attachment_revision,
            kind,
            state=current if kind is PlanningSceneDeltaKind.STATE else None,
            attachments=current.attachments if kind is PlanningSceneDeltaKind.ATTACHMENT else (),
        )

    def resolve_planning_geometry(
        self,
        geometry_id: str,
        representation: PlanningGeometryRepresentation | None = None,
        environment_index: int = 0,
    ) -> PlanningGeometryLease:
        operation = "world.resolve_planning_geometry"
        self._planning_require_authority(operation)
        environment_index = self._planning_environment_index(environment_index, operation)
        if type(geometry_id) is not str or not geometry_id:
            raise PlanningSceneContractError(
                "planning geometry_id must be a non-empty string",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        runtime = self._planning_environments[environment_index]
        geometry = next((item for item in runtime.catalog.geometries if item.geometry_id == geometry_id), None)
        if geometry is None:
            raise PlanningSceneNotFoundError(
                "planning geometry is absent from the current catalog",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        if representation is not None:
            if (
                type(representation) is not PlanningGeometryRepresentation
                or representation is not geometry.representation
            ):
                raise PlanningSceneRepresentationError(
                    "requested planning geometry representation does not match the catalog",
                    operation=operation,
                    backend_id=self._session.descriptor.provider_id,
                    world_id=self.world_id,
                ) from None
        key = geometry.resolution_key
        if key is None or geometry.resource_layout is None or geometry.content_profile is None:
            raise PlanningSceneRepresentationError(
                "inline planning geometry has no resource payload",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        content = runtime.storage_cache.get(key)
        if content is None:
            native = self._native_planning_call(
                operation,
                lambda: self._planning_native.planning_resource(geometry_id, environment_index),
            )
            content = self._validated_resource_content(native, geometry, operation)
            runtime.storage_cache[key] = content
        if runtime.lease_serial >= _MAX_COUNTER:
            raise PlanningSceneContractError(
                "planning geometry lease counter is exhausted",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        runtime.lease_serial += 1
        descriptor = PlanningGeometryResourceDescriptor(
            self._session.descriptor.provider_id,
            self.world_id,
            runtime.generation,
            environment_index,
            runtime.catalog.catalog_revision,
            runtime.catalog.geometry_revision,
            runtime.catalog.content_sha256,
            f"lease-{runtime.lease_serial}",
            cast(str, geometry.resource_id),
            geometry.geometry_id,
            geometry.representation,
            PlanningGeometryStorageKind.IMMUTABLE_MEMORY,
            f"mem-{cast(str, geometry.sha256)[:24]}",
            geometry.content_profile,
            "m",
            PlanningGeometryAxisConvention.RIGHT_HANDED_Z_UP,
            geometry.resource_layout,
            len(content),
            cast(str, geometry.sha256),
        )
        descriptor.validate_against(runtime.catalog)
        return _PlanningLease(descriptor, content, runtime.lease_epoch)

    def reset(self, environment_indices: Iterable[int] | None = None) -> ResetResult:
        operation = "world.reset"
        self._planning_require_authority(operation)
        selected = self._indices(
            environment_indices,
            self._spec.environments.count,
            "environment_indices",
            operation=operation,
        )
        result = super().reset(selected)
        replacements: dict[int, _PlanningEnvironment] = {}
        try:
            for environment_index in selected:
                current = self._planning_environments[environment_index]
                if current.generation >= _MAX_COUNTER:
                    raise PlanningSceneStaleGenerationError(
                        "planning generation is exhausted",
                        operation="planning_scene.preflight",
                        backend_id=self._session.descriptor.provider_id,
                        world_id=self.world_id,
                    ) from None
                replacements[environment_index] = self._build_environment(
                    environment_index,
                    current.generation + 1,
                    force_resync=True,
                )
        except BaseException:
            self._close(notify_session=True)
            raise
        for environment_index, replacement in replacements.items():
            _revoke(self._planning_environments[environment_index])
            self._planning_environments[environment_index] = replacement
        return result

    def _close(self, *, notify_session: bool) -> None:
        for environment in self._planning_environments.values():
            _revoke(environment)
        self._planning_environments.clear()
        super()._close(notify_session=notify_session)


__all__ = ["IsaacLabPlanningWorld", "planning_scene_demanded", "validate_planning_build_spec"]

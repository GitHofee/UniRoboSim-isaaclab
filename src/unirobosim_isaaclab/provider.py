"""Lazy provider and transactional session lifecycle."""

from __future__ import annotations

import hashlib
import itertools
import os
import stat
from collections.abc import Callable, Iterable
from types import TracebackType
from typing import cast

from unirobosim import (
    ASSET_DEPENDENCY_INCOMPLETE,
    ASSET_IDENTITY_CHANGED,
    WORLD_SCHEMA_UNSUPPORTED,
    BuildInput,
    CapabilityNegotiationError,
    CapabilityRequirement,
    LifecycleError,
    NegotiationReport,
    PlanningSceneIncompleteError,
    ProbeReport,
    ProviderDescriptor,
    ProviderSelectionError,
    SessionState,
    UniRoboSimError,
    UnsupportedCapabilityError,
    ValidationError,
    WorldBuildError,
    WorldSpec,
)

from .config import IsaacLabAdapterConfig
from .descriptor import descriptor_for_config
from .native_protocols import NativePlanningError, NativePlanningWorldDriver, NativeRuntime
from .planning_scene import IsaacLabPlanningWorld, planning_scene_demanded, validate_planning_build_spec
from .probe import probe_environment
from .world import IsaacLabWorld

RuntimeFactory = Callable[[IsaacLabAdapterConfig], NativeRuntime]
ProbeFunction = Callable[[IsaacLabAdapterConfig, ProviderDescriptor], ProbeReport]
_SESSION_IDS = itertools.count(1)


def _scrub_exception(error: BaseException) -> None:
    try:
        error.__traceback__ = None
        error.__cause__ = None
        error.__context__ = None
    except BaseException:
        pass


def _default_runtime_factory(config: IsaacLabAdapterConfig) -> NativeRuntime:
    from .worker import IsaacLabWorkerRuntime

    return IsaacLabWorkerRuntime(config)


class IsaacLabProvider:
    def __init__(
        self,
        config: IsaacLabAdapterConfig | None = None,
        *,
        runtime_factory: RuntimeFactory | None = None,
        probe_function: ProbeFunction = probe_environment,
    ) -> None:
        self._config = config if config is not None else IsaacLabAdapterConfig()
        if not isinstance(self._config, IsaacLabAdapterConfig):
            raise ValidationError("config must be an IsaacLabAdapterConfig", operation="isaaclab.provider.init")
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._probe_function = probe_function
        self._descriptor = descriptor_for_config(self._config)
        self._active_session: IsaacLabSession | None = None

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def probe(self) -> ProbeReport:
        return self._probe_function(self._config, self.descriptor)

    def open(self) -> IsaacLabSession:
        if self._active_session is not None and self._active_session.state is not SessionState.CLOSED:
            raise LifecycleError(
                "this provider already owns a live session",
                operation="provider.open",
                backend_id=self.descriptor.provider_id,
            )
        probe = self.probe()
        if not probe.available:
            raise ProviderSelectionError(
                "Isaac Lab compatibility profile is unavailable",
                operation="provider.open",
                backend_id=self.descriptor.provider_id,
                details={"reason": probe.reason, "probe": probe.details.to_dict()},
            )
        try:
            native = self._runtime_factory(self._config)
        except Exception as exc:
            raise ProviderSelectionError(
                "failed to launch Isaac Lab",
                operation="provider.open",
                backend_id=self.descriptor.provider_id,
                cause=exc,
            ) from exc
        session = IsaacLabSession(self.descriptor, native, config=self._config, on_close=self._session_closed)
        self._active_session = session
        return session

    def _session_closed(self, session: IsaacLabSession) -> None:
        if self._active_session is session:
            self._active_session = None


class IsaacLabSession:
    def __init__(
        self,
        descriptor: ProviderDescriptor,
        native: NativeRuntime,
        *,
        config: IsaacLabAdapterConfig,
        on_close: Callable[[IsaacLabSession], None],
    ) -> None:
        self._descriptor = descriptor
        self._native = native
        self._config = config
        self._on_close = on_close
        self._session_id = f"isaaclab-session-{next(_SESSION_IDS)}"
        self._state = SessionState.OPEN
        self._generation = 0
        self._active_world: IsaacLabWorld | None = None

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def config(self) -> IsaacLabAdapterConfig:
        return self._config

    def _ensure_open(self, operation: str, *, allow_ready: bool = False) -> None:
        accepted = {SessionState.OPEN, SessionState.READY} if allow_ready else {SessionState.OPEN}
        if self._state not in accepted:
            raise LifecycleError(
                "session is not in a valid state for this operation",
                operation=operation,
                backend_id=self.descriptor.provider_id,
                details={"state": self._state.value, "accepted": sorted(item.value for item in accepted)},
            )

    def negotiate(self, requirements: Iterable[CapabilityRequirement]) -> NegotiationReport:
        self._ensure_open("session.negotiate", allow_ready=True)
        return self.descriptor.capabilities.negotiate(tuple(requirements))

    @staticmethod
    def _validate_build_sources(build_input: BuildInput) -> tuple[str, str | None, str | None] | None:
        for source in build_input.sources:
            descriptors: list[int] = []
            failure: tuple[str, str | None, str | None] | None = None
            try:
                directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                current = os.open(os.path.sep, directory_flags)
                descriptors.append(current)
                for part in source.source_root.split(os.path.sep):
                    if not part:
                        continue
                    current = os.open(part, directory_flags, dir_fd=current)
                    descriptors.append(current)
                parts = source.relative_source_path.split("/")
                for part in parts[:-1]:
                    current = os.open(part, directory_flags, dir_fd=current)
                    descriptors.append(current)
                file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
                descriptors.append(file_descriptor)
                before = os.fstat(file_descriptor)
                identity = source.expected_identity
                actual_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                expected_identity = (
                    identity.device,
                    identity.inode,
                    identity.mode,
                    identity.byte_size,
                    identity.mtime_ns,
                    identity.ctime_ns,
                )
                digest = hashlib.sha256()
                byte_size = 0
                while chunk := os.read(file_descriptor, 1024 * 1024):
                    byte_size += len(chunk)
                    digest.update(chunk)
                after = os.fstat(file_descriptor)
                final_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if (
                    not stat.S_ISREG(before.st_mode)
                    or actual_identity != expected_identity
                    or final_identity != actual_identity[:-1]
                    or after.st_ctime_ns != before.st_ctime_ns
                    or byte_size != identity.byte_size
                    or digest.hexdigest() != source.expected_sha256
                ):
                    failure = (source.resource_id, source.expected_sha256[:12], digest.hexdigest()[:12])
            except OSError as caught:
                _scrub_exception(caught)
                failure = (source.resource_id, None, None)
            finally:
                for descriptor in reversed(descriptors):
                    try:
                        os.close(descriptor)
                    except OSError as caught:
                        _scrub_exception(caught)
                        failure = (source.resource_id, None, None)
            if failure is not None:
                return failure
        return None

    def build(self, spec: WorldSpec, *, build_input: BuildInput | None = None) -> IsaacLabWorld:
        self._ensure_open("session.build")
        if not isinstance(spec, WorldSpec):
            raise ValidationError("build requires a WorldSpec", operation="session.build")
        if spec.schema_version not in self.descriptor.supported_world_schema_versions:
            build_input = None
            raise UnsupportedCapabilityError(
                "provider does not support the requested World schema",
                operation="session.build",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
                details={
                    "detail_code": WORLD_SCHEMA_UNSUPPORTED,
                    "requested_schema": spec.schema_version,
                    "supported_world_schema_versions": self.descriptor.supported_world_schema_versions,
                },
            ) from None
        if spec.build_resource_manifest_sha256 is None:
            if build_input is not None:
                build_input = None
                raise ValidationError(
                    "asset-free World cannot receive BuildInput",
                    operation="session.build",
                    details={"detail_code": ASSET_DEPENDENCY_INCOMPLETE},
                ) from None
        elif type(build_input) is not BuildInput or build_input.manifest.sha256 != spec.build_resource_manifest_sha256:
            build_input = None
            raise ValidationError(
                "World and BuildInput manifest identities do not match",
                operation="session.build",
                details={"detail_code": ASSET_DEPENDENCY_INCOMPLETE},
            ) from None
        else:
            source_failure = self._validate_build_sources(build_input)
            build_input = None
            if source_failure is not None:
                resource_id, expected_prefix, actual_prefix = source_failure
                details: dict[str, object] = {
                    "detail_code": ASSET_IDENTITY_CHANGED,
                    "resource_id": resource_id,
                }
                if expected_prefix is not None and actual_prefix is not None:
                    details["expected_sha256_prefix"] = expected_prefix
                    details["actual_sha256_prefix"] = actual_prefix
                raise ValidationError(
                    "build source identity changed",
                    operation="session.build",
                    details=details,
                ) from None
        negotiation = self.negotiate(spec.requirements)
        if not negotiation.accepted:
            raise CapabilityNegotiationError(
                "world requirements are not satisfied",
                operation="session.build",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
                details={"negotiation": negotiation.to_dict()},
            )
        IsaacLabWorld.validate_build_spec(spec, backend_id=self.descriptor.provider_id)
        planning_demanded = planning_scene_demanded(spec)
        if planning_demanded:
            validate_planning_build_spec(spec, backend_id=self.descriptor.provider_id)
        planning_admission_failed = False
        native_world = None
        try:
            native_world = self._native.build_world(spec)
        except NativePlanningError as caught:
            _scrub_exception(caught)
            planning_admission_failed = True
        except UniRoboSimError as caught:
            if planning_demanded:
                _scrub_exception(caught)
                planning_admission_failed = True
            else:
                raise
        except Exception as caught:
            if planning_demanded:
                _scrub_exception(caught)
                planning_admission_failed = True
            else:
                raise WorldBuildError(
                    "Isaac Lab world build failed",
                    operation="session.build",
                    backend_id=self.descriptor.provider_id,
                    world_id=spec.world_id,
                    cause=caught,
                ) from caught
        if planning_admission_failed:
            raise PlanningSceneIncompleteError(
                "Isaac Lab planning admission failed",
                operation="planning_scene.preflight",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
            ) from None
        if native_world is None:
            raise WorldBuildError(
                "Isaac Lab world build produced no native world",
                operation="session.build",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
            ) from None
        generation = self._generation + 1
        world: IsaacLabWorld
        if planning_demanded:
            world = IsaacLabPlanningWorld(
                self,
                spec,
                generation,
                cast(NativePlanningWorldDriver, native_world),
            )
        else:
            world = IsaacLabWorld(self, spec, generation, native_world)
        self._generation = generation
        self._active_world = world
        self._state = SessionState.READY
        return world

    def _world_closed(self, world: IsaacLabWorld) -> None:
        if self._active_world is world:
            self._active_world = None
            if self._state is not SessionState.CLOSED:
                self._state = SessionState.OPEN

    def close(self) -> None:
        if self._state is SessionState.CLOSED:
            return
        world = self._active_world
        self._active_world = None
        self._state = SessionState.CLOSED
        try:
            if world is not None:
                world._close(notify_session=False)
        finally:
            try:
                self._native.close()
            finally:
                self._on_close(self)

    def __enter__(self) -> IsaacLabSession:
        self._ensure_open("session.enter", allow_ready=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

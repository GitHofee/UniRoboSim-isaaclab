"""Lazy provider and transactional session lifecycle."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from types import TracebackType

from unirobosim import (
    CapabilityNegotiationError,
    CapabilityRequirement,
    LifecycleError,
    NegotiationReport,
    ProbeReport,
    ProviderDescriptor,
    ProviderSelectionError,
    SessionState,
    UniRoboSimError,
    ValidationError,
    WorldBuildError,
    WorldSpec,
)

from .config import IsaacLabAdapterConfig
from .descriptor import DESCRIPTOR
from .native_protocols import NativeRuntime
from .probe import probe_environment
from .world import IsaacLabWorld

RuntimeFactory = Callable[[IsaacLabAdapterConfig], NativeRuntime]
ProbeFunction = Callable[[IsaacLabAdapterConfig, ProviderDescriptor], ProbeReport]
_SESSION_IDS = itertools.count(1)


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
        self._active_session: IsaacLabSession | None = None

    @property
    def descriptor(self) -> ProviderDescriptor:
        return DESCRIPTOR

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
        session = IsaacLabSession(self.descriptor, native, on_close=self._session_closed)
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
        on_close: Callable[[IsaacLabSession], None],
    ) -> None:
        self._descriptor = descriptor
        self._native = native
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

    def build(self, spec: WorldSpec) -> IsaacLabWorld:
        self._ensure_open("session.build")
        if not isinstance(spec, WorldSpec):
            raise ValidationError("build requires a WorldSpec", operation="session.build")
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
        try:
            native_world = self._native.build_world(spec)
        except UniRoboSimError:
            raise
        except Exception as exc:
            raise WorldBuildError(
                "Isaac Lab world build failed",
                operation="session.build",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
                cause=exc,
            ) from exc
        self._generation += 1
        world = IsaacLabWorld(self, spec, self._generation, native_world)
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

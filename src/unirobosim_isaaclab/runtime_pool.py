"""Opt-in process-owned idle Isaac worker, independent of Run/Session identity."""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Callable

from unirobosim import WorldSpec

from .config import IsaacLabAdapterConfig
from .native_protocols import NativeRuntime, NativeWorldDriver

REUSE_WORKER_ENV = "UNIROBOSIM_ISAACLAB_REUSE_WORKER"


def _reusable(runtime: NativeRuntime) -> bool:
    # Worker-side health is a cheap local predicate, never an RPC or full scan.
    try:
        return getattr(runtime, "reusable", False) is True
    except Exception:
        return False


class RuntimeLease:
    """A single Session's ownership; returning it does not keep that Session alive."""

    def __init__(self, pool: RuntimePool, config: IsaacLabAdapterConfig, runtime: NativeRuntime) -> None:
        self._pool = pool
        self._config = config
        self._runtime = runtime
        self._closed = False
        self._invalid = False
        self._pid = os.getpid()

    def invalidate(self) -> None:
        self._invalid = True

    def build_world(self, spec: WorldSpec) -> NativeWorldDriver:
        if self._closed or self._pid != os.getpid():
            raise RuntimeError("Isaac worker lease is closed or belongs to another process")
        try:
            return self._runtime.build_world(spec)
        except BaseException:
            self.invalidate()
            raise

    def close(self) -> None:
        self._pool.release(self)


class RuntimePool:
    """At most one idle compatible worker; active leases are never shared."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._closed = False
        self._idle: tuple[IsaacLabAdapterConfig, NativeRuntime] | None = None
        self._active: dict[RuntimeLease, NativeRuntime] = {}

    def _check_process(self) -> None:
        if self._pid != os.getpid():
            # A forked process must neither reuse nor terminate its parent's worker.
            self._lock = threading.Lock()
            self._pid = os.getpid()
            self._idle = None
            self._active = {}
            self._closed = False

    def acquire(
        self, config: IsaacLabAdapterConfig, factory: Callable[[IsaacLabAdapterConfig], NativeRuntime]
    ) -> RuntimeLease:
        self._check_process()
        with self._lock:
            if self._closed:
                raise RuntimeError("Isaac worker pool is closed")
            idle, self._idle = self._idle, None
        runtime = None
        if idle is not None:
            old_config, candidate = idle
            if old_config == config and _reusable(candidate):
                runtime = candidate
            else:
                try:
                    candidate.close()
                except Exception:
                    # close() reports abnormal exit even after disposing the dead
                    # child. That old failure must not reject the next Run.
                    # A still-live or unknown child remains a cleanup failure.
                    if getattr(candidate, "terminated", False) is not True:
                        raise
        if runtime is None:
            runtime = factory(config)
        lease = RuntimeLease(self, config, runtime)
        with self._lock:
            closed = self._closed
            if not closed:
                self._active[lease] = runtime
        if closed:
            runtime.close()
            raise RuntimeError("Isaac worker pool closed during acquisition")
        return lease

    def release(self, lease: RuntimeLease) -> None:
        self._check_process()
        with self._lock:
            if lease._closed:
                return
            lease._closed = True
            runtime = self._active.pop(lease, None)
            if runtime is None:
                return
            if not self._closed and self._idle is None and not lease._invalid and _reusable(runtime):
                self._idle = lease._config, runtime
                return
        runtime.close()

    def close(self) -> None:
        self._check_process()
        with self._lock:
            self._closed = True
            runtimes = list(self._active.values())
            for lease in self._active:
                lease._closed = True
            self._active.clear()
            if self._idle is not None:
                runtimes.append(self._idle[1])
                self._idle = None
        first_error: BaseException | None = None
        for runtime in runtimes:
            try:
                runtime.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


PROCESS_RUNTIME_POOL = RuntimePool()
atexit.register(PROCESS_RUNTIME_POOL.close)

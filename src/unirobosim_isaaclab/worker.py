"""Process-isolated native runtime for Isaac Sim's process-owning lifecycle."""

from __future__ import annotations

import faulthandler
import multiprocessing
import os
import signal
import sys
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Protocol, cast

from unirobosim import CommandMode, DebugBatch, EntityPath, PointCommandMode, WorldSpec

from .config import IsaacLabAdapterConfig
from .native_protocols import (
    Matrix,
    NativeRuntime,
    NativeSensorSample,
    NativeWorldDriver,
    PointBatch,
    Quaternion,
    Vector3,
)

_CALL_TIMEOUT_SECONDS = 300.0
_SHUTDOWN_TIMEOUT_SECONDS = 30.0

Request = tuple[str, tuple[Any, ...]]
Reply = tuple[str, Any]


class NativeWorkerError(RuntimeError):
    """A native worker failed or returned an exception."""


class _ProcessHandle(Protocol):
    @property
    def pid(self) -> int | None: ...

    @property
    def exitcode(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...


WorkerFactory = Callable[[IsaacLabAdapterConfig], tuple[Connection, _ProcessHandle]]


def _error_reply(exc: Exception) -> Reply:
    return (
        "error",
        {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _require_world(world: NativeWorldDriver | None, operation: str) -> NativeWorldDriver:
    if world is None:
        raise RuntimeError(f"native worker has no world for operation {operation!r}")
    return world


def _dispatch(
    runtime: NativeRuntime,
    world: NativeWorldDriver | None,
    request: Request,
) -> tuple[NativeWorldDriver | None, Any, bool]:
    """Execute one trusted local IPC request and return updated worker state."""

    operation, args = request
    if operation == "build_world":
        if world is not None:
            raise RuntimeError("native worker already owns a world")
        return runtime.build_world(cast(WorldSpec, args[0])), None, False
    if operation == "close_runtime":
        if world is not None:
            world.close()
        return None, None, True

    active = _require_world(world, operation)
    if operation == "reset":
        active.reset(cast(tuple[int, ...], args[0]))
        return active, None, False
    if operation == "apply_articulation":
        active.apply_articulation(
            cast(EntityPath, args[0]),
            cast(CommandMode, args[1]),
            cast(Matrix, args[2]),
            cast(tuple[int, ...], args[3]),
            cast(tuple[int, ...], args[4]),
        )
        return active, None, False
    if operation == "read_articulation":
        return active, active.read_articulation(cast(EntityPath, args[0])), False
    if operation == "apply_rigid_body_wrench":
        active.apply_rigid_body_wrench(
            cast(EntityPath, args[0]),
            cast(Matrix, args[1]),
            cast(Matrix, args[2]),
            cast(tuple[int, ...], args[3]),
        )
        return active, None, False
    if operation == "read_rigid_body":
        return active, active.read_rigid_body(cast(EntityPath, args[0])), False
    if operation == "set_rigid_body_pose":
        active.set_rigid_body_pose(
            cast(EntityPath, args[0]),
            cast(Vector3, args[1]),
            cast(Quaternion, args[2]),
            cast(int, args[3]),
        )
        return active, None, False
    if operation == "read_contact":
        return active, active.read_contact(cast(EntityPath, args[0])), False
    if operation == "apply_deformable_position":
        active.apply_deformable_position(
            cast(EntityPath, args[0]),
            cast(PointBatch, args[1]),
            cast(tuple[int, ...], args[2]),
            cast(tuple[int, ...], args[3]),
        )
        return active, None, False
    if operation == "read_deformable":
        return active, active.read_deformable(cast(EntityPath, args[0])), False
    if operation == "apply_particle_fluid":
        active.apply_particle_fluid(
            cast(EntityPath, args[0]),
            cast(PointCommandMode, args[1]),
            cast(PointBatch, args[2]),
            cast(tuple[int, ...], args[3]),
            cast(tuple[int, ...], args[4]),
        )
        return active, None, False
    if operation == "read_particle_fluid":
        return active, active.read_particle_fluid(cast(EntityPath, args[0])), False
    if operation == "read_sensor":
        return active, active.read_sensor(cast(EntityPath, args[0])), False
    if operation == "publish_debug":
        return active, active.publish_debug(cast(DebugBatch, args[0])), False
    if operation == "clear_debug":
        return (
            active,
            active.clear_debug(
                cast(str | None, args[0]),
                cast(str | None, args[1]),
                cast(str | None, args[2]),
            ),
            False,
        )
    if operation == "step":
        active.step(cast(int, args[0]))
        return active, None, False
    if operation == "close_world":
        active.close()
        return None, None, False
    raise RuntimeError(f"unknown native worker operation {operation!r}")


def _worker_main(connection: Connection, config: IsaacLabAdapterConfig) -> None:  # pragma: no cover
    """Own AppLauncher and every native simulator import inside one child process."""

    if hasattr(os, "setsid"):
        os.setsid()
    if hasattr(signal, "SIGUSR1"):
        # A stalled native SDK call can otherwise leave only the parent IPC wait
        # visible.  SIGUSR1 emits every Python-thread stack without changing the
        # worker state, making native acceptance runs diagnosable and repeatable.
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    runtime: NativeRuntime | None = None
    world: NativeWorldDriver | None = None
    try:
        try:
            from .native import IsaacLabNativeRuntime

            runtime = IsaacLabNativeRuntime(config, process_isolated=True)
        except Exception as exc:
            connection.send(_error_reply(exc))
            return
        connection.send(("ok", None))
        while True:
            try:
                request = cast(Request, connection.recv())
            except EOFError:
                break
            try:
                world, payload, should_stop = _dispatch(runtime, world, request)
            except Exception as exc:
                connection.send(_error_reply(exc))
                continue
            connection.send(("ok", payload))
            if should_stop:
                break
    finally:
        connection.close()
        if runtime is not None:
            # Fast shutdown terminates only this adapter-owned worker process.
            runtime.close()


def _spawn_worker(config: IsaacLabAdapterConfig) -> tuple[Connection, _ProcessHandle]:  # pragma: no cover
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process: BaseProcess = context.Process(
        target=_worker_main,
        args=(child, config),
        name="unirobosim-isaaclab",
        daemon=False,
    )
    process.start()
    child.close()
    return parent, process


def _proc_stat_session(stat: str) -> int | None:
    """Extract the Linux session id while tolerating spaces in process names."""

    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = stat[closing_parenthesis + 1 :].split()
    if len(fields) < 4:
        return None
    try:
        return int(fields[3])
    except ValueError:
        return None


def _session_member_pids(session_id: int) -> tuple[int, ...]:
    members: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return ()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if _proc_stat_session(stat) == session_id:
            members.append(int(entry.name))
    return tuple(sorted(members, reverse=True))


def _terminate_worker_tree(process: _ProcessHandle) -> None:
    """Terminate the worker session and any Kit helpers that outlive it."""

    pid = process.pid
    if pid is not None and pid > 1 and pid != os.getsid(0):
        members = _session_member_pids(pid)
        for member in members:
            if member == os.getpid():
                continue
            try:
                os.kill(member, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if member == pid and process.is_alive():
                    process.terminate()
        if members:
            return
    if pid is not None and pid > 1 and hasattr(os, "killpg") and pid != os.getpgrp():
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.is_alive():
                process.terminate()
        return
    if process.is_alive():
        process.terminate()


class IsaacLabWorkerRuntime:
    """NativeRuntime proxy that keeps Isaac Sim outside the caller's process."""

    def __init__(
        self,
        config: IsaacLabAdapterConfig,
        *,
        worker_factory: WorkerFactory = _spawn_worker,
    ) -> None:
        self._connection, self._process = worker_factory(config)
        self._closed = False
        self._active_world: IsaacLabWorkerWorld | None = None
        try:
            self._receive("worker startup")
        except Exception:
            self._abort()
            raise

    def _receive(self, operation: str) -> Any:
        try:
            if not self._connection.poll(_CALL_TIMEOUT_SECONDS):
                raise NativeWorkerError(f"timed out after {_CALL_TIMEOUT_SECONDS:g}s during {operation}")
            reply = cast(Reply, self._connection.recv())
        except (EOFError, OSError) as exc:
            raise NativeWorkerError(
                f"native worker disconnected during {operation}; exitcode={self._process.exitcode}"
            ) from exc
        status, payload = reply
        if status == "ok":
            return payload
        if status != "error" or not isinstance(payload, dict):
            raise NativeWorkerError(f"invalid native worker reply during {operation}: {reply!r}")
        remote_type = payload.get("type", "Exception")
        message = payload.get("message", "")
        remote_traceback = payload.get("traceback", "")
        raise NativeWorkerError(f"{remote_type} during {operation}: {message}\n{remote_traceback}".rstrip())

    def _request(self, operation: str, *args: Any) -> Any:
        if self._closed:
            raise NativeWorkerError(f"native worker is closed during {operation}")
        if not self._process.is_alive():
            raise NativeWorkerError(f"native worker exited before {operation}; exitcode={self._process.exitcode}")
        try:
            self._connection.send((operation, args))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise NativeWorkerError(
                f"failed to contact native worker during {operation}; exitcode={self._process.exitcode}"
            ) from exc
        return self._receive(operation)

    def build_world(self, spec: WorldSpec) -> IsaacLabWorkerWorld:
        if self._active_world is not None and not self._active_world.closed:
            raise NativeWorkerError("native worker already owns a world")
        self._request("build_world", spec)
        world = IsaacLabWorkerWorld(self)
        self._active_world = world
        return world

    def _world_closed(self, world: IsaacLabWorkerWorld) -> None:
        if self._active_world is world:
            self._active_world = None

    def _abort(self) -> None:
        try:
            self._connection.close()
        finally:
            _terminate_worker_tree(self._process)
            self._process.join(_SHUTDOWN_TIMEOUT_SECONDS)

    def close(self) -> None:
        if self._closed:
            return
        error: Exception | None = None
        try:
            if self._process.is_alive():
                try:
                    self._request("close_runtime")
                except Exception as exc:
                    error = exc
        finally:
            self._closed = True
            if self._active_world is not None:
                self._active_world._mark_closed()
                self._active_world = None
            self._connection.close()
            self._process.join(_SHUTDOWN_TIMEOUT_SECONDS)
            if self._process.is_alive():
                _terminate_worker_tree(self._process)
                self._process.join(_SHUTDOWN_TIMEOUT_SECONDS)
            else:
                # Fast shutdown may leave telemetry and crash-report helpers alive.
                _terminate_worker_tree(self._process)
        if error is not None:
            raise error
        if self._process.exitcode not in {0, None}:
            raise NativeWorkerError(f"native worker exited with status {self._process.exitcode}")


class IsaacLabWorkerWorld:
    """NativeWorldDriver proxy bound to its owning worker runtime."""

    def __init__(self, runtime: IsaacLabWorkerRuntime) -> None:
        self._runtime = runtime
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self, operation: str) -> None:
        if self._closed:
            raise NativeWorkerError(f"native world is closed during {operation}")

    def reset(self, environment_indices: tuple[int, ...]) -> None:
        self._ensure_open("reset")
        self._runtime._request("reset", environment_indices)

    def apply_articulation(
        self,
        path: EntityPath,
        mode: CommandMode,
        targets: Matrix,
        environment_indices: tuple[int, ...],
        degree_of_freedom_indices: tuple[int, ...],
    ) -> None:
        self._ensure_open("apply_articulation")
        self._runtime._request(
            "apply_articulation",
            path,
            mode,
            targets,
            environment_indices,
            degree_of_freedom_indices,
        )

    def read_articulation(self, path: EntityPath) -> tuple[Matrix, Matrix]:
        self._ensure_open("read_articulation")
        return cast(tuple[Matrix, Matrix], self._runtime._request("read_articulation", path))

    def apply_rigid_body_wrench(
        self,
        path: EntityPath,
        forces_n: Matrix,
        torques_n_m: Matrix,
        environment_indices: tuple[int, ...],
    ) -> None:
        self._ensure_open("apply_rigid_body_wrench")
        self._runtime._request(
            "apply_rigid_body_wrench",
            path,
            forces_n,
            torques_n_m,
            environment_indices,
        )

    def read_rigid_body(self, path: EntityPath) -> tuple[Matrix, Matrix, Matrix, Matrix]:
        self._ensure_open("read_rigid_body")
        return cast(
            tuple[Matrix, Matrix, Matrix, Matrix],
            self._runtime._request("read_rigid_body", path),
        )

    def set_rigid_body_pose(
        self,
        path: EntityPath,
        position_m: Vector3,
        orientation_xyzw: Quaternion,
        environment_index: int,
    ) -> None:
        self._ensure_open("set_rigid_body_pose")
        self._runtime._request(
            "set_rigid_body_pose",
            path,
            position_m,
            orientation_xyzw,
            environment_index,
        )

    def read_contact(self, path: EntityPath) -> Matrix:
        self._ensure_open("read_contact")
        return cast(Matrix, self._runtime._request("read_contact", path))

    def apply_deformable_position(
        self,
        path: EntityPath,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        point_indices: tuple[int, ...],
    ) -> None:
        self._ensure_open("apply_deformable_position")
        self._runtime._request(
            "apply_deformable_position",
            path,
            targets,
            environment_indices,
            point_indices,
        )

    def read_deformable(self, path: EntityPath) -> tuple[PointBatch, PointBatch]:
        self._ensure_open("read_deformable")
        return cast(tuple[PointBatch, PointBatch], self._runtime._request("read_deformable", path))

    def apply_particle_fluid(
        self,
        path: EntityPath,
        mode: PointCommandMode,
        targets: PointBatch,
        environment_indices: tuple[int, ...],
        particle_indices: tuple[int, ...],
    ) -> None:
        self._ensure_open("apply_particle_fluid")
        self._runtime._request(
            "apply_particle_fluid",
            path,
            mode,
            targets,
            environment_indices,
            particle_indices,
        )

    def read_particle_fluid(self, path: EntityPath) -> tuple[PointBatch, PointBatch]:
        self._ensure_open("read_particle_fluid")
        return cast(tuple[PointBatch, PointBatch], self._runtime._request("read_particle_fluid", path))

    def read_sensor(self, path: EntityPath) -> NativeSensorSample:
        self._ensure_open("read_sensor")
        return cast(NativeSensorSample, self._runtime._request("read_sensor", path))

    def publish_debug(self, batch: DebugBatch) -> tuple[int, int, int]:
        self._ensure_open("publish_debug")
        return cast(tuple[int, int, int], self._runtime._request("publish_debug", batch))

    def clear_debug(self, layer: str | None, group: str | None, primitive_id: str | None) -> int:
        self._ensure_open("clear_debug")
        return cast(int, self._runtime._request("clear_debug", layer, group, primitive_id))

    def step(self, count: int) -> None:
        self._ensure_open("step")
        self._runtime._request("step", count)

    def _mark_closed(self) -> None:
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._runtime._request("close_world")
        finally:
            self._mark_closed()
            self._runtime._world_closed(self)

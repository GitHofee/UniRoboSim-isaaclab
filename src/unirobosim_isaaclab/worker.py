"""Process-isolated native runtime for Isaac Sim's process-owning lifecycle."""

from __future__ import annotations

import faulthandler
import multiprocessing
import os
import signal
import subprocess
import sys
import traceback
import warnings
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Protocol, cast

import unirobosim
from unirobosim import CommandMode, DebugBatch, EntityPath, PointCommandMode, WorldSpec

from ._version import DISTRIBUTION_VERSION
from .config import IsaacLabAdapterConfig
from .native_protocols import (
    Matrix,
    NativeArticulationCommand,
    NativeCameraCalibration,
    NativePhysicsDiagnostics,
    NativePlanningCatalog,
    NativePlanningError,
    NativePlanningResource,
    NativePlanningState,
    NativePlanningWorldDriver,
    NativeRuntime,
    NativeSensorSample,
    NativeWorldDriver,
    PointBatch,
    Quaternion,
    Vector3,
)
from .planning_scene import planning_scene_demanded

_CALL_TIMEOUT_SECONDS = 300.0
_STARTUP_TIMEOUT_SECONDS = 30.0
_STARTUP_ATTEMPTS = 2
_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_WORKER_HANDSHAKE_SCHEMA = "unirobosim-isaaclab-worker-startup/1"
_WORKER_PROTOCOL_VERSION = 1

Request = tuple[str, tuple[Any, ...]]
Reply = tuple[str, Any]


class NativeWorkerError(RuntimeError):
    """A native worker failed or returned an exception."""


class _NativeWorkerTimeout(NativeWorkerError):
    """An otherwise-live worker did not reply before its operation deadline."""


def _module_origin(module: object, name: str) -> Path:
    raw_origin = getattr(module, "__file__", None)
    if type(raw_origin) is not str or not raw_origin:
        raise NativeWorkerError(f"{name} package has no concrete origin")
    origin = Path(raw_origin).resolve()
    if not origin.is_file():
        raise NativeWorkerError(f"{name} package origin is not a file: {origin}")
    return origin


def _worker_startup_fingerprint() -> dict[str, object]:
    """Describe the exact Core and adapter packages loaded by this process."""

    adapter = sys.modules.get(__package__)
    if adapter is None:
        raise NativeWorkerError("adapter package is not loaded")
    core_version = getattr(unirobosim, "__version__", None)
    if type(core_version) is not str or not core_version:
        raise NativeWorkerError("UniRoboSim Core package has no valid version")
    return {
        "schema": _WORKER_HANDSHAKE_SCHEMA,
        "worker_protocol": _WORKER_PROTOCOL_VERSION,
        "core": {
            "version": core_version,
            "origin": str(_module_origin(unirobosim, "unirobosim")),
        },
        "adapter": {
            "version": DISTRIBUTION_VERSION,
            "origin": str(_module_origin(adapter, "unirobosim_isaaclab")),
        },
    }


def _worker_package_roots() -> tuple[Path, ...]:
    """Return ordered import roots for the exact packages loaded by the parent."""

    adapter = sys.modules.get(__package__)
    if adapter is None:
        raise NativeWorkerError("adapter package is not loaded")
    origins = (
        _module_origin(adapter, "unirobosim_isaaclab"),
        _module_origin(unirobosim, "unirobosim"),
    )
    roots: list[Path] = []
    for origin in origins:
        root = origin.parent.parent
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _worker_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Build a child-only environment with exact package roots taking precedence."""

    child = dict(os.environ if environ is None else environ)
    entries = list(_worker_package_roots())
    for raw_entry in child.get("PYTHONPATH", "").split(os.pathsep):
        if not raw_entry:
            continue
        candidate = Path(raw_entry)
        if not candidate.is_absolute():
            continue
        resolved = candidate.resolve()
        if resolved not in entries:
            entries.append(resolved)
    child["PYTHONPATH"] = os.pathsep.join(str(entry) for entry in entries)
    child["PYTHONSAFEPATH"] = "1"
    return child


def _worker_command(connection_descriptor: int) -> tuple[str, ...]:
    return (
        sys.executable,
        "-P",
        "-B",
        "-m",
        "unirobosim_isaaclab.worker_bootstrap",
        str(connection_descriptor),
    )


def _validate_worker_startup(value: object) -> None:
    expected = _worker_startup_fingerprint()
    if type(value) is not dict or value != expected:
        raise NativeWorkerError(f"native worker startup fingerprint differs: expected {expected!r}, got {value!r}")


class _ProcessHandle(Protocol):
    @property
    def pid(self) -> int | None: ...

    @property
    def exitcode(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...


WorkerFactory = Callable[[IsaacLabAdapterConfig], tuple[Connection, _ProcessHandle]]


class _SubprocessHandle:
    """Adapt ``subprocess.Popen`` to the small worker process protocol."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        return self._process.poll()

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self) -> None:
        self._process.terminate()


def _error_reply(exc: Exception) -> Reply:
    return (
        "error",
        {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _planning_error_reply(exc: BaseException) -> Reply:
    """Return the only failure envelope allowed across the planning IPC seam."""

    code = exc.code if isinstance(exc, NativePlanningError) else "native_failure"
    try:
        exc.__traceback__ = None
        exc.__cause__ = None
        exc.__context__ = None
    except BaseException:
        pass
    return "planning_error", {"code": code}


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
    if operation == "physics_diagnostics":
        return active, active.physics_diagnostics(), False
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
    if operation == "apply_articulation_commands_and_step":
        active.apply_articulation_commands_and_step(
            cast(tuple[NativeArticulationCommand, ...], args[0]),
            cast(int, args[1]),
        )
        return active, None, False
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
    if operation == "camera_calibration":
        return active, active.camera_calibration(cast(EntityPath, args[0])), False
    if operation == "planning_catalog":
        planning = cast(NativePlanningWorldDriver, active)
        return active, planning.planning_catalog(cast(int, args[0])), False
    if operation == "planning_state":
        planning = cast(NativePlanningWorldDriver, active)
        return active, planning.planning_state(cast(int, args[0])), False
    if operation == "planning_resource":
        planning = cast(NativePlanningWorldDriver, active)
        return active, planning.planning_resource(cast(str, args[0]), cast(int, args[1])), False
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

    if hasattr(os, "setsid") and os.getsid(0) != os.getpid():
        # ``subprocess.Popen(start_new_session=True)`` already made the clean
        # bootstrap process a session leader. The multiprocessing fallback has
        # not, so only that path still needs ``setsid`` here.
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
        connection.send(("ok", _worker_startup_fingerprint()))
        while True:
            try:
                request = cast(Request, connection.recv())
            except EOFError:
                break
            try:
                world, payload, should_stop = _dispatch(runtime, world, request)
            except Exception as exc:
                operation, args = request
                demanded_build = (
                    operation == "build_world"
                    and bool(args)
                    and isinstance(args[0], WorldSpec)
                    and planning_scene_demanded(args[0])
                )
                if operation.startswith("planning_") or demanded_build or isinstance(exc, NativePlanningError):
                    connection.send(_planning_error_reply(exc))
                else:
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
    if os.name == "posix":
        # A multiprocessing "spawn" child imports the parent's __main__ module
        # before calling the target. For an MCP/FastSim parent that can preload
        # libraries which conflict with Kit. A clean interpreter bootstrap keeps
        # the native worker import boundary real while retaining the same Pipe.
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        descriptor = child.fileno()
        process: _SubprocessHandle | None = None
        try:
            bootstrap_process = subprocess.Popen(
                _worker_command(descriptor),
                close_fds=True,
                env=_worker_environment(),
                pass_fds=(descriptor,),
                start_new_session=True,
            )
            process = _SubprocessHandle(bootstrap_process)
            child.close()
            parent.send(config)
            return parent, process
        except BaseException:
            child.close()
            parent.close()
            if process is not None:
                _terminate_worker_tree(process)
                process.join(_SHUTDOWN_TIMEOUT_SECONDS)
            raise
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    spawned_process: BaseProcess = context.Process(
        target=_worker_main,
        args=(child, config),
        name="unirobosim-isaaclab",
        daemon=False,
    )
    spawned_process.start()
    child.close()
    return parent, spawned_process


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
        self._closed = False
        self._active_world: IsaacLabWorkerWorld | None = None
        for attempt in range(1, _STARTUP_ATTEMPTS + 1):
            self._connection, self._process = worker_factory(config)
            try:
                startup = self._receive(
                    "worker startup",
                    timeout_seconds=_STARTUP_TIMEOUT_SECONDS,
                )
                _validate_worker_startup(startup)
                break
            except _NativeWorkerTimeout as exc:
                self._abort()
                if attempt == _STARTUP_ATTEMPTS:
                    raise NativeWorkerError(
                        "native worker startup timed out after "
                        f"{_STARTUP_ATTEMPTS} attempts of {_STARTUP_TIMEOUT_SECONDS:g}s"
                    ) from exc
                warnings.warn(
                    "Isaac Kit did not complete native startup within "
                    f"{_STARTUP_TIMEOUT_SECONDS:g}s; the isolated worker was cleaned up "
                    f"and startup will be retried ({attempt + 1}/{_STARTUP_ATTEMPTS})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception:
                self._abort()
                raise

    def _receive(self, operation: str, *, timeout_seconds: float | None = None) -> Any:
        timeout = _CALL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        try:
            if not self._connection.poll(timeout):
                raise _NativeWorkerTimeout(f"timed out after {timeout:g}s during {operation}")
            reply = cast(Reply, self._connection.recv())
        except (EOFError, OSError) as exc:
            raise NativeWorkerError(
                f"native worker disconnected during {operation}; exitcode={self._process.exitcode}"
            ) from exc
        status, payload = reply
        if status == "ok":
            return payload
        if status == "planning_error" and isinstance(payload, dict):
            raise NativePlanningError(cast(str, payload.get("code", "native_failure"))) from None
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
        world = IsaacLabWorkerPlanningWorld(self) if planning_scene_demanded(spec) else IsaacLabWorkerWorld(self)
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

    def physics_diagnostics(self) -> NativePhysicsDiagnostics:
        self._ensure_open("physics_diagnostics")
        return cast(
            NativePhysicsDiagnostics,
            self._runtime._request("physics_diagnostics"),
        )

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

    def apply_articulation_commands_and_step(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
    ) -> None:
        self._ensure_open("apply_articulation_commands_and_step")
        self._runtime._request("apply_articulation_commands_and_step", commands, count)

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

    def camera_calibration(self, path: EntityPath) -> NativeCameraCalibration:
        self._ensure_open("camera_calibration")
        return cast(NativeCameraCalibration, self._runtime._request("camera_calibration", path))

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


class IsaacLabWorkerPlanningWorld(IsaacLabWorkerWorld):
    """Planning-only native proxy; ordinary worker worlds have no such methods."""

    def planning_catalog(self, environment_index: int = 0) -> NativePlanningCatalog:
        self._ensure_open("planning_catalog")
        return cast(NativePlanningCatalog, self._runtime._request("planning_catalog", environment_index))

    def planning_state(self, environment_index: int = 0) -> NativePlanningState:
        self._ensure_open("planning_state")
        return cast(NativePlanningState, self._runtime._request("planning_state", environment_index))

    def planning_resource(self, geometry_id: str, environment_index: int = 0) -> NativePlanningResource:
        self._ensure_open("planning_resource")
        return cast(
            NativePlanningResource,
            self._runtime._request("planning_resource", geometry_id, environment_index),
        )

"""Process-isolated native runtime for Isaac Sim's process-owning lifecycle."""

from __future__ import annotations

import faulthandler
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Protocol, cast

import unirobosim
from unirobosim import CommandMode, DebugBatch, EntityPath, KinematicTarget, PointCommandMode, Pose, WorldSpec

from ._version import DISTRIBUTION_VERSION
from .config import IsaacLabAdapterConfig
from .native_protocols import (
    Matrix,
    NativeArticulationCommand,
    NativeCameraCalibration,
    NativeEncodedSensorFrame,
    NativeEncodedSensorRequest,
    NativeEntityPrimState,
    NativeKinematicState,
    NativePhysicsDiagnostics,
    NativePlanningCatalog,
    NativePlanningError,
    NativePlanningResource,
    NativePlanningState,
    NativePlanningWorldDriver,
    NativeRenderStateFrame,
    NativeRuntime,
    NativeSensorBatch,
    NativeSensorSample,
    NativeWorldDriver,
    PointBatch,
    Quaternion,
    Vector3,
)
from .planning_scene import planning_scene_demanded

_CALL_TIMEOUT_SECONDS = 300.0
_STARTUP_ATTEMPTS = 2
_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_WORKER_HANDSHAKE_SCHEMA = "unirobosim-isaaclab-worker-startup/2"
_WORKER_PROGRESS_SCHEMA = "unirobosim-isaaclab-worker-progress/1"
_WORKER_PROTOCOL_VERSION = 5
_PACKED_RGB_BATCH_STATUS = "packed_rgb_batch"
_SHARED_RGB_BATCH_STATUS = "shared_rgb_batch"
_SHARED_STEP_STATE_RGB_STATUS = "shared_step_state_rgb"
_MAX_PACKED_RGB_SAMPLE_BYTES = 1024 * 1024 * 1024

# Earlier interpreter/import phases normally complete in well under a second and
# retain small fixed idle limits.  Kit launch is the exceptional phase: a cold
# container can legitimately spend tens of seconds compiling/loading native state,
# so its idle limit and the per-worker hard limit come from the validated adapter
# configuration.  Progress remains a fixed, strictly ordered sequence: it can
# never act as an unbounded heartbeat or extend the configured hard limit.
_STARTUP_PHASES = (
    "bootstrap_connected",
    "config_received",
    "worker_imported",
    "native_module_loading",
    "native_module_loaded",
    "sdk_importing",
    "kit_launching",
    "kit_ready",
    "runtime_importing",
    "runtime_ready",
)
_STARTUP_PHASE_IDLE_TIMEOUT_SECONDS = {
    "process_spawned": 8.0,
    "bootstrap_connected": 8.0,
    "config_received": 8.0,
    "worker_imported": 8.0,
    "native_module_loading": 8.0,
    "native_module_loaded": 8.0,
    "sdk_importing": 8.0,
    "kit_ready": 10.0,
    "runtime_importing": 10.0,
    "runtime_ready": 5.0,
}

Request = tuple[str, tuple[Any, ...]]
Reply = tuple[str, Any]


class NativeWorkerError(RuntimeError):
    """A native worker failed or returned an exception."""


class _NativeWorkerTimeout(NativeWorkerError):
    """An otherwise-live worker did not reply before its operation deadline."""


@dataclass(frozen=True)
class _WorkerStartupProgress:
    phase: str


@dataclass(frozen=True, slots=True)
class _SharedRgbReply:
    metadata: tuple[tuple[tuple[int, ...], int, int], ...]


@dataclass(frozen=True, slots=True)
class _SharedStepStateRgbReply:
    states: tuple[tuple[Matrix, Matrix], ...]
    metadata: tuple[tuple[tuple[int, ...], int, int], ...]


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


def _startup_progress_reply(phase: str) -> Reply:
    """Return one bounded, versioned startup progress event."""

    if phase not in _STARTUP_PHASES:
        raise ValueError(f"unknown native worker startup phase {phase!r}")
    return (
        "startup_progress",
        {
            "schema": _WORKER_PROGRESS_SCHEMA,
            "phase": phase,
        },
    )


def _validate_startup_progress(value: object) -> _WorkerStartupProgress:
    if type(value) is not dict or set(value) != {"schema", "phase"}:
        raise NativeWorkerError(f"invalid native worker startup progress payload: {value!r}")
    schema = value.get("schema")
    phase = value.get("phase")
    if schema != _WORKER_PROGRESS_SCHEMA or type(phase) is not str or phase not in _STARTUP_PHASES:
        raise NativeWorkerError(f"invalid native worker startup progress payload: {value!r}")
    return _WorkerStartupProgress(phase)


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
    bootstrap = Path(__file__).resolve().with_name("worker_bootstrap.py")
    if not bootstrap.is_file():
        raise NativeWorkerError(f"native worker bootstrap is missing: {bootstrap}")
    return (
        sys.executable,
        "-P",
        "-B",
        str(bootstrap),
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


def _packed_rgb_batch_metadata(payload: object) -> tuple[tuple[tuple[int, ...], int], ...] | None:
    """Describe an RGB-only batch whose byte payloads can bypass pickle."""

    if type(payload) is not tuple:
        return None
    metadata: list[tuple[tuple[int, ...], int]] = []
    for sample in payload:
        if type(sample) is not tuple or len(sample) != 1:
            return None
        channel = sample[0]
        if type(channel) is not tuple or len(channel) != 3 or channel[0] is not unirobosim.CameraModality.RGB:
            return None
        shape, values = channel[1], channel[2]
        if (
            type(shape) is not tuple
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
            or type(values) is not bytes
        ):
            return None
        expected = math.prod(shape)
        if expected > _MAX_PACKED_RGB_SAMPLE_BYTES or len(values) != expected:
            return None
        metadata.append((shape, expected))
    return tuple(metadata)


def _send_worker_reply(
    connection: Connection,
    operation: str,
    payload: object,
    rgb_transport: shared_memory.SharedMemory | None,
) -> None:
    """Send large packed RGB buffers without re-pickling their contents."""

    if isinstance(payload, _SharedRgbReply):
        connection.send((_SHARED_RGB_BATCH_STATUS, payload.metadata))
        return
    if isinstance(payload, _SharedStepStateRgbReply):
        connection.send((_SHARED_STEP_STATE_RGB_STATUS, (payload.states, payload.metadata)))
        return
    metadata = _packed_rgb_batch_metadata(payload) if operation == "read_sensors" else None
    if metadata is None:
        connection.send(("ok", payload))
        return
    if rgb_transport is not None:
        transport_buffer = cast(memoryview, rgb_transport.buf)
        offset = 0
        shared_metadata: list[tuple[tuple[int, ...], int, int]] = []
        samples = cast(NativeSensorBatch, payload)
        for (shape, size), sample in zip(metadata, samples, strict=True):
            end = offset + size
            if end > rgb_transport.size:
                raise RuntimeError("packed RGB batch exceeds its shared transport capacity")
            transport_buffer[offset:end] = cast(bytes, sample[0][2])
            shared_metadata.append((shape, offset, size))
            offset = end
        connection.send((_SHARED_RGB_BATCH_STATUS, tuple(shared_metadata)))
        return
    connection.send((_PACKED_RGB_BATCH_STATUS, metadata))
    samples = cast(NativeSensorBatch, payload)
    for sample in samples:
        connection.send_bytes(cast(bytes, sample[0][2]))


def _require_world(world: NativeWorldDriver | None, operation: str) -> NativeWorldDriver:
    if world is None:
        raise RuntimeError(f"native worker has no world for operation {operation!r}")
    return world


def _dispatch(
    runtime: NativeRuntime,
    world: NativeWorldDriver | None,
    request: Request,
    rgb_transport: shared_memory.SharedMemory | None = None,
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
    if operation == "apply_render_state":
        active.apply_render_state(cast(NativeRenderStateFrame, args[0]))
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
    if operation == "apply_articulation_commands_step_and_read":
        states = active.apply_articulation_commands_step_and_read(
            cast(tuple[NativeArticulationCommand, ...], args[0]),
            cast(int, args[1]),
            cast(tuple[EntityPath, ...], args[2]),
        )
        return active, states, False
    if operation == "apply_articulation_commands_step_and_read_sensors":
        states = active.apply_articulation_commands_step_and_read(
            cast(tuple[NativeArticulationCommand, ...], args[0]),
            cast(int, args[1]),
            cast(tuple[EntityPath, ...], args[2]),
        )
        sensor_paths = cast(tuple[EntityPath, ...], args[3])
        shared_reader = getattr(active, "read_sensors_into_shared", None)
        if rgb_transport is not None and callable(shared_reader):
            metadata = shared_reader(sensor_paths, cast(memoryview, rgb_transport.buf))
            if metadata is not None:
                return active, _SharedStepStateRgbReply(states, metadata), False
        return active, (states, active.read_sensors(sensor_paths)), False
    if operation == "apply_articulation_commands_step_and_read_encoded_sensors":
        states = active.apply_articulation_commands_step_and_read(
            cast(tuple[NativeArticulationCommand, ...], args[0]),
            cast(int, args[1]),
            cast(tuple[EntityPath, ...], args[2]),
        )
        encoded_reader = getattr(active, "read_encoded_sensors", None)
        if not callable(encoded_reader):
            raise RuntimeError("native world does not support encoded sensors")
        frames = encoded_reader(cast(tuple[NativeEncodedSensorRequest, ...], args[3]))
        return active, (states, frames), False
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
    if operation == "read_entity_prim_states":
        return active, active.read_entity_prim_states(cast(tuple[EntityPath, ...], args[0])), False
    if operation == "set_entity_prim_pose":
        active.set_entity_prim_pose(
            cast(EntityPath, args[0]),
            cast(Vector3, args[1]),
            cast(Quaternion, args[2]),
            cast(int, args[3]),
        )
        return active, None, False
    if operation == "attach_rigid_body":
        relative = active.attach_rigid_body(
            cast(str, args[0]),
            cast(EntityPath, args[1]),
            cast(str | None, args[2]),
            cast(EntityPath, args[3]),
            cast(str | None, args[4]),
            cast(int, args[5]),
            cast(Pose | None, args[6]),
        )
        return active, relative, False
    if operation == "detach_rigid_body":
        active.detach_rigid_body(
            cast(str, args[0]),
            cast(EntityPath, args[1]),
            cast(int, args[2]),
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
    if operation == "read_sensors":
        paths = cast(tuple[EntityPath, ...], args[0])
        shared_reader = getattr(active, "read_sensors_into_shared", None)
        if rgb_transport is not None and callable(shared_reader):
            metadata = shared_reader(paths, cast(memoryview, rgb_transport.buf))
            if metadata is not None:
                return active, _SharedRgbReply(metadata), False
        return active, active.read_sensors(paths), False
    if operation == "read_encoded_sensors":
        encoded_reader = getattr(active, "read_encoded_sensors", None)
        if not callable(encoded_reader):
            raise RuntimeError("native world does not support encoded sensors")
        return active, encoded_reader(cast(tuple[NativeEncodedSensorRequest, ...], args[0])), False
    if operation == "camera_calibration":
        return active, active.camera_calibration(cast(EntityPath, args[0])), False
    if operation == "read_selected_kinematics":
        return (
            active,
            active.read_selected_kinematics(
                cast(tuple[KinematicTarget, ...], args[0]),
                cast(int, args[1]),
            ),
            False,
        )
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
    rgb_transport: shared_memory.SharedMemory | None = None
    try:
        try:
            connection.send(_startup_progress_reply("native_module_loading"))
            from .native import IsaacLabNativeRuntime

            connection.send(_startup_progress_reply("native_module_loaded"))
            runtime = IsaacLabNativeRuntime(
                config,
                process_isolated=True,
                startup_progress=lambda phase: connection.send(_startup_progress_reply(phase)),
            )
        except Exception as exc:
            connection.send(_error_reply(exc))
            return
        connection.send(("ok", _worker_startup_fingerprint()))
        while True:
            try:
                request = cast(Request, connection.recv())
            except EOFError:
                break
            operation = request[0]
            if operation == "build_world" and len(request[1]) == 2 and request[1][1] is not None:
                descriptor = request[1][1]
                if (
                    type(descriptor) is not tuple
                    or len(descriptor) != 2
                    or type(descriptor[0]) is not str
                    or type(descriptor[1]) is not int
                    or descriptor[1] <= 0
                ):
                    connection.send(_error_reply(ValueError("invalid shared RGB transport descriptor")))
                    continue
                rgb_transport = shared_memory.SharedMemory(name=descriptor[0], create=False)
                resource_tracker.unregister(cast(Any, rgb_transport)._name, "shared_memory")
                if rgb_transport.size != descriptor[1]:
                    rgb_transport.close()
                    rgb_transport = None
                    connection.send(_error_reply(ValueError("shared RGB transport capacity differs")))
                    continue
            try:
                world, payload, should_stop = _dispatch(runtime, world, request, rgb_transport)
            except Exception as exc:
                _, args = request
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
            _send_worker_reply(connection, operation, payload, rgb_transport)
            if should_stop:
                break
    finally:
        connection.close()
        if runtime is not None:
            # Fast shutdown terminates only this adapter-owned worker process.
            runtime.close()
        if rgb_transport is not None:
            rgb_transport.close()


def _spawned_worker_main(connection: Connection, config: IsaacLabAdapterConfig) -> None:  # pragma: no cover
    """Supply the same progress contract for the non-POSIX spawn fallback."""

    connection.send(_startup_progress_reply("bootstrap_connected"))
    connection.send(_startup_progress_reply("config_received"))
    connection.send(_startup_progress_reply("worker_imported"))
    _worker_main(connection, config)


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
        target=_spawned_worker_main,
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
        self._rgb_transport: shared_memory.SharedMemory | None = None
        for attempt in range(1, _STARTUP_ATTEMPTS + 1):
            self._connection, self._process = worker_factory(config)
            try:
                startup = self._receive_startup(config)
                _validate_worker_startup(startup)
                break
            except _NativeWorkerTimeout as exc:
                self._abort()
                if attempt == _STARTUP_ATTEMPTS:
                    raise NativeWorkerError(
                        f"native worker startup timed out after {_STARTUP_ATTEMPTS} attempts; last timeout: {exc}"
                    ) from exc
                warnings.warn(
                    f"Isaac Kit worker {exc}; the isolated worker was cleaned up "
                    f"and startup will be retried ({attempt + 1}/{_STARTUP_ATTEMPTS})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception:
                self._abort()
                raise

    def _receive_startup(self, config: IsaacLabAdapterConfig) -> Any:
        started = time.monotonic()
        hard_limit = config.worker_startup_hard_timeout_s
        hard_deadline = started + hard_limit
        phase = "process_spawned"
        next_phase_index = 0
        while True:
            now = time.monotonic()
            idle_limit = (
                config.worker_kit_launch_idle_timeout_s
                if phase == "kit_launching"
                else _STARTUP_PHASE_IDLE_TIMEOUT_SECONDS[phase]
            )
            timeout = min(idle_limit, max(0.0, hard_deadline - now))
            if timeout <= 0.0:
                raise _NativeWorkerTimeout(f"stalled in startup phase {phase!r} at the {hard_limit:g}s hard limit")
            try:
                value = self._receive(
                    "worker startup",
                    timeout_seconds=timeout,
                    allow_startup_progress=True,
                )
            except _NativeWorkerTimeout as exc:
                elapsed = time.monotonic() - started
                limit_kind = "hard" if elapsed >= hard_limit else "idle"
                raise _NativeWorkerTimeout(
                    f"stalled in startup phase {phase!r} after {elapsed:.3f}s "
                    f"({limit_kind} limit, phase idle limit {idle_limit:g}s, "
                    f"hard limit {hard_limit:g}s)"
                ) from exc
            if isinstance(value, _WorkerStartupProgress):
                expected_phase = _STARTUP_PHASES[next_phase_index] if next_phase_index < len(_STARTUP_PHASES) else None
                if value.phase != expected_phase:
                    raise NativeWorkerError(
                        "native worker startup progress is not strictly ordered: "
                        f"expected {expected_phase!r}, got {value.phase!r}"
                    )
                phase = value.phase
                next_phase_index += 1
                continue
            if next_phase_index != len(_STARTUP_PHASES):
                expected_phase = _STARTUP_PHASES[next_phase_index]
                raise NativeWorkerError(
                    "native worker returned its startup fingerprint before completing progress: "
                    f"expected phase {expected_phase!r}"
                )
            return value

    def _receive(
        self,
        operation: str,
        *,
        timeout_seconds: float | None = None,
        allow_startup_progress: bool = False,
    ) -> Any:
        timeout = _CALL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + timeout
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
        if status == _PACKED_RGB_BATCH_STATUS:
            return self._receive_packed_rgb_batch(payload, operation=operation, deadline=deadline)
        if status == _SHARED_RGB_BATCH_STATUS:
            return self._receive_shared_rgb_batch(payload, operation=operation)
        if status == _SHARED_STEP_STATE_RGB_STATUS:
            if type(payload) is not tuple or len(payload) != 2 or type(payload[0]) is not tuple:
                raise NativeWorkerError(f"invalid shared step/state/RGB payload during {operation}: {payload!r}")
            return payload[0], self._receive_shared_rgb_batch(payload[1], operation=operation)
        if status == "startup_progress" and allow_startup_progress:
            return _validate_startup_progress(payload)
        if status == "planning_error" and isinstance(payload, dict):
            raise NativePlanningError(cast(str, payload.get("code", "native_failure"))) from None
        if status != "error" or not isinstance(payload, dict):
            raise NativeWorkerError(f"invalid native worker reply during {operation}: {reply!r}")
        remote_type = payload.get("type", "Exception")
        message = payload.get("message", "")
        remote_traceback = payload.get("traceback", "")
        raise NativeWorkerError(f"{remote_type} during {operation}: {message}\n{remote_traceback}".rstrip())

    def _receive_packed_rgb_batch(
        self,
        value: object,
        *,
        operation: str,
        deadline: float,
    ) -> NativeSensorBatch:
        if type(value) is not tuple:
            raise NativeWorkerError(f"invalid packed RGB metadata during {operation}: {value!r}")
        result: list[NativeSensorSample] = []
        for entry in value:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not tuple
                or not entry[0]
                or any(type(size) is not int or size <= 0 for size in entry[0])
                or type(entry[1]) is not int
                or entry[1] <= 0
                or entry[1] > _MAX_PACKED_RGB_SAMPLE_BYTES
                or math.prod(entry[0]) != entry[1]
            ):
                raise NativeWorkerError(f"invalid packed RGB metadata during {operation}: {entry!r}")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not self._connection.poll(remaining):
                raise _NativeWorkerTimeout(f"timed out after payload header during {operation}")
            try:
                packed = self._connection.recv_bytes()
            except (EOFError, OSError) as exc:
                raise NativeWorkerError(
                    f"native worker disconnected during packed RGB payload for {operation}; "
                    f"exitcode={self._process.exitcode}"
                ) from exc
            if len(packed) != entry[1]:
                raise NativeWorkerError(
                    f"packed RGB payload size differs during {operation}: expected {entry[1]}, got {len(packed)}"
                )
            result.append(((unirobosim.CameraModality.RGB, entry[0], packed),))
        return tuple(result)

    def _receive_shared_rgb_batch(self, value: object, *, operation: str) -> NativeSensorBatch:
        transport = self._rgb_transport
        if transport is None or type(value) is not tuple:
            raise NativeWorkerError(f"invalid shared RGB metadata during {operation}: {value!r}")
        result: list[NativeSensorSample] = []
        transport_buffer = cast(memoryview, transport.buf)
        previous_end = 0
        for entry in value:
            if (
                type(entry) is not tuple
                or len(entry) != 3
                or type(entry[0]) is not tuple
                or not entry[0]
                or any(type(size) is not int or size <= 0 for size in entry[0])
                or type(entry[1]) is not int
                or type(entry[2]) is not int
                or entry[1] != previous_end
                or entry[2] <= 0
                or entry[2] > _MAX_PACKED_RGB_SAMPLE_BYTES
                or math.prod(entry[0]) != entry[2]
                or entry[1] + entry[2] > transport.size
            ):
                raise NativeWorkerError(f"invalid shared RGB metadata during {operation}: {entry!r}")
            end = entry[1] + entry[2]
            packed = bytes(transport_buffer[entry[1] : end])
            result.append(((unirobosim.CameraModality.RGB, entry[0], packed),))
            previous_end = end
        return tuple(result)

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
        capacity = sum(
            entity.camera.width_px * entity.camera.height_px * 3
            for entity in spec.entities
            if entity.camera is not None and unirobosim.CameraModality.RGB in entity.camera.modalities
        )
        if capacity > 0:
            self._rgb_transport = shared_memory.SharedMemory(create=True, size=capacity)
        descriptor = (
            None
            if self._rgb_transport is None
            else (self._rgb_transport.name, self._rgb_transport.size)
        )
        try:
            self._request("build_world", spec, descriptor)
        except BaseException:
            self._close_rgb_transport()
            raise
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
            self._close_rgb_transport()

    def _close_rgb_transport(self) -> None:
        transport = self._rgb_transport
        self._rgb_transport = None
        if transport is not None:
            transport.close()
            transport.unlink()

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
            self._close_rgb_transport()
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

    def apply_render_state(self, frame: NativeRenderStateFrame) -> None:
        self._ensure_open("apply_render_state")
        self._runtime._request("apply_render_state", frame)

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

    def apply_articulation_commands_step_and_read(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[Matrix, Matrix], ...]:
        self._ensure_open("apply_articulation_commands_step_and_read")
        return cast(
            tuple[tuple[Matrix, Matrix], ...],
            self._runtime._request("apply_articulation_commands_step_and_read", commands, count, paths),
        )

    def apply_articulation_commands_step_and_read_sensors(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
        sensor_paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[tuple[Matrix, Matrix], ...], NativeSensorBatch]:
        self._ensure_open("apply_articulation_commands_step_and_read_sensors")
        return cast(
            tuple[tuple[tuple[Matrix, Matrix], ...], NativeSensorBatch],
            self._runtime._request(
                "apply_articulation_commands_step_and_read_sensors",
                commands,
                count,
                paths,
                sensor_paths,
            ),
        )

    def apply_articulation_commands_step_and_read_encoded_sensors(
        self,
        commands: tuple[NativeArticulationCommand, ...],
        count: int,
        paths: tuple[EntityPath, ...],
        sensor_requests: tuple[NativeEncodedSensorRequest, ...],
    ) -> tuple[tuple[tuple[Matrix, Matrix], ...], tuple[NativeEncodedSensorFrame, ...]]:
        self._ensure_open("apply_articulation_commands_step_and_read_encoded_sensors")
        return cast(
            tuple[tuple[tuple[Matrix, Matrix], ...], tuple[NativeEncodedSensorFrame, ...]],
            self._runtime._request(
                "apply_articulation_commands_step_and_read_encoded_sensors",
                commands,
                count,
                paths,
                sensor_requests,
            ),
        )

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

    def read_entity_prim_states(
        self,
        paths: tuple[EntityPath, ...],
    ) -> tuple[tuple[NativeEntityPrimState, ...], ...]:
        self._ensure_open("read_entity_prim_states")
        return cast(
            tuple[tuple[NativeEntityPrimState, ...], ...],
            self._runtime._request("read_entity_prim_states", paths),
        )

    def set_entity_prim_pose(
        self,
        path: EntityPath,
        position_m: Vector3,
        orientation_xyzw: Quaternion,
        environment_index: int,
    ) -> None:
        self._ensure_open("set_entity_prim_pose")
        self._runtime._request(
            "set_entity_prim_pose",
            path,
            position_m,
            orientation_xyzw,
            environment_index,
        )

    def attach_rigid_body(
        self,
        attachment_id: str,
        parent_path: EntityPath,
        parent_link_name: str | None,
        child_path: EntityPath,
        child_link_name: str | None,
        environment_index: int,
        parent_T_child: Pose | None,
    ) -> Pose:
        self._ensure_open("attach_rigid_body")
        return cast(
            Pose,
            self._runtime._request(
                "attach_rigid_body",
                attachment_id,
                parent_path,
                parent_link_name,
                child_path,
                child_link_name,
                environment_index,
                parent_T_child,
            ),
        )

    def detach_rigid_body(
        self,
        attachment_id: str,
        child_path: EntityPath,
        environment_index: int,
    ) -> None:
        self._ensure_open("detach_rigid_body")
        self._runtime._request(
            "detach_rigid_body",
            attachment_id,
            child_path,
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

    def read_sensors(self, paths: tuple[EntityPath, ...]) -> NativeSensorBatch:
        self._ensure_open("read_sensors")
        return cast(NativeSensorBatch, self._runtime._request("read_sensors", paths))

    def read_encoded_sensors(
        self,
        requests: tuple[NativeEncodedSensorRequest, ...],
    ) -> tuple[NativeEncodedSensorFrame, ...]:
        self._ensure_open("read_encoded_sensors")
        return cast(
            tuple[NativeEncodedSensorFrame, ...],
            self._runtime._request("read_encoded_sensors", requests),
        )

    def camera_calibration(self, path: EntityPath) -> NativeCameraCalibration:
        self._ensure_open("camera_calibration")
        return cast(NativeCameraCalibration, self._runtime._request("camera_calibration", path))

    def read_selected_kinematics(
        self,
        targets: tuple[KinematicTarget, ...],
        environment_index: int = 0,
    ) -> tuple[NativeKinematicState, ...]:
        self._ensure_open("read_selected_kinematics")
        return cast(
            tuple[NativeKinematicState, ...],
            self._runtime._request("read_selected_kinematics", targets, environment_index),
        )

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

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from unirobosim import (
    ArrayValue,
    CameraModality,
    CameraSpec,
    CommandMode,
    DebugBatch,
    DebugPrimitive,
    DebugPrimitiveKind,
    EntityKind,
    EntityPath,
    EntitySpec,
    ParticleFluidSpec,
    PointCommandMode,
    WorldSpec,
)

from unirobosim_isaaclab import worker as worker_module
from unirobosim_isaaclab.config import IsaacLabAdapterConfig
from unirobosim_isaaclab.native_protocols import NativePlanningError
from unirobosim_isaaclab.worker import (
    IsaacLabWorkerPlanningWorld,
    IsaacLabWorkerRuntime,
    IsaacLabWorkerWorld,
    NativeWorkerError,
    WorkerFactory,
    _dispatch,
    _error_reply,
    _planning_error_reply,
    _proc_stat_session,
    _session_member_pids,
    _SubprocessHandle,
    _terminate_worker_tree,
)

from .helpers import FakeNativeRuntime, FakeNativeWorld, make_articulation_asset, make_world
from .test_planning_scene import _make_planning_fixture


def extended_world(asset: Path) -> WorldSpec:
    base = make_world(asset)
    return WorldSpec(
        base.world_id,
        (
            *base.entities,
            EntitySpec(
                EntityPath("/fluid"),
                EntityKind.PARTICLE_FLUID,
                particle_fluid=ParticleFluidSpec(ArrayValue.from_nested(((0.0, 0.0, 1.0),))),
            ),
            EntitySpec(
                EntityPath("/camera"),
                EntityKind.CAMERA_SENSOR,
                camera=CameraSpec(width_px=2, height_px=2),
            ),
        ),
        environments=base.environments,
    )


def debug_batch() -> DebugBatch:
    return DebugBatch(
        (
            DebugPrimitive(
                "point",
                "test",
                DebugPrimitiveKind.POINT_SET,
                ArrayValue.from_nested([[[0.0, 0.0, 0.0]]]),
                (0,),
            ),
        )
    )


class FakeProcess:
    def __init__(self, *, alive: bool = True, exitcode: int | None = 0, pid: int | None = None) -> None:
        self.alive = alive
        self._exitcode = exitcode
        self._pid = pid
        self.join_calls: list[float | None] = []
        self.terminate_calls = 0

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        self.alive = False

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False


class FakeConnection:
    def __init__(self, replies: list[object], process: FakeProcess, *, poll_result: bool = True) -> None:
        self.replies = replies
        self.process = process
        self.poll_result = poll_result
        self.sent: list[object] = []
        self.closed = False
        self.send_error: OSError | None = None

    def poll(self, timeout: float) -> bool:
        del timeout
        return self.poll_result

    def recv(self) -> object:
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def send(self, value: object) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


class FakePopen:
    def __init__(self) -> None:
        self.pid = 1234
        self.returncode: int | None = None
        self.wait_calls: list[float | None] = []
        self.terminate_calls = 0
        self.timeout = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.timeout:
            raise subprocess.TimeoutExpired("worker", timeout)
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15


def fake_worker_factory(
    replies: list[object], *, process: FakeProcess | None = None, poll_result: bool = True
) -> tuple[WorkerFactory, FakeConnection, FakeProcess]:
    handle = process if process is not None else FakeProcess()
    connection = FakeConnection(replies, handle, poll_result=poll_result)

    def factory(config: IsaacLabAdapterConfig) -> tuple[Any, Any]:
        del config
        return connection, handle

    return cast(WorkerFactory, factory), connection, handle


def test_dispatches_complete_native_world_protocol(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    spec = extended_world(make_articulation_asset(tmp_path / "robot.usda"))

    world, result, stop = _dispatch(runtime, None, ("build_world", (spec,)))
    assert result is None and not stop and world is runtime.worlds[0]
    assert isinstance(world, FakeNativeWorld)

    world, result, stop = _dispatch(runtime, world, ("reset", ((0, 1),)))
    assert result is None and not stop
    assert runtime.worlds[0].calls[-1] == ("reset", (0, 1))

    world, _, _ = _dispatch(
        runtime,
        world,
        (
            "apply_articulation",
            (EntityPath("/robots/arm"), CommandMode.POSITION, ((0.1, 0.2),), (0,), (0, 1)),
        ),
    )
    world, articulation, _ = _dispatch(runtime, world, ("read_articulation", (EntityPath("/robots/arm"),)))
    assert articulation[0] == ((0.2, -0.1), (0.2, -0.1))

    forces = ((1.0, 0.0, 0.0),)
    torques = ((0.0, 0.0, 0.5),)
    world, _, _ = _dispatch(
        runtime,
        world,
        ("apply_rigid_body_wrench", (EntityPath("/props/marker"), forces, torques, (1,))),
    )
    world, rigid, _ = _dispatch(runtime, world, ("read_rigid_body", (EntityPath("/props/marker"),)))
    assert rigid[0] == ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    world, _, _ = _dispatch(
        runtime,
        world,
        ("set_rigid_body_pose", (EntityPath("/props/marker"), (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 1)),
    )
    world, moved, _ = _dispatch(runtime, world, ("read_rigid_body", (EntityPath("/props/marker"),)))
    assert moved[0][1] == (1.0, 2.0, 3.0)
    world, contact, _ = _dispatch(runtime, world, ("read_contact", (EntityPath("/props/marker"),)))
    assert contact == ((0.0, 0.0, 9.81), (0.0, 0.0, 9.81))

    targets = (((0.0, 0.0, 1.0),),)
    world, _, _ = _dispatch(
        runtime,
        world,
        ("apply_deformable_position", (EntityPath("/soft/jelly"), targets, (0,), (0,))),
    )
    world, deformable, _ = _dispatch(runtime, world, ("read_deformable", (EntityPath("/soft/jelly"),)))
    assert len(deformable[0]) == 2

    fluid_targets = (((0.25, 0.0, 1.0),),)
    world, _, _ = _dispatch(
        runtime,
        world,
        ("apply_particle_fluid", (EntityPath("/fluid"), PointCommandMode.POSITION, fluid_targets, (0,), (0,))),
    )
    world, fluid, _ = _dispatch(runtime, world, ("read_particle_fluid", (EntityPath("/fluid"),)))
    assert fluid[0][0][0] == (0.0, 0.0, 1.0)
    world, sensor, _ = _dispatch(runtime, world, ("read_sensor", (EntityPath("/camera"),)))
    assert tuple(channel[0] for channel in sensor) == (CameraModality.RGB, CameraModality.DEPTH)
    batch = debug_batch()
    world, report, _ = _dispatch(runtime, world, ("publish_debug", (batch,)))
    assert report == (1, 0, 1)
    world, cleared, _ = _dispatch(runtime, world, ("clear_debug", ("test", "default", "point")))
    assert cleared == 1

    world, _, _ = _dispatch(runtime, world, ("step", (3,)))
    last_call: Any = runtime.worlds[0].calls[-1]
    assert last_call == ("step", 3)
    world, _, _ = _dispatch(runtime, world, ("close_world", ()))
    assert world is None and runtime.worlds[0].closed

    world, _, stop = _dispatch(runtime, world, ("close_runtime", ()))
    assert world is None and stop


def test_dispatch_rejects_invalid_state_and_operation(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    with pytest.raises(RuntimeError, match="has no world"):
        _dispatch(runtime, None, ("step", (1,)))
    world = runtime.build_world(make_world(make_articulation_asset(tmp_path / "robot.usda")))
    with pytest.raises(RuntimeError, match="already owns"):
        _dispatch(runtime, world, ("build_world", (world.spec,)))
    with pytest.raises(RuntimeError, match="unknown"):
        _dispatch(runtime, world, ("bad", ()))
    closed, _, stop = _dispatch(runtime, world, ("close_runtime", ()))
    assert closed is None and stop and world.closed


def test_error_reply_preserves_remote_diagnostics() -> None:
    try:
        raise ValueError("bad native state")
    except ValueError as exc:
        status, payload = _error_reply(exc)
    assert status == "error"
    assert payload["type"] == "ValueError"
    assert payload["message"] == "bad native state"
    assert "ValueError: bad native state" in payload["traceback"]
    assert issubclass(NativeWorkerError, RuntimeError)


def test_planning_error_reply_is_bounded_and_contains_no_native_diagnostics() -> None:
    status, payload = _planning_error_reply(RuntimeError("private native traceback detail"))
    assert status == "planning_error"
    assert payload == {"code": "native_failure"}
    assert "private" not in repr((status, payload))

    status, payload = _planning_error_reply(NativePlanningError("frame_missing"))
    assert status == "planning_error"
    assert payload == {"code": "frame_missing"}


def test_proc_stat_session_parser_handles_spaces_and_invalid_input() -> None:
    assert _proc_stat_session("447389 (omni telemetry) S 1 447389 447342 0 -1") == 447342
    assert _proc_stat_session("invalid") is None
    assert _proc_stat_session("1 (x) S") is None
    assert _proc_stat_session("1 (x) S 0 1 nope") is None


def test_subprocess_handle_adapts_lifecycle_and_timeout() -> None:
    process = FakePopen()
    handle = _SubprocessHandle(cast(Any, process))
    assert handle.pid == 1234 and handle.exitcode is None and handle.is_alive()
    process.timeout = True
    handle.join(0.25)
    assert process.wait_calls == [0.25] and handle.is_alive()
    process.timeout = False
    handle.join()
    assert handle.exitcode == 0 and not handle.is_alive()
    process.returncode = None
    handle.terminate()
    assert process.terminate_calls == 1 and handle.exitcode == -15


def test_session_member_scan_finds_current_process() -> None:
    assert os.getpid() in _session_member_pids(os.getsid(0))


def test_worker_runtime_and_world_proxy_complete_protocol(tmp_path: Path) -> None:
    positions = ((0.2, -0.1), (0.2, -0.1))
    velocities = ((0.0, 0.0), (0.0, 0.0))
    points = (((0.0, 0.0, 1.0),), ((0.0, 0.0, 1.0),))
    zeros = (((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0),))
    rigid_positions = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    orientations = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    rigid_zeros = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    contacts = ((0.0, 0.0, 9.81), (0.0, 0.0, 9.81))
    sensor = (
        (CameraModality.RGB, (2, 2, 2, 3), (17,) * 24),
        (CameraModality.DEPTH, (2, 2, 2), (1.25,) * 8),
    )
    factory, connection, process = fake_worker_factory(
        [
            ("ok", None),
            ("ok", None),
            ("ok", None),
            ("ok", None),
            ("ok", (positions, velocities)),
            ("ok", None),
            ("ok", (rigid_positions, orientations, rigid_zeros, rigid_zeros)),
            ("ok", None),
            ("ok", contacts),
            ("ok", None),
            ("ok", (points, zeros)),
            ("ok", None),
            ("ok", (points, zeros)),
            ("ok", sensor),
            ("ok", (1, 0, 1)),
            ("ok", 1),
            ("ok", None),
            ("ok", None),
            ("ok", None),
        ]
    )
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    spec = extended_world(make_articulation_asset(tmp_path / "robot.usda"))
    world = runtime.build_world(spec)
    assert type(world) is IsaacLabWorkerWorld
    assert not isinstance(world, IsaacLabWorkerPlanningWorld)
    assert "planning_catalog" not in dir(world)
    with pytest.raises(NativeWorkerError, match="already owns"):
        runtime.build_world(spec)

    world.reset((0, 1))
    world.apply_articulation(EntityPath("/robots/arm"), CommandMode.POSITION, ((0.1,),), (0,), (0,))
    assert world.read_articulation(EntityPath("/robots/arm")) == (positions, velocities)
    world.apply_rigid_body_wrench(
        EntityPath("/props/marker"),
        ((1.0, 0.0, 0.0),),
        ((0.0, 0.0, 0.5),),
        (1,),
    )
    assert world.read_rigid_body(EntityPath("/props/marker")) == (
        rigid_positions,
        orientations,
        rigid_zeros,
        rigid_zeros,
    )
    world.set_rigid_body_pose(EntityPath("/props/marker"), (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 1)
    assert world.read_contact(EntityPath("/props/marker")) == contacts
    world.apply_deformable_position(EntityPath("/soft/jelly"), (((0.0, 0.0, 1.0),),), (0,), (0,))
    assert world.read_deformable(EntityPath("/soft/jelly")) == (points, zeros)
    world.apply_particle_fluid(EntityPath("/fluid"), PointCommandMode.VELOCITY, (((0.1, 0.0, 0.0),),), (0,), (0,))
    assert world.read_particle_fluid(EntityPath("/fluid")) == (points, zeros)
    assert world.read_sensor(EntityPath("/camera")) == sensor
    batch = debug_batch()
    assert world.publish_debug(batch) == (1, 0, 1)
    assert world.clear_debug("test", "default", "point") == 1
    world.step(2)
    world.close()
    world.close()
    assert world.closed
    with pytest.raises(NativeWorkerError, match="closed"):
        world.step(1)

    runtime.close()
    runtime.close()
    assert connection.closed and not process.alive
    with pytest.raises(NativeWorkerError, match="closed"):
        runtime._request("step", 1)
    operations = [cast(tuple[Any, ...], request)[0] for request in connection.sent]
    assert operations == [
        "build_world",
        "reset",
        "apply_articulation",
        "read_articulation",
        "apply_rigid_body_wrench",
        "read_rigid_body",
        "set_rigid_body_pose",
        "read_contact",
        "apply_deformable_position",
        "read_deformable",
        "apply_particle_fluid",
        "read_particle_fluid",
        "read_sensor",
        "publish_debug",
        "clear_debug",
        "step",
        "close_world",
        "close_runtime",
    ]


def test_worker_planning_proxy_is_demand_only_and_complete(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    resource = next(iter(fixture.resources.values()))
    factory, connection, process = fake_worker_factory(
        [
            ("ok", None),
            ("ok", None),
            ("ok", fixture.catalog),
            ("ok", fixture.state),
            ("ok", resource),
            ("ok", None),
            ("ok", None),
        ]
    )
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    world = runtime.build_world(fixture.spec)
    assert type(world) is IsaacLabWorkerPlanningWorld
    assert world.planning_catalog() == fixture.catalog
    assert world.planning_state() == fixture.state
    assert world.planning_resource(resource.geometry_id) == resource
    world.close()
    runtime.close()
    assert connection.closed and not process.alive
    operations = [cast(tuple[Any, ...], request)[0] for request in connection.sent]
    assert operations == [
        "build_world",
        "planning_catalog",
        "planning_state",
        "planning_resource",
        "close_world",
        "close_runtime",
    ]


def test_worker_planning_failure_maps_only_bounded_code(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    factory, connection, _ = fake_worker_factory(
        [
            ("ok", None),
            ("planning_error", {"code": "frame_missing", "message": "must be ignored"}),
            ("ok", None),
        ]
    )
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    with pytest.raises(NativePlanningError) as caught:
        runtime.build_world(fixture.spec)
    assert caught.value.code == "frame_missing"
    assert "must be ignored" not in str(caught.value)
    runtime.close()
    assert [cast(tuple[Any, ...], request)[0] for request in connection.sent] == ["build_world", "close_runtime"]


@pytest.mark.parametrize(
    ("startup_reply", "message"),
    [
        (("error", {"type": "ValueError", "message": "boom", "traceback": "remote trace"}), "ValueError"),
        (("unexpected", None), "invalid native worker reply"),
        (EOFError(), "disconnected"),
    ],
)
def test_worker_startup_failures_abort(startup_reply: object, message: str) -> None:
    factory, connection, process = fake_worker_factory([startup_reply])
    with pytest.raises(NativeWorkerError, match=message):
        IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    assert connection.closed and not process.alive


def test_worker_startup_timeout_aborts() -> None:
    factory, connection, process = fake_worker_factory([], poll_result=False)
    with pytest.raises(NativeWorkerError, match="timed out"):
        IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    assert connection.closed and not process.alive


def test_worker_request_transport_failures(tmp_path: Path) -> None:
    spec = make_world(make_articulation_asset(tmp_path / "robot.usda"))

    dead_process = FakeProcess(alive=False, exitcode=9)
    factory, _, _ = fake_worker_factory([("ok", None)], process=dead_process)
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    with pytest.raises(NativeWorkerError, match="exited before"):
        runtime.build_world(spec)

    factory, connection, _ = fake_worker_factory([("ok", None)])
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    connection.send_error = BrokenPipeError("gone")
    with pytest.raises(NativeWorkerError, match="failed to contact"):
        runtime.build_world(spec)


def test_worker_remote_world_error_and_close_error(tmp_path: Path) -> None:
    spec = make_world(make_articulation_asset(tmp_path / "robot.usda"))
    remote_error = ("error", {"type": "RuntimeError", "message": "bad build", "traceback": "trace"})
    factory, _, _ = fake_worker_factory([("ok", None), remote_error])
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    with pytest.raises(NativeWorkerError, match="bad build"):
        runtime.build_world(spec)

    factory, _, _ = fake_worker_factory([("ok", None), remote_error])
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    with pytest.raises(NativeWorkerError, match="bad build"):
        runtime.close()


def test_worker_reports_nonzero_exit_after_close() -> None:
    process = FakeProcess(exitcode=7)
    factory, _, _ = fake_worker_factory([("ok", None), ("ok", None)], process=process)
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    with pytest.raises(NativeWorkerError, match="status 7"):
        runtime.close()


def test_worker_close_marks_active_world_and_terminates_stubborn_process(tmp_path: Path) -> None:
    class StubbornProcess(FakeProcess):
        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)
            if len(self.join_calls) > 1:
                self.alive = False

    process = StubbornProcess(pid=None)
    factory, _, _ = fake_worker_factory(
        [("ok", None), ("ok", None), ("ok", None)],
        process=process,
    )
    runtime = IsaacLabWorkerRuntime(IsaacLabAdapterConfig(), worker_factory=factory)
    world = runtime.build_world(make_world(make_articulation_asset(tmp_path / "robot.usda")))
    runtime.close()
    assert world.closed
    assert process.terminate_calls == 1
    assert len(process.join_calls) == 2


def test_worker_tree_cleanup_targets_only_owned_session(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess(pid=900_001)
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "getsid", lambda pid: 100)
    monkeypatch.setattr(worker_module, "_session_member_pids", lambda session_id: (os.getpid(), 900_002, 900_001))
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

    _terminate_worker_tree(process)

    assert killed == [(900_002, signal.SIGKILL), (900_001, signal.SIGKILL)]
    assert process.terminate_calls == 0


def test_worker_tree_cleanup_falls_back_to_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeProcess(pid=900_003)
    killed_groups: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "getsid", lambda pid: 100)
    monkeypatch.setattr(worker_module, "_session_member_pids", lambda session_id: ())
    monkeypatch.setattr(os, "getpgrp", lambda: 100)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed_groups.append((pid, sig)))

    _terminate_worker_tree(process)

    assert killed_groups == [(900_003, signal.SIGKILL)]


def test_worker_tree_cleanup_without_pid_terminates_live_process() -> None:
    process = FakeProcess(pid=None)
    _terminate_worker_tree(process)
    assert process.terminate_calls == 1

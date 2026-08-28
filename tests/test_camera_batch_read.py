from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from unirobosim import (
    ArrayValue,
    CameraModality,
    CameraSpec,
    CommandError,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    ParticleFluidSpec,
    ValidationError,
    WorldSpec,
)

from unirobosim_isaaclab import IsaacLabProvider
from unirobosim_isaaclab.config import IsaacLabAdapterConfig
from unirobosim_isaaclab.native import IsaacLabNativeWorld

from .helpers import FakeNativeRuntime, available_probe


class _HostArray:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self, *, order: str) -> bytes:
        assert order == "C"
        return self._payload


class _Tensor:
    def __init__(self, shape: tuple[int, ...], payload: bytes, counters: dict[str, int]) -> None:
        self.shape = shape
        self.payload = payload
        self._counters = counters
        self.device = SimpleNamespace(type="cuda", index=0)
        self.dtype = "uint8"

    def to(self, *, dtype: object) -> _Tensor:
        del dtype
        self._counters["to"] += 1
        return self

    def contiguous(self) -> _Tensor:
        return self

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        self._counters["cpu"] += 1
        return self

    def numpy(self) -> _HostArray:
        return _HostArray(self.payload)


class _Torch:
    uint8 = "uint8"

    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters
        self.cuda = SimpleNamespace(current_stream=lambda *, device: _Stream(counters))

    def stack(self, tensors: tuple[_Tensor, ...], *, dim: int) -> _Tensor:
        self._counters["stack"] += 1
        raise AssertionError("RGB staging must not stack GPU tensors")

    def empty(
        self,
        shape: tuple[int, ...],
        *,
        device: str,
        dtype: object,
        pin_memory: bool,
    ) -> _HostBatch:
        assert device == "cpu" and dtype == self.uint8 and pin_memory
        self._counters["empty"] += 1
        return _HostBatch(shape[0], self._counters)


class _Stream:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters

    def synchronize(self) -> None:
        self._counters["sync"] += 1


class _HostBatch:
    def __init__(self, count: int, counters: dict[str, int]) -> None:
        self._rows = [_HostRow(counters) for _ in range(count)]

    def __getitem__(self, index: int) -> _HostRow:
        return self._rows[index]


class _HostRow:
    def __init__(self, counters: dict[str, int]) -> None:
        self._counters = counters
        self._payload = b""

    def copy_(self, tensor: _Tensor, *, non_blocking: bool) -> None:
        assert non_blocking
        self._counters["copy"] += 1
        self._payload = tensor.payload

    def numpy(self) -> _HostArray:
        return _HostArray(self._payload)


class _Camera:
    def __init__(self, tensor: _Tensor) -> None:
        self.data = SimpleNamespace(output={"rgb": tensor})
        self.update_count = 0

    def update(self, dt: float, *, force_recompute: bool) -> None:
        assert dt == 0.0 and force_recompute
        self.update_count += 1


class _Simulation:
    def __init__(self) -> None:
        self.render_count = 0

    def render(self) -> None:
        self.render_count += 1


def _native_camera_world(
    entities: tuple[EntitySpec, ...],
    payload: Callable[[int, EntitySpec], bytes],
) -> tuple[IsaacLabNativeWorld, _Simulation, dict[str, int], tuple[_Camera, ...]]:
    world = object.__new__(IsaacLabNativeWorld)
    counters = {"to": 0, "cpu": 0, "stack": 0, "empty": 0, "copy": 0, "sync": 0}
    cameras = tuple(
        _Camera(
            _Tensor(
                (1, entity.camera.height_px, entity.camera.width_px, 3),
                payload(index, entity),
                counters,
            )
        )
        for index, entity in enumerate(entities)
        if entity.camera is not None
    )
    simulation = _Simulation()
    world._spec = SimpleNamespace(entities=entities)
    world._cameras = {entity.path: camera for entity, camera in zip(entities, cameras, strict=True)}
    world._m = SimpleNamespace(torch=_Torch(counters))
    world._rgb_host_staging = {}
    world._sim = simulation
    world._render_revision = 0
    world._rendered_revision = -1
    world._sync_all_mounted_cameras = lambda: None
    return world, simulation, counters, cameras


def _rgb_camera(path: str, *, width: int = 2, height: int = 2) -> EntitySpec:
    return EntitySpec(
        EntityPath(path),
        EntityKind.CAMERA_SENSOR,
        camera=CameraSpec(width_px=width, height_px=height, modalities=(CameraModality.RGB,)),
    )


def test_public_batch_preserves_order_tick_and_packed_bytes_with_atomic_validation() -> None:
    runtime = FakeNativeRuntime()
    provider = IsaacLabProvider(
        IsaacLabAdapterConfig(enable_cameras=True, render=True),
        runtime_factory=lambda config: runtime,
        probe_function=available_probe,
    )
    session = provider.open()
    left = _rgb_camera("/left")
    right = _rgb_camera("/right")
    fluid = EntitySpec(
        EntityPath("/fluid"),
        EntityKind.PARTICLE_FLUID,
        particle_fluid=ParticleFluidSpec(ArrayValue.from_nested(((0.0, 0.0, 1.0),))),
    )
    world = session.build(WorldSpec("camera-batch", (left, right, fluid), environments=EnvironmentSpec(1)))
    handles = (world.resolve(right.path), world.resolve(left.path))
    before = world.tick

    samples = world.read_sensors(handles)

    assert tuple(sample.handle for sample in samples) == handles
    assert samples[0].tick is samples[1].tick
    assert samples[0].tick == before and world.tick == before
    assert all(sample.channel(CameraModality.RGB).is_packed for sample in samples)
    assert samples[0].channel(CameraModality.RGB).to_bytes() == bytes((17,)) * 12
    assert runtime.worlds[0].calls[-1] == ("read_sensors", (right.path, left.path))

    call_count = len(runtime.worlds[0].calls)
    with pytest.raises(CommandError, match="not a camera"):
        world.read_sensors((handles[0], world.resolve(fluid.path)))
    assert len(runtime.worlds[0].calls) == call_count
    with pytest.raises(ValidationError, match="iterable"):
        world.read_sensors(None)  # type: ignore[arg-type]
    assert world.read_sensors(()) == ()
    assert len(runtime.worlds[0].calls) == call_count
    session.close()


def test_native_compatible_rgb_batch_uses_one_render_update_and_host_transfer() -> None:
    entities = (_rgb_camera("/a"), _rgb_camera("/b"), _rgb_camera("/c"))
    sample_size = 12
    world, simulation, counters, cameras = _native_camera_world(
        entities,
        lambda index, entity: bytes((index + 1,)) * sample_size,
    )

    samples = world.read_sensors(tuple(entity.path for entity in reversed(entities)))

    assert simulation.render_count == 1
    assert tuple(camera.update_count for camera in cameras) == (1, 1, 1)
    assert counters == {"to": 3, "cpu": 0, "stack": 0, "empty": 1, "copy": 3, "sync": 1}
    assert tuple(sample[0][2] for sample in samples) == (
        bytes((3,)) * sample_size,
        bytes((2,)) * sample_size,
        bytes((1,)) * sample_size,
    )

    world.read_sensors(tuple(entity.path for entity in reversed(entities)))
    assert counters["empty"] == 1
    assert counters["copy"] == 6 and counters["sync"] == 2 and counters["stack"] == 0


def test_native_fallback_keeps_one_render_and_single_camera_remains_compatible() -> None:
    mixed_shapes = (_rgb_camera("/wide", width=2, height=1), _rgb_camera("/narrow", width=1, height=1))
    world, simulation, counters, cameras = _native_camera_world(
        mixed_shapes,
        lambda index, entity: bytes((9 + index,)) * (entity.camera.width_px * entity.camera.height_px * 3),
    )

    samples = world.read_sensors(tuple(entity.path for entity in mixed_shapes))

    assert simulation.render_count == 1
    assert tuple(camera.update_count for camera in cameras) == (1, 1)
    assert counters["stack"] == 0 and counters["cpu"] == 2
    assert samples[0][0][1] == (1, 1, 2, 3)
    assert samples[1][0][1] == (1, 1, 1, 3)

    single = (_rgb_camera("/only", width=1, height=1),)
    legacy_world, _, _, _ = _native_camera_world(single, lambda index, entity: b"\x2a" * 3)
    batch_world, _, _, _ = _native_camera_world(single, lambda index, entity: b"\x2a" * 3)
    assert legacy_world.read_sensor(single[0].path) == batch_world.read_sensors((single[0].path,))[0]


def test_native_batch_validates_every_path_before_rendering() -> None:
    entities = (_rgb_camera("/valid"),)
    world, simulation, counters, cameras = _native_camera_world(entities, lambda index, entity: b"\x01" * 12)

    with pytest.raises(KeyError, match="/missing"):
        world.read_sensors((entities[0].path, EntityPath("/missing")))

    assert simulation.render_count == 0
    assert cameras[0].update_count == 0
    assert counters == {"to": 0, "cpu": 0, "stack": 0, "empty": 0, "copy": 0, "sync": 0}

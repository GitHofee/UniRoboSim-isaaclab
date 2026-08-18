from __future__ import annotations

import ast
import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from unirobosim import CapabilityId, FrozenMap, ProbeReport, Provider, ValidationError

import unirobosim_isaaclab
from unirobosim_isaaclab import CAPABILITIES, DESCRIPTOR, IsaacLabAdapterConfig, IsaacLabProvider
from unirobosim_isaaclab import probe as probe_module
from unirobosim_isaaclab.native import (
    _environment_origins,
    _launcher_kwargs,
    _native_name,
    _rotate_xyzw,
    _surface_from_tetrahedra,
)
from unirobosim_isaaclab.probe import probe_environment

from .helpers import available_probe


def test_public_identity_and_protocol() -> None:
    provider = unirobosim_isaaclab.create_provider(IsaacLabAdapterConfig(device="cpu"))
    assert isinstance(provider, Provider)
    assert provider.descriptor is DESCRIPTOR
    assert unirobosim_isaaclab.__version__ == "0.4.0a0"
    assert DESCRIPTOR.provider_id == "nvidia.isaaclab"
    assert DESCRIPTOR.contract_version == "v0alpha4"
    assert CAPABILITIES.get(CapabilityId("state.rigid_body@1")) is not None
    assert CAPABILITIES.get(CapabilityId("control.rigid_body.wrench@1")) is not None
    assert CAPABILITIES.get(CapabilityId("contact.net_normal_force@1")) is not None
    assert CAPABILITIES.get(CapabilityId("state.fluid.particles@1")) is not None
    assert CAPABILITIES.get(CapabilityId("control.fluid.particles@1")) is not None
    assert CAPABILITIES.get(CapabilityId("debug.sink.native_overlay@1")) is not None
    assert CAPABILITIES.get(CapabilityId("render.browser-scene@1")) is not None
    assert CAPABILITIES.get(CapabilityId("sensor.camera.rgb@1")) is None
    camera_provider = unirobosim_isaaclab.create_provider(
        IsaacLabAdapterConfig(device="cpu", enable_cameras=True, render=True)
    )
    assert camera_provider.descriptor is not DESCRIPTOR
    assert camera_provider.descriptor.capabilities.get(CapabilityId("sensor.camera.rgb@1")) is not None
    control = CAPABILITIES.get(CapabilityId("control.deformable.points@1"))
    assert control is not None
    assert control.properties == FrozenMap(
        {"frame": "environment-local-world", "modes": ["position"], "topologies": ["volume"]}
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:12"])
def test_valid_config(device: str) -> None:
    config = IsaacLabAdapterConfig(device=device, environment_spacing_m=2, position_stiffness=2)
    assert config.environment_spacing_m == 2.0
    assert config.position_stiffness == 2.0


@pytest.mark.parametrize("device", ["", "gpu", "cuda:-1", "CUDA:0", 7])
def test_invalid_device(device: object) -> None:
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(device=device)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", [0, -1, float("inf"), float("nan"), True, "bad"])
def test_invalid_spacing(spacing: object) -> None:
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(environment_spacing_m=spacing)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["headless", "enable_cameras", "render"])
def test_invalid_boolean_flags(field: str) -> None:
    values = {field: 1}
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("gain", [-1, float("inf"), float("nan"), True, "bad"])
def test_invalid_gains(gain: object) -> None:
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(position_damping=gain)  # type: ignore[arg-type]


def test_render_and_experience_validation() -> None:
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(render=True, headless=True, enable_cameras=False)
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(experience="")
    for invalid in (0, 15, 1_000_001, True, 1.5):
        with pytest.raises(ValidationError):
            IsaacLabAdapterConfig(max_cached_scene_commands=invalid)  # type: ignore[arg-type]
    assert IsaacLabAdapterConfig(render=True, enable_cameras=True).render


def test_probe_cpu_success_and_version_failures() -> None:
    versions: dict[str, str | None] = {
        "isaaclab": "6.1.17",
        "isaaclab_physx": "1.1.3",
        "isaacsim": "6.0.1.0",
        "torch": "2.10.0",
    }
    report = probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get)
    assert report.available
    assert report.reason is None
    assert report.details["versions"] == FrozenMap(versions)

    versions["isaacsim"] = "5.1.0.0"
    versions["isaaclab_physx"] = None
    versions["torch"] = "2.11.0"
    report = probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get)
    assert not report.available
    assert "isaacsim==6.0.1.0" in (report.reason or "")
    assert "isaaclab_physx is not installed" in (report.reason or "")
    assert "torch==2.10.0" in (report.reason or "")

    versions.update({"isaacsim": "6.0.1.0", "isaaclab_physx": "1.1.3", "torch": "2.10.0+cu128"})
    assert probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get).available


def test_probe_rejects_other_python(monkeypatch: pytest.MonkeyPatch) -> None:
    class Version:
        major = 3
        minor = 11
        micro = 9

        def __getitem__(self, item: slice) -> tuple[int, int]:
            assert item == slice(None, 2)
            return (self.major, self.minor)

    versions = {**probe_module._EXPECTED, "torch": "2.10.0"}
    monkeypatch.setattr(sys, "version_info", Version())
    report = probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get)
    assert not report.available
    assert "Python 3.12 is required" in (report.reason or "")


def test_probe_cuda_success_and_command_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {**probe_module._EXPECTED, "torch": "2.10.0"}

    def successful(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="RTX 5090, 580.1\n", stderr="")

    monkeypatch.setattr(subprocess, "run", successful)
    report = probe_environment(IsaacLabAdapterConfig(), DESCRIPTOR, version_reader=versions.get)
    assert report.available
    assert report.details["gpu"] == "RTX 5090, 580.1"

    def failed(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="", stderr="driver unavailable")

    monkeypatch.setattr(subprocess, "run", failed)
    report = probe_environment(IsaacLabAdapterConfig(), DESCRIPTOR, version_reader=versions.get)
    assert not report.available
    assert "driver unavailable" in (report.reason or "")

    def missing(*args: object, **kwargs: object) -> None:
        raise OSError("not found")

    monkeypatch.setattr(subprocess, "run", missing)
    report = probe_environment(IsaacLabAdapterConfig(), DESCRIPTOR, version_reader=versions.get)
    assert "nvidia-smi failed" in (report.reason or "")


def test_distribution_reader_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert probe_module._distribution_version("missing") is None


def test_provider_custom_probe_is_used() -> None:
    provider = IsaacLabProvider(probe_function=available_probe, runtime_factory=lambda config: object())  # type: ignore[arg-type,return-value]
    assert isinstance(provider.probe(), ProbeReport)
    with pytest.raises(ValidationError):
        IsaacLabProvider(config="bad")  # type: ignore[arg-type]


def test_default_runtime_factory_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    from unirobosim_isaaclab import provider as provider_module
    from unirobosim_isaaclab import worker as worker_module

    sentinel = object()
    monkeypatch.setattr(worker_module, "IsaacLabWorkerRuntime", lambda config: sentinel)
    assert provider_module._default_runtime_factory(IsaacLabAdapterConfig(device="cpu")) is sentinel


def test_lightweight_modules_have_no_top_level_native_imports() -> None:
    package = Path(unirobosim_isaaclab.__file__).parent
    forbidden = {"isaaclab", "isaacsim", "omni", "pxr", "torch"}
    for name in ("__init__.py", "config.py", "descriptor.py", "probe.py", "provider.py", "world.py"):
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden
    assert not forbidden & set(sys.modules)
    importlib.reload(unirobosim_isaaclab)
    assert not forbidden & set(sys.modules)


def test_native_pure_conversions() -> None:
    assert _environment_origins(5, 2.0) == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (2.0, 2.0, 0.0),
    )
    assert _native_name(_path := __import__("unirobosim").EntityPath("/a/thing-x")).startswith("thing_x_")
    assert _native_name(_path) == _native_name(_path)
    rotated = _rotate_xyzw((1.0, 0.0, 0.0), (0.0, 0.0, 2**-0.5, 2**-0.5))
    assert rotated == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    faces = _surface_from_tetrahedra(((0, 1, 2, 3), (0, 2, 1, 4)))
    assert len(faces) == 6


def test_launcher_disables_process_terminating_fast_shutdown() -> None:
    config = IsaacLabAdapterConfig(headless=True, device="cuda:0", enable_cameras=True)
    assert _launcher_kwargs(config) == {
        "headless": True,
        "device": "cuda:0",
        "enable_cameras": True,
        "fast_shutdown": False,
    }

    configured = IsaacLabAdapterConfig(experience="/tmp/custom.kit")
    assert _launcher_kwargs(configured)["experience"] == "/tmp/custom.kit"
    assert _launcher_kwargs(config, process_isolated=True)["fast_shutdown"] is True

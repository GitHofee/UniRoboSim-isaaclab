from __future__ import annotations

import ast
import importlib
import importlib.metadata
import pickle
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
    _camera_launcher_settings,
    _environment_origins,
    _launcher_kwargs,
    _native_name,
    _render_interval_steps,
    _render_step_enabled,
    _rotate_xyzw,
    _surface_from_tetrahedra,
)
from unirobosim_isaaclab.probe import probe_environment

from .helpers import available_probe


def test_public_identity_and_protocol() -> None:
    provider = unirobosim_isaaclab.create_provider(IsaacLabAdapterConfig(device="cpu"))
    assert isinstance(provider, Provider)
    assert provider.descriptor is DESCRIPTOR
    assert unirobosim_isaaclab.__version__ == "0.10.17"
    assert DESCRIPTOR.version == unirobosim_isaaclab.__version__
    assert DESCRIPTOR.provider_id == "nvidia.isaaclab"
    assert DESCRIPTOR.contract_version == "v0alpha6"
    assert [profile["id"] for profile in DESCRIPTOR.metadata["runtime_profiles"]] == [
        "source-isaaclab-3.0.0-beta2",
        "ngc-isaaclab-3.0.0",
    ]
    assert CAPABILITIES.get(CapabilityId("state.rigid_body@1")) is not None
    render_state = CAPABILITIES.get(CapabilityId("render.state.apply@1"))
    assert render_state is not None
    assert render_state.properties["physics_advance"] is False
    assert "packed-float32-le" in render_state.properties["fluid_payloads"]
    assert CAPABILITIES.get(CapabilityId("control.rigid_body.wrench@1")) is not None
    assert CAPABILITIES.get(CapabilityId("contact.net_normal_force@1")) is not None
    assert CAPABILITIES.get(CapabilityId("state.fluid.particles@1")) is not None
    assert CAPABILITIES.get(CapabilityId("control.fluid.particles@1")) is not None
    fluid_state = CAPABILITIES.get(CapabilityId("state.fluid.particles@1"))
    assert fluid_state is not None
    assert any("raw USD/Omni Physics" in limitation for limitation in fluid_state.limitations)
    assert any("particle reaction loads" in limitation for limitation in fluid_state.limitations)
    assert CAPABILITIES.get(CapabilityId("debug.sink.native_overlay@1")) is None
    assert CAPABILITIES.get(CapabilityId("scene.snapshot@1")) is not None
    assert CAPABILITIES.get(CapabilityId("scene.delta@1")) is not None
    pose_command = CAPABILITIES.get(CapabilityId("scene.command.pose@1"))
    assert pose_command is not None
    drag_command = CAPABILITIES.get(CapabilityId("scene.command.drag@1"))
    assert drag_command is not None
    assert drag_command.properties == FrozenMap({"entity_kinds": ["rigid_body"], "modes": ["kinematic"]})
    assert any("constraint drag" in limitation for limitation in drag_command.limitations)
    assert CAPABILITIES.get(CapabilityId("render.browser-scene@1")) is not None
    asset_formats = CAPABILITIES.get(CapabilityId("asset.formats@1"))
    assert asset_formats is not None
    assert asset_formats.properties["static_scene"] == ("model/vnd.usd",)
    assert asset_formats.properties["composite_scene"] == ("model/vnd.usd",)
    assert CAPABILITIES.get(CapabilityId("scene.static@1")) is not None
    assert CAPABILITIES.get(CapabilityId("scene.composite@1")) is not None
    assert CAPABILITIES.get(CapabilityId("entity.embedded-binding@1")) is not None
    assert CAPABILITIES.get(CapabilityId("entity.scale.rigid@1")) is not None
    assert CAPABILITIES.get(CapabilityId("entity.scale.articulation.uniform@1")) is not None
    assert CAPABILITIES.get(CapabilityId("entity.scale.static_scene@1")) is not None
    assert CAPABILITIES.get(CapabilityId("entity.scale.composite_scene@1")) is not None
    planning = CAPABILITIES.get(CapabilityId("planning.scene@2"))
    assert planning is not None
    assert planning.properties["collision_authority"] == "composed-usd-and-physx-effective"
    assert planning.properties["geometry_read_limit_bytes"] == 64 * 1024 * 1024
    assert planning.properties["representation_fallback"] is False
    assert planning.properties["single_representation_per_geometry"] is True
    normalization = CAPABILITIES.get(CapabilityId("asset.normalization@1"))
    assert normalization is not None
    assert normalization.properties["rigid_body"] == FrozenMap(
        {"media_type": "model/vnd.usd", "profile": "isaaclab.dynamic-rigid-usd@1"}
    )
    assert CAPABILITIES.get(CapabilityId("sensor.camera.rgb@1")) is None
    camera_provider = unirobosim_isaaclab.create_provider(
        IsaacLabAdapterConfig(device="cpu", enable_cameras=True, render=True)
    )
    assert camera_provider.descriptor is not DESCRIPTOR
    assert camera_provider.descriptor.capabilities.get(CapabilityId("sensor.camera.rgb@1")) is not None
    assert camera_provider.descriptor.capabilities.get(CapabilityId("sensor.camera.normals@1")) is not None
    native_debug = camera_provider.descriptor.capabilities.get(CapabilityId("debug.sink.native_overlay@1"))
    assert native_debug is not None
    assert native_debug.properties["offscreen"] is True
    assert any("headless-physics" in limitation for limitation in native_debug.limitations)
    assert camera_provider.descriptor.capabilities.get(CapabilityId("scene.command.pose@1")) is pose_command
    assert camera_provider.descriptor.capabilities.get(CapabilityId("scene.command.drag@1")) is drag_command
    camera_profile = camera_provider.descriptor.capabilities.get(CapabilityId("sensor.camera@1"))
    assert camera_profile is not None
    assert camera_profile.properties["mount_parent_kinds"] == ("articulation",)
    assert camera_provider.descriptor.metadata["camera_anti_aliasing"] == "fxaa"
    assert camera_provider.descriptor.metadata["camera_texture_streaming"] is False
    assert camera_provider.descriptor.metadata["render_on_step"] is True
    assert camera_provider.descriptor.metadata["max_render_hz"] is None
    assert camera_provider.descriptor.metadata["fluid_render_mode"] == "particles"
    physics_only = unirobosim_isaaclab.create_easy_provider(launch_profile="headless-physics")
    assert physics_only.descriptor.capabilities.get(CapabilityId("debug.sink.native_overlay@1")) is None
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
    assert config.anti_aliasing == "fxaa"


def test_fluid_surface_color_is_normalized_and_bounded() -> None:
    assert IsaacLabAdapterConfig(fluid_surface_color_rgb=(1, 0.5, 0)).fluid_surface_color_rgb == (1.0, 0.5, 0.0)
    with pytest.raises(ValidationError, match="fluid_surface_color_rgb"):
        IsaacLabAdapterConfig(fluid_surface_color_rgb=(1.1, 0.5, 0.0))


def test_default_position_gains_preserve_authored_asset_values() -> None:
    config = IsaacLabAdapterConfig()
    assert config.position_stiffness is None
    assert config.position_damping is None


def test_default_worker_startup_budget_covers_cold_kit_without_unbounded_wait() -> None:
    config = IsaacLabAdapterConfig()
    assert config.worker_startup_hard_timeout_s == 120.0
    assert config.worker_kit_launch_idle_timeout_s == 90.0


def test_worker_startup_budget_is_normalized_and_serializable() -> None:
    config = IsaacLabAdapterConfig(
        worker_startup_hard_timeout_s=300,
        worker_kit_launch_idle_timeout_s=45,
    )
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    assert restored.worker_startup_hard_timeout_s == 300.0
    assert restored.worker_kit_launch_idle_timeout_s == 45.0


@pytest.mark.parametrize("field", ["worker_startup_hard_timeout_s", "worker_kit_launch_idle_timeout_s"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "120", "bad", 301])
def test_invalid_worker_startup_budget(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(**{field: value})  # type: ignore[arg-type]


def test_kit_launch_idle_budget_cannot_exceed_worker_hard_budget() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        IsaacLabAdapterConfig(
            worker_startup_hard_timeout_s=20,
            worker_kit_launch_idle_timeout_s=21,
        )


@pytest.mark.parametrize("device", ["", "gpu", "cuda:-1", "CUDA:0", 7])
def test_invalid_device(device: object) -> None:
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(device=device)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", [0, -1, float("inf"), float("nan"), True, "bad"])
def test_invalid_spacing(spacing: object) -> None:
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(environment_spacing_m=spacing)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["headless", "enable_cameras", "render", "texture_streaming", "render_on_step"])
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
    assert IsaacLabAdapterConfig(max_render_hz=60).max_render_hz == 60.0
    invalid_render_rates: tuple[object, ...] = (
        0,
        -1,
        float("inf"),
        float("nan"),
        True,
        "bad",
    )
    for invalid in invalid_render_rates:
        with pytest.raises(ValidationError):
            IsaacLabAdapterConfig(max_render_hz=invalid)  # type: ignore[arg-type]
    assert IsaacLabAdapterConfig(anti_aliasing="FXAA").anti_aliasing == "fxaa"
    for invalid in ("", "msaa", 2):
        with pytest.raises(ValidationError):
            IsaacLabAdapterConfig(anti_aliasing=invalid)  # type: ignore[arg-type]
    assert IsaacLabAdapterConfig(fluid_render_mode="ISOSURFACE").fluid_render_mode == "isosurface"
    for invalid in ("", "mesh", 2):
        with pytest.raises(ValidationError):
            IsaacLabAdapterConfig(fluid_render_mode=invalid)  # type: ignore[arg-type]


def test_render_rate_cap_maps_to_native_step_interval_without_overspeed() -> None:
    assert _render_interval_steps(1.0 / 240.0, None) == 1
    assert _render_interval_steps(1.0 / 240.0, 240.0) == 1
    assert _render_interval_steps(1.0 / 240.0, 60.0) == 4
    assert _render_interval_steps(1.0 / 240.0, 59.0) == 5
    assert _render_interval_steps(1.0 / 30.0, 60.0) == 1


def test_native_step_render_schedule_preserves_full_physics_cadence() -> None:
    config = IsaacLabAdapterConfig(
        headless=False,
        enable_cameras=True,
        render=True,
        max_render_hz=60.0,
    )
    assert [_render_step_enabled(config, native_step_index, 4) for native_step_index in range(1, 9)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert not _render_step_enabled(
        IsaacLabAdapterConfig(headless=True, enable_cameras=False, render=False),
        4,
        4,
    )


def test_probe_cpu_success_and_version_failures() -> None:
    versions: dict[str, str | None] = {
        "isaaclab": "6.1.17",
        "isaaclab_physx": "1.1.3",
        "isaacsim": "6.0.1.0",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "torchaudio": "2.11.0",
    }
    report = probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get)
    assert report.available
    assert report.reason is None
    assert report.details["versions"] == FrozenMap(versions)

    versions["isaacsim"] = "5.1.0.0"
    versions["isaaclab_physx"] = None
    versions["torch"] = "2.10.0"
    versions["torchvision"] = "0.25.0"
    versions["torchaudio"] = "2.10.0"
    report = probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get)
    assert not report.available
    assert "isaacsim==6.0.1.0" in (report.reason or "")
    assert "isaaclab_physx is not installed" in (report.reason or "")
    assert "torch==2.11.0" in (report.reason or "")
    assert "torchvision==0.26.0" in (report.reason or "")
    assert "torchaudio==2.11.0" in (report.reason or "")

    versions.update(
        {
            "isaacsim": "6.0.1.0",
            "isaaclab_physx": "1.1.3",
            "torch": "2.11.0+cu128",
            "torchvision": "0.26.0+cu128",
            "torchaudio": "2.11.0+cu128",
        }
    )
    assert probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get).available


def test_recommended_startup_budgets_are_larger_only_for_verified_ngc_bundle() -> None:
    source_versions = dict(probe_module._EXPECTED)
    ngc_versions: dict[str, str | None] = {
        **source_versions,
        **probe_module._OFFICIAL_NGC_EXPECTED,
        "isaacsim": None,
        "torchaudio": None,
    }
    accepted_bundle = probe_module._OfficialBundleEvidence((), {"isaacsim_release": "6.0.1"})
    rejected_bundle = probe_module._OfficialBundleEvidence(("bundle mismatch",), {})

    assert probe_module.recommended_startup_budgets(version_reader=source_versions.get) == (120.0, 90.0)
    assert probe_module.recommended_startup_budgets(
        version_reader=ngc_versions.get,
        official_bundle_inspector=lambda: accepted_bundle,
    ) == (300.0, 300.0)
    assert probe_module.recommended_startup_budgets(
        version_reader=ngc_versions.get,
        official_bundle_inspector=lambda: rejected_bundle,
    ) == (120.0, 90.0)


def test_probe_accepts_only_validated_official_ngc_bundle_profile() -> None:
    versions: dict[str, str | None] = {
        "isaaclab": "6.1.11",
        "isaaclab_physx": "1.1.3",
        "isaacsim": None,
        "torch": "2.10.0+cu128",
        "torchvision": "0.25.0+cu128",
        "torchaudio": None,
    }
    calls = 0

    def inspect() -> probe_module._OfficialBundleEvidence:
        nonlocal calls
        calls += 1
        return probe_module._OfficialBundleEvidence(
            (),
            {
                "isaaclab_release": "3.0.0",
                "isaacsim_release": "6.0.1",
                "isaacsim_build": "6.0.1-alpha.17+develop.42429.af8ceaf7.gl",
            },
        )

    report = probe_environment(
        IsaacLabAdapterConfig(device="cpu"),
        DESCRIPTOR,
        version_reader=versions.get,
        official_bundle_inspector=inspect,
    )
    assert report.available
    assert report.reason is None
    assert calls == 1
    assert report.details["runtime_profile"] == "ngc-isaaclab-3.0.0"
    assert report.details["runtime_profile_evidence"]["isaacsim_release"] == "6.0.1"


def test_probe_rejects_ngc_fingerprint_when_bundle_capabilities_are_incomplete() -> None:
    versions: dict[str, str | None] = {
        "isaaclab": "6.1.11",
        "isaaclab_physx": "1.1.3",
        "isaacsim": None,
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "torchaudio": None,
    }
    report = probe_environment(
        IsaacLabAdapterConfig(device="cpu"),
        DESCRIPTOR,
        version_reader=versions.get,
        official_bundle_inspector=lambda: probe_module._OfficialBundleEvidence(
            ("official NGC bundle is missing required adapter API modules: sensors/camera/__init__.py",),
            {"missing_api_modules": ("sensors/camera/__init__.py",)},
        ),
    )
    assert not report.available
    assert report.details["runtime_profile"] is None
    assert "sensors/camera" in (report.reason or "")


def test_official_ngc_bundle_inspection_is_file_based_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isaaclab_root = tmp_path / "workspace" / "isaaclab"
    isaaclab_package = isaaclab_root / "source" / "isaaclab" / "isaaclab"
    physx_package = isaaclab_root / "source" / "isaaclab_physx" / "isaaclab_physx"
    isaacsim_root = tmp_path / "isaac-sim"
    isaacsim_package = isaacsim_root / "python_packages" / "isaacsim"
    (isaaclab_root / "VERSION").parent.mkdir(parents=True)
    (isaaclab_root / "VERSION").write_text("3.0.0\n", encoding="utf-8")
    for relative in (
        "app/__init__.py",
        "sim/__init__.py",
        "actuators/__init__.py",
        "assets/__init__.py",
        "assets/articulation/__init__.py",
        "assets/rigid_object/__init__.py",
        "sensors/camera/__init__.py",
        "sensors/contact_sensor/__init__.py",
        "sim/schemas/__init__.py",
    ):
        path = isaaclab_package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for relative in ("physics/__init__.py", "sim/schemas/__init__.py"):
        path = physx_package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (isaacsim_root / "docs" / "py").mkdir(parents=True)
    (isaacsim_root / "docs" / "py" / "VERSION").write_text("6.0.1\n", encoding="utf-8")
    (isaacsim_root / "VERSION").write_text("6.0.1-alpha.17+develop.42429.af8ceaf7.gl\n", encoding="utf-8")
    isaacsim_package.mkdir(parents=True)
    extension = isaacsim_root / "extscache" / "isaacsim.util.debug_draw-3.2.3"
    (extension / "bin").mkdir(parents=True)
    (extension / "isaacsim").mkdir()
    package_paths = {
        "isaaclab": isaaclab_package,
        "isaaclab_physx": physx_package,
        "isaacsim": isaacsim_package,
    }
    monkeypatch.setattr(probe_module, "_package_directory", package_paths.get)

    evidence = probe_module._inspect_official_ngc_bundle()

    assert evidence.issues == ()
    assert evidence.details["isaaclab_release"] == "3.0.0"
    assert evidence.details["isaacsim_release"] == "6.0.1"
    assert evidence.details["debug_draw_extension"] == str(extension)


def test_probe_rejects_other_python(monkeypatch: pytest.MonkeyPatch) -> None:
    class Version:
        major = 3
        minor = 11
        micro = 9

        def __getitem__(self, item: slice) -> tuple[int, int]:
            assert item == slice(None, 2)
            return (self.major, self.minor)

    versions = dict(probe_module._EXPECTED)
    monkeypatch.setattr(sys, "version_info", Version())
    report = probe_environment(IsaacLabAdapterConfig(device="cpu"), DESCRIPTOR, version_reader=versions.get)
    assert not report.available
    assert "Python 3.12 is required" in (report.reason or "")


def test_probe_cuda_success_and_command_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = dict(probe_module._EXPECTED)

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
    forbidden = {"isaaclab", "isaacsim", "omni", "pxr", "torch", "torchvision", "torchaudio"}
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
        "anti_aliasing": 2,
        "renderer": "RaytracedLighting",
    }

    isosurface = IsaacLabAdapterConfig(enable_cameras=True, fluid_render_mode="isosurface")
    assert _launcher_kwargs(isosurface)["renderer"] == "RealTimePathTracing"

    configured = IsaacLabAdapterConfig(experience="/tmp/custom.kit")
    assert _launcher_kwargs(configured)["experience"] == "/tmp/custom.kit"
    assert _launcher_kwargs(config, process_isolated=True)["fast_shutdown"] is True
    visible = _launcher_kwargs(IsaacLabAdapterConfig(headless=False))
    assert visible["headless"] is False
    assert visible["visualizer"] == ["kit"]
    assert visible["visualizer_explicit"] is True


def test_camera_launcher_texture_residency_settings() -> None:
    fidelity = _camera_launcher_settings(IsaacLabAdapterConfig(enable_cameras=True))
    assert "--/rtx-transient/resourcemanager/enableTextureStreaming=false" in fidelity
    assert "--/rtx-transient/resourcemanager/texturestreaming/async=false" in fidelity
    assert "--/persistent/rtx/modes/rt/enabled=true" in fidelity

    streaming = _camera_launcher_settings(IsaacLabAdapterConfig(enable_cameras=True, texture_streaming=True))
    assert "--/rtx-transient/resourcemanager/enableTextureStreaming=true" in streaming
    assert "--/rtx-transient/resourcemanager/texturestreaming/async=false" not in streaming

    isosurface = _camera_launcher_settings(IsaacLabAdapterConfig(enable_cameras=True, fluid_render_mode="isosurface"))
    assert "--/persistent/rtx/modes/rt/enabled=true" not in isosurface

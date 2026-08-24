from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from unirobosim import (
    ARTICULATION_AXIS_UNITS_MISMATCH,
    PHYSICAL_WORLD_SCHEMA_VERSION,
    WORLD_SCHEMA_VERSION,
    ArrayValue,
    ArticulationCommand,
    BoxGeometrySpec,
    BuildInput,
    BuildResourceEntry,
    BuildResourceManifest,
    BuildSourceEntry,
    CameraModality,
    CameraSpec,
    CapabilityId,
    CommandError,
    CommandMode,
    EntityKind,
    EntityPath,
    EntitySpec,
    LocalSourceIdentity,
    PhysicsSpec,
    ValidationError,
    WorldSpec,
)

import unirobosim_isaaclab.droid_acceptance as droid_acceptance
from unirobosim_isaaclab.descriptor import DESCRIPTOR
from unirobosim_isaaclab.droid_acceptance import (
    DroidAcceptanceBackendRun,
    _acceptance_world_spec,
    _DroidTelemetryProbe,
    _effective_camera_calibration,
    _look_at_xyzw,
    create_backend_run,
)
from unirobosim_isaaclab.native import _declared_joint_map
from unirobosim_isaaclab.native_protocols import NativeCameraCalibration

from .helpers import FakeNativeRuntime
from .test_lifecycle_world import open_test_session


def _build_input(asset: Path) -> BuildInput:
    payload = asset.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    stat = asset.stat()
    entry = BuildResourceEntry(
        entity_id="droid",
        component_id="robot.droid",
        resource_id="resource.droid",
        role="simulation",
        media_type="model/vnd.usd",
        requested_uri=str(asset),
        resolved_uri=str(asset),
        canonical_source_identity=f"sha256:{digest}",
        byte_size=len(payload),
        sha256=digest,
        selected_simulation_input=True,
        purposes=("collision", "planning", "simulation", "visual"),
        relative_bundle_path="droid/droid.usd",
    )
    source = BuildSourceEntry(
        resource_id=entry.resource_id,
        source_kind="local-file",
        source_root=str(asset.parent),
        relative_source_path=asset.name,
        expected_identity=LocalSourceIdentity(
            stat.st_dev,
            stat.st_ino,
            stat.st_mode,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ),
        expected_sha256=digest,
    )
    return BuildInput(manifest=BuildResourceManifest((entry,)), sources=(source,))


def _physical_world(asset: Path, build_input: BuildInput) -> WorldSpec:
    return WorldSpec(
        "droid-physical",
        (
            EntitySpec(
                EntityPath("/droid"),
                EntityKind.ARTICULATION,
                joint_names=("revolute", "prismatic"),
                initial_joint_positions=(0.0, 0.1),
                joint_position_units=("rad", "m"),
                asset_uri=str(asset),
            ),
        ),
        schema_version=PHYSICAL_WORLD_SCHEMA_VERSION,
        build_resource_manifest_sha256=build_input.manifest.sha256,
    )


def test_descriptor_and_session_expose_the_core_090_contract() -> None:
    assert DESCRIPTOR.supported_world_schema_versions == (WORLD_SCHEMA_VERSION, PHYSICAL_WORLD_SCHEMA_VERSION)
    assert DESCRIPTOR.capabilities.get(CapabilityId("state.articulation.axis-units@1")) is not None
    assert DESCRIPTOR.capabilities.get(CapabilityId("control.articulation.position.axis-units@1")) is not None
    signature = inspect.signature(open_test_session(FakeNativeRuntime())[1].build)
    assert signature.parameters["build_input"].kind is inspect.Parameter.KEYWORD_ONLY


def test_v5_build_input_and_articulation_state_are_self_describing(tmp_path: Path) -> None:
    asset = tmp_path / "droid.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    build_input = _build_input(asset)
    spec = _physical_world(asset, build_input)
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    with pytest.raises(ValidationError):
        session.build(spec)
    world = session.build(spec, build_input=build_input)
    state = world.read_articulation(world.resolve(EntityPath("/droid")))
    assert state.entity_id == "/droid"
    assert state.generation == world.generation
    assert state.joint_names == ("revolute", "prismatic")
    assert state.joint_position_units == ("rad", "m")
    assert state.joint_velocity_units == ("rad/s", "m/s")


def test_tick_batches_disjoint_resource_groups_before_one_native_step(tmp_path: Path) -> None:
    asset = tmp_path / "droid.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    build_input = _build_input(asset)
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    world = session.build(_physical_world(asset, build_input), build_input=build_input)
    handle = world.resolve(EntityPath("/droid"))
    world.apply_articulation_command(
        ArticulationCommand(
            handle,
            CommandMode.POSITION,
            ArrayValue.from_rows(((0.25,),)),
            degree_of_freedom_indices=(0,),
            target_units=("rad",),
        )
    )
    world.apply_articulation_command(
        ArticulationCommand(
            handle,
            CommandMode.POSITION,
            ArrayValue.from_rows(((0.04,),)),
            degree_of_freedom_indices=(1,),
            target_units=("m",),
        )
    )
    assert runtime.worlds[0].calls == []
    world.step()
    assert [call[0] for call in runtime.worlds[0].calls] == [
        "articulation_batch",
        "articulation",
        "articulation",
        "step",
    ]


def test_failed_second_command_discards_the_uncommitted_tick(tmp_path: Path) -> None:
    asset = tmp_path / "droid.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    build_input = _build_input(asset)
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    world = session.build(_physical_world(asset, build_input), build_input=build_input)
    handle = world.resolve(EntityPath("/droid"))
    world.apply_articulation_command(
        ArticulationCommand(
            handle,
            CommandMode.POSITION,
            ArrayValue.from_rows(((0.25,),)),
            degree_of_freedom_indices=(0,),
            target_units=("rad",),
        )
    )
    with pytest.raises(CommandError) as caught:
        world.apply_articulation_command(
            ArticulationCommand(
                handle,
                CommandMode.POSITION,
                ArrayValue.from_rows(((0.04,),)),
                degree_of_freedom_indices=(1,),
                target_units=("rad",),
            )
        )
    assert caught.value.details["detail_code"] == ARTICULATION_AXIS_UNITS_MISMATCH
    world.step()
    batch = runtime.worlds[0].calls[0]
    assert batch[0] == "articulation_batch" and batch[1] == ()
    assert [call[0] for call in runtime.worlds[0].calls] == ["articulation_batch", "step"]


def test_declared_joint_map_accepts_arm_and_gripper_subsets_without_robot_assumptions() -> None:
    native = ("arm_1", "arm_2", "finger_left", "finger_right")
    assert _declared_joint_map(EntityPath("/generic"), native, ("arm_1", "arm_2")) == (0, 1)
    assert _declared_joint_map(EntityPath("/generic"), native, ("finger_left", "finger_right")) == (2, 3)
    with pytest.raises(ValueError, match="not a subset"):
        _declared_joint_map(EntityPath("/generic"), native, ("missing",))


def test_droid_acceptance_world_pins_cross_backend_zero_gravity() -> None:
    base = WorldSpec(
        "droid-acceptance-gravity",
        (
            EntitySpec(
                EntityPath("/marker"),
                EntityKind.RIGID_BODY,
                box=BoxGeometrySpec((0.1, 0.1, 0.1), 0.1),
            ),
        ),
        physics=PhysicsSpec(time_step_seconds=1.0 / 240.0, gravity_m_s2=(0.0, 0.0, -9.81)),
    )
    effective = _acceptance_world_spec(
        base,
        entities=base.entities,
        requirements=(),
        gravity_m_s2=(0.0, 0.0, 0.0),
    )
    assert base.physics.gravity_m_s2 == (0.0, 0.0, -9.81)
    assert effective.physics.gravity_m_s2 == (0.0, 0.0, 0.0)
    assert effective.physics.time_step_seconds == 1.0 / 240.0


def test_droid_acceptance_entry_point_closes_metadata_and_factory() -> None:
    provider = object()
    entry_point = droid_acceptance._entry_point(provider)
    assert entry_point.name == "isaaclab"
    assert entry_point.group == "unirobosim.backends"
    assert entry_point.value == "unirobosim_isaaclab:create_easy_provider"
    assert (entry_point.dist.name, entry_point.dist.version) == ("unirobosim-isaaclab", "0.9.5")
    factory = entry_point.load()
    assert callable(factory)
    assert factory() is provider


def test_droid_acceptance_entry_point_passes_fastsim_adapter_discovery() -> None:
    adapter_module = pytest.importorskip(
        "fastsim.integrations.unirobosim.adapter",
        reason="FastSim is an acceptance-only integration dependency",
        exc_type=ImportError,
    )
    aliases_module = pytest.importorskip("fastsim.integrations.unirobosim.aliases", exc_type=ImportError)
    projection_module = pytest.importorskip("fastsim.integrations.unirobosim.projection", exc_type=ImportError)
    provider = object()
    entry_point = droid_acceptance._entry_point(provider)
    world = WorldSpec(
        "entry-point-discovery",
        (
            EntitySpec(
                EntityPath("/fixture"),
                EntityKind.RIGID_BODY,
                box=BoxGeometrySpec((0.1, 0.1, 0.1), 0.1),
            ),
        ),
    )
    projection = projection_module.UniRoboSimProjection(
        plan_digest="a" * 64,
        plan_content_digest="a" * 64,
        backend=aliases_module.backend_alias("isaaclab"),
        world_spec=world,
        build_input=None,
        entities=(),
        articulations=(),
        fluids=(),
        cameras=(),
        default_entity_id=None,
        physics_dt_seconds=1.0 / 60.0,
        control_hz=60.0,
        rate_policy="exact",
        initial_generation_seed=1,
        planning_reads_demanded=False,
    )
    adapter = adapter_module._UniRoboSimAdapter(projection, entry_points=lambda: (entry_point,))
    assert adapter._select_entry_point() is entry_point
    assert adapter.diagnostics["selected_entry_point"] == {
        "name": "isaaclab",
        "group": "unirobosim.backends",
        "value": "unirobosim_isaaclab:create_easy_provider",
        "distribution": "unirobosim-isaaclab",
        "version": "0.9.5",
    }
    assert entry_point.load()() is provider


def test_droid_effective_camera_metadata_is_derived_from_native_calibration() -> None:
    eye = (2.2, -2.2, 1.6)
    look_at = (0.0, 0.0, 0.65)
    up = (0.0, 0.0, 1.0)
    focus_distance = math.sqrt(sum((look_at[index] - eye[index]) ** 2 for index in range(3)))
    focal_px = 1920.0 / (2.0 * math.tan(math.radians(60.0) / 2.0))
    native = NativeCameraCalibration(
        resolution_px=(1920, 1080),
        intrinsic_matrix=(focal_px, 0.0, 960.0, 0.0, focal_px, 540.0, 0.0, 0.0, 1.0),
        projection="perspective",
        focal_length=18.147302994931884,
        horizontal_aperture=20.955,
        clipping_range_m=(0.05, 20.0),
        position_m=eye,
        orientation_opengl_xyzw=_look_at_xyzw(eye, look_at, up),
    )
    effective = _effective_camera_calibration(native, focus_distance_m=focus_distance, up_reference=up)
    assert effective["schema_version"] == "unirobosim-effective-camera-calibration/1"
    assert effective["K_row_major"] == native.intrinsic_matrix
    assert effective["resolution_px"] == (1920, 1080)
    projection = effective["projection"]
    extrinsics = effective["extrinsics"]
    assert isinstance(projection, Mapping)
    assert isinstance(extrinsics, Mapping)
    assert projection["horizontal_fov_deg"] == pytest.approx(60.0)
    assert projection["vertical_fov_deg"] == pytest.approx(35.98339777135764)
    assert extrinsics["eye_m"] == pytest.approx(eye)
    assert extrinsics["look_at_m"] == pytest.approx(look_at)
    assert extrinsics["up"] == up


def test_droid_acceptance_visible_window_is_keyword_only_and_defaults_off() -> None:
    parameter = inspect.signature(create_backend_run).parameters["visible_window"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def test_droid_scene_camera_cache_never_crosses_runtime_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    class Kernel:
        snapshot = SimpleNamespace(generation=1, tick=0)

    kernel = Kernel()
    probe = _DroidTelemetryProbe(object(), kernel, 8, 1.0, (0.0, 0.0, 1.0), 1280, 720)
    submissions: list[tuple[int, int]] = []

    def submit(generation: int, tick: int) -> Mapping[str, object]:
        submissions.append((generation, tick))
        calibration: Mapping[str, object] = MappingProxyType({"generation": generation})
        with probe._lock:
            probe._camera_calibration = (generation, calibration)
        return MappingProxyType({"generation": generation, "tick": tick})

    monkeypatch.setattr(probe, "_submit", submit)
    assert probe.sample(0)["generation"] == 1
    assert probe.camera_calibration()["generation"] == 1

    kernel.snapshot = SimpleNamespace(generation=2, tick=0)
    assert probe.sample(0)["generation"] == 2
    assert probe.camera_calibration()["generation"] == 2
    assert submissions == [(1, 0), (2, 0)]


def test_droid_stale_authority_read_does_not_fail_runtime() -> None:
    runtime = pytest.importorskip("fastsim.runtime", exc_type=ImportError)

    class Backend:
        _local_tick = 0

        def prepare(self, _plan: object, *, seed: int) -> object:
            return runtime.BackendTick(0, 0.0)

        def reset(self, *, seed: int) -> object:
            self._local_tick = 0
            return runtime.BackendTick(0, 0.0)

        def step(self) -> object:
            self._local_tick += 1
            return runtime.BackendTick(self._local_tick, self._local_tick / 240.0)

        def read_observation(self, _entity_id: str, _provider_key: str) -> object:
            raise LookupError

        def close(self) -> None:
            return None

    backend = Backend()
    kernel = runtime.RuntimeKernel(
        SimpleNamespace(runtime={"physics_hz": 240.0}),
        backend,
        run_id="droid-stale-telemetry-test",
        options=runtime.RuntimeOptions(step_pacing_seconds=0.0),
    )
    probe = _DroidTelemetryProbe(backend, kernel, 8, 1.0, (0.0, 0.0, 1.0), 1280, 720)
    kernel.prepare(timeout=2.0)
    try:
        with pytest.raises(RuntimeError, match="stale runtime state"):
            probe._submit(2, 0)
        assert kernel.state is runtime.RuntimeState.READY
        assert kernel.failure is None
    finally:
        kernel.close(timeout=2.0)


def test_droid_compose_keeps_exact_adapter_and_planning_gate() -> None:
    adapter_module = pytest.importorskip("fastsim.integrations.unirobosim.adapter", exc_type=ImportError)
    aliases_module = pytest.importorskip("fastsim.integrations.unirobosim.aliases", exc_type=ImportError)
    planning_module = pytest.importorskip("fastsim.integrations.unirobosim._planning_raw", exc_type=ImportError)
    projection_module = pytest.importorskip("fastsim.integrations.unirobosim.projection", exc_type=ImportError)
    world = WorldSpec(
        "droid-compose-exact-adapter",
        (
            EntitySpec(
                EntityPath("/camera"),
                EntityKind.CAMERA_SENSOR,
                camera=CameraSpec(
                    width_px=1280,
                    height_px=720,
                    modalities=(CameraModality.RGB,),
                ),
            ),
        ),
    )
    projection = projection_module.UniRoboSimProjection(
        plan_digest="a" * 64,
        plan_content_digest="a" * 64,
        backend=aliases_module.backend_alias("isaaclab"),
        world_spec=world,
        build_input=None,
        entities=(),
        articulations=(),
        fluids=(),
        cameras=(),
        default_entity_id=None,
        physics_dt_seconds=1.0 / 240.0,
        control_hz=240.0,
        rate_policy="exact",
        initial_generation_seed=1,
        planning_reads_demanded=False,
    )
    bundle, adapter = droid_acceptance._compose(
        SimpleNamespace(runtime={"physics_hz": 240.0}),
        projection,
        object(),
        8,
        1.0,
        (0.0, 0.0, 1.0),
    )
    try:
        assert type(adapter) is adapter_module._UniRoboSimAdapter
        assert planning_module._authority_adapter(adapter) is adapter
        assert type(adapter._droid_telemetry_probe) is _DroidTelemetryProbe
        assert bundle.planning_raw is not None
        assert bundle._UniRoboSimRuntimeBundle__executor._plan_digest == projection.plan_digest
    finally:
        bundle.close(timeout=2.0)


def test_droid_acceptance_default_window_evidence_does_not_query_x11(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        droid_acceptance,
        "_wait_for_native_window",
        lambda *_args, **_kwargs: pytest.fail("headless default must not query X11"),
    )
    run = DroidAcceptanceBackendRun(
        object(),
        object(),
        (0.0, 0.0, 0.0),
        False,
        None,
        frozenset(),
        tmp_path,
        "rulebased_blocking",
    )
    assert run.window_evidence == {
        "schema_version": "fastsim-visible-window-evidence/1",
        "requested": False,
        "headless": True,
        "observed": False,
        "display": "not requested",
        "native_window_id": "not requested",
        "window_title": "not requested",
        "source": "visible_window=False; xwininfo was not invoked",
    }


def test_droid_acceptance_records_viewable_window_and_clean_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    details = 'xwininfo: Window id: 0x123 "Isaac Sim"\n  Map State: IsViewable\n'
    monkeypatch.setattr(
        droid_acceptance,
        "_wait_for_native_window",
        lambda *_args, **_kwargs: ("0x123", "Isaac Sim", details),
    )
    monkeypatch.setattr(droid_acceptance, "_wait_for_window_close", lambda *_args, **_kwargs: True)
    run = DroidAcceptanceBackendRun(
        object(),
        object(),
        (0.0, 0.0, 0.0),
        True,
        ":1",
        frozenset({"0x100"}),
        tmp_path,
        "model_servo_preempt",
    )
    assert run.window_evidence == {
        "schema_version": "fastsim-visible-window-evidence/1",
        "requested": True,
        "headless": False,
        "observed": True,
        "display": ":1",
        "native_window_id": "0x123",
        "window_title": "Isaac Sim",
        "source": "xwininfo -display :1 -id 0x123: Map State: IsViewable",
    }
    assert "Map State: IsViewable" in (tmp_path / "droid-model_servo_preempt-isaaclab.window-xwininfo.txt").read_text(
        encoding="utf-8"
    )
    run.close()
    close_evidence = (tmp_path / "droid-model_servo_preempt-isaaclab.window-close.json").read_text(encoding="utf-8")
    assert '"observed_closed": true' in close_evidence

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
from unirobosim import (
    PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION,
    PLANNING_SYSTEM_ENTITY_PATH,
    ArrayValue,
    BoxGeometrySpec,
    CameraSpec,
    CapabilityId,
    CapabilityRequirement,
    DeformableBodySpec,
    DeformableTopology,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    FrozenMap,
    PlanningEntityKind,
    PlanningFrameKind,
    PlanningGeometryRepresentation,
    PlanningGeometryResourceRevokedError,
    PlanningSceneContractError,
    PlanningSceneDeltaContinuityError,
    PlanningSceneDeltaKind,
    PlanningSceneHashMismatchError,
    PlanningSceneIncompleteError,
    PlanningSceneRepresentationError,
    PlanningSceneWorld,
    SessionState,
    WorldSpec,
)
from unirobosim.testing import FakeProvider

from unirobosim_isaaclab.config import IsaacLabAdapterConfig
from unirobosim_isaaclab.descriptor import DESCRIPTOR, descriptor_for_config
from unirobosim_isaaclab.native_protocols import (
    NativePlanningCatalog,
    NativePlanningError,
    NativePlanningResource,
    NativePlanningState,
)
from unirobosim_isaaclab.planning_scene import IsaacLabPlanningWorld
from unirobosim_isaaclab.provider import IsaacLabSession
from unirobosim_isaaclab.world import IsaacLabWorld

from .helpers import FakeNativeWorld


def _planning_requirement() -> CapabilityRequirement:
    return CapabilityRequirement(CapabilityId("planning.scene@2"))


def _planning_descriptor():
    return DESCRIPTOR


@dataclass(frozen=True, slots=True)
class _PlanningFixture:
    spec: WorldSpec
    catalog: NativePlanningCatalog
    state: NativePlanningState
    resources: dict[str, NativePlanningResource]


def _make_planning_fixture(tmp_path: Path) -> _PlanningFixture:
    asset = tmp_path / "planning.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    asset_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
    spec = WorldSpec(
        "isaaclab-planning-test",
        (
            EntitySpec(
                EntityPath("/mesh"),
                EntityKind.RIGID_BODY,
                asset_uri=str(asset),
                metadata=FrozenMap({"fake_planning_collision_authority": "effective_native"}),
            ),
            EntitySpec(
                EntityPath("/robots/arm"),
                EntityKind.ARTICULATION,
                joint_names=("shoulder", "elbow"),
                initial_joint_positions=(0.1, -0.2),
                asset_uri=str(asset),
                metadata=FrozenMap(
                    {
                        "planning_entity_kind": "robot",
                        "planning_frame_declarations": {
                            "schema": PLANNING_FRAME_DECLARATIONS_SCHEMA_VERSION,
                            "component_sha256": asset_sha256,
                            "entries": (
                                {
                                    "name": "wrist_mount",
                                    "owner_link": "elbow child",
                                    "source": {"kind": "link", "name": "elbow child"},
                                },
                            ),
                        },
                    }
                ),
            ),
            EntitySpec(
                EntityPath("/fixtures/door"),
                EntityKind.ARTICULATION,
                joint_names=("hinge",),
                initial_joint_positions=(0.3,),
                asset_uri=str(asset),
            ),
            EntitySpec(EntityPath("/box"), EntityKind.RIGID_BODY, box=BoxGeometrySpec()),
        ),
        requirements=(_planning_requirement(),),
    )
    session = FakeProvider().open()
    world = cast(PlanningSceneWorld, session.build(spec))
    try:
        catalog = world.planning_scene_catalog()
        state = world.planning_scene_state()
        resources: dict[str, NativePlanningResource] = {}
        for geometry in catalog.geometries:
            if geometry.resource_id is None:
                continue
            lease = world.resolve_planning_geometry(geometry.geometry_id)
            try:
                content = lease.read()
            finally:
                lease.close()
            resources[geometry.geometry_id] = NativePlanningResource(
                geometry.geometry_id,
                geometry.representation,
                content,
                hashlib.sha256(content).hexdigest(),
            )
        return _PlanningFixture(
            spec,
            NativePlanningCatalog(
                catalog.entities,
                catalog.links,
                catalog.joints,
                catalog.frames,
                catalog.geometries,
            ),
            NativePlanningState(
                state.tick.step_index,
                state.entities,
                state.links,
                state.frames,
                state.articulations,
                state.geometry_transforms,
                state.attachments,
            ),
            resources,
        )
    finally:
        session.close()


class _FakeNativePlanningWorld(FakeNativeWorld):
    def __init__(self, fixture: _PlanningFixture) -> None:
        super().__init__(fixture.spec)
        self.fixture = fixture
        self.native_step_index = 0
        self.planning_calls: list[tuple[str, object]] = []
        self.catalog_error: BaseException | None = None
        self.state_error: BaseException | None = None
        self.resource_error: BaseException | None = None

    def planning_catalog(self, environment_index: int = 0) -> NativePlanningCatalog:
        self.planning_calls.append(("catalog", environment_index))
        if self.catalog_error is not None:
            raise self.catalog_error
        return self.fixture.catalog

    def planning_state(self, environment_index: int = 0) -> NativePlanningState:
        self.planning_calls.append(("state", environment_index))
        if self.state_error is not None:
            raise self.state_error
        return replace(self.fixture.state, step_index=self.native_step_index)

    def planning_resource(self, geometry_id: str, environment_index: int = 0) -> NativePlanningResource:
        self.planning_calls.append(("resource", (geometry_id, environment_index)))
        if self.resource_error is not None:
            raise self.resource_error
        try:
            return self.fixture.resources[geometry_id]
        except KeyError:
            raise NativePlanningError("resource_missing") from None

    def step(self, count: int) -> None:
        super().step(count)
        self.native_step_index += count


class _FakePlanningRuntime:
    def __init__(self, fixture: _PlanningFixture) -> None:
        self.fixture = fixture
        self.worlds: list[FakeNativeWorld] = []
        self.closed = False
        self.build_error: BaseException | None = None
        self.catalog_error: BaseException | None = None

    def build_world(self, spec: WorldSpec) -> FakeNativeWorld:
        if self.build_error is not None:
            error = self.build_error
            self.build_error = None
            raise error
        if any(item.capability == CapabilityId("planning.scene@2") for item in spec.requirements):
            world = _FakeNativePlanningWorld(self.fixture)
            world.catalog_error = self.catalog_error
        else:
            world = FakeNativeWorld(spec)
        self.worlds.append(world)
        return world

    def close(self) -> None:
        self.closed = True


def _open_session(runtime: _FakePlanningRuntime) -> IsaacLabSession:
    return IsaacLabSession(
        _planning_descriptor(),
        runtime,
        config=IsaacLabAdapterConfig(),
        on_close=lambda session: None,
    )


def test_planning_demand_selects_only_public_subtype_and_validates_graph(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    world = session.build(fixture.spec)
    try:
        assert type(world) is IsaacLabPlanningWorld
        assert isinstance(world, PlanningSceneWorld)
        catalog = world.planning_scene_catalog()
        state = world.planning_scene_state()
        state.validate_against(catalog)
        assert {item.path: item.kind for item in catalog.entities} == {
            PLANNING_SYSTEM_ENTITY_PATH: PlanningEntityKind.OTHER,
            "/box": PlanningEntityKind.RIGID_OBJECT,
            "/fixtures/door": PlanningEntityKind.ARTICULATION,
            "/mesh": PlanningEntityKind.RIGID_OBJECT,
            "/robots/arm": PlanningEntityKind.ROBOT,
        }
        robot = next(item for item in catalog.entities if item.path == "/robots/arm")
        robot_state = next(item for item in state.articulations if item.entity_id == robot.entity_id)
        assert robot_state.joint_ids == robot.joint_ids
        assert len(robot_state.positions) == len(robot_state.velocities) == len(robot_state.position_units) == 2
        named = next(
            item
            for item in catalog.frames
            if item.owner_entity_id == robot.entity_id and item.kind is PlanningFrameKind.NAMED
        )
        assert named.name == "wrist_mount"
        assert named.owner_link_id is not None
        assert named.parent_frame_id == next(
            item.frame_id for item in catalog.links if item.link_id == named.owner_link_id
        )
    finally:
        session.close()


def test_no_demand_keeps_exact_base_world_and_never_calls_planning(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    spec = replace(fixture.spec, world_id="no-planning-demand", requirements=())
    world = session.build(spec)
    try:
        assert type(world) is IsaacLabWorld
        assert not isinstance(world, PlanningSceneWorld)
        assert "planning_scene_catalog" not in dir(world)
        assert not any("planning" in name for name in world.__dict__)
        world.step(2)
        world.reset()
        native = runtime.worlds[0]
        assert type(native) is FakeNativeWorld
        assert not hasattr(native, "planning_calls")
    finally:
        session.close()


def test_supported_optional_requirement_is_an_explicit_planning_demand(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    optional = replace(
        fixture.spec,
        world_id="optional-planning-demand",
        requirements=(CapabilityRequirement(CapabilityId("planning.scene@2"), required=False),),
    )
    world = session.build(optional)
    try:
        assert type(world) is IsaacLabPlanningWorld
        assert isinstance(world, PlanningSceneWorld)
        assert world.planning_scene_catalog().world_id == optional.world_id
    finally:
        session.close()


def test_nonphysical_camera_can_coexist_without_entering_planning_catalog(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    camera_path = EntityPath("/sensors/overview")
    spec = replace(
        fixture.spec,
        world_id="planning-with-camera",
        entities=fixture.spec.entities + (EntitySpec(camera_path, EntityKind.CAMERA_SENSOR, camera=CameraSpec()),),
    )
    runtime = _FakePlanningRuntime(fixture)
    config = IsaacLabAdapterConfig(enable_cameras=True, render=True)
    session = IsaacLabSession(
        descriptor_for_config(config),
        runtime,
        config=config,
        on_close=lambda active: None,
    )
    world = cast(PlanningSceneWorld, session.build(spec))
    try:
        catalog = world.planning_scene_catalog()
        assert camera_path.value not in {entity.path for entity in catalog.entities}
        world.planning_scene_state().validate_against(catalog)
    finally:
        session.close()


def test_multiple_environments_are_independent_and_partial_reset_revokes_only_selected_leases(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    fixture = replace(
        fixture,
        spec=replace(fixture.spec, world_id="planning-multiple-environments", environments=EnvironmentSpec(2)),
    )
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    world = cast(PlanningSceneWorld, session.build(fixture.spec))
    first_catalog = world.planning_scene_catalog(0)
    second_catalog = world.planning_scene_catalog(1)
    assert first_catalog.environment_index == 0
    assert second_catalog.environment_index == 1
    geometry_id = next(item.geometry_id for item in first_catalog.geometries if item.resource_id is not None)
    first_lease = world.resolve_planning_geometry(geometry_id, environment_index=0)
    second_lease = world.resolve_planning_geometry(geometry_id, environment_index=1)
    try:
        cast(IsaacLabWorld, world).reset((1,))
        assert first_lease.read(0, 1)
        with pytest.raises(PlanningGeometryResourceRevokedError):
            second_lease.read(0, 1)
        assert world.planning_scene_catalog(0).generation == first_catalog.generation
        assert world.planning_scene_catalog(1).generation == second_catalog.generation + 1
    finally:
        first_lease.close()
        second_lease.close()
        session.close()


def test_state_delta_continuity_and_reset_resync(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    world = cast(PlanningSceneWorld, session.build(fixture.spec))
    try:
        initial = world.planning_scene_state()
        assert initial.sequence == 1
        with pytest.raises(PlanningSceneDeltaContinuityError):
            world.planning_scene_delta(initial.sequence)
        cast(IsaacLabWorld, world).step(3)
        current = world.planning_scene_state()
        delta = world.planning_scene_delta(initial.sequence)
        assert delta.kind is PlanningSceneDeltaKind.STATE
        assert delta.state == current
        assert delta.sequence == current.sequence == 2
        assert delta.transform_revision == initial.transform_revision + 1
        assert delta.catalog_content_sha256 == current.catalog_content_sha256
        cast(IsaacLabWorld, world).reset()
        reset_state = world.planning_scene_state()
        assert reset_state.generation == initial.generation + 1
        resync = world.planning_scene_delta(reset_state.sequence)
        assert resync.kind is PlanningSceneDeltaKind.RESYNC
        assert resync.resync_required and resync.state is None and resync.catalog is None
    finally:
        session.close()


def test_geometry_is_lazy_hash_verified_cached_and_revoked(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    world = cast(PlanningSceneWorld, session.build(fixture.spec))
    native = cast(_FakeNativePlanningWorld, runtime.worlds[0])
    catalog = world.planning_scene_catalog()
    geometry = next(item for item in catalog.geometries if item.resource_id is not None)
    assert not any(call[0] == "resource" for call in native.planning_calls)
    first = world.resolve_planning_geometry(geometry.geometry_id)
    second = world.resolve_planning_geometry(geometry.geometry_id, geometry.representation)
    initial_generation = first.descriptor.generation
    try:
        assert sum(call[0] == "resource" for call in native.planning_calls) == 1
        assert first.descriptor.resolution_key == second.descriptor.resolution_key
        assert first.descriptor.resource_layout == geometry.resource_layout
        assert first.descriptor.sha256 == hashlib.sha256(first.read()).hexdigest()
        assert first.read(3, 11) == first.read()[3:14]
        first.descriptor.validate_against(catalog)
        with pytest.raises(PlanningSceneRepresentationError):
            world.resolve_planning_geometry(geometry.geometry_id, PlanningGeometryRepresentation.CONVEX_MESH)
        cast(IsaacLabWorld, world).reset()
        with pytest.raises(PlanningGeometryResourceRevokedError):
            first.read()
        assert first.closed and second.closed
        replacement = world.resolve_planning_geometry(geometry.geometry_id)
        try:
            assert replacement.descriptor.generation == initial_generation + 1
            assert sum(call[0] == "resource" for call in native.planning_calls) == 2
        finally:
            replacement.close()
    finally:
        session.close()


def test_close_revokes_live_geometry_lease(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    session = _open_session(_FakePlanningRuntime(fixture))
    world = cast(PlanningSceneWorld, session.build(fixture.spec))
    geometry = next(item for item in world.planning_scene_catalog().geometries if item.resource_id is not None)
    lease = world.resolve_planning_geometry(geometry.geometry_id)
    cast(IsaacLabWorld, world).close()
    assert lease.closed
    with pytest.raises(PlanningGeometryResourceRevokedError):
        lease.read()
    session.close()


def test_hash_mismatch_and_native_errors_are_typed_and_scrubbed(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    world = cast(PlanningSceneWorld, session.build(fixture.spec))
    native = cast(_FakeNativePlanningWorld, runtime.worlds[0])
    geometry = next(item for item in world.planning_scene_catalog().geometries if item.resource_id is not None)
    native.resource_error = RuntimeError("private native resource detail")
    with pytest.raises(PlanningSceneContractError) as caught:
        world.resolve_planning_geometry(geometry.geometry_id)
    assert caught.value.__cause__ is None
    assert "private native resource detail" not in str(caught.value)
    native.resource_error = None
    original = native.fixture.resources[geometry.geometry_id]
    native.fixture.resources[geometry.geometry_id] = replace(original, content=b"corrupt")
    with pytest.raises(PlanningSceneHashMismatchError):
        world.resolve_planning_geometry(geometry.geometry_id)

    class AdversarialScalar:
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("private scalar detail")

    native.fixture.resources[geometry.geometry_id] = replace(
        original,
        sha256=cast(str, AdversarialScalar()),
    )
    with pytest.raises(PlanningSceneHashMismatchError) as caught:
        world.resolve_planning_geometry(geometry.geometry_id)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "private scalar detail" not in str(caught.value)
    session.close()


def test_admission_failure_is_transactional_scrubbed_and_retryable(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    runtime.build_error = RuntimeError("private build detail")
    session = _open_session(runtime)
    with pytest.raises(PlanningSceneIncompleteError) as caught:
        session.build(fixture.spec)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.operation == "planning_scene.preflight"
    assert "private build detail" not in str(caught.value)
    assert session.state is SessionState.OPEN
    world = session.build(fixture.spec)
    assert world.generation == 1
    world.close()

    runtime.catalog_error = NativePlanningError("frame_missing")
    with pytest.raises(PlanningSceneIncompleteError) as caught:
        session.build(fixture.spec)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "frame_missing" in str(caught.value)
    assert runtime.worlds[-1].closed
    assert session.state is SessionState.OPEN
    session.close()


def test_catalog_resource_exceeding_advertised_limit_fails_admission(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    resource_geometry = next(item for item in fixture.catalog.geometries if item.resource_layout is not None)
    assert resource_geometry.resource_layout is not None
    oversized_layout = replace(resource_geometry.resource_layout, vertex_shape=(6_000_000, 3))
    assert oversized_layout.decoded_byte_size > 64 * 1024 * 1024
    oversized_geometry = replace(resource_geometry, resource_layout=oversized_layout)
    fixture = replace(
        fixture,
        catalog=replace(
            fixture.catalog,
            geometries=tuple(
                oversized_geometry if item.geometry_id == oversized_geometry.geometry_id else item
                for item in fixture.catalog.geometries
            ),
        ),
    )
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    with pytest.raises(PlanningSceneIncompleteError) as caught:
        session.build(fixture.spec)
    assert caught.value.operation == "planning_scene.preflight"
    assert runtime.worlds[-1].closed
    assert session.state is SessionState.OPEN
    session.close()


def test_soft_matter_fails_before_native_build(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    runtime = _FakePlanningRuntime(fixture)
    session = _open_session(runtime)
    soft_spec = WorldSpec(
        "planning-soft-rejected",
        (
            EntitySpec(EntityPath("/box"), EntityKind.RIGID_BODY, box=BoxGeometrySpec()),
            EntitySpec(
                EntityPath("/cloth"),
                EntityKind.SURFACE_DEFORMABLE,
                deformable=DeformableBodySpec(
                    DeformableTopology.SURFACE,
                    ArrayValue.from_nested(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
                    surface_triangles=ArrayValue.from_nested(((0, 1, 2),), dtype="int64"),
                ),
            ),
        ),
        requirements=(_planning_requirement(),),
    )
    with pytest.raises(PlanningSceneIncompleteError) as caught:
        session.build(soft_spec)
    assert caught.value.entity_path == "/cloth"
    assert runtime.worlds == []
    session.close()


def test_world_calls_are_authority_thread_only_but_lease_reads_are_safe(tmp_path: Path) -> None:
    fixture = _make_planning_fixture(tmp_path)
    session = _open_session(_FakePlanningRuntime(fixture))
    world = cast(PlanningSceneWorld, session.build(fixture.spec))
    geometry = next(item for item in world.planning_scene_catalog().geometries if item.resource_id is not None)
    lease = world.resolve_planning_geometry(geometry.geometry_id)
    results: list[object] = []

    def invoke() -> None:
        results.append(lease.read(0, 8))
        try:
            world.planning_scene_state()
        except BaseException as error:
            results.append(error)

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()
    assert type(results[0]) is bytes and len(cast(bytes, results[0])) == 8
    assert type(results[1]) is PlanningSceneContractError
    lease.close()
    session.close()

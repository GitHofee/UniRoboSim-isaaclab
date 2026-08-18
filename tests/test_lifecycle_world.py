from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    CameraModality,
    CameraSpec,
    CapabilityId,
    CapabilityNegotiationError,
    CapabilityRequirement,
    CommandError,
    CommandMode,
    DebugBatch,
    DebugPrimitive,
    DebugPrimitiveKind,
    DeformableCommand,
    EntityKind,
    EntityNotFoundError,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    LifecycleError,
    ParticleFluidCommand,
    ParticleFluidSpec,
    PointCommandMode,
    ProviderSelectionError,
    RigidBodyCommand,
    SessionState,
    StaleHandleError,
    UniRoboSimError,
    UnsupportedCapabilityError,
    ValidationError,
    World,
    WorldBuildError,
    WorldSpec,
    WorldState,
)

from unirobosim_isaaclab import IsaacLabProvider
from unirobosim_isaaclab.config import IsaacLabAdapterConfig
from unirobosim_isaaclab.provider import IsaacLabSession

from .helpers import (
    FakeNativeRuntime,
    FakeNativeWorld,
    available_probe,
    make_articulation_asset,
    make_world,
    unavailable_probe,
)


def open_test_session(runtime: FakeNativeRuntime) -> tuple[IsaacLabProvider, IsaacLabSession]:
    provider = IsaacLabProvider(runtime_factory=lambda config: runtime, probe_function=available_probe)
    return provider, provider.open()


def test_unavailable_and_launch_failure() -> None:
    provider = IsaacLabProvider(probe_function=unavailable_probe)
    with pytest.raises(ProviderSelectionError) as caught:
        provider.open()
    assert caught.value.details["reason"] == "disabled for test"

    def fail(config: object) -> FakeNativeRuntime:
        raise RuntimeError("kit failed")

    provider = IsaacLabProvider(runtime_factory=fail, probe_function=available_probe)
    with pytest.raises(ProviderSelectionError) as caught:
        provider.open()
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_session_world_lifecycle_and_rebuild(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    provider, session = open_test_session(runtime)
    assert session.state.value == SessionState.OPEN.value
    with pytest.raises(LifecycleError):
        provider.open()
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usda")))
    assert isinstance(world, World)
    assert session.state.value == SessionState.READY.value
    assert world.state is WorldState.READY
    assert world.build_report.environment_count == 2
    assert world.build_report.entity_count == 4
    with pytest.raises(LifecycleError):
        session.build(make_world(tmp_path / "arm.usda", world_id="second"))
    assert session.negotiate(()).accepted
    world.close()
    assert world.state.value == WorldState.CLOSED.value
    world.close()
    second = session.build(make_world(tmp_path / "arm.usda", world_id="second"))
    assert second.generation == 2
    session.close()
    assert second.state is WorldState.CLOSED
    assert session.state is SessionState.CLOSED
    assert runtime.closed
    session.close()
    replacement = provider.open()
    replacement.close()


def test_context_managers_and_closed_guards(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    provider, session = open_test_session(runtime)
    with session:
        with session.build(make_world(make_articulation_asset(tmp_path / "arm.usd"))) as world:
            assert world.tick.step_index == 0
    assert session.state is SessionState.CLOSED
    with pytest.raises(LifecycleError):
        session.__enter__()
    with pytest.raises(LifecycleError):
        session.negotiate(())
    with pytest.raises(LifecycleError):
        world.__enter__()
    with pytest.raises(LifecycleError):
        world.step()


def test_build_failure_is_transactional(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime(build_failures=1)
    _, session = open_test_session(runtime)
    spec = make_world(make_articulation_asset(tmp_path / "arm.usdc"))
    with pytest.raises(WorldBuildError) as caught:
        session.build(spec)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert session.state is SessionState.OPEN
    world = session.build(spec)
    assert world.generation == 1
    world.close()
    session.close()


def test_build_type_and_preflight_failures(tmp_path: Path) -> None:
    _, session = open_test_session(FakeNativeRuntime())
    with pytest.raises(ValidationError):
        session.build("bad")  # type: ignore[arg-type]

    missing = tmp_path / "missing.usd"
    with pytest.raises(WorldBuildError):
        session.build(make_world(missing))
    wrong = make_articulation_asset(tmp_path / "arm.urdf")
    with pytest.raises(WorldBuildError):
        session.build(make_world(wrong))
    remote_spec = make_world(make_articulation_asset(tmp_path / "arm.usd"))
    arm = next(entity for entity in remote_spec.entities if entity.kind is EntityKind.ARTICULATION)
    remote = replace(
        remote_spec,
        entities=tuple(
            replace(item, asset_uri="https://x/arm.usd") if item is arm else item for item in remote_spec.entities
        ),
    )
    with pytest.raises(WorldBuildError):
        session.build(remote)
    session.close()


def test_capability_and_structured_build_errors(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    spec = make_world(make_articulation_asset(tmp_path / "arm.usd"))
    unsupported = replace(
        spec,
        requirements=(*spec.requirements, CapabilityRequirement(CapabilityId("sensor.camera.rgb@1"))),
    )
    with pytest.raises(CapabilityNegotiationError):
        session.build(unsupported)

    class StructuredRuntime(FakeNativeRuntime):
        def build_world(self, spec: WorldSpec) -> FakeNativeWorld:
            raise WorldBuildError("native detail", operation="native.build")

    session.close()
    _, structured = open_test_session(StructuredRuntime())
    with pytest.raises(WorldBuildError, match="native detail"):
        structured.build(spec)
    structured.close()


def test_all_preflight_rejections(tmp_path: Path) -> None:
    usd = make_articulation_asset(tmp_path / "arm.usda")
    _, session = open_test_session(FakeNativeRuntime())
    no_asset = WorldSpec(
        "no-asset",
        (EntitySpec(EntityPath("/arm"), EntityKind.ARTICULATION, joint_names=("joint",)),),
    )
    with pytest.raises(WorldBuildError):
        session.build(no_asset)

    rigid_no_asset = WorldSpec(
        "rigid-no-asset",
        (EntitySpec(EntityPath("/box"), EntityKind.RIGID_BODY),),
    )
    with pytest.raises(WorldBuildError):
        session.build(rigid_no_asset)

    rigid_bad = WorldSpec(
        "rigid-bad",
        (EntitySpec(EntityPath("/box"), EntityKind.RIGID_BODY, asset_uri=str(tmp_path / "box.obj")),),
    )
    with pytest.raises(WorldBuildError):
        session.build(rigid_bad)

    base = make_world(usd)
    cloth = next(entity for entity in base.entities if entity.kind is EntityKind.SURFACE_DEFORMABLE)
    assert cloth.deformable is not None
    kinematic_cloth = replace(cloth, deformable=replace(cloth.deformable, kinematic_node_indices=(0,)))
    bad_cloth = replace(
        base, world_id="bad-cloth", entities=tuple(kinematic_cloth if item is cloth else item for item in base.entities)
    )
    with pytest.raises(UnsupportedCapabilityError):
        session.build(bad_cloth)

    file_uri = replace(
        base,
        world_id="file-uri",
        entities=tuple(
            replace(item, asset_uri=usd.as_uri()) if item.kind is EntityKind.ARTICULATION else item
            for item in base.entities
        ),
    )
    world = session.build(file_uri)
    world.close()
    session.close()


def test_resolve_reset_step_and_handle_generation(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usda")))
    arm = world.resolve(EntityPath("/robots/arm"))
    assert arm.entity_kind is EntityKind.ARTICULATION
    with pytest.raises(ValidationError):
        world.resolve("/robots/arm")  # type: ignore[arg-type]
    with pytest.raises(EntityNotFoundError):
        world.resolve(EntityPath("/missing"))
    with pytest.raises(StaleHandleError):
        world.read_articulation("bad")  # type: ignore[arg-type]
    default_result = world.reset()
    assert default_result.environment_indices == (0, 1)
    result = world.reset([1])
    assert result.environment_indices == (1,)
    assert result.reset_count == 2
    assert runtime.worlds[0].calls[-1] == ("reset", (1,))
    for invalid in ([], [2], [-1], [0, 0], [True]):
        with pytest.raises(ValidationError):
            world.reset(invalid)
    assert world.step(3).step_index == 3
    assert world.tick.sim_time_seconds == pytest.approx(0.05)
    invalid_counts: tuple[object, ...] = (0, -1, True, 1.5)
    for invalid_count in invalid_counts:
        with pytest.raises(ValidationError):
            world.step(invalid_count)  # type: ignore[arg-type]
    world.close()
    second = session.build(make_world(tmp_path / "arm.usda", world_id="same-id"))
    with pytest.raises(StaleHandleError):
        second.read_articulation(arm)
    second.close()
    session.close()


def test_native_runtime_errors_are_structured(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime(close_error=True)
    _, session = open_test_session(runtime)
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usd")))

    runtime.worlds[0].step_error = RuntimeError("step failed")
    with pytest.raises(UniRoboSimError) as caught:
        world.step()
    assert caught.value.operation == "world.step"
    assert isinstance(caught.value.__cause__, RuntimeError)

    original = ValidationError("preserved", operation="native.reset")

    runtime.worlds[0].reset_error = original
    with pytest.raises(ValidationError) as caught_validation:
        world.reset()
    assert caught_validation.value is original
    with pytest.raises(UniRoboSimError) as close_error:
        world.close()
    assert close_error.value.operation == "world.close"
    assert session.state is SessionState.OPEN
    session.close()


@pytest.mark.parametrize("mode", list(CommandMode))
def test_articulation_commands_and_read(mode: CommandMode, tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usd")))
    handle = world.resolve(EntityPath("/robots/arm"))
    command = ArticulationCommand(
        handle,
        mode,
        ArrayValue.from_nested(((1.0,),)),
        environment_indices=(1,),
        degree_of_freedom_indices=(0,),
    )
    world.apply_articulation_command(command)
    call = runtime.worlds[0].calls[-1]
    assert call[0] == "articulation"
    assert call[1][1] is mode
    state = world.read_articulation(handle)
    assert state.joint_positions.shape == (2, 2)
    assert state.tick == world.tick
    world.close()
    session.close()


def test_articulation_validation(tmp_path: Path) -> None:
    _, session = open_test_session(FakeNativeRuntime())
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usd")))
    handle = world.resolve(EntityPath("/robots/arm"))
    cloth = world.resolve(EntityPath("/soft/cloth"))
    with pytest.raises(CommandError):
        world.apply_articulation_command("bad")  # type: ignore[arg-type]
    with pytest.raises(CommandError):
        world.apply_articulation_command(
            ArticulationCommand(handle, CommandMode.POSITION, ArrayValue.from_nested(((1.0, 2.0),)), (0,), (0,))
        )
    with pytest.raises(CommandError):
        world.read_articulation(cloth)
    world.close()
    session.close()


def test_rigid_body_state_wrench_and_contact(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usd")))
    handle = world.resolve(EntityPath("/props/marker"))

    initial = world.read_rigid_body(handle)
    assert initial.positions_m.shape == (2, 3)
    assert initial.orientations_xyzw.rows() == ((0.0, 0.0, 0.0, 1.0),) * 2
    assert initial.tick == world.tick

    world.apply_rigid_body_command(
        RigidBodyCommand(
            handle,
            ArrayValue.from_nested(((2.0, 0.0, 0.0),)),
            ArrayValue.from_nested(((0.0, 0.0, 0.5),)),
            environment_indices=(1,),
        )
    )
    name, payload = runtime.worlds[0].calls[-1]
    assert name == "rigid_wrench"
    assert payload[1:] == (((2.0, 0.0, 0.0),), ((0.0, 0.0, 0.5),), (1,))

    contact = world.read_contact(handle)
    assert contact.net_normal_forces_n.shape == (2, 3)
    assert contact.in_contact.values == (True, True)
    assert world.read_contact(handle, force_threshold_n=10.0).in_contact.values == (False, False)

    world.close()
    session.close()


def test_rigid_body_validation(tmp_path: Path) -> None:
    _, session = open_test_session(FakeNativeRuntime())
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usd")))
    rigid = world.resolve(EntityPath("/props/marker"))
    arm = world.resolve(EntityPath("/robots/arm"))
    with pytest.raises(CommandError):
        world.apply_rigid_body_command("bad")  # type: ignore[arg-type]
    with pytest.raises(CommandError):
        world.apply_rigid_body_command(
            RigidBodyCommand(
                rigid,
                ArrayValue.from_nested(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
                ArrayValue.from_nested(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
                environment_indices=(0,),
            )
        )
    with pytest.raises(CommandError):
        world.read_rigid_body(arm)
    with pytest.raises(CommandError):
        world.read_contact(arm)
    for threshold in (-1.0, float("inf"), True, "bad"):
        with pytest.raises(ValidationError):
            world.read_contact(rigid, threshold)  # type: ignore[arg-type]
    world.close()
    session.close()


def test_deformable_state_and_kinematic_command(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usd")))
    jelly = world.resolve(EntityPath("/soft/jelly"))
    command = DeformableCommand(
        jelly,
        PointCommandMode.POSITION,
        ArrayValue.from_nested((((0.25, 0.5, 1.0),),)),
        environment_indices=(1,),
        node_indices=(0,),
    )
    world.apply_deformable_command(command)
    assert runtime.worlds[0].calls[-1][0] == "deformable"
    state = world.read_deformable(jelly)
    assert state.node_positions_m.shape == (2, 4, 3)
    cloth_state = world.read_deformable(world.resolve(EntityPath("/soft/cloth")))
    assert cloth_state.node_positions_m.shape == (2, 3, 3)
    world.close()
    session.close()


def test_deformable_command_rejections(tmp_path: Path) -> None:
    _, session = open_test_session(FakeNativeRuntime())
    world = session.build(make_world(make_articulation_asset(tmp_path / "arm.usd")))
    jelly = world.resolve(EntityPath("/soft/jelly"))
    cloth = world.resolve(EntityPath("/soft/cloth"))
    with pytest.raises(CommandError):
        world.apply_deformable_command("bad")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedCapabilityError):
        world.apply_deformable_command(
            DeformableCommand(
                cloth, PointCommandMode.POSITION, ArrayValue.from_nested((((0.0, 0.0, 0.0),),)), (0,), (0,)
            )
        )
    with pytest.raises(UnsupportedCapabilityError):
        world.apply_deformable_command(
            DeformableCommand(
                jelly, PointCommandMode.VELOCITY, ArrayValue.from_nested((((0.0, 0.0, 0.0),),)), (0,), (0,)
            )
        )
    with pytest.raises(CommandError):
        world.apply_deformable_command(
            DeformableCommand(
                jelly, PointCommandMode.POSITION, ArrayValue.from_nested((((0.0, 0.0, 0.0),),)), (0,), (1,)
            )
        )
    with pytest.raises(CommandError):
        world.apply_deformable_command(
            DeformableCommand(
                jelly,
                PointCommandMode.POSITION,
                ArrayValue.from_nested((((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)),
                (0,),
                (0,),
            )
        )
    with pytest.raises(CommandError):
        world.read_deformable(world.resolve(EntityPath("/robots/arm")))
    world.close()
    session.close()


def test_particle_fluid_state_command_and_validation(tmp_path: Path) -> None:
    del tmp_path
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    fluid = EntitySpec(
        EntityPath("/fluid"),
        EntityKind.PARTICLE_FLUID,
        particle_fluid=ParticleFluidSpec(ArrayValue.from_nested(((0.0, 0.0, 1.0), (0.1, 0.0, 1.0)))),
    )
    spec = WorldSpec("fluid", (fluid,), environments=EnvironmentSpec(1))
    world = session.build(spec)
    handle = world.resolve(EntityPath("/fluid"))
    with pytest.raises(CommandError):
        world.apply_particle_fluid_command("bad")  # type: ignore[arg-type]
    command = ParticleFluidCommand(
        handle,
        PointCommandMode.POSITION,
        ArrayValue.from_nested((((0.0, 0.0, 1.0),),)),
        environment_indices=(0,),
        particle_indices=(0,),
    )
    world.apply_particle_fluid_command(command)
    assert runtime.worlds[0].calls[-1][0] == "fluid"
    state = world.read_particle_fluid(handle)
    assert state.particle_positions_m.shape == (1, 2, 3)
    assert state.tick == world.tick
    with pytest.raises(UnsupportedCapabilityError):
        world.apply_particle_fluid_command(
            ParticleFluidCommand(
                handle,
                PointCommandMode.FORCE,
                ArrayValue.from_nested((((0.0, 0.0, 0.0),),)),
                environment_indices=(0,),
                particle_indices=(0,),
            )
        )
    with pytest.raises(CommandError):
        world.apply_particle_fluid_command(
            ParticleFluidCommand(
                handle,
                PointCommandMode.POSITION,
                ArrayValue.from_nested((((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),)),
                environment_indices=(0,),
                particle_indices=(0,),
            )
        )
    world.close()
    session.close()


def test_camera_and_debug_endpoints_are_strict_and_backend_neutral() -> None:
    runtime = FakeNativeRuntime()
    provider = IsaacLabProvider(
        IsaacLabAdapterConfig(enable_cameras=True, render=True),
        runtime_factory=lambda config: runtime,
        probe_function=available_probe,
    )
    session = provider.open()
    spec = WorldSpec(
        "camera-debug",
        (
            EntitySpec(
                EntityPath("/camera"),
                EntityKind.CAMERA_SENSOR,
                camera=CameraSpec(width_px=4, height_px=3),
            ),
            EntitySpec(
                EntityPath("/fluid"),
                EntityKind.PARTICLE_FLUID,
                particle_fluid=ParticleFluidSpec(ArrayValue.from_nested(((0.0, 0.0, 1.0),))),
            ),
        ),
        environments=EnvironmentSpec(2),
    )
    world = session.build(spec)
    camera = world.resolve(EntityPath("/camera"))
    fluid = world.resolve(EntityPath("/fluid"))
    before = world.tick
    sample = world.read_sensor(camera)
    assert sample.tick == before and world.tick == before
    assert sample.channel(CameraModality.RGB).shape == (2, 3, 4, 3)
    assert sample.channel(CameraModality.DEPTH).shape == (2, 3, 4)
    with pytest.raises(CommandError):
        world.read_sensor(fluid)
    with pytest.raises(CommandError):
        world.read_particle_fluid(camera)
    with pytest.raises(ValidationError):
        world.apply_particle_fluid_command(
            ParticleFluidCommand(
                camera,
                PointCommandMode.POSITION,
                ArrayValue.from_nested((((0.0, 0.0, 0.0),),)),
            )
        )

    primitive = DebugPrimitive(
        "goal",
        "planning",
        DebugPrimitiveKind.POINT_SET,
        ArrayValue.from_nested([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        (0, 1),
    )
    report = world.publish_debug(DebugBatch((primitive,)))
    assert (report.accepted_count, report.dropped_count, report.active_count) == (1, 0, 1)
    assert world.clear_debug(layer="planning", primitive_id="goal") == 1
    with pytest.raises(ValidationError):
        world.publish_debug(object())  # type: ignore[arg-type]
    out_of_range = DebugPrimitive(
        "bad",
        "planning",
        DebugPrimitiveKind.POINT_SET,
        ArrayValue.from_nested([[[0.0, 0.0, 0.0]]]),
        (2,),
    )
    with pytest.raises(ValidationError):
        world.publish_debug(DebugBatch((out_of_range,)))
    with pytest.raises(ValidationError):
        world.clear_debug(layer="")
    session.close()


def test_camera_modality_order_from_native_is_verified() -> None:
    class WrongOrderWorld(FakeNativeWorld):
        def read_sensor(self, path: EntityPath):
            channels = super().read_sensor(path)
            return tuple(reversed(channels))

    class WrongOrderRuntime(FakeNativeRuntime):
        def build_world(self, spec: WorldSpec) -> FakeNativeWorld:
            world = WrongOrderWorld(spec)
            self.worlds.append(world)
            return world

    runtime = WrongOrderRuntime()
    provider = IsaacLabProvider(
        IsaacLabAdapterConfig(enable_cameras=True, render=True),
        runtime_factory=lambda config: runtime,
        probe_function=available_probe,
    )
    session = provider.open()
    world = session.build(
        WorldSpec(
            "bad-camera-order",
            (
                EntitySpec(
                    EntityPath("/camera"),
                    EntityKind.CAMERA_SENSOR,
                    camera=CameraSpec(width_px=2, height_px=2),
                ),
            ),
        )
    )
    with pytest.raises(UniRoboSimError, match="invalid order"):
        world.read_sensor(world.resolve(EntityPath("/camera")))
    session.close()

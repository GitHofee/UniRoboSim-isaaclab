from __future__ import annotations

import math
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from unirobosim import (
    ArrayValue,
    EntityKind,
    EntityPath,
    EntitySpec,
    PackedFloat32Array,
    ParticleFluidSpec,
    RenderArticulationState,
    RenderParticleFluidState,
    RenderRigidBodyState,
    RenderStateFrame,
    WorldSpec,
)

from unirobosim_isaaclab.native import IsaacLabNativeWorld
from unirobosim_isaaclab.native_protocols import (
    NativeRenderArticulationState,
    NativeRenderParticleFluidState,
    NativeRenderRigidBodyState,
    NativeRenderStateFrame,
)

from .helpers import FakeNativeRuntime, make_articulation_asset, make_world
from .test_lifecycle_world import open_test_session


def _public_world(asset: Path) -> WorldSpec:
    base = make_world(asset)
    return WorldSpec(
        base.world_id,
        (
            *base.entities,
            EntitySpec(
                EntityPath("/fluid"),
                EntityKind.PARTICLE_FLUID,
                particle_fluid=ParticleFluidSpec(
                    ArrayValue.from_nested(((0.0, 0.0, 1.0), (0.1, 0.0, 1.0)))
                ),
            ),
        ),
        environments=base.environments,
    )


def test_public_world_maps_one_render_frame_to_one_native_call_without_advancing_tick(tmp_path: Path) -> None:
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    try:
        world = session.build(_public_world(make_articulation_asset(tmp_path / "robot.usda")))
        tick = world.tick
        packed = PackedFloat32Array((1, 2, 3), struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        frame = RenderStateFrame(
            articulations=(
                RenderArticulationState(
                    world.resolve(EntityPath("/robots/arm")),
                    ArrayValue.from_rows(((0.7, -0.4),)),
                    ArrayValue.from_rows(((0.1, 0.2),)),
                    environment_indices=(1,),
                    root_positions_m=ArrayValue.from_rows(((9.0, 8.0, 7.0),)),
                    root_orientations_xyzw=ArrayValue.from_rows(((0.0, 0.0, 0.0, 1.0),)),
                    root_linear_velocities_m_s=ArrayValue.from_rows(((0.6, 0.5, 0.4),)),
                    root_angular_velocities_rad_s=ArrayValue.from_rows(((0.3, 0.2, 0.1),)),
                ),
            ),
            rigid_bodies=(
                RenderRigidBodyState(
                    world.resolve(EntityPath("/props/marker")),
                    ArrayValue.from_rows(((1.0, 2.0, 3.0),)),
                    ArrayValue.from_rows(((0.0, 0.0, 0.0, 1.0),)),
                    ArrayValue.from_rows(((0.1, 0.2, 0.3),)),
                    ArrayValue.from_rows(((0.4, 0.5, 0.6),)),
                    environment_indices=(0,),
                ),
            ),
            particle_fluids=(
                RenderParticleFluidState(
                    world.resolve(EntityPath("/fluid")),
                    packed,
                    environment_indices=(1,),
                ),
            ),
        )

        result = world.apply_render_state(frame)

        assert result.tick == tick == world.tick
        assert result.state_revision == 1
        name, native_frame = runtime.worlds[0].calls[-1]
        assert name == "render_state"
        assert isinstance(native_frame, NativeRenderStateFrame)
        assert native_frame.articulations[0].root_positions_m == ((9.0, 8.0, 7.0),)
        assert native_frame.particle_fluids[0].positions_m is packed
        assert native_frame.particle_fluids[0].positions_m.data is packed.data
    finally:
        session.close()


class _Simulation:
    device = "cpu"

    def __init__(self) -> None:
        self.forward_count = 0
        self.render_count = 0
        self.step_count = 0

    def forward(self) -> None:
        self.forward_count += 1

    def render(self) -> None:
        self.render_count += 1

    def step(self, *, render: bool) -> None:
        del render
        self.step_count += 1


class _ArticulationData:
    def __init__(self) -> None:
        self.joint_pos = SimpleNamespace(torch=torch.zeros((2, 2), dtype=torch.float32))
        self.joint_vel = SimpleNamespace(torch=torch.zeros((2, 2), dtype=torch.float32))


class _Articulation:
    def __init__(self) -> None:
        self.data = _ArticulationData()
        self.writes: dict[str, torch.Tensor] = {}

    def write_joint_position_to_sim_index(self, *, position: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.writes["positions"] = position.clone()

    def write_root_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.writes["root_pose"] = root_pose.clone()

    def write_root_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.writes["root_velocity"] = root_velocity.clone()

    def write_joint_velocity_to_sim_index(self, *, velocity: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.writes["velocities"] = velocity.clone()

    def set_joint_position_target_index(self, *, target: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.writes["position_targets"] = target.clone()

    def set_joint_velocity_target_index(self, *, target: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.writes["velocity_targets"] = target.clone()

    def set_joint_effort_target_index(self, *, target: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.writes["efforts"] = target.clone()


class _WrenchComposer:
    def __init__(self) -> None:
        self.calls = 0

    def set_forces_and_torques_index(self, **kwargs: object) -> None:
        del kwargs
        self.calls += 1


class _Rigid:
    def __init__(self) -> None:
        self.poses: torch.Tensor | None = None
        self.velocities: torch.Tensor | None = None
        self.permanent_wrench_composer = _WrenchComposer()

    def write_root_pose_to_sim_index(self, *, root_pose: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.poses = root_pose.clone()

    def write_root_link_velocity_to_sim_index(self, *, root_velocity: torch.Tensor, env_ids: object) -> None:
        del env_ids
        self.velocities = root_velocity.clone()


class _Attribute:
    def __init__(self, value: object) -> None:
        self.value = value
        self.set_count = 0

    def Get(self) -> object:
        return self.value

    def Set(self, value: object) -> None:
        self.value = value
        self.set_count += 1


class _Points:
    def __init__(self) -> None:
        self.positions = _Attribute(((0.0, 0.0, 1.0), (0.1, 0.0, 1.0)))
        self.velocities = _Attribute(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    def GetPointsAttr(self) -> _Attribute:
        return self.positions

    def GetVelocitiesAttr(self) -> _Attribute:
        return self.velocities


class _Vec3fArray:
    @classmethod
    def FromNumpy(cls, value: object) -> object:
        return value


def _native_world() -> tuple[IsaacLabNativeWorld, _Simulation, _Articulation, _Rigid, _Points, list[str]]:
    global torch
    import torch

    world = object.__new__(IsaacLabNativeWorld)
    simulation = _Simulation()
    articulation = _Articulation()
    rigid = _Rigid()
    points = _Points()
    events: list[str] = []
    robot_path = EntityPath("/robot")
    rigid_path = EntityPath("/box")
    fluid_path = EntityPath("/fluid")
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=2))
    world._m = SimpleNamespace(torch=torch, Vt=SimpleNamespace(Vec3fArray=_Vec3fArray))
    world._config = SimpleNamespace(fluid_render_mode="particles")
    world._sim = simulation
    world._step_index = 9
    world._joint_maps = {robot_path: (0, 1)}
    world._usd_articulation_views = {}
    world._articulations = {robot_path: articulation}
    world._articulation_control_modes = {robot_path: [[None, None], [None, None]]}
    world._kinematic_rigids = {rigid_path: False}
    world._usd_rigid_views = {}
    world._usd_rigid_wrenches = {}
    world._rigids = {rigid_path: rigid}
    world._origins = torch.zeros((2, 3), dtype=torch.float32)
    world._origins_cpu = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    world._fluids = {
        fluid_path: tuple(
            SimpleNamespace(
                points=points if environment == 0 else _Points(),
                initial_positions=((0.0, 0.0, 1.0), (0.1, 0.0, 1.0)),
            )
            for environment in range(2)
        )
    }
    world._update_assets = lambda dt: events.append(f"update:{dt}")
    world._sync_all_mounted_cameras = lambda: events.append("sync")
    world._render_revision = 4
    world._rendered_revision = 4
    return world, simulation, articulation, rigid, points, events


def _native_frame(*, fluid_payload: PackedFloat32Array) -> NativeRenderStateFrame:
    return NativeRenderStateFrame(
        articulations=(
            NativeRenderArticulationState(
                EntityPath("/robot"),
                ((1.0, -1.0),),
                ((0.2, -0.2),),
                (0,),
                (0, 1),
                ((9.0, 8.0, 7.0),),
                ((0.0, 0.0, 0.0, 1.0),),
                ((0.6, 0.5, 0.4),),
                ((0.3, 0.2, 0.1),),
            ),
        ),
        rigid_bodies=(
            NativeRenderRigidBodyState(
                EntityPath("/box"),
                ((3.0, 2.0, 1.0),),
                ((0.0, 0.0, 0.0, 1.0),),
                ((0.1, 0.2, 0.3),),
                ((0.4, 0.5, 0.6),),
                (0,),
            ),
        ),
        particle_fluids=(
            NativeRenderParticleFluidState(EntityPath("/fluid"), fluid_payload, None, (0,), 0),
        ),
    )


def _assert_native_render_state_batches_writes_and_syncs_once_without_physics() -> None:
    world, simulation, articulation, rigid, points, events = _native_world()
    payload = PackedFloat32Array((1, 2, 3), struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    world.apply_render_state(_native_frame(fluid_payload=payload))

    assert simulation.forward_count == 1
    assert simulation.step_count == 0
    assert world._step_index == 9
    assert events == ["update:0.0", "sync"]
    assert world._render_revision == 5
    assert articulation.writes["positions"].tolist() == [[1.0, -1.0]]
    assert articulation.writes["velocities"].tolist()[0] == pytest.approx([0.2, -0.2])
    assert articulation.writes["root_pose"].tolist() == [[9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0]]
    assert articulation.writes["root_velocity"].tolist()[0] == pytest.approx([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert rigid.poses is not None and rigid.poses.tolist() == [[3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    assert rigid.velocities is not None
    assert rigid.velocities.tolist()[0] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert points.positions.set_count == 1
    assert points.velocities.set_count == 0


def _assert_native_render_state_rejects_nonfinite_packed_payload_before_any_write() -> None:
    world, simulation, articulation, rigid, points, events = _native_world()
    payload = PackedFloat32Array(
        (1, 2, 3),
        struct.pack("<6f", math.nan, 2.0, 3.0, 4.0, 5.0, 6.0),
    )

    with pytest.raises(ValueError, match="finite"):
        world.apply_render_state(_native_frame(fluid_payload=payload))

    assert articulation.writes == {}
    assert rigid.poses is None and rigid.velocities is None
    assert points.positions.set_count == 0
    assert simulation.forward_count == simulation.step_count == 0
    assert events == []


def _assert_direct_rigid_pose_invalidates_an_already_rendered_revision() -> None:
    world, simulation, _, rigid, _, events = _native_world()

    world.set_rigid_body_pose(EntityPath("/box"), (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 0)
    world._ensure_camera_render()

    assert simulation.forward_count == 1
    assert simulation.step_count == 0
    assert simulation.render_count == 1
    assert rigid.poses is not None
    assert events == ["update:0.0", "sync", "sync"]
    assert world._render_revision == world._rendered_revision == 5


@pytest.mark.parametrize(
    "assertion",
    (
        "_assert_native_render_state_batches_writes_and_syncs_once_without_physics",
        "_assert_native_render_state_rejects_nonfinite_packed_payload_before_any_write",
        "_assert_direct_rigid_pose_invalidates_an_already_rendered_revision",
    ),
)
def test_native_render_state_in_an_isolated_torch_process(assertion: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"from tests.test_render_state import {assertion}; {assertion}()"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

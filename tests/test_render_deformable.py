import struct
from types import SimpleNamespace

import pytest
from unirobosim import EntityPath, PackedFloat32Array

from unirobosim_isaaclab.native_protocols import NativeRenderDeformableState, NativeRenderStateFrame

from .test_render_state import _Attribute, _native_world, _Points


class Mesh(_Points):
    def CreateExtentAttr(self):
        self.extent = _Attribute(None)
        return self.extent


def _assert_native_deformable_replay_updates_only_selected_mesh_without_physics():
    world, sim, *_ = _native_world()
    path = EntityPath("/cloth")
    meshes = [Mesh(), Mesh()]
    world._render_deformables = {path: meshes}
    world._entity_specs = {path: SimpleNamespace(deformable=SimpleNamespace(node_count=3))}

    class Vectors:
        def __new__(cls, value):
            return value

        @staticmethod
        def FromNumpy(value):
            return value

    world._m.Vt.Vec3fArray = Vectors
    positions = PackedFloat32Array((1, 3, 3), struct.pack("<9f", *range(9)))
    update = NativeRenderDeformableState(path, positions, None, (1,))
    world.apply_render_state(NativeRenderStateFrame((), (), (), (update,)))
    assert meshes[0].positions.set_count == 0
    assert meshes[1].positions.set_count == 1
    assert meshes[1].positions.value.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    assert sim.step_count == 0
    assert world._step_index == 9
    bad = NativeRenderDeformableState(
        path, PackedFloat32Array((1, 3, 3), struct.pack("<9f", float("nan"), *range(8))), None, (1,)
    )
    with pytest.raises(ValueError, match="finite"):
        world.apply_render_state(NativeRenderStateFrame((), (), (), (bad,)))
    assert meshes[1].positions.set_count == 1


def test_native_deformable_replay_isolated():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_render_deformable import "
            "_assert_native_deformable_replay_updates_only_selected_mesh_without_physics as check; check()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_render_only_profile_is_explicit_and_no_physics_capability():
    from unirobosim import CapabilityId, ValidationError

    from unirobosim_isaaclab.config import IsaacLabAdapterConfig
    from unirobosim_isaaclab.descriptor import descriptor_for_config

    config = IsaacLabAdapterConfig(deformable_mode="render_only")
    descriptor = descriptor_for_config(config)
    assert descriptor.metadata.to_dict()["deformable_mode"] == "render_only"
    assert descriptor.capabilities.get(CapabilityId("physics.deformable.self-collision@1")) is None
    with pytest.raises(ValidationError):
        IsaacLabAdapterConfig(deformable_mode="guess")


def test_public_deformable_replay_validates_topology_before_native_call():
    from unirobosim import (
        ArrayValue,
        CommandError,
        DeformableBodySpec,
        DeformableTopology,
        EntityKind,
        EntitySpec,
        RenderDeformableState,
        RenderStateFrame,
        WorldSpec,
    )

    from unirobosim_isaaclab import IsaacLabProvider
    from unirobosim_isaaclab.config import IsaacLabAdapterConfig

    from .helpers import FakeNativeRuntime, available_probe

    runtime = FakeNativeRuntime()
    session = IsaacLabProvider(
        IsaacLabAdapterConfig(deformable_mode="render_only"),
        runtime_factory=lambda config: runtime,
        probe_function=available_probe,
    ).open()
    try:
        spec = DeformableBodySpec(
            DeformableTopology.SURFACE,
            ArrayValue.from_rows(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            ArrayValue.from_rows(((0, 1, 2),), dtype="int64"),
        )
        world = session.build(
            WorldSpec("soft", (EntitySpec(EntityPath("/cloth"), EntityKind.SURFACE_DEFORMABLE, deformable=spec),))
        )
        handle = world.resolve(EntityPath("/cloth"))
        value = PackedFloat32Array((1, 3, 3), struct.pack("<9f", *range(9)))
        result = world.apply_render_state(RenderStateFrame(deformables=(RenderDeformableState(handle, value),)))
        assert result.deformable_count == 1
        assert result.tick == world.tick
        native = runtime.worlds[0].calls[-1][1]
        assert native.deformables[0].positions_m is value
        before = len(runtime.worlds[0].calls)
        with pytest.raises(CommandError):
            world.apply_render_state(
                RenderStateFrame(
                    deformables=(
                        RenderDeformableState(handle, PackedFloat32Array((1, 4, 3), struct.pack("<12f", *range(12)))),
                    )
                )
            )
        assert len(runtime.worlds[0].calls) == before
    finally:
        session.close()


def test_particle_world_procedural_box_uses_raw_rigid_bridge():
    from unirobosim import BoxGeometrySpec, EntityKind, EntitySpec, Pose

    from unirobosim_isaaclab.native import IsaacLabNativeWorld

    calls = []

    class Cfg(SimpleNamespace):
        InitialStateCfg = SimpleNamespace

    def spawn(path, cfg, **kwargs):
        calls.append((path, cfg, kwargs))

    prim = SimpleNamespace(HasAPI=lambda api: True)
    utils = SimpleNamespace(
        CuboidCfg=lambda **kwargs: SimpleNamespace(func=spawn, **kwargs),
        RigidBodyPropertiesCfg=SimpleNamespace,
        MassPropertiesCfg=SimpleNamespace,
        CollisionPropertiesCfg=SimpleNamespace,
        PreviewSurfaceCfg=SimpleNamespace,
        RigidBodyMaterialCfg=SimpleNamespace,
        get_current_stage=lambda: SimpleNamespace(GetPrimAtPath=lambda path: prim),
    )
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(RigidObjectCfg=Cfg, sim_utils=utils, UsdPhysics=SimpleNamespace(RigidBodyAPI=object()))
    world._has_fluid = True
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=2))
    world._usd_rigids = {}
    world._kinematic_rigids = {}
    entity = EntitySpec(
        EntityPath("/box"),
        EntityKind.RIGID_BODY,
        pose=Pose((1.0, 2.0, 3.0)),
        box=BoxGeometrySpec(dimensions_m=(0.2, 0.3, 0.4), mass_kg=0.7),
    )
    world._author_rigid(entity)
    assert len(world._usd_rigids[entity.path]) == 2
    assert calls[0][1].mass_props.mass == 0.7
    assert calls[0][1].size == (0.2, 0.3, 0.4)
    assert calls[0][2]["translation"] == (1.0, 2.0, 3.0)
    assert world._kinematic_rigids[entity.path] is False


def test_initial_deformable_points_bake_scale_and_pose_once():
    import math

    from unirobosim import ArrayValue, DeformableBodySpec, DeformableTopology, Pose

    from unirobosim_isaaclab.native import _render_deformable_initial_points

    spec = DeformableBodySpec(
        DeformableTopology.SURFACE,
        ArrayValue.from_rows(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        ArrayValue.from_rows(((0, 1, 2),), dtype="int64"),
    )
    # EntitySpec currently rejects scaled soft matter. Exercise the pure
    # coordinate helper without widening that public contract.
    entity = SimpleNamespace(
        pose=Pose((10.0, 20.0, 30.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))),
        scale_xyz=(2.0, 3.0, 4.0),
        deformable=spec,
    )
    values = _render_deformable_initial_points(entity)
    assert values[0] == pytest.approx((10.0, 22.0, 30.0))
    assert values[1] == pytest.approx((7.0, 20.0, 30.0))
    assert values[2] == pytest.approx((10.0, 20.0, 34.0))


def test_actual_usd_numpy_and_tuple_vec3_paths():
    import subprocess
    import sys

    code = """
import numpy as np
from pxr import Vt
points=np.arange(9,dtype=np.float32).reshape(3,3)
fast=Vt.Vec3fArray.FromNumpy(points)
slow=Vt.Vec3fArray([tuple(row) for row in points.tolist()])
assert list(fast)==list(slow)
assert tuple(fast[2])==(6.,7.,8.)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_actual_usd_deformable_parent_origin_is_applied_once():
    import subprocess
    import sys

    code = """
from types import SimpleNamespace
from pxr import Usd,UsdGeom,Vt,Gf
from unirobosim import *
from unirobosim_isaaclab.native import IsaacLabNativeWorld
stage=Usd.Stage.CreateInMemory()
for i in range(2):
    UsdGeom.Xform.Define(stage,f'/World/env_{i}').AddTranslateOp().Set(Gf.Vec3d(i*20,0,0))
world=object.__new__(IsaacLabNativeWorld)
world._spec=SimpleNamespace(environments=SimpleNamespace(count=2))
world._m=SimpleNamespace(sim_utils=SimpleNamespace(get_current_stage=lambda:stage),UsdGeom=UsdGeom,Vt=Vt)
world._render_deformables={}
body=DeformableBodySpec(DeformableTopology.SURFACE,ArrayValue.from_rows(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))),ArrayValue.from_rows(((0,1,2),),dtype='int64'))
entity=EntitySpec(EntityPath('/soft'),EntityKind.SURFACE_DEFORMABLE,pose=Pose((1.,2.,3.)),deformable=body)
world._author_render_deformable(entity)
meshes=world._render_deformables[entity.path]
assert tuple(meshes[1].GetPointsAttr().Get()[0])==(1.,2.,3.)
matrix=UsdGeom.XformCache().GetLocalToWorldTransform(meshes[1].GetPrim())
assert tuple(matrix.Transform(Gf.Vec3d(*meshes[1].GetPointsAttr().Get()[0])))==(21.,2.,3.)
assert not meshes[1].GetPrim().GetAppliedSchemas()
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def _assert_unwrapped_joint_render_state_does_not_send_drive_targets():
    import torch

    from unirobosim_isaaclab.native_protocols import NativeRenderArticulationState

    for raw in (False, True):
        world, simulation, articulation, *_ = _native_world()
        writes = articulation.writes
        if raw:

            def forbidden(*args):
                raise AssertionError("render replay must not send drive targets")

            view = SimpleNamespace(
                get_dof_positions=lambda: torch.zeros((2, 2)),
                get_dof_velocities=lambda: torch.zeros((2, 2)),
                set_dof_positions=lambda value, indices, writes=writes: writes.update(positions=value.clone()),
                set_dof_velocities=lambda value, indices, writes=writes: writes.update(velocities=value.clone()),
                set_dof_actuation_forces=lambda value, indices: None,
                set_dof_position_targets=forbidden,
                set_dof_velocity_targets=forbidden,
            )
            world._usd_articulation_views = {EntityPath("/robot"): view}
            world._usd_simulation_view = lambda: SimpleNamespace(update_articulations_kinematic=lambda: None)
            world._m.omni_physx = SimpleNamespace(
                get_physx_interface=lambda: SimpleNamespace(update_transformations=lambda *args: None)
            )
        frame = NativeRenderStateFrame(
            (NativeRenderArticulationState(EntityPath("/robot"), ((7.5, -8.0),), ((1.5, 0.0),), (0,), (0, 1)),), (), ()
        )
        world.apply_render_state(frame)
        assert writes["positions"].tolist() == [[7.5, -8.0]]
        assert "position_targets" not in writes
        assert "velocity_targets" not in writes
        assert world._articulation_control_modes[EntityPath("/robot")][0] == [None, None]
        assert simulation.step_count == 0


def test_unwrapped_joint_render_state_without_drive_commands():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_render_deformable import "
            "_assert_unwrapped_joint_render_state_does_not_send_drive_targets as check; check()",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

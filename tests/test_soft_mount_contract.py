from types import SimpleNamespace as NS

import pytest
from unirobosim import (
    ArrayValue,
    DeformableBodySpec,
    DeformableRigidMount,
    DeformableTopology,
    EntityKind,
    EntityPath,
    EntitySpec,
    Pose,
)

from unirobosim_isaaclab.deformable_mounts import author_rigid_mount


@pytest.mark.parametrize(
    "failure", ["omniphysics:src0", "omniphysics:src1", "omniphysics:vtxIndicesSrc0", "omniphysics:localPositionsSrc1"]
)
@pytest.mark.parametrize("missing", [False, True])
def test_failed_mount_authoring_removes_partial_constraint(failure, missing):
    class Field:
        def __init__(self, name):
            self.name = name

        def __bool__(self):
            return not (missing and self.name == failure)

        def Set(self, _value):
            return self.name != failure

        def SetTargets(self, _value):
            return self.name != failure

    class Prim:
        def GetRelationship(self, name):
            return Field(name)

        def GetAttribute(self, name):
            return Field(name)

    class Matrix:
        def __mul__(self, _other):
            return self

        def GetInverse(self):
            return self

        def Transform(self, value):
            return value

    prims = {"/pad": Prim(), "/parent": Prim()}

    def define(path, _type):
        prims[path] = Prim()
        return prims[path]

    stage = NS(GetPrimAtPath=lambda path: prims.get(path), DefinePrim=define, RemovePrim=lambda path: prims.pop(path))
    modules = NS(
        UsdGeom=NS(XformCache=lambda: NS(GetLocalToWorldTransform=lambda _: Matrix())),
        Gf=NS(Vec3d=lambda *v: v),
        Sdf=NS(Path=str),
        Vt=NS(IntArray=tuple, Vec3fArray=tuple),
    )
    with pytest.raises(RuntimeError, match="native structural mount"):
        author_rigid_mount(modules, stage, "/pad", "/parent", (0,), ((0, 0, 0),))
    assert set(prims) == {"/pad", "/parent"}


def test_mounted_partial_reset_rotates_initial_velocity_and_adds_parent_origin_twist():
    import torch

    from .test_native_kinematic_rigid import _bare_world

    world, _ = _bare_world(kinematic=False, raw=False)
    world._m = NS(torch=torch)
    world._origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    world._origins_cpu = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    world._rigids = {}
    world._contacts = {}
    parent = EntitySpec(EntityPath("/parent"), EntityKind.RIGID_BODY)
    pad = EntitySpec(
        EntityPath("/pad"),
        EntityKind.VOLUME_DEFORMABLE,
        pose=Pose((0, 0, 1)),
        deformable=DeformableBodySpec(
            DeformableTopology.VOLUME,
            ArrayValue.from_rows(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
            tetrahedra=ArrayValue.from_rows(((0, 1, 2, 3),), dtype="int64"),
            initial_node_velocities_m_s=ArrayValue.from_rows(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            ),
            rigid_mount=DeformableRigidMount(parent.path, node_indices=(0,), coupling="kinematic_boundary"),
        ),
    )
    world._entity_specs = {parent.path: parent, pad.path: pad}
    world._spec = NS(entities=(pad, parent))
    state = torch.zeros(2, 4, 6)
    target = torch.ones(2, 4, 4)
    target[:, 0, 3] = 0
    world._initial_deformable = {pad.path: (state, target)}
    calls = []
    world._deformables = {
        pad.path: NS(
            write_nodal_state_to_sim_index=lambda s, env_ids: calls.append((s.clone(), list(env_ids))),
            write_nodal_kinematic_target_to_sim_index=lambda *a, **k: None,
            reset=lambda **k: None,
            update=lambda dt: None,
        )
    }
    world._initial_entity_prim_poses = {pad.path: (Pose(), Pose())}
    world._write_entity_prim_pose = lambda *a: pytest.fail("recooked live deformable")
    world.read_rigid_body = lambda path: (
        ((2.0, 0.0, 0.0),) * 2,
        ((0.0, 0.0, 2**-0.5, 2**-0.5),) * 2,
        ((1.0, 2.0, 3.0),) * 2,
        ((0.0, 0.0, 2.0),) * 2,
    )
    world.reset((1,))
    written, ids = calls[0]
    assert ids == [1]
    assert written[0, 1, :3].tolist() == pytest.approx([12.0, 1.0, 1.0])
    assert written[0, 1, 3:].tolist() == pytest.approx([-1.0, 3.0, 3.0])
    assert written[0, 0, 3:].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert torch.equal(state, torch.zeros_like(state)), "reset mutated stored/other environment state"

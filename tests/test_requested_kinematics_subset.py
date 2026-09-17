"""Transfer volume follows deduplicated requested axes, not asset size."""

from types import SimpleNamespace as NS

import pytest
from unirobosim import EntityPath, KinematicTarget

from unirobosim_isaaclab.native import IsaacLabNativeWorld


class Tensor:
    def __init__(self, values, transfers):
        self.values = values
        self.transfers = transfers

    def __getitem__(self, index):
        if isinstance(index, tuple):
            rows, columns = index
            return Tensor([[row[i] for i in columns] for row in self.values[rows]], self.transfers)
        return Tensor(
            [self.values[i] for i in index] if isinstance(index, list) else self.values[index], self.transfers
        )

    def detach(self):
        return self

    def cpu(self):
        self.transfers.append((len(self.values), len(self.values[0]) if self.values else 0))
        return self

    def tolist(self):
        return [row[:] for row in self.values]


def test_sparse_multiasset_multienvironment_reversed_duplicate_and_same_tick_reads():
    world = object.__new__(IsaacLabNativeWorld)
    world._spec = NS(environments=NS(count=2))
    world._origins_cpu = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    world._articulations = {}
    transfers = []
    paths = tuple(EntityPath(f"/robot{i}") for i in range(2))
    poses = None
    for path in paths:
        names = tuple(f"body{i}" for i in range(4096))
        poses = [[[float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for i in range(4096)] for _ in range(2)]
        velocities = [[[float(i)] * 6 for i in range(4096)] for _ in range(2)]
        world._articulations[path] = NS(
            body_names=names,
            data=NS(
                body_link_pose_w=NS(torch=Tensor(poses, transfers)),
                body_link_vel_w=NS(torch=Tensor(velocities, transfers)),
            ),
        )
    targets = (
        KinematicTarget("z", paths[1], "body4095"),
        KinematicTarget("a", paths[0], "body9"),
        KinematicTarget("z-again", paths[1], "body4095"),
        KinematicTarget("b", paths[1], "body3"),
    )
    states = world.read_selected_kinematics(targets, 1)
    assert [s.position_m[0] for s in states] == [4085.0, -1.0, 4085.0, -7.0]
    assert tuple(s.target_id for s in states) == tuple(t.target_id for t in targets)
    assert transfers == [(2, 7), (2, 6), (1, 7), (1, 6)]
    poses[1][4095][0] = 25.0
    assert world.read_selected_kinematics(targets, 1)[0].position_m[0] == 15.0
    assert world.read_selected_kinematics(targets, 0)[0].position_m[0] == 4095.0
    for invalid in (True, -1, 2, 0.0):
        with pytest.raises(IndexError):
            world.read_selected_kinematics(targets, invalid)


@pytest.mark.parametrize("raw", [False, True])
def test_joint_read_deduplicates_on_device_before_transfer(raw):
    path = EntityPath("/robot")
    world = object.__new__(IsaacLabNativeWorld)
    transfers = []
    positions = Tensor([[float(i) for i in range(4096)]] * 2, transfers)
    velocities = Tensor([[float(-i) for i in range(4096)]] * 2, transfers)
    world._joint_maps = {path: (4095, 1, 4095)}
    world._articulations = (
        {} if raw else {path: NS(data=NS(joint_pos=NS(torch=positions), joint_vel=NS(torch=velocities)))}
    )
    world._usd_articulation_views = (
        {path: NS(get_dof_positions=lambda: positions, get_dof_velocities=lambda: velocities)} if raw else {}
    )
    assert world.read_articulation(path) == (
        ((4095.0, 1.0, 4095.0),) * 2,
        ((-4095.0, -1.0, -4095.0),) * 2,
    )
    assert transfers == [(2, 2), (2, 2)]

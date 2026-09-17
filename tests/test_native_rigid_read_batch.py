"""Rigid readback keeps exact values while transferring each tensor only once."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from unirobosim import EntityPath

from unirobosim_isaaclab.native import IsaacLabNativeWorld


class _Tensor:
    device = "cuda:0"
    dtype = "float32"

    def __init__(self, values: list[list[float]], transfers: list[str], label: str) -> None:
        self.values = [list(row) for row in values]
        self.transfers = transfers
        self.label = label

    def clone(self) -> _Tensor:
        return _Tensor(self.values, self.transfers, self.label)

    def __getitem__(self, key: tuple[slice, slice]) -> _Tensor:
        rows, columns = key
        return _Tensor([[row[index] for index in columns] if isinstance(columns, list) else row[columns]
                        for row in self.values[rows]], self.transfers, self.label)

    def __setitem__(self, key: tuple[slice, slice], value: _Tensor) -> None:
        rows, columns = key
        for index, row in zip(range(len(self.values))[rows], value.values, strict=True):
            self.values[index][columns] = row

    def __isub__(self, other: _Tensor) -> _Tensor:
        self.values = [
            [left - right for left, right in zip(row, origin, strict=True)]
            for row, origin in zip(self.values, other.values, strict=True)
        ]
        return self

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        self.transfers.append(self.label)
        return self

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self.values]


def _world(raw: bool, count: int) -> tuple[IsaacLabNativeWorld, _Tensor, _Tensor, list[str]]:
    transfers: list[str] = []
    poses = _Tensor([
        [11.0, 22.0, 33.0, 0.0, 0.0, 0.0, 1.0],
        [41.0, 52.0, 63.0, 0.0, 0.0, 1.0, 0.0],
    ][:count], transfers, "pose")
    velocities = _Tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    ][:count], transfers, "velocity")
    origins = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]][:count]
    path = EntityPath("/objects/mug")
    world = object.__new__(IsaacLabNativeWorld)
    world._origins_cpu = tuple(tuple(row) for row in origins)
    world._origins = _Tensor(origins, transfers, "origin")
    world._m = SimpleNamespace(torch=SimpleNamespace(
        tensor=lambda values, *, device, dtype: _Tensor(values, transfers, "origin"),
    ))
    world._usd_rigid_views = {path: SimpleNamespace(
        get_transforms=lambda: poses,
        get_velocities=lambda: velocities,
    )} if raw else {}
    world._rigids = {} if raw else {path: SimpleNamespace(data=SimpleNamespace(
        root_link_pose_w=SimpleNamespace(torch=poses),
        root_link_vel_w=SimpleNamespace(torch=velocities),
    ))}
    return world, poses, velocities, transfers


@pytest.mark.parametrize("raw", [False, True], ids=["asset", "usd-view"])
@pytest.mark.parametrize("count", [1, 2], ids=["one-environment", "two-environments"])
def test_rigid_read_transfers_pose_and_velocity_once_without_changing_values(raw: bool, count: int) -> None:
    world, poses, velocities, transfers = _world(raw, count)
    original_poses = poses.tolist()
    original_velocities = velocities.tolist()

    result = world.read_rigid_body(EntityPath("/objects/mug"))

    assert result == (
        ((1.0, 2.0, 3.0),) * count,
        ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0))[:count],
        ((1.0, 2.0, 3.0), (-1.0, -2.0, -3.0))[:count],
        ((4.0, 5.0, 6.0), (-4.0, -5.0, -6.0))[:count],
    )
    assert poses.values == original_poses
    assert velocities.values == original_velocities
    assert transfers == ["pose", "velocity"]

    poses.values[0][0] = 12.0
    velocities.values[0][0] = 7.0
    following = world.read_rigid_body(EntityPath("/objects/mug"))
    assert following[0][0] == (2.0, 2.0, 3.0)
    assert following[2][0] == (7.0, 2.0, 3.0)
    assert result[0][0] == (1.0, 2.0, 3.0)
    assert result[2][0] == (1.0, 2.0, 3.0)
    assert transfers == ["pose", "velocity", "pose", "velocity"]


@pytest.mark.parametrize("raw", [False, True], ids=["asset", "usd-view"])
def test_unknown_rigid_path_is_rejected_before_transfer(raw: bool) -> None:
    world, _, _, transfers = _world(raw, 1)
    with pytest.raises(KeyError):
        world.read_rigid_body(EntityPath("/objects/missing"))
    assert transfers == []

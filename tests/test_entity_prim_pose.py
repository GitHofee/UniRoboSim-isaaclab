from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from unirobosim import EntityPath, Pose

from unirobosim_isaaclab.native import (
    IsaacLabNativeWorld,
    _normalized_quaternion_xyzw,
    _pose_from_world_matrix,
    _retarget_physical_root_pose,
)


class _Quaternion:
    def __init__(self, real: float, imaginary: tuple[float, float, float]) -> None:
        self._real = real
        self._imaginary = imaginary

    def GetReal(self) -> float:
        return self._real

    def GetImaginary(self) -> tuple[float, float, float]:
        return self._imaginary


class _PoseMatrix:
    def ExtractRotationQuat(self) -> _Quaternion:
        # Deliberately retain a small numeric drift so the regression also
        # proves explicit quaternion normalization.
        return _Quaternion(0.5000005, (0.5000005, 0.5000005, 0.5000005))


class _ScaledCoffeeMachineMatrix:
    def __init__(self) -> None:
        self.remove_scale_shear_calls = 0

    def ExtractTranslation(self) -> tuple[float, float, float]:
        return (0.63285, 3.14843, 1.37485)

    def ExtractRotationQuat(self) -> _Quaternion:
        # This is the invalid value observed when the scale-1.5 world matrix
        # was incorrectly treated as a pure rotation matrix.
        return _Quaternion(0.75, (0.75, 0.75, 0.5))

    def RemoveScaleShear(self) -> _PoseMatrix:
        self.remove_scale_shear_calls += 1
        return _PoseMatrix()


def test_entity_prim_pose_removes_scale_before_extracting_rotation() -> None:
    matrix = _ScaledCoffeeMachineMatrix()

    pose = _pose_from_world_matrix(matrix, (0.0, 0.0, 0.0))

    assert matrix.remove_scale_shear_calls == 1
    assert pose.position == pytest.approx((0.63285, 3.14843, 1.37485))
    assert pose.orientation_xyzw == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert math.sqrt(sum(component * component for component in pose.orientation_xyzw)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "quaternion",
    (
        _Quaternion(0.0, (0.0, 0.0, 0.0)),
        _Quaternion(float("nan"), (0.0, 0.0, 1.0)),
    ),
)
def test_entity_prim_pose_rejects_degenerate_rotation(quaternion: _Quaternion) -> None:
    with pytest.raises(ValueError, match="invalid rotation quaternion"):
        _normalized_quaternion_xyzw(quaternion)


def test_physical_root_retarget_preserves_the_entity_local_root_transform() -> None:
    source_entity = Pose((1.0, 2.0, 0.0))
    source_root = Pose((1.1, 2.0, 0.040822))
    target_entity = Pose((4.0, 5.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))

    target_root = _retarget_physical_root_pose(target_entity, source_entity, source_root)

    assert target_root.position == pytest.approx((4.0, 5.1, 0.040822))
    assert target_root.orientation_xyzw == pytest.approx(target_entity.orientation_xyzw)


@dataclass(frozen=True)
class _InitialState:
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]

    def replace(self, **changes: object) -> _InitialState:
        return replace(self, **changes)


def test_high_level_initial_state_targets_the_physical_root_not_the_entity_prim() -> None:
    path = EntityPath("/robots/g2")
    entity_pose = Pose((6.554862, 5.51, 0.0), (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)))
    root_pose = Pose((6.554862, 5.51, 0.040822), entity_pose.orientation_xyzw)
    world = object.__new__(IsaacLabNativeWorld)
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=2))
    world._read_usd_entity_prim_pose = lambda _path, _environment: entity_pose
    world._read_usd_prim_pose = lambda _prim_path, _environment: root_pose
    asset = SimpleNamespace(
        cfg=SimpleNamespace(
            init_state=_InitialState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        )
    )
    entity = SimpleNamespace(path=path, pose=entity_pose)

    world._configure_high_level_initial_root_pose(
        entity,
        asset,
        ("/World/env_0/g2/base_x_link", "/World/env_1/g2/base_x_link"),
    )

    assert asset.cfg.init_state.pos == pytest.approx((6.554862, 5.51, 0.040822))
    assert asset.cfg.init_state.rot == pytest.approx(entity_pose.orientation_xyzw)

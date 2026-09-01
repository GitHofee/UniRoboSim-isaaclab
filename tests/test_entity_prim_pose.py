from __future__ import annotations

import math

import pytest

from unirobosim_isaaclab.native import _normalized_quaternion_xyzw, _pose_from_world_matrix


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

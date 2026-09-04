from __future__ import annotations

from dataclasses import dataclass

import pytest

from unirobosim_isaaclab.physics_activation import (
    DynamicRigidBodyCandidate,
    PhysicsActivationController,
    _CollisionCandidate,
    _point_aabb_distance_squared,
)


@dataclass
class _Attribute:
    value: bool = True
    writes: int = 0

    def Set(self, value: bool) -> None:
        self.value = value
        self.writes += 1


class _RigidBodyView:
    def __init__(self, torch, position: tuple[float, float, float]) -> None:
        self._torch = torch
        self.transforms = torch.tensor([(*position, 0.0, 0.0, 0.0, 1.0)], dtype=torch.float32)
        self.disabled = torch.zeros((1, 1), dtype=torch.uint8)
        self.count = 1
        self.wake_count = 0

    def get_transforms(self):
        return self.transforms

    def get_disable_simulations(self):
        return self.disabled

    def set_disable_simulations(self, data, _indices) -> None:
        self.disabled.copy_(data)

    def wake_up(self, _indices) -> None:
        self.wake_count += 1


def test_point_aabb_distance_is_zero_inside_and_euclidean_outside() -> None:
    minimum = (1.0, 2.0, 3.0)
    maximum = (2.0, 4.0, 6.0)

    assert _point_aabb_distance_squared((1.5, 3.0, 4.0), minimum, maximum) == 0.0
    assert _point_aabb_distance_squared((4.0, 1.0, 3.5), minimum, maximum) == 5.0


def test_activation_uses_bounded_updates_hysteresis_and_diff_only_writes() -> None:
    attribute = _Attribute()
    anchors = [[(0.0, 0.0, 0.0)]]
    controller = PhysicsActivationController(
        candidates=(
            _CollisionCandidate(
                path="/World/env_0/scene/collider",
                collision_enabled=attribute,
                minimum=(2.5, -0.1, -0.1),
                maximum=(2.6, 0.1, 0.1),
            ),
        ),
        protected_count=4,
        radius_m=2.0,
        hysteresis_m=1.0,
        update_interval_steps=12,
        anchor_points=lambda: tuple(anchors[0]),
    )

    controller.update(0, force=True)
    assert attribute.value is False
    assert attribute.writes == 1

    anchors[0] = [(1.0, 0.0, 0.0)]
    controller.update(6)
    assert attribute.value is False
    assert attribute.writes == 1
    controller.update(12)
    assert attribute.value is True
    assert attribute.writes == 2

    anchors[0] = [(0.0, 0.0, 0.0)]
    controller.update(24)
    assert attribute.value is True
    assert attribute.writes == 2
    anchors[0] = [(-1.0, 0.0, 0.0)]
    controller.update(36)
    assert attribute.value is False
    assert attribute.writes == 3

    assert controller.diagnostics.candidate_count == 1
    assert controller.diagnostics.disabled_count == 1
    assert controller.diagnostics.protected_count == 4
    assert controller.diagnostics.update_count == 4


def test_far_rigid_body_freezes_then_near_or_commanded_body_stays_active() -> None:
    torch = pytest.importorskip("torch")
    anchors = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    view = _RigidBodyView(torch, (5.0, 0.0, 0.0))
    controller = PhysicsActivationController(
        candidates=(),
        protected_count=0,
        radius_m=1.5,
        hysteresis_m=0.5,
        update_interval_steps=1,
        anchor_points=lambda: anchors,
        torch_module=torch,
    )
    controller.configure_dynamic_candidates((DynamicRigidBodyCandidate("/objects/cup", view),))

    controller.update(0, force=True)
    assert view.disabled.tolist() == [[1]]
    assert controller.diagnostics.dynamic_disabled_count == 1

    view.transforms[0, :3] = torch.tensor((1.0, 0.0, 0.0))
    controller.update(1)
    assert view.disabled.tolist() == [[0]]
    assert view.wake_count == 1

    view.transforms[0, :3] = torch.tensor((8.0, 0.0, 0.0))
    controller.update(2)
    assert view.disabled.tolist() == [[0]]

    controller.reset_dynamic_state()
    controller.update(3, force=True)
    assert view.disabled.tolist() == [[1]]
    controller.pin("/objects/cup")
    assert view.disabled.tolist() == [[0]]
    assert view.wake_count == 2

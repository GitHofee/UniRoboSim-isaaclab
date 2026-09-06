"""Lightweight sibling-entry regression for high-level articulation reset."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from unirobosim import EntityPath

from unirobosim_isaaclab.native import IsaacLabNativeWorld


class _Tensor:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def __getitem__(self, indices: list[int]) -> _Tensor:
        return _Tensor([self.rows[index] for index in indices])


class _Asset:
    def __init__(self) -> None:
        self.data = SimpleNamespace(default_root_vel=SimpleNamespace(torch=_Tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        ])))
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def call(**kwargs: Any) -> None:
            self.calls.append((name, kwargs))
        return call


@pytest.mark.parametrize("environments", ((0,), (1,), (1, 0)))
def test_high_level_reset_writes_selected_default_root_velocity(environments: tuple[int, ...]) -> None:
    path = EntityPath("/robot")
    asset = _Asset()
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=SimpleNamespace(zeros_like=lambda value: value))
    world._articulations = {path: asset}
    world._initial_articulation = {path: (_Tensor([[0.0] * 7] * 2), _Tensor([[0.0]] * 2), _Tensor([[0.0]] * 2))}
    world._initial_articulation_gains = {path: (_Tensor([[1.0]] * 2), _Tensor([[1.0]] * 2))}
    world._articulation_control_modes = {path: [[None], [None]]}
    world._usd_articulation_views = {}
    world._rigids = {}
    world._usd_rigid_views = {}
    world._deformables = {}
    world._fluids = {}
    world._cameras = {}
    world._debug_lifetimes = {}
    world._sim = SimpleNamespace(forward=lambda: None)
    world._update_assets = lambda dt: None
    world._sync_all_mounted_cameras = lambda: None
    world._invalidate_render = lambda: None

    world.reset(environments)

    writes = [kwargs for name, kwargs in asset.calls if name == "write_root_velocity_to_sim_index"]
    assert len(writes) == 1
    assert writes[0]["env_ids"] == list(environments)
    assert writes[0]["root_velocity"].rows == [asset.data.default_root_vel.torch.rows[index] for index in environments]

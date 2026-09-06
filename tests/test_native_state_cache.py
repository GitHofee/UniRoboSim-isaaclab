"""Dependency/layout tests without importing any optional simulator package."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from unirobosim import EntityPath

from unirobosim_isaaclab.native import IsaacLabNativeWorld
from unirobosim_isaaclab.native_state import _ARTICULATION_STATE_BUFFERS, ArticulationStateCache


def _data() -> SimpleNamespace:
    data_type = type(
        "ArticulationData",
        (SimpleNamespace,),
        {
            "__module__": "isaaclab_physx.assets.articulation.articulation_data",
        },
    )
    buffer_type = type(
        "TimestampedBufferWarp",
        (SimpleNamespace,),
        {
            "__module__": "isaaclab.utils.buffers.timestamped_buffer_warp",
        },
    )
    return data_type(
        _sim_timestamp=0.25,
        **{name: buffer_type(timestamp=0.25, data=object()) for name in _ARTICULATION_STATE_BUFFERS},
    )


def test_invalidation_is_lazy_and_does_not_touch_data_time_or_excluded_state() -> None:
    data = _data()
    data._joint_acc = SimpleNamespace(timestamp=0.25, data=object())
    data._body_com_acc_w = SimpleNamespace(timestamp=0.25, data=object())
    data.default_root_vel = object()
    original = {name: getattr(data, name).data for name in _ARTICULATION_STATE_BUFFERS}
    cache = ArticulationStateCache(data)
    assert all(getattr(data, name).timestamp == 0.25 for name in original)

    cache.invalidate()

    assert data._sim_timestamp == 0.25
    assert data._joint_acc.timestamp == data._body_com_acc_w.timestamp == 0.25
    assert all(getattr(data, name).timestamp == -1.0 for name in original)
    assert all(getattr(data, name).data is payload for name, payload in original.items())


@pytest.mark.parametrize("name", _ARTICULATION_STATE_BUFFERS)
def test_missing_dependency_fails_closed(name: str) -> None:
    data = _data()
    delattr(data, name)
    with pytest.raises(RuntimeError, match=name):
        ArticulationStateCache(data)


@pytest.mark.parametrize("value", (None, True, "0", float("nan"), float("inf")))
def test_malformed_buffer_timestamp_fails_closed(value: object) -> None:
    data = _data()
    data._body_link_vel_w.timestamp = value
    with pytest.raises(RuntimeError, match="_body_link_vel_w"):
        ArticulationStateCache(data)


@pytest.mark.parametrize("value", (-1.0, None, True, float("nan"), float("inf")))
def test_malformed_simulation_timestamp_fails_closed(value: object) -> None:
    data = _data()
    data._sim_timestamp = value
    with pytest.raises(RuntimeError, match="simulation timestamp"):
        ArticulationStateCache(data)


def test_unknown_data_or_buffer_layout_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="cache layout"):
        ArticulationStateCache(SimpleNamespace())
    data = _data()
    data._root_link_vel_w = SimpleNamespace(timestamp=0.25, data=object())
    with pytest.raises(RuntimeError, match="_root_link_vel_w"):
        ArticulationStateCache(data)


def test_build_prepares_every_cache_before_any_initial_state_write() -> None:
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=object())
    world._sim = object()
    world._articulations = {
        EntityPath("/valid"): SimpleNamespace(data=_data()),
        EntityPath("/unknown"): SimpleNamespace(data=SimpleNamespace()),
    }
    # The valid asset has no joint metadata/writers: reaching initialization of
    # even that first asset would fail. Unknown layout must be rejected first.
    with pytest.raises(RuntimeError, match="cache layout"):
        world._initialize_articulations()


@pytest.mark.parametrize("dt", (0.0, 1 / 120))
def test_only_same_tick_asset_updates_invalidate_state(dt: float) -> None:
    calls = []
    data = _data()
    world = object.__new__(IsaacLabNativeWorld)
    path = EntityPath("/robot")
    world._articulations = {path: SimpleNamespace(update=lambda value: calls.append(value))}
    world._articulation_state_caches = {path: ArticulationStateCache(data)}
    world._rigids = world._contacts = world._deformables = {}

    world._update_assets(dt)

    assert calls == [dt]
    assert data._sim_timestamp == 0.25
    assert all(getattr(data, name).timestamp == (-1.0 if dt == 0.0 else 0.25) for name in _ARTICULATION_STATE_BUFFERS)

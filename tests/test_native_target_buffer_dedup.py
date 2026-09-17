"""Persistent target buffer writes must not repeat unchanged CPU commands."""

from types import SimpleNamespace

import pytest
from unirobosim import CommandMode

from unirobosim_isaaclab import IsaacLabAdapterConfig

from .test_articulation_drive_gains import _Torch, _world


def _instrument():
    world, path, asset = _world(IsaacLabAdapterConfig())
    calls = []
    asset.target_state_revision = 0

    class CountingTorch(_Torch):
        @staticmethod
        def tensor(rows, **kwargs):
            calls.append("allocate_target")
            return _Torch.tensor(rows, **kwargs)

        @staticmethod
        def zeros_like(target):
            calls.append("allocate_zero")
            return _Torch.zeros_like(target)

    world._m = SimpleNamespace(torch=CountingTorch)
    for name in ("position", "velocity", "effort"):

        def setter(*, target, joint_ids, env_ids, channel=name):
            calls.append((channel, target.rows, joint_ids, env_ids))
            asset.target_state_revision += 1

        setattr(asset, f"set_joint_{name}_target_index", setter)
    return world, path, asset, calls


@pytest.mark.parametrize("mode", [CommandMode.POSITION, CommandMode.VELOCITY])
def test_unchanged_hold_does_not_allocate_or_write_target_buffers(mode):
    world, path, _, calls = _instrument()
    world.apply_articulation(path, mode, ((0.1, 0.2),), (0,), (0, 2))
    calls.clear()
    world.apply_articulation(path, mode, ((0.2, 0.1),), (0,), (2, 0))
    assert calls == []


@pytest.mark.parametrize("mode,channel", [(CommandMode.POSITION, "position"), (CommandMode.VELOCITY, "velocity")])
def test_changed_primary_target_preserves_known_zero_buffers(mode, channel):
    world, path, _, calls = _instrument()
    world.apply_articulation(path, mode, ((0.1,),), (0,), (0,))
    calls.clear()
    world.apply_articulation(path, mode, ((0.2,),), (0,), (0,))
    assert calls == ["allocate_target", (channel, [[0.2]], [0], [0])]


def test_mode_switch_and_state_write_invalidation_restore_full_buffers():
    world, path, asset, calls = _instrument()
    world.apply_articulation(path, CommandMode.EFFORT, ((0.1,),), (0,), (0,))
    calls.clear()
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    assert [call[0] for call in calls if isinstance(call, tuple)] == ["position", "velocity", "effort"]
    assert len(asset.stiffness_writes) == len(asset.damping_writes) == 2
    world._joint_target_versions[path].invalidate()
    calls.clear()
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    assert [call[0] for call in calls if isinstance(call, tuple)] == ["position", "velocity", "effort"]


def test_effort_commands_continue_writing_each_time():
    world, path, _, calls = _instrument()
    world.apply_articulation(path, CommandMode.EFFORT, ((0.1,),), (0,), (0,))
    calls.clear()
    world.apply_articulation(path, CommandMode.EFFORT, ((0.1,),), (0,), (0,))
    assert calls == ["allocate_target", ("effort", [[0.1]], [0], [0])]


def test_failed_setter_does_not_leave_a_trusted_previous_buffer_value():
    world, path, asset, calls = _instrument()
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    normal_setter = asset.set_joint_position_target_index

    def fail_after_write(**kwargs):
        normal_setter(**kwargs)
        raise RuntimeError("native write failed")

    asset.set_joint_position_target_index = fail_after_write
    with pytest.raises(RuntimeError, match="native write failed"):
        world.apply_articulation(path, CommandMode.POSITION, ((0.2,),), (0,), (0,))
    asset.set_joint_position_target_index = normal_setter
    calls.clear()
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    assert [call[0] for call in calls if isinstance(call, tuple)] == ["position", "velocity", "effort"]


@pytest.mark.parametrize("channel", ["position", "velocity", "effort"])
def test_direct_native_target_mutation_invalidates_cached_modes_and_values(channel):
    world, path, asset, calls = _instrument()
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    getattr(asset, f"set_joint_{channel}_target_index")(
        target=_Torch.tensor(((0.9,),)),
        joint_ids=[0],
        env_ids=[0],
    )
    calls.clear()
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    assert [call[0] for call in calls if isinstance(call, tuple)] == ["position", "velocity", "effort"]
    assert len(asset.stiffness_writes) == 2


def test_legacy_sdk_never_skips_persistent_setters_without_mutation_revision():
    world, path, asset, calls = _instrument()
    del asset.target_state_revision
    for name in ("position", "velocity", "effort"):
        setattr(asset, f"set_joint_{name}_target_index", lambda *, channel=name, **kw: calls.append((channel,)))
    for _ in range(2):
        calls.clear()
        world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
        assert [call[0] for call in calls if isinstance(call, tuple)] == ["position", "velocity", "effort"]


def test_environment_target_ownership_does_not_confuse_identical_axes():
    world, path, asset, calls = _instrument()
    world._articulation_control_modes[path].append([None] * 3)
    world._initial_articulation_gains[path] = (_Torch.tensor(((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))),) * 2
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    world.apply_articulation(path, CommandMode.POSITION, ((0.2,),), (1,), (0,))
    calls.clear()
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (1,), (0,))
    assert any(isinstance(call, tuple) and call[0] == "position" for call in calls)

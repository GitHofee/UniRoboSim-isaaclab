"""Target ownership, public optional dispatch and CPU-side joint ordering."""

from types import SimpleNamespace

import pytest
from unirobosim import CommandMode, EntityPath

from unirobosim_isaaclab import IsaacLabAdapterConfig
from unirobosim_isaaclab.native import IsaacLabNativeWorld
from unirobosim_isaaclab.native_targets import JointTargetVersions

from .test_articulation_drive_gains import _world
from .test_native_rigid_read_batch import _Tensor


def test_versions_follow_physical_axes_without_changing_for_reordered_commands():
    versions = JointTargetVersions()
    versions.update("position", ((1.0, 2.0),), (0,), (1, 3))
    assert (versions.position, versions.velocity) == (1, 0)
    versions.update("position", ((2.0, 1.0),), (0,), (3, 1))
    assert versions.position == 1
    versions.update("position", ((2.0,),), (1,), (3,))
    assert versions.position == 2
    versions.invalidate()
    assert (versions.position, versions.velocity) == (3, 1)
    versions.update("position", ((2.0,),), (1,), (3,))
    assert versions.position == 4
    with pytest.raises(ValueError):
        versions.update("effort", ((0.0,),), (0,), (0,))


def test_same_mode_zeros_keep_velocity_version_and_mode_switch_invalidates():
    world, path, asset = _world(IsaacLabAdapterConfig())
    asset.target_state_revision = 0
    world.apply_articulation(path, CommandMode.POSITION, ((0.1,),), (0,), (0,))
    versions = world._joint_target_versions[path]
    before = (versions.position, versions.velocity)
    world.apply_articulation(path, CommandMode.POSITION, ((0.2,),), (0,), (0,))
    assert versions.position == before[0] + 1
    assert versions.velocity == before[1]
    world.apply_articulation(path, CommandMode.EFFORT, ((0.2,),), (0,), (0,))
    assert versions.position == before[0] + 2
    assert versions.velocity == before[1] + 1


def test_public_versioned_writer_and_original_profile_dispatch():
    world = object.__new__(IsaacLabNativeWorld)
    world._joint_target_versions = {}
    world._versioned_target_writers = {}
    calls = []

    class Updated:
        target_state_revision = 0

        def write_data_to_sim(self, *, position_target_version=None, velocity_target_version=None):
            calls.append((position_target_version, velocity_target_version))

    class Original:
        def write_data_to_sim(self):
            calls.append("original")

    world._write_articulation_data(EntityPath("/new"), Updated())
    world._write_articulation_data(EntityPath("/old"), Original())
    assert calls == [(1, 1), "original"]


@pytest.mark.parametrize("raw", [True, False])
def test_joint_reads_transfer_once_per_field_then_reorder_on_cpu(raw):
    transfers = []
    position = _Tensor([[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]], transfers, "position")
    velocity = _Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], transfers, "velocity")
    path = EntityPath("/robot")
    world = object.__new__(IsaacLabNativeWorld)
    world._joint_maps = {path: (2, 0)}
    world._usd_articulation_views = (
        {
            path: SimpleNamespace(
                get_dof_positions=lambda: position,
                get_dof_velocities=lambda: velocity,
            )
        }
        if raw
        else {}
    )
    world._articulations = (
        {}
        if raw
        else {
            path: SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=SimpleNamespace(torch=position),
                    joint_vel=SimpleNamespace(torch=velocity),
                )
            )
        }
    )
    assert world.read_articulation(path) == (((33.0, 11.0), (66.0, 44.0)), ((3.0, 1.0), (6.0, 4.0)))
    assert transfers == ["position", "velocity"]
    position.values[0][2] = 77.0
    assert world.read_articulation(path)[0][0] == (77.0, 11.0)


def test_external_revision_before_substep_forces_new_versions_and_failed_writer_invalidates():
    world = object.__new__(IsaacLabNativeWorld)
    path = EntityPath("/robot")
    world._articulation_control_modes = {path: [[CommandMode.POSITION]]}
    calls = []

    class Asset:
        target_state_revision = 0
        fail = False

        def write_data_to_sim(self, *, position_target_version=None, velocity_target_version=None):
            calls.append((position_target_version, velocity_target_version))
            if self.fail:
                raise RuntimeError("partial native write")

    asset = Asset()
    world._write_articulation_data(path, asset)
    world._write_articulation_data(path, asset)
    assert calls[0] == calls[1]
    asset.target_state_revision += 1
    world._write_articulation_data(path, asset)
    assert calls[-1] != calls[0]
    asset.fail = True
    with pytest.raises(RuntimeError, match="partial native write"):
        world._write_articulation_data(path, asset)
    failed = calls[-1]
    asset.fail = False
    world._write_articulation_data(path, asset)
    assert calls[-1] != failed
    assert world._articulation_control_modes[path] == [[None]]


def test_signature_negotiation_is_once_and_noninspectable_or_positional_only_falls_back(monkeypatch):
    from unirobosim_isaaclab import native

    world = object.__new__(IsaacLabNativeWorld)
    calls = []
    signatures = []
    inspect_signature = native.inspect.signature

    class Asset:
        target_state_revision = 0

        def write_data_to_sim(self, position_target_version=None, velocity_target_version=None, /):
            calls.append((position_target_version, velocity_target_version))

    def inspect(method):
        signatures.append(method)
        return inspect_signature(method)

    monkeypatch.setattr(native.inspect, "signature", inspect)
    asset = Asset()
    world._write_articulation_data(EntityPath("/robot"), asset)
    world._write_articulation_data(EntityPath("/robot"), asset)
    assert len(signatures) == 1 and calls == [(None, None), (None, None)]

    def unavailable(method):
        raise ValueError("native callable has no signature")

    monkeypatch.setattr(native.inspect, "signature", unavailable)
    world._write_articulation_data(EntityPath("/other"), asset)
    assert calls[-1] == (None, None)

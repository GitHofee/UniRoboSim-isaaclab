"""Explicit subprocess-only numeric/call-site gates; not pytest auto-discovery.

Invoked by test_checkpoint_velocity_frame.py. Importing torch in the parent
pytest collection would invalidate existing lightweight SDK-isolation gates.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from unirobosim import EntityPath

from unirobosim_isaaclab.native import IsaacLabNativeWorld
from unirobosim_isaaclab.native_velocity import root_com_velocity_from_link

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("translation", (0.0, 1e3, 1e5))
@pytest.mark.parametrize("order", ((0, 1), (1, 0)))
def test_root_com_conversion_uses_saved_orientation_local_offset_and_selected_order(dtype, translation, order):
    poses = torch.tensor(
        [[translation, 2 * translation, -translation, 0, 0, math.sqrt(0.5), math.sqrt(0.5)], [0, 0, 0, 0, 0, 0, 1]],
        dtype=dtype,
    )[list(order)]
    velocity = torch.tensor([[0.4, -0.3, 0.2, 0.2, -0.1, 0.7], [0.1, 0.2, 0.3, -0.4, 0.5, -0.6]], dtype=dtype)[
        list(order)
    ]
    offset = torch.tensor([[0.03, 0.02, 0.01], [0.02, -0.04, 0.05]], dtype=dtype)[list(order)]
    originals = tuple(value.clone() for value in (poses, velocity, offset))

    actual = root_com_velocity_from_link(torch, poses, velocity, offset)

    expected = torch.tensor(
        [[0.378, -0.316, 0.204, 0.2, -0.1, 0.7], [0.101, 0.208, 0.306, -0.4, 0.5, -0.6]], dtype=dtype
    )[list(order)]
    assert actual.dtype == dtype and actual.device == velocity.device
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    assert all(
        torch.equal(actual_input, saved)
        for actual_input, saved in zip((poses, velocity, offset), originals, strict=True)
    )


@pytest.mark.parametrize("zero_offset", (False, True))
def test_zero_angular_or_zero_com_offset_needs_no_linear_shift(zero_offset):
    pose = torch.tensor([[0.0, 0, 0, 0, 0, math.sqrt(0.5), math.sqrt(0.5)]])
    velocity = torch.tensor([[0.4, -0.3, 0.2, 0.2, -0.1, 0.7]])
    offset = torch.tensor([[0.03, 0.02, 0.01]])
    if zero_offset:
        offset.zero_()
    else:
        velocity[:, 3:] = 0
    assert torch.equal(root_com_velocity_from_link(torch, pose, velocity, offset), velocity)


@pytest.mark.parametrize(
    "bad_offset", (torch.zeros(1, 4), torch.tensor([[float("nan"), 0, 0]]), torch.zeros(1, 3, dtype=torch.float64))
)
def test_malformed_native_com_is_rejected_before_any_write(bad_offset):
    with pytest.raises(ValueError, match="root COM conversion"):
        root_com_velocity_from_link(torch, torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]]), torch.zeros(1, 6), bad_offset)


@pytest.mark.parametrize("intermediate", (False, True))
def test_finite_extreme_inputs_reject_computation_overflow(intermediate):
    pose = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]])
    velocity = torch.tensor([[0.0, 0, 0, 0, 0, 2e38]])
    offset = torch.tensor([[2.0, 0, 0]])
    if intermediate:
        pose[:, 5:7] = math.sqrt(0.5)
        velocity.zero_()
        offset[:, 0] = 2.6e38
    assert all(bool(torch.isfinite(value).all()) for value in (pose, velocity, offset))
    with pytest.raises(ValueError, match="non-finite COM velocity"):
        root_com_velocity_from_link(torch, pose, velocity, offset)


def _checkpoint_world():
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=torch)
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=2))
    world._usd_articulation_views = world._rigids = world._usd_rigid_views = {}
    world._deformables = world._fluids = {}
    world._composite_rigid_states = world._composite_articulation_states = []
    world._runtime_attachments = {}
    path = EntityPath("/robot")
    poses = torch.tensor([[0.0, 0, 0, 0, 0, math.sqrt(0.5), math.sqrt(0.5)], [0, 0, 0, 0, 0, 0, 1]])
    velocities = torch.tensor([[0.4, -0.3, 0.2, 0.2, -0.1, 0.7], [0.1, 0.2, 0.3, -0.4, 0.5, -0.6]])
    references = {"root_pose": poses, "root_velocity": velocities}
    fields = {
        "root_pose": "root_link_pose_w",
        "root_velocity": "root_link_vel_w",
        "joint_position": "joint_pos",
        "joint_velocity": "joint_vel",
        "position_target": "joint_pos_target",
        "velocity_target": "joint_vel_target",
        "effort_target": "joint_effort_target",
        "stiffness": "joint_stiffness",
        "damping": "joint_damping",
    }
    references.update({key: torch.zeros(2, 1) for key in fields if key not in references})
    data = SimpleNamespace(**{name: SimpleNamespace(torch=references[key]) for key, name in fields.items()})
    data.body_com_pose_b = SimpleNamespace(
        torch=torch.tensor([[[0.03, 0.02, 0.01, 0, 0, 0, 1]], [[0.02, -0.04, 0.05, 0, 0, 0, 1]]])
    )
    calls = []
    asset = SimpleNamespace(
        data=data,
        reset=lambda **kw: calls.append(("reset", kw)),
        write_root_pose_to_sim_index=lambda **kw: calls.append(("pose", kw)),
        write_root_com_velocity_to_sim_index=lambda **kw: calls.append(("com", kw)),
    )
    world._articulations = {path: asset}
    world._articulation_control_modes = {path: [[None], [None]]}
    record = {key: value.tolist() for key, value in references.items()}
    record.update(kind="isaaclab", control_modes=[[None], [None]])
    state = {
        "schema": "nvidia.isaaclab.native-state/1",
        "articulations": {path.value: record},
        "rigids": {},
        "deformables": {},
        "fluids": {},
        "composite_rigids": [],
        "composite_articulations": [],
        "attachments": [],
    }
    return world, asset, state, calls


def test_staging_preserves_link_payload_and_application_uses_com_writer():
    world, asset, state, calls = _checkpoint_world()
    staged = world._stage_checkpoint(state)
    record = staged["articulations"][EntityPath("/robot")]
    assert calls == []
    assert torch.equal(record["root_velocity"], asset.data.root_link_vel_w.torch)
    assert torch.allclose(
        record["root_com_velocity"],
        torch.tensor([[0.378, -0.316, 0.204, 0.2, -0.1, 0.7], [0.101, 0.208, 0.306, -0.4, 0.5, -0.6]]),
        atol=1e-7,
        rtol=0,
    )
    assert "root_com_velocity" not in state["articulations"]["/robot"]
    world._m.sim_utils = SimpleNamespace(get_current_stage=lambda: object())

    # Stop immediately after COM write so this test isolates the call site, not
    # the unrelated joint/rigid/checkpoint application implementation.
    class AfterVelocity(Exception):
        pass

    def stop(**_kwargs):
        raise AfterVelocity

    asset.write_joint_position_to_sim_index = stop
    with pytest.raises(AfterVelocity):
        world._apply_staged_checkpoint(staged)
    assert [call[0] for call in calls] == ["reset", "pose", "com"]
    assert calls[-1][1]["env_ids"] == [0, 1]
    assert torch.equal(calls[-1][1]["root_velocity"], record["root_com_velocity"])


@pytest.mark.parametrize(
    "fault",
    (
        "writer",
        "missing_com",
        "shape",
        "environments",
        "empty_bodies",
        "nan",
        "dtype",
        "final_overflow",
        "intermediate_overflow",
    ),
)
def test_native_capability_failure_precedes_all_checkpoint_writes(fault):
    world, asset, state, calls = _checkpoint_world()
    _first_world, first_asset, first_state, first_calls = _checkpoint_world()
    first_path = EntityPath("/first_valid")
    world._articulations = {first_path: first_asset, **world._articulations}
    world._articulation_control_modes[first_path] = [[None], [None]]
    state["articulations"][first_path.value] = first_state["articulations"]["/robot"]
    if fault == "writer":
        asset.write_root_com_velocity_to_sim_index = None
    elif fault == "missing_com":
        del asset.data.body_com_pose_b
    elif fault in {"final_overflow", "intermediate_overflow"}:
        asset.data.body_com_pose_b.torch[0, 0, :3] = torch.tensor([2.0, 0, 0])
        record = state["articulations"]["/robot"]
        record["root_velocity"][0] = [0.0, 0, 0, 0, 0, 2e38]
        if fault == "intermediate_overflow":
            record["root_velocity"][0] = [0.0] * 6
            asset.data.body_com_pose_b.torch[0, 0, 0] = 2.6e38
    else:
        value = {
            "shape": torch.zeros(2, 7),
            "environments": torch.zeros(1, 1, 7),
            "empty_bodies": torch.zeros(2, 0, 7),
            "nan": torch.full((2, 1, 7), float("nan")),
            "dtype": torch.zeros(2, 1, 7, dtype=torch.float64),
        }[fault]
        asset.data.body_com_pose_b.torch = value
    with pytest.raises((RuntimeError, ValueError), match="COM|local COM"):
        world.restore_checkpoint(state)
    assert calls == first_calls == []

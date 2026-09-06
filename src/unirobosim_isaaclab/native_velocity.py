"""Batched frame-point conversion without importing optional tensor packages."""

from __future__ import annotations

from typing import Any


def root_com_velocity_from_link(torch: Any, root_pose: Any, link_velocity: Any, local_com_position: Any) -> Any:
    """Convert saved world link twist to world COM twist without world subtraction.

    Used while staging a checkpoint, before any physical write. Saved pose and
    velocity are already finite-validated by the checkpoint parser; validate the
    additional native local COM input here. Inputs remain unchanged.
    """
    if not all(torch.is_tensor(value) for value in (root_pose, link_velocity, local_com_position)):
        raise ValueError("root COM conversion requires native tensors")
    if len(link_velocity.shape) != 2 or link_velocity.shape[1] != 6:
        raise ValueError("root COM conversion requires [environment,6] link velocity")
    count = link_velocity.shape[0]
    if tuple(root_pose.shape) != (count, 7) or tuple(local_com_position.shape) != (count, 3):
        raise ValueError("root COM conversion pose/local COM shape differs")
    if any(
        value.dtype != link_velocity.dtype or value.device != link_velocity.device
        for value in (root_pose, local_com_position)
    ):
        raise ValueError("root COM conversion tensor dtype/device differs")
    if not link_velocity.is_floating_point() or not bool(torch.isfinite(local_com_position).all()):
        raise ValueError("root COM conversion requires finite floating local COM values")
    quaternion_vector = root_pose[:, 3:6]
    twice_cross = 2.0 * torch.cross(quaternion_vector, local_com_position, dim=-1)
    world_offset = (
        local_com_position + root_pose[:, 6:7] * twice_cross + torch.cross(quaternion_vector, twice_cross, dim=-1)
    )
    velocity = link_velocity.clone()
    velocity[:, :3] += torch.cross(link_velocity[:, 3:], world_offset, dim=-1)
    if not bool(torch.isfinite(velocity).all()):
        raise ValueError("root COM conversion produced non-finite COM velocity")
    return velocity

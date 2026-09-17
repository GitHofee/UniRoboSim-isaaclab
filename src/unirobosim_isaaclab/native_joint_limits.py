"""Publish effective native drive limits rather than authored planning hints."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from unirobosim import EntityKind, PlanningJointDescriptor

from .native_protocols import NativePlanningError


def effective_joint_drive_limits(
    world: Any,
    spec: Any,
    binding: Any,
    joints: tuple[PlanningJointDescriptor, ...],
) -> tuple[PlanningJointDescriptor, ...]:
    """Read immutable admission limits once, with exact physical DOF ownership."""

    movable = tuple(item for item in binding.joint_bindings if item.movable_name is not None)
    if spec.kind is not EntityKind.ARTICULATION or not movable:
        return joints
    asset = world._articulations.get(spec.path)
    if asset is not None:
        names = tuple(asset.joint_names)
        indices = {name: index for index, name in enumerate(names)}
        if len(indices) != len(names):
            raise NativePlanningError("catalog_invalid")
        velocities = asset.data.joint_vel_limits.torch.detach().cpu().tolist()
        efforts = asset.data.joint_effort_limits.torch.detach().cpu().tolist()
    else:
        view = world._usd_articulation_views.get(spec.path)
        if view is None:
            raise NativePlanningError("native_failure")
        indices = dict(zip(spec.joint_names, world._joint_maps[spec.path], strict=True))
        velocities = view.get_dof_max_velocities().detach().cpu().tolist()
        efforts = view.get_dof_max_forces().detach().cpu().tolist()
    count = world._spec.environments.count
    if len(velocities) != count or len(efforts) != count:
        raise NativePlanningError("catalog_invalid")
    widths = {len(row) for row in (*velocities, *efforts)}
    if len(widths) != 1 or any(type(index) is not int or index < 0 for index in indices.values()):
        raise NativePlanningError("catalog_invalid")
    if len(indices) != len(set(indices.values())):
        raise NativePlanningError("catalog_invalid")
    updated = {}
    for item in movable:
        index = indices.get(item.movable_name)
        if index is None:
            raise NativePlanningError("catalog_invalid")
        limits = []
        for rows in (velocities, efforts):
            if any(index >= len(row) for row in rows):
                raise NativePlanningError("catalog_invalid")
            values = tuple(float(row[index]) for row in rows)
            if any(math.isnan(value) or value < 0.0 for value in values):
                raise NativePlanningError("catalog_invalid")
            if any(value != values[0] for value in values[1:]):
                raise NativePlanningError("catalog_invalid")
            # The public contract represents positive finite limits; zero passive
            # effort and native unbounded values do not assert a positive cap.
            limits.append(values[0] if 0.0 < values[0] < math.inf else None)
        updated[item.descriptor.joint_id] = replace(item.descriptor, max_velocity=limits[0], max_effort=limits[1])
    return tuple(updated.get(joint.joint_id, joint) for joint in joints)

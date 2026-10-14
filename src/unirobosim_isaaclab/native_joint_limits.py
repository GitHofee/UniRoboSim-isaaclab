"""Publish effective native joint limits rather than authored planning hints."""

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
        positions = asset.data.joint_pos_limits.torch.detach().cpu().tolist()
        velocities = asset.data.joint_vel_limits.torch.detach().cpu().tolist()
        efforts = asset.data.joint_effort_limits.torch.detach().cpu().tolist()
    else:
        view = world._usd_articulation_views.get(spec.path)
        if view is None:
            raise NativePlanningError("native_failure")
        names = tuple(spec.joint_names)
        mapped = world._joint_maps.get(spec.path)
        if mapped is None or len(names) != len(mapped) or len(set(names)) != len(names):
            raise NativePlanningError("catalog_invalid")
        indices = dict(zip(names, mapped, strict=True))
        positions = view.get_dof_limits().detach().cpu().tolist()
        velocities = view.get_dof_max_velocities().detach().cpu().tolist()
        efforts = view.get_dof_max_forces().detach().cpu().tolist()
    count = world._spec.environments.count
    if any(not isinstance(rows, (tuple, list)) or len(rows) != count for rows in (positions, velocities, efforts)):
        raise NativePlanningError("catalog_invalid")
    if any(not isinstance(row, (tuple, list)) for rows in (positions, velocities, efforts) for row in rows):
        raise NativePlanningError("catalog_invalid")
    widths = {len(row) for row in (*positions, *velocities, *efforts)}
    if len(widths) != 1 or any(type(index) is not int or index < 0 for index in indices.values()):
        raise NativePlanningError("catalog_invalid")
    if len(indices) != len(set(indices.values())):
        raise NativePlanningError("catalog_invalid")
    updated = {}
    for item in movable:
        index = indices.get(item.movable_name)
        if index is None:
            raise NativePlanningError("catalog_invalid")
        lower, upper = _position_bounds(positions, index)
        limits = []
        for rows in (velocities, efforts):
            if any(index >= len(row) for row in rows):
                raise NativePlanningError("catalog_invalid")
            values = tuple(_native_number(row[index]) for row in rows)
            if any(math.isnan(value) or value < 0.0 for value in values):
                raise NativePlanningError("catalog_invalid")
            if any(value != values[0] for value in values[1:]):
                raise NativePlanningError("catalog_invalid")
            # The public contract represents positive finite limits; zero passive
            # effort and native unbounded values do not assert a positive cap.
            limits.append(values[0] if 0.0 < values[0] < math.inf else None)
        updated[item.descriptor.joint_id] = replace(
            item.descriptor, lower=lower, upper=upper, max_velocity=limits[0], max_effort=limits[1]
        )
    return tuple(updated.get(joint.joint_id, joint) for joint in joints)


def _position_bounds(rows: Any, index: int) -> tuple[float | None, float | None]:
    """Read SDK SI values exactly; the SDK owns degree conversion and precision."""
    if any(index >= len(row) for row in rows):
        raise NativePlanningError("catalog_invalid")
    pairs = tuple(row[index] for row in rows)
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in pairs):
        raise NativePlanningError("catalog_invalid")
    pairs = tuple(tuple(_native_number(value) for value in pair) for pair in pairs)
    if not pairs or any(pair != pairs[0] for pair in pairs[1:]):
        raise NativePlanningError("catalog_invalid")
    lower, upper = (float(value) for value in pairs[0])
    if lower == -math.inf and upper == math.inf:
        return None, None
    # The public descriptor supports two finite endpoints or a fully unbounded
    # joint. Never erase a remaining finite endpoint on a one-sided interval.
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise NativePlanningError("catalog_invalid")
    return lower, upper


def _native_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativePlanningError("catalog_invalid")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise NativePlanningError("catalog_invalid") from exc
    if math.isnan(result):
        raise NativePlanningError("catalog_invalid")
    return result

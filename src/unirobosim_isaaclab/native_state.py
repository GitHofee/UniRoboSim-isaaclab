"""Narrow, lazy state-cache compatibility for the verified Isaac Lab 3 layout.

The 3.0.0-beta2 PhysX articulation writers omit parts of the timestamp dependency
closure. This adapter-local shim never changes physics, data or simulation time.
It is prepared before state writes and runs only after same-tick mutations.
"""

from __future__ import annotations

import math
from typing import Any

# Root/body pose and velocity sources, their COM/link transforms, deprecated
# aggregate states, and root-orientation-derived directions/velocities.
_ARTICULATION_STATE_BUFFERS = (
    "_root_link_pose_w",
    "_root_com_pose_w",
    "_root_com_vel_w",
    "_root_link_vel_w",
    "_body_link_pose_w",
    "_body_com_pose_w",
    "_body_com_vel_w",
    "_body_link_vel_w",
    "_root_state_w",
    "_root_link_state_w",
    "_root_com_state_w",
    "_body_state_w",
    "_body_link_state_w",
    "_body_com_state_w",
    "_projected_gravity_b",
    "_heading_w",
    "_root_link_lin_vel_b",
    "_root_link_ang_vel_b",
    "_root_com_lin_vel_b",
    "_root_com_ang_vel_b",
)
# Exclusions: local COM calibration and defaults do not change in these writes;
# joint acceleration has a finite-difference timebase that must not be touched;
# body acceleration is explicitly zeroed by the velocity writer. Joint writers
# already invalidate task-space Jacobian/mass/gravity buffers as appropriate.


class ArticulationStateCache:
    """Pre-validated timestamp-buffer references; no per-read/step introspection."""

    def __init__(self, data: Any) -> None:
        layout = (type(data).__module__, type(data).__name__)
        if layout != ("isaaclab_physx.assets.articulation.articulation_data", "ArticulationData"):
            raise RuntimeError(f"unsupported Isaac articulation same-tick cache layout: {layout!r}")
        timestamp = getattr(data, "_sim_timestamp", None)
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0.0
        ):
            raise RuntimeError("unsupported Isaac articulation simulation timestamp")
        buffers: list[Any] = []
        for name in _ARTICULATION_STATE_BUFFERS:
            buffer = getattr(data, name, None)
            layout = (type(buffer).__module__, type(buffer).__name__)
            timestamp = getattr(buffer, "timestamp", None)
            if (
                layout != ("isaaclab.utils.buffers.timestamped_buffer_warp", "TimestampedBufferWarp")
                or isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or getattr(buffer, "data", None) is None
            ):
                raise RuntimeError(f"unsupported Isaac articulation same-tick cache buffer: {name}")
            buffers.append(buffer)
        self._buffers = tuple(buffers)

    def invalidate(self) -> None:
        for buffer in self._buffers:
            buffer.timestamp = -1.0

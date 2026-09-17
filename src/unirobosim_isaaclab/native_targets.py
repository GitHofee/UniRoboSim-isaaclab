"""CPU-side authority versions for joint targets written through the adapter."""

from __future__ import annotations


class JointTargetVersions:
    """Track physical axes independently of public command grouping or order."""

    def __init__(self) -> None:
        self.native_revision: int | None = None
        self.position = 0
        self.velocity = 0
        self._values: dict[tuple[str, int, int], float] = {}

    def invalidate(self) -> None:
        self.native_revision = None
        self._values.clear()
        self.position += 1
        self.velocity += 1

    def contains(
        self,
        channel: str,
        environments: tuple[int, ...],
        joints: tuple[int, ...],
    ) -> bool:
        """Whether all selected axes have a successfully established input."""
        if channel not in {"position", "velocity"}:
            raise ValueError("unsupported joint target channel")
        return all((channel, environment, joint) in self._values for environment in environments for joint in joints)

    def matches(
        self,
        channel: str,
        values: tuple[tuple[float, ...], ...],
        environments: tuple[int, ...],
        joints: tuple[int, ...],
    ) -> bool:
        """Compare commands without committing them or reading device tensors."""
        if channel not in {"position", "velocity"}:
            raise ValueError("unsupported joint target channel")
        matched = True
        for environment, row in zip(environments, values, strict=True):
            for joint, value in zip(joints, row, strict=True):
                matched = (
                    (channel, environment, joint) in self._values
                    and self._values[(channel, environment, joint)] == value
                ) and matched
        return matched

    def update(
        self,
        channel: str,
        values: tuple[tuple[float, ...], ...],
        environments: tuple[int, ...],
        joints: tuple[int, ...],
    ) -> None:
        if channel not in {"position", "velocity"}:
            raise ValueError("unsupported joint target channel")
        changed = False
        for environment, row in zip(environments, values, strict=True):
            for joint, value in zip(joints, row, strict=True):
                key = (channel, environment, joint)
                changed = changed or self._values.get(key) != value
                self._values[key] = value
        if changed:
            if channel == "position":
                self.position += 1
            else:
                self.velocity += 1

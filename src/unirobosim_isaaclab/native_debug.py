"""SDK-independent state bridge for an Isaac native debug-draw interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from unirobosim import DebugPrimitive

Vector3 = tuple[float, float, float]
Color = tuple[float, float, float, float]
DebugKey = tuple[str, str, str]


class DebugDrawInterface(Protocol):
    """Narrow subset of ``isaacsim.util.debug_draw`` used by the adapter."""

    def clear_points(self) -> None: ...

    def clear_lines(self) -> None: ...

    def draw_points(
        self,
        points: tuple[Vector3, ...],
        colors: tuple[Color, ...],
        sizes: tuple[float, ...],
    ) -> None: ...

    def draw_lines(
        self,
        starts: tuple[Vector3, ...],
        ends: tuple[Vector3, ...],
        colors: tuple[Color, ...],
        widths: tuple[float, ...],
    ) -> None: ...


@dataclass(frozen=True)
class NativeDebugPayload:
    """Fully lowered payload accepted by Isaac Sim's native debug-draw interface."""

    points: tuple[Vector3, ...]
    point_colors: tuple[Color, ...]
    point_sizes: tuple[float, ...]
    line_starts: tuple[Vector3, ...]
    line_ends: tuple[Vector3, ...]
    line_colors: tuple[Color, ...]
    line_widths: tuple[float, ...]


class NativeDebugOverlay:
    """Own stable-key state and atomically redraw one process-local native overlay.

    Isaac's interface exposes global append/clear operations rather than keyed updates. The bridge
    therefore keeps portable state independently and rebuilds one batched payload after mutations.
    Physics/lifetime policy remains in the world driver and SDK imports remain outside this module.
    """

    def __init__(
        self,
        interface: DebugDrawInterface,
        lower: Callable[[Iterable[DebugPrimitive]], NativeDebugPayload],
    ) -> None:
        self._interface = interface
        self._lower = lower
        self._primitives: dict[DebugKey, DebugPrimitive] = {}
        self._closed = False

    @property
    def active_count(self) -> int:
        return len(self._primitives)

    @property
    def keys(self) -> tuple[DebugKey, ...]:
        return tuple(self._primitives)

    def upsert(self, primitives: Iterable[DebugPrimitive]) -> int:
        self._ensure_open()
        count = 0
        for primitive in primitives:
            self._primitives[primitive.key] = primitive
            count += 1
        self._redraw()
        return count

    def remove(self, keys: Iterable[DebugKey]) -> int:
        self._ensure_open()
        removed = 0
        for key in keys:
            if key in self._primitives:
                self._primitives.pop(key)
                removed += 1
        if removed:
            self._redraw()
        return removed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._primitives.clear()
        self._interface.clear_points()
        self._interface.clear_lines()

    def _redraw(self) -> None:
        self._interface.clear_points()
        self._interface.clear_lines()
        payload = self._lower(self._primitives.values())
        if payload.points:
            self._interface.draw_points(payload.points, payload.point_colors, payload.point_sizes)
        if payload.line_starts:
            self._interface.draw_lines(
                payload.line_starts,
                payload.line_ends,
                payload.line_colors,
                payload.line_widths,
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("native debug overlay is closed")

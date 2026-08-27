from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from unirobosim import ArrayValue, DebugLifetime, DebugPrimitive, DebugPrimitiveKind

from unirobosim_isaaclab.native import _find_debug_extension
from unirobosim_isaaclab.native_debug import NativeDebugOverlay, NativeDebugPayload


class FakeDraw:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def clear_points(self) -> None:
        self.calls.append(("clear_points", 0))

    def clear_lines(self) -> None:
        self.calls.append(("clear_lines", 0))

    def draw_points(self, points, colors, sizes) -> None:  # type: ignore[no-untyped-def]
        assert len(points) == len(colors) == len(sizes)
        self.calls.append(("draw_points", len(points)))

    def draw_lines(self, starts, ends, colors, widths) -> None:  # type: ignore[no-untyped-def]
        assert len(starts) == len(ends) == len(colors) == len(widths)
        self.calls.append(("draw_lines", len(starts)))


def primitive(value: float = 0.0) -> DebugPrimitive:
    return DebugPrimitive(
        primitive_id="stable",
        layer="layer",
        group="group",
        kind=DebugPrimitiveKind.POINT_SET,
        geometry_m=ArrayValue.from_nested([[[value, 0.0, 0.0]]]),
        environment_indices=(0,),
        lifetime=DebugLifetime.persistent(),
    )


def lower(primitives: Iterable[DebugPrimitive]) -> NativeDebugPayload:
    count = len(tuple(primitives))
    return NativeDebugPayload(
        points=((0.0, 0.0, 0.0),) * count,
        point_colors=((1.0, 1.0, 1.0, 1.0),) * count,
        point_sizes=(5.0,) * count,
        line_starts=((0.0, 0.0, 0.0),) * count,
        line_ends=((1.0, 0.0, 0.0),) * count,
        line_colors=((1.0, 0.0, 0.0, 1.0),) * count,
        line_widths=(2.0,) * count,
    )


def test_overlay_batches_stable_key_updates_and_removal() -> None:
    draw = FakeDraw()
    overlay = NativeDebugOverlay(draw, lower)
    assert overlay.upsert((primitive(),)) == 1
    assert overlay.upsert((primitive(1.0),)) == 1
    assert overlay.active_count == 1
    assert overlay.keys == (("layer", "group", "stable"),)
    assert (
        draw.calls
        == [
            ("clear_points", 0),
            ("clear_lines", 0),
            ("draw_points", 1),
            ("draw_lines", 1),
        ]
        * 2
    )
    assert overlay.remove((("missing", "group", "stable"),)) == 0
    assert overlay.remove(overlay.keys) == 1
    assert overlay.active_count == 0
    assert draw.calls[-2:] == [("clear_points", 0), ("clear_lines", 0)]


def test_overlay_close_is_idempotent_and_rejects_mutation() -> None:
    draw = FakeDraw()
    overlay = NativeDebugOverlay(draw, lower)
    overlay.close()
    overlay.close()
    assert draw.calls == [("clear_points", 0), ("clear_lines", 0)]
    with pytest.raises(RuntimeError, match="closed"):
        overlay.upsert((primitive(),))
    with pytest.raises(RuntimeError, match="closed"):
        overlay.remove((primitive().key,))


def test_debug_extension_lookup_is_explicit_about_sdk_cache(tmp_path: Path) -> None:
    root = tmp_path / "isaacsim"
    extension = root / "extscache" / "isaacsim.util.debug_draw-3.2.3"
    (extension / "bin").mkdir(parents=True)
    (extension / "isaacsim").mkdir()
    assert _find_debug_extension(root) == extension
    with pytest.raises(RuntimeError, match="isaacsim-extscache-kit-sdk"):
        _find_debug_extension(tmp_path / "missing")


def test_debug_extension_lookup_supports_official_ngc_bundle_layout(tmp_path: Path) -> None:
    package_root = tmp_path / "isaac-sim" / "python_packages" / "isaacsim"
    package_root.mkdir(parents=True)
    extension = tmp_path / "isaac-sim" / "extscache" / "isaacsim.util.debug_draw-3.2.3"
    (extension / "bin").mkdir(parents=True)
    (extension / "isaacsim").mkdir()
    assert _find_debug_extension(package_root) == extension

from __future__ import annotations

from unirobosim import ArrayValue, DebugLifetime, DebugPrimitive, DebugPrimitiveKind

from unirobosim_isaaclab.native import (
    _debug_draw_payload,
    _debug_line_groups,
    _debug_points,
    _text_segments,
)

ORIGINS = ((10.0, 20.0, 30.0),)


def make_primitive(kind: DebugPrimitiveKind) -> DebugPrimitive:
    common = {
        "primitive_id": kind.value,
        "layer": "test",
        "group": "lowering",
        "kind": kind,
        "environment_indices": (0,),
        "color_rgba": (0.2, 0.4, 0.6, 0.8),
        "size": 0.5,
        "lifetime": DebugLifetime.persistent(),
    }
    if kind is DebugPrimitiveKind.POINT_SET:
        return DebugPrimitive(geometry_m=ArrayValue.from_nested([[[1.0, 2.0, 3.0]]]), **common)
    if kind is DebugPrimitiveKind.LINE_LIST:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]),
            **common,
        )
    if kind is DebugPrimitiveKind.COORDINATE_AXES:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]),
            **common,
        )
    if kind is DebugPrimitiveKind.TEXT:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[0.0, 0.0, 1.0]]]),
            text=(("A中",),),
            **common,
        )
    if kind is DebugPrimitiveKind.BOUNDING_BOX:
        return DebugPrimitive(
            geometry_m=ArrayValue.from_nested([[[0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 1.0]]]),
            **common,
        )
    return DebugPrimitive(
        geometry_m=ArrayValue.from_nested([[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]]),
        sample_times_s=ArrayValue.from_nested([[0.0, 0.5, 1.0]]),
        **common,
    )


def count_segments(primitive: DebugPrimitive) -> int:
    return sum(len(segments) for segments in _debug_line_groups(primitive, ORIGINS).values())


def test_point_and_line_lowering_apply_environment_origin() -> None:
    points = _debug_points(make_primitive(DebugPrimitiveKind.POINT_SET), ORIGINS)
    assert points == ((11.0, 22.0, 33.0),)
    line_groups = _debug_line_groups(make_primitive(DebugPrimitiveKind.LINE_LIST), ORIGINS)
    assert tuple(line_groups.values()) == ((((10.0, 20.0, 30.0), (11.0, 20.0, 30.0)),),)


def test_axes_box_trajectory_and_text_lower_to_visible_line_segments() -> None:
    axes = _debug_line_groups(make_primitive(DebugPrimitiveKind.COORDINATE_AXES), ORIGINS)
    assert len(axes) == 3 and count_segments(make_primitive(DebugPrimitiveKind.COORDINATE_AXES)) == 3
    assert count_segments(make_primitive(DebugPrimitiveKind.BOUNDING_BOX)) == 12
    assert count_segments(make_primitive(DebugPrimitiveKind.TRAJECTORY)) == 2
    text_count = count_segments(make_primitive(DebugPrimitiveKind.TEXT))
    assert text_count > 20
    assert _text_segments((0.0, 0.0, 0.0), "中", 0.7) == _text_segments((0.0, 0.0, 0.0), "?", 0.7)


def test_nonmatching_lowering_helpers_return_empty_geometry() -> None:
    line = make_primitive(DebugPrimitiveKind.LINE_LIST)
    point = make_primitive(DebugPrimitiveKind.POINT_SET)
    assert _debug_points(line, ORIGINS) == ()
    assert _debug_line_groups(point, ORIGINS) == {}


def test_native_debug_payload_aggregates_all_six_kinds() -> None:
    primitives = tuple(make_primitive(kind) for kind in DebugPrimitiveKind)
    payload = _debug_draw_payload(primitives, ORIGINS)
    assert payload.points == ((11.0, 22.0, 33.0),)
    assert payload.point_colors == ((0.2, 0.4, 0.6, 0.8),)
    assert payload.point_sizes == (25.0,)
    expected_line_count = sum(count_segments(primitive) for primitive in primitives)
    assert len(payload.line_starts) == expected_line_count
    assert len(payload.line_ends) == expected_line_count
    assert len(payload.line_colors) == expected_line_count
    assert len(payload.line_widths) == expected_line_count
    assert all(width >= 1.0 for width in payload.line_widths)

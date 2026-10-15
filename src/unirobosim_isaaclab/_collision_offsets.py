"""Validate PhysX contact response without changing collision shape geometry."""

from __future__ import annotations

import math
from typing import Any

from .native_protocols import NativePlanningError

_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")


def _offset(value: Any, *, contact: bool) -> float | str:
    # USD uses negative infinity to request automatic offset selection. It is
    # not an invalid numeric offset, unlike NaN and positive infinity.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativePlanningError("collision_geometry_unsupported")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise NativePlanningError("collision_geometry_unsupported") from None
    if number == -math.inf:
        return "auto"
    if not math.isfinite(number) or abs(number) >= _FLOAT32_MAX or (contact and number < 0.0):
        raise NativePlanningError("collision_geometry_unsupported")
    return number


def collision_offset_provenance(physx: Any, meters_per_unit: float) -> dict[str, Any] | None:
    """Return authored response parameters for the geometry provenance digest.

    Shapes describe their uninflated geometric surface. Contact offsets select
    predictive contact generation, while rest offsets select solver separation;
    neither changes that surface. The native USD attributes remain untouched.
    Automatic values stay symbolic: no shape-dependent SDK default is invented.
    """
    if not physx:
        return None
    contact_attr, rest_attr = physx.GetContactOffsetAttr(), physx.GetRestOffsetAttr()
    authored = (contact_attr.HasAuthoredValueOpinion(), rest_attr.HasAuthoredValueOpinion())
    if not any(authored):
        return None
    if isinstance(meters_per_unit, bool) or not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise NativePlanningError("collision_geometry_unsupported")
    contact = _offset(contact_attr.Get(), contact=True)
    rest = _offset(rest_attr.Get(), contact=False)
    # Installed USD PhysX accepts legacy (0, 0), selecting a small positive
    # effective contact distance. Preserve the authored pair, not that runtime default.
    if isinstance(contact, float) and isinstance(rest, float) and contact <= rest and (contact, rest) != (0.0, 0.0):
        raise NativePlanningError("collision_geometry_unsupported")
    values: dict[str, Any] = {
        "profile": "physx-contact-response-uninflated-shape/1",
        "meters_per_unit": meters_per_unit,
    }
    for name, value, explicit in zip(("contact", "rest"), (contact, rest), authored, strict=True):
        distance = value * meters_per_unit if isinstance(value, float) else value
        if isinstance(distance, float) and not math.isfinite(distance):
            raise NativePlanningError("collision_geometry_unsupported")
        values[name] = {"authored": bool(explicit), "stage_units": value, "metres": distance}
    return values

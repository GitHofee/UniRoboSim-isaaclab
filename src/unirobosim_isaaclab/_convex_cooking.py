"""Snapshot and reproduce effective PhysX convex cooking configuration."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .native_protocols import NativePlanningError

type CookingValue = bool | int | float


@dataclass(frozen=True, slots=True)
class ConvexCookingParameters:
    schema: str
    attributes: tuple[tuple[str, CookingValue], ...]

    @property
    def identity(self) -> str:
        return json.dumps((self.schema, self.attributes), separators=(",", ":"), allow_nan=False)


def convex_cooking_parameters(modules: Any, prim: Any, approximation: str) -> ConvexCookingParameters | None:
    """Read composed values and registered defaults without mutating the source."""
    schema_name = {
        "convexHull": "PhysxConvexHullCollisionAPI",
        "convexDecomposition": "PhysxConvexDecompositionCollisionAPI",
    }.get(approximation)
    if schema_name is None:
        return None
    schema = getattr(modules.PhysxSchema, schema_name)
    definition = modules.Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_name)
    if definition is None:
        raise NativePlanningError("collision_geometry_unsupported")
    values = []
    for name in sorted(schema.GetSchemaAttributeNames(False)):
        attr = prim.GetAttribute(name)
        if attr and attr.ValueMightBeTimeVarying():
            raise NativePlanningError("collision_geometry_unsupported")
        value = attr.Get() if attr else None
        if value is None:
            if attr and attr.HasAuthoredValueOpinion():
                raise NativePlanningError("collision_geometry_unsupported")
            value = definition.GetAttributeFallbackValue(name)
        if type(value) not in (bool, int, float) or (type(value) is float and not math.isfinite(value)):
            raise NativePlanningError("collision_geometry_unsupported")
        values.append((str(name), value))
    if not values:
        raise NativePlanningError("collision_geometry_unsupported")
    return ConvexCookingParameters(schema_name, tuple(values))


def apply_convex_cooking_parameters(modules: Any, prim: Any, parameters: ConvexCookingParameters) -> None:
    """Apply only the captured convex cooking schema to a temporary mesh."""
    schema = getattr(modules.PhysxSchema, parameters.schema).Apply(prim)
    if not schema:
        raise NativePlanningError("collision_cooking_failed")
    for name, value in parameters.attributes:
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.Set(value):
            raise NativePlanningError("collision_cooking_failed")

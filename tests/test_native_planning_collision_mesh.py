from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from unirobosim_isaaclab._collision_mesh import single_exact_convex_mesh
from unirobosim_isaaclab.native_protocols import NativePlanningError


class _MeshSchema:
    pass


class _CollisionSchema:
    pass


class _Prim:
    def __init__(self, name: str, *, mesh: bool = False, collision: bool = False) -> None:
        self.name = name
        self.mesh = mesh
        self.collision = collision

    def IsA(self, schema: object) -> bool:
        return schema is _MeshSchema and self.mesh

    def HasAPI(self, schema: object) -> bool:
        return schema is _CollisionSchema and self.collision


_MODULES = SimpleNamespace(
    UsdGeom=SimpleNamespace(Mesh=_MeshSchema),
    UsdPhysics=SimpleNamespace(CollisionAPI=_CollisionSchema),
)


def test_exact_convex_accepts_mesh_carrier_itself() -> None:
    carrier = _Prim("chassis_rounded_bbox", mesh=True, collision=True)

    assert single_exact_convex_mesh(_MODULES, carrier, (carrier,)) is carrier


def test_exact_convex_accepts_one_mesh_below_container_carrier() -> None:
    carrier = _Prim("collision", collision=True)
    mesh = _Prim("mesh", mesh=True)

    assert single_exact_convex_mesh(_MODULES, carrier, (carrier, mesh)) is mesh


@pytest.mark.parametrize(
    "subtree",
    (
        lambda carrier: (carrier,),
        lambda carrier: (carrier, _Prim("mesh_a", mesh=True), _Prim("mesh_b", mesh=True)),
        lambda carrier: (carrier, _Prim("nested", mesh=True, collision=True)),
        lambda carrier: (carrier, _Prim("mesh", mesh=True), _Prim("nested", collision=True)),
    ),
    ids=("no-mesh", "multiple-meshes", "nested-collider-mesh", "mesh-and-nested-collider"),
)
def test_exact_convex_rejects_missing_or_ambiguous_meshes(
    subtree: Callable[[_Prim], tuple[_Prim, ...]],
) -> None:
    carrier = _Prim("collision", collision=True)

    with pytest.raises(NativePlanningError, match="collision_cooking_failed"):
        single_exact_convex_mesh(_MODULES, carrier, subtree(carrier))


def test_exact_convex_rejects_nested_collider_below_self_mesh() -> None:
    carrier = _Prim("chassis_rounded_bbox", mesh=True, collision=True)
    nested = _Prim("nested", collision=True)

    with pytest.raises(NativePlanningError, match="collision_cooking_failed"):
        single_exact_convex_mesh(_MODULES, carrier, (carrier, nested))

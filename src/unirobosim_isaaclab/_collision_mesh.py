"""SDK-free collision Mesh selection shared by native planning paths."""

from __future__ import annotations

from typing import Any

from .native_protocols import NativePlanningError


def single_exact_convex_mesh(modules: Any, carrier: Any, subtree: tuple[Any, ...]) -> Any:
    """Resolve the one Mesh represented by a convex-hull collision carrier.

    USD permits ``PhysicsCollisionAPI`` and ``PhysicsMeshCollisionAPI`` to be
    authored directly on a ``UsdGeom.Mesh``. It also permits the collision carrier
    to be a container with one Mesh below it that does not carry its own
    ``PhysicsCollisionAPI``. Both forms describe one collision geometry and are
    accepted here; nested collision carriers and multi-Mesh carriers remain
    ambiguous and fail closed.
    """

    meshes = tuple(prim for prim in subtree if prim.IsA(modules.UsdGeom.Mesh))
    nested_colliders = tuple(
        prim for prim in subtree if prim != carrier and prim.HasAPI(modules.UsdPhysics.CollisionAPI)
    )
    if len(meshes) != 1 or nested_colliders:
        raise NativePlanningError("collision_cooking_failed")
    return meshes[0]

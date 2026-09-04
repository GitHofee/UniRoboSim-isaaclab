"""Optional spatial activation for large static/composite collision scenes."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_METADATA_KEY = "fastsim_physics_activation"
_ORIGINAL_COLLISION_ENABLED_ATTRIBUTE = "unirobosim:physicsActivationOriginalCollisionEnabled"


@dataclass(slots=True)
class _CollisionCandidate:
    path: str
    collision_enabled: Any
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    enabled: bool = True


@dataclass(slots=True)
class DynamicRigidBodyCandidate:
    path: str
    view: Any
    enabled: bool = True
    activated: bool = False
    pinned: bool = False


@dataclass(frozen=True, slots=True)
class PhysicsActivationDiagnostics:
    candidate_count: int
    enabled_count: int
    disabled_count: int
    protected_count: int
    dynamic_candidate_count: int
    dynamic_enabled_count: int
    dynamic_disabled_count: int
    update_count: int


class PhysicsActivationController:
    """Toggle only changed environment colliders at a bounded update cadence."""

    def __init__(
        self,
        *,
        candidates: tuple[_CollisionCandidate, ...],
        protected_count: int,
        radius_m: float,
        hysteresis_m: float,
        update_interval_steps: int,
        anchor_points: Callable[[], Any],
        torch_module: Any | None = None,
    ) -> None:
        self._candidates = candidates
        self._protected_count = protected_count
        self._radius_m = radius_m
        self._hysteresis_m = hysteresis_m
        self._update_interval_steps = update_interval_steps
        self._anchor_points = anchor_points
        self._torch = torch_module
        self._minimum_tensor: Any | None = None
        self._maximum_tensor: Any | None = None
        self._enabled_tensor: Any | None = None
        self._dynamic_candidates: tuple[DynamicRigidBodyCandidate, ...] = ()
        self._dynamic_by_path: dict[str, DynamicRigidBodyCandidate] = {}
        self._pending_dynamic_paths: set[str] = set()
        self._next_update_step = 0
        self._update_count = 0

    def configure_dynamic_candidates(self, candidates: tuple[DynamicRigidBodyCandidate, ...]) -> None:
        if self._dynamic_candidates:
            raise RuntimeError("physics activation dynamic candidates may be configured only once")
        if len({candidate.path for candidate in candidates}) != len(candidates):
            raise RuntimeError("physics activation dynamic candidate paths must be unique")
        self._dynamic_candidates = candidates
        self._dynamic_by_path = {candidate.path: candidate for candidate in candidates}
        self._pending_dynamic_paths = set(self._dynamic_by_path)

    def pin(self, path: str) -> None:
        candidate = self._dynamic_by_path.get(path)
        if candidate is None:
            return
        candidate.pinned = True
        candidate.activated = True
        self._pending_dynamic_paths.discard(path)
        self._set_dynamic_enabled(candidate, True)

    def reset_dynamic_state(self) -> None:
        for candidate in self._dynamic_candidates:
            candidate.activated = False
            candidate.pinned = False
            self._set_dynamic_enabled(candidate, True)
        self._pending_dynamic_paths = set(self._dynamic_by_path)
        self._next_update_step = 0

    def update(self, step_index: int, *, force: bool = False) -> None:
        if not force and step_index < self._next_update_step:
            return
        anchors = self._anchor_points()
        empty = anchors.numel() == 0 if self._torch is not None else not anchors
        if empty:
            raise RuntimeError("proximity physics activation requires at least one live robot anchor")
        desired_states = self._desired_states(anchors)
        for candidate, desired in zip(self._candidates, desired_states, strict=True):
            if desired == candidate.enabled:
                continue
            candidate.collision_enabled.Set(desired)
            candidate.enabled = desired
        self._update_dynamic_candidates(anchors)
        self._update_count += 1
        self._next_update_step = step_index + self._update_interval_steps

    def _desired_states(self, anchors: Any) -> tuple[bool, ...]:
        if self._torch is None:
            return tuple(
                any(
                    _point_aabb_distance_squared(anchor, candidate.minimum, candidate.maximum)
                    <= (
                        self._radius_m
                        + (self._hysteresis_m if self._update_count > 0 and candidate.enabled else 0.0)
                    )
                    ** 2
                    for anchor in anchors
                )
                for candidate in self._candidates
            )
        torch = self._torch
        if self._minimum_tensor is None or self._minimum_tensor.device != anchors.device:
            self._minimum_tensor = torch.tensor(
                [candidate.minimum for candidate in self._candidates],
                device=anchors.device,
                dtype=anchors.dtype,
            )
            self._maximum_tensor = torch.tensor(
                [candidate.maximum for candidate in self._candidates],
                device=anchors.device,
                dtype=anchors.dtype,
            )
            self._enabled_tensor = torch.ones(len(self._candidates), device=anchors.device, dtype=torch.bool)
        if not self._candidates:
            return ()
        assert self._maximum_tensor is not None and self._enabled_tensor is not None
        lower = self._minimum_tensor[:, None, :] - anchors[None, :, :]
        upper = anchors[None, :, :] - self._maximum_tensor[:, None, :]
        distances = torch.maximum(torch.maximum(lower, upper), torch.zeros((), device=anchors.device))
        distance_squared = torch.sum(distances * distances, dim=-1)
        if self._update_count == 0:
            thresholds = torch.full(
                (len(self._candidates),),
                self._radius_m,
                device=anchors.device,
                dtype=anchors.dtype,
            )
        else:
            thresholds = torch.where(
                self._enabled_tensor,
                self._radius_m + self._hysteresis_m,
                self._radius_m,
            )
        desired = torch.any(distance_squared <= thresholds[:, None] * thresholds[:, None], dim=1)
        self._enabled_tensor = desired
        return tuple(bool(value) for value in desired.detach().cpu().tolist())

    def _update_dynamic_candidates(self, anchors: Any) -> None:
        if not self._pending_dynamic_paths:
            return
        if self._torch is None:
            raise RuntimeError("dynamic physics activation requires the native tensor backend")
        torch = self._torch
        pending = tuple(
            candidate for candidate in self._dynamic_candidates if candidate.path in self._pending_dynamic_paths
        )
        distances: list[Any] = []
        for candidate in pending:
            positions = candidate.view.get_transforms()[..., :3].reshape(-1, 3)
            deltas = positions[:, None, :] - anchors[None, :, :]
            distances.append(torch.min(torch.sum(deltas * deltas, dim=-1)))
        near = torch.stack(distances) <= self._radius_m * self._radius_m
        desired_states = tuple(bool(value) for value in near.detach().cpu().tolist())
        for candidate, desired in zip(pending, desired_states, strict=True):
            if desired:
                candidate.activated = True
                self._pending_dynamic_paths.discard(candidate.path)
            self._set_dynamic_enabled(candidate, candidate.pinned or candidate.activated)

    def _set_dynamic_enabled(self, candidate: DynamicRigidBodyCandidate, enabled: bool) -> None:
        if candidate.enabled == enabled:
            return
        if self._torch is None:
            raise RuntimeError("dynamic physics activation requires the native tensor backend")
        view = candidate.view
        # PhysX exposes poses on the simulation device, but its disable flags
        # and index buffer are CPU tensors even for a CUDA simulation. Derive
        # the device and shape from that property rather than from transforms.
        disabled = view.get_disable_simulations().clone()
        disabled.fill_(0 if enabled else 1)
        indices = self._torch.arange(view.count, device=disabled.device, dtype=self._torch.int64)
        view.set_disable_simulations(disabled, indices)
        if enabled:
            view.wake_up(indices)
        candidate.enabled = enabled

    @property
    def diagnostics(self) -> PhysicsActivationDiagnostics:
        enabled_count = sum(candidate.enabled for candidate in self._candidates)
        return PhysicsActivationDiagnostics(
            candidate_count=len(self._candidates),
            enabled_count=enabled_count,
            disabled_count=len(self._candidates) - enabled_count,
            protected_count=self._protected_count,
            dynamic_candidate_count=len(self._dynamic_candidates),
            dynamic_enabled_count=sum(candidate.enabled for candidate in self._dynamic_candidates),
            dynamic_disabled_count=sum(not candidate.enabled for candidate in self._dynamic_candidates),
            update_count=self._update_count,
        )


def build_physics_activation_controller(
    modules: Any,
    world_spec: Any,
    *,
    scene_roots: tuple[str, ...],
    protected_roots: tuple[str, ...],
    anchor_points: Callable[[], Any],
) -> PhysicsActivationController | None:
    raw = world_spec.metadata.get(_METADATA_KEY)
    if raw is None:
        return None
    expected = {"anchor_paths", "hysteresis_m", "managed_paths", "mode", "radius_m", "update_interval_steps"}
    if set(raw) != expected or raw["mode"] != "proximity":
        raise RuntimeError("FastSim physics activation metadata is invalid")
    if not scene_roots:
        raise RuntimeError("proximity physics activation requires one static or composite scene")
    if not raw["anchor_paths"]:
        raise RuntimeError("proximity physics activation requires at least one configured robot")

    radius_m = _positive_float(raw["radius_m"], "radius_m")
    hysteresis_m = _nonnegative_float(raw["hysteresis_m"], "hysteresis_m")
    update_interval_steps = raw["update_interval_steps"]
    if type(update_interval_steps) is not int or update_interval_steps <= 0:
        raise RuntimeError("physics activation update_interval_steps must be a positive integer")

    stage = modules.sim_utils.get_current_stage()
    bbox_cache = modules.UsdGeom.BBoxCache(
        modules.Usd.TimeCode.Default(),
        [modules.UsdGeom.Tokens.default_, modules.UsdGeom.Tokens.render, modules.UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    protected = tuple(sorted(set(protected_roots)))
    candidates: list[_CollisionCandidate] = []
    protected_count = 0
    for scene_root in scene_roots:
        root = stage.GetPrimAtPath(scene_root)
        if not root or not root.IsValid():
            raise RuntimeError(f"physics activation scene root is unavailable: {scene_root}")
        for prim in modules.Usd.PrimRange(root):
            if not prim.HasAPI(modules.UsdPhysics.CollisionAPI):
                continue
            path = str(prim.GetPath())
            collision_api = modules.UsdPhysics.CollisionAPI(prim)
            collision_enabled = collision_api.GetCollisionEnabledAttr()
            if collision_enabled.Get() is False:
                continue
            if any(_path_is_at_or_under(path, protected_root) for protected_root in protected):
                protected_count += 1
                continue
            if _has_dynamic_rigid_owner(modules, prim, scene_root):
                protected_count += 1
                continue
            collision_enabled = collision_api.CreateCollisionEnabledAttr()
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum_value = aligned.GetMin()
            maximum_value = aligned.GetMax()
            minimum = tuple(float(minimum_value[axis]) for axis in range(3))
            maximum = tuple(float(maximum_value[axis]) for axis in range(3))
            if not all(math.isfinite(value) for value in (*minimum, *maximum)):
                raise RuntimeError(f"physics activation collider has no finite world bound: {path}")
            prim.CreateAttribute(
                _ORIGINAL_COLLISION_ENABLED_ATTRIBUTE,
                modules.Sdf.ValueTypeNames.Bool,
                custom=True,
            ).Set(True)
            candidates.append(
                _CollisionCandidate(
                    path=path,
                    collision_enabled=collision_enabled,
                    minimum=minimum,
                    maximum=maximum,
                )
            )
    controller = PhysicsActivationController(
        candidates=tuple(candidates),
        protected_count=protected_count,
        radius_m=radius_m,
        hysteresis_m=hysteresis_m,
        update_interval_steps=update_interval_steps,
        anchor_points=anchor_points,
        torch_module=modules.torch,
    )
    controller.update(0, force=True)
    return controller


def collision_enabled_for_planning(usd_physics: Any, prim: Any) -> bool:
    """Return logical collision intent, ignoring runtime-only activation state."""

    get_attribute = getattr(prim, "GetAttribute", None)
    marker = get_attribute(_ORIGINAL_COLLISION_ENABLED_ATTRIBUTE) if get_attribute is not None else None
    if marker and marker.Get() is True:
        return True
    return usd_physics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False


def _point_aabb_distance_squared(
    point: tuple[float, float, float],
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> float:
    total = 0.0
    for axis in range(3):
        coordinate = point[axis]
        delta = minimum[axis] - coordinate if coordinate < minimum[axis] else coordinate - maximum[axis]
        if minimum[axis] <= coordinate <= maximum[axis]:
            delta = 0.0
        total += delta * delta
    return total


def _path_is_at_or_under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _has_dynamic_rigid_owner(modules: Any, prim: Any, scene_root: str) -> bool:
    current = prim
    while current and current.IsValid() and _path_is_at_or_under(str(current.GetPath()), scene_root):
        if current.HasAPI(modules.UsdPhysics.RigidBodyAPI):
            rigid = modules.UsdPhysics.RigidBodyAPI(current)
            enabled = rigid.GetRigidBodyEnabledAttr().Get() is not False
            kinematic = rigid.GetKinematicEnabledAttr().Get() is True
            return enabled and not kinematic
        current = current.GetParent()
    return False


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"physics activation {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"physics activation {name} must be finite and positive")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"physics activation {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"physics activation {name} must be finite and non-negative")
    return result

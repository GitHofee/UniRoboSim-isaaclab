from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from unirobosim import FrozenMap


class _Attribute:
    def __init__(self, value: int) -> None:
        self.value = value
        self.writes: list[int] = []

    def Get(self) -> int:
        return self.value

    def Set(self, value: int) -> bool:
        self.value = value
        self.writes.append(value)
        return True


class _PhysxArticulationAPI:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def GetSolverPositionIterationCountAttr(self) -> _Attribute:
        return self.prim.position

    def GetSolverVelocityIterationCountAttr(self) -> _Attribute:
        return self.prim.velocity

    @classmethod
    def Apply(cls, prim: _Prim) -> _PhysxArticulationAPI:
        prim.physx = True
        return cls(prim)


class _UsdPhysics:
    ArticulationRootAPI = object()


class _PhysxSchema:
    PhysxArticulationAPI = _PhysxArticulationAPI


class _Prim:
    def __init__(self, position: int, velocity: int, *, articulation: bool = True, physx: bool = True) -> None:
        self.position = _Attribute(position)
        self.velocity = _Attribute(velocity)
        self.articulation = articulation
        self.physx = physx

    def HasAPI(self, api: object) -> bool:
        if api is _UsdPhysics.ArticulationRootAPI:
            return self.articulation
        if api is _PhysxArticulationAPI:
            return self.physx
        return False


class _SimUtils:
    def __init__(self, prims: list[_Prim]) -> None:
        self.prims = prims
        self.roots: list[str] = []

    def get_all_matching_child_prims(self, root: str, predicate: object) -> list[_Prim]:
        self.roots.append(root)
        return [prim for prim in self.prims if predicate(prim)]  # type: ignore[operator]


def _world(profile: object, prims: list[_Prim]) -> tuple[Any, _SimUtils]:
    from unirobosim_isaaclab.native import IsaacLabNativeWorld

    sim_utils = _SimUtils(prims)
    world = object.__new__(IsaacLabNativeWorld)
    world._spec = SimpleNamespace(metadata=FrozenMap({"fastsim_physics_profile": profile}))
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_UsdPhysics, PhysxSchema=_PhysxSchema)
    return world, sim_utils


def test_balanced_caps_high_solver_counts_without_raising_lower_counts() -> None:
    expensive = _Prim(32, 4)
    already_lower = _Prim(8, 0)
    unrelated = _Prim(64, 8, articulation=False, physx=False)
    legacy_link_override = _Prim(32, 4, articulation=False, physx=True)
    inherited = _Prim(32, 1, physx=False)
    world, sim_utils = _world(
        "balanced",
        [expensive, already_lower, inherited, legacy_link_override, unrelated],
    )

    world._apply_runtime_physics_profile()

    assert sim_utils.roots == ["/World"]
    assert (expensive.position.value, expensive.velocity.value) == (16, 1)
    assert expensive.position.writes == [16]
    assert expensive.velocity.writes == [1]
    assert (already_lower.position.value, already_lower.velocity.value) == (8, 0)
    assert already_lower.position.writes == []
    assert already_lower.velocity.writes == []
    assert inherited.physx is True
    assert (inherited.position.value, inherited.velocity.value) == (16, 1)
    assert (legacy_link_override.position.value, legacy_link_override.velocity.value) == (16, 1)
    assert (unrelated.position.value, unrelated.velocity.value) == (64, 8)


@pytest.mark.parametrize("profile", [None, "accurate"])
def test_non_fastsim_and_accurate_worlds_preserve_authored_solver_counts(profile: object) -> None:
    from unirobosim_isaaclab.native import IsaacLabNativeWorld

    prim = _Prim(32, 4)
    sim_utils = _SimUtils([prim])
    world = object.__new__(IsaacLabNativeWorld)
    metadata = FrozenMap() if profile is None else FrozenMap({"fastsim_physics_profile": profile})
    world._spec = SimpleNamespace(metadata=metadata)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_UsdPhysics, PhysxSchema=_PhysxSchema)

    world._apply_runtime_physics_profile()

    assert sim_utils.roots == []
    assert (prim.position.value, prim.velocity.value) == (32, 4)


def test_unknown_physics_profile_fails_before_first_simulation_reset() -> None:
    world, _ = _world("fast", [_Prim(32, 4)])

    with pytest.raises(RuntimeError, match="unsupported FastSim physics profile"):
        world._apply_runtime_physics_profile()

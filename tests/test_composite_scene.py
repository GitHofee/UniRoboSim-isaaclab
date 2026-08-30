from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from unirobosim import (
    COMPOSITE_WORLD_SCHEMA_VERSION,
    CapabilityId,
    EmbeddedEntityBinding,
    EmbeddedPrimBinding,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    FrozenMap,
    WorldBuildError,
    WorldSpec,
)

from unirobosim_isaaclab.descriptor import DESCRIPTOR
from unirobosim_isaaclab.native import (
    IsaacLabNativeWorld,
    _CompositeArticulationState,
    _CompositeRigidState,
    _declared_joint_path_map,
    _native_name,
)
from unirobosim_isaaclab.world import IsaacLabWorld

FRAME_BODY = "mechanism/base"
ROOT_BODY = "mechanism/door"
JOINT = "mechanism/door/RevoluteJoint"
_UNSET = object()


def _composite_world(
    asset: Path,
    *,
    environments: int = 2,
    unbound_mode: object = _UNSET,
    include_embedded_rigid: bool = False,
) -> WorldSpec:
    container_path = EntityPath("/scene")
    metadata = FrozenMap() if unbound_mode is _UNSET else FrozenMap({"composite_unbound_rigid_mode": unbound_mode})
    container = EntitySpec(
        container_path,
        EntityKind.COMPOSITE_SCENE,
        asset_uri=asset.as_uri(),
        metadata=metadata,
    )
    door = EntitySpec(
        EntityPath("/scene/door"),
        EntityKind.ARTICULATION,
        joint_names=("door_hinge",),
        initial_joint_positions=(0.0,),
        embedded_binding=EmbeddedEntityBinding(
            container_path=container_path,
            root_body_prim_path=ROOT_BODY,
            link_prims=(
                EmbeddedPrimBinding("base", FRAME_BODY),
                EmbeddedPrimBinding("door", ROOT_BODY),
            ),
            joint_prims=(EmbeddedPrimBinding("door_hinge", JOINT),),
        ),
    )
    entities = [door, container]
    if include_embedded_rigid:
        entities.insert(
            1,
            EntitySpec(
                EntityPath("/scene/bound_prop"),
                EntityKind.RIGID_BODY,
                embedded_binding=EmbeddedEntityBinding(
                    container_path=container_path,
                    root_body_prim_path="props/bound_prop",
                    link_prims=(EmbeddedPrimBinding("body", "props/bound_prop"),),
                ),
            ),
        )
    return WorldSpec(
        "composite-scene-test",
        tuple(entities),
        environments=EnvironmentSpec(environments),
        schema_version=COMPOSITE_WORLD_SCHEMA_VERSION,
        build_resource_manifest_sha256="a" * 64,
    )


class _RecordedUsdFileCfg:
    instances: list[_RecordedUsdFileCfg] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, tuple[float, float, float], tuple[float, float, float, float]]] = []
        self.instances.append(self)

    def func(
        self,
        root: str,
        cfg: object,
        *,
        translation: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        assert cfg is self
        self.calls.append((root, translation, orientation))


class _Attribute:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def Get(self) -> bool:
        return self.prim.kinematic

    def Set(self, value: bool) -> bool:
        self.prim.kinematic_writes.append(value)
        if not self.prim.kinematic_write_succeeds:
            return False
        self.prim.kinematic = value
        return True


class _JointEnabledAttribute:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def Get(self) -> bool:
        return self.prim.joint_enabled

    def Set(self, value: bool) -> bool:
        self.prim.joint_enabled_writes.append(value)
        if not self.prim.joint_write_succeeds:
            return False
        self.prim.joint_enabled = value
        return True


class _CollisionEnabledAttribute:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def Get(self) -> bool:
        return self.prim.collision_enabled

    def Set(self, value: bool) -> bool:
        self.prim.collision_enabled_writes.append(value)
        if not self.prim.collision_enabled_write_succeeds:
            return False
        self.prim.collision_enabled = value
        return True


class _RigidBodyAPI:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def GetKinematicEnabledAttr(self) -> _Attribute:
        return _Attribute(self.prim)

    def CreateKinematicEnabledAttr(self) -> _Attribute:
        return _Attribute(self.prim)


class _ArticulationRootAPI:
    pass


class _CollisionAPI:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def GetCollisionEnabledAttr(self) -> _CollisionEnabledAttribute:
        return _CollisionEnabledAttribute(self.prim)

    def CreateCollisionEnabledAttr(self) -> _CollisionEnabledAttribute:
        return _CollisionEnabledAttribute(self.prim)


class _Joint:
    def __new__(cls, prim: _Prim) -> Any:
        return prim


class _Physics:
    RigidBodyAPI = _RigidBodyAPI
    ArticulationRootAPI = _ArticulationRootAPI
    CollisionAPI = _CollisionAPI
    Joint = _Joint


class _Path:
    def __init__(self, value: str) -> None:
        self.pathString = value

    def __str__(self) -> str:
        return self.pathString


class _Relationship:
    def __init__(self, target: str | None) -> None:
        self.target = target

    def GetTargets(self) -> tuple[str, ...]:
        return () if self.target is None else (self.target,)


class _Prim:
    def __init__(
        self,
        path: str,
        *,
        rigid: bool = False,
        articulation_root: bool = False,
        joint: bool = False,
        collision: bool = False,
        kinematic: bool = False,
        body0: str | None = None,
        body1: str | None = None,
        kinematic_write_succeeds: bool = True,
        rigid_api_remove_succeeds: bool = True,
        collision_enabled_write_succeeds: bool = True,
        joint_write_succeeds: bool = True,
    ) -> None:
        self.path = path
        self.rigid = rigid
        self.articulation_root = articulation_root
        self.joint = joint
        self.collision = collision
        self.kinematic = kinematic
        self.body0 = body0
        self.body1 = body1
        self.kinematic_write_succeeds = kinematic_write_succeeds
        self.rigid_api_remove_succeeds = rigid_api_remove_succeeds
        self.collision_enabled_write_succeeds = collision_enabled_write_succeeds
        self.joint_write_succeeds = joint_write_succeeds
        self.collision_enabled = True
        self.joint_enabled = True
        self.kinematic_writes: list[bool] = []
        self.rigid_api_removals: list[object] = []
        self.collision_enabled_writes: list[bool] = []
        self.joint_enabled_writes: list[bool] = []

    def __bool__(self) -> bool:
        return True

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> _Path:
        return _Path(self.path)

    def HasAPI(self, api: object) -> bool:
        if api is _Physics.RigidBodyAPI:
            return self.rigid
        if api is _Physics.ArticulationRootAPI:
            return self.articulation_root
        if api is _Physics.CollisionAPI:
            return self.collision
        return False

    def IsA(self, schema: object) -> bool:
        return schema is _Physics.Joint and self.joint

    def RemoveAPI(self, api: object) -> bool:
        self.rigid_api_removals.append(api)
        if api is not _Physics.RigidBodyAPI or not self.rigid_api_remove_succeeds:
            return False
        self.rigid = False
        return True

    def GetBody0Rel(self) -> _Relationship:
        return _Relationship(self.body0)

    def GetBody1Rel(self) -> _Relationship:
        return _Relationship(self.body1)

    def GetJointEnabledAttr(self) -> _JointEnabledAttribute:
        return _JointEnabledAttribute(self)

    def CreateJointEnabledAttr(self) -> _JointEnabledAttribute:
        return _JointEnabledAttribute(self)


class _Stage:
    def __init__(self, prims: tuple[_Prim, ...]) -> None:
        self.prims = {prim.path: prim for prim in prims}

    def GetPrimAtPath(self, path: str) -> _Prim | None:
        return self.prims.get(path)


class _SimUtils:
    UsdFileCfg = _RecordedUsdFileCfg

    def __init__(self, stage: _Stage) -> None:
        self.stage = stage
        self.match_calls = 0

    def get_current_stage(self) -> _Stage:
        return self.stage

    def get_all_matching_child_prims(self, root: str, predicate: Any) -> list[_Prim]:
        self.match_calls += 1
        return [
            prim
            for prim in self.stage.prims.values()
            if (prim.path == root or prim.path.startswith(f"{root}/")) and predicate(prim)
        ]


def _native_roots(spec: WorldSpec) -> tuple[str, ...]:
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)
    name = _native_name(container.path)
    return tuple(f"/World/env_{index}/{name}" for index in range(spec.environments.count))


def _binding_prims(roots: tuple[str, ...], *, root_api: bool = True) -> tuple[_Prim, ...]:
    result: list[_Prim] = []
    for root in roots:
        result.extend(
            (
                _Prim(f"{root}/{ROOT_BODY}", rigid=True, articulation_root=root_api),
                _Prim(f"{root}/{FRAME_BODY}", rigid=True, kinematic=True),
                _Prim(
                    f"{root}/{JOINT}",
                    joint=True,
                    body0=f"{root}/{FRAME_BODY}",
                    body1=f"{root}/{ROOT_BODY}/door",
                ),
            )
        )
    return tuple(result)


def test_descriptor_and_preflight_admit_v6_composite_and_embedded_entities(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset)

    IsaacLabWorld.validate_build_spec(spec, backend_id="nvidia.isaaclab")

    assert COMPOSITE_WORLD_SCHEMA_VERSION in DESCRIPTOR.supported_world_schema_versions
    assert DESCRIPTOR.contract_version == "v0alpha6"
    assert DESCRIPTOR.capabilities.get(CapabilityId("scene.composite@1")) is not None
    unbound_mode = DESCRIPTOR.capabilities.get(CapabilityId("scene.composite.unbound-rigid-mode@1"))
    assert unbound_mode is not None
    assert unbound_mode.properties["modes"] == ("authored", "kinematic", "static")
    assert unbound_mode.properties["default"] == "authored"
    assert (
        unbound_mode.properties["embedded_link_protection"] == "exact-link-or-nearest-rigid-ancestor-within-composite"
    )
    assert unbound_mode.properties["private_joint_bodies"] == FrozenMap(
        {"kinematic": "kinematic", "static": "static-collider"}
    )
    assert unbound_mode.properties["embedded_joint_protection"] == "exact-relative-prim-path"
    assert unbound_mode.properties["private_joint_prims"] == "disabled-in-current-stage-before-first-reset"
    assert unbound_mode.properties["static_authoring"] == "remove-unbound-rigid-body-api-preserve-collision"
    assert DESCRIPTOR.capabilities.get(CapabilityId("entity.embedded-binding@1")) is not None
    formats = DESCRIPTOR.capabilities.get(CapabilityId("asset.formats@1"))
    assert formats is not None and formats.properties["composite_scene"] == ("model/vnd.usd",)

    asset.unlink()
    with pytest.raises(WorldBuildError, match="existing local USD"):
        IsaacLabWorld.validate_build_spec(spec, backend_id="nvidia.isaaclab")


def test_composite_authors_once_then_binds_exact_prims_without_respawn(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset)
    roots = _native_roots(spec)
    modules = SimpleNamespace(sim_utils=_SimUtils(_Stage(_binding_prims(roots))), UsdPhysics=_Physics)
    world = object.__new__(IsaacLabNativeWorld)
    world._m = modules
    world._spec = spec
    world._composite_scene_roots = {}
    world._usd_articulations = {}
    world._usd_rigids = {}
    world._embedded_joint_paths = {}
    world._kinematic_rigids = {}
    _RecordedUsdFileCfg.instances.clear()
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    world._author_composite_scene(container)
    authored_count = len(_RecordedUsdFileCfg.instances)
    world._bind_embedded_entities()

    assert authored_count == spec.environments.count
    assert len(_RecordedUsdFileCfg.instances) == authored_count
    assert world._composite_scene_roots[container.path] == roots
    embedded = EntityPath("/scene/door")
    assert tuple(item.root_prim.path for item in world._usd_articulations[embedded]) == tuple(
        f"{root}/{ROOT_BODY}" for root in roots
    )
    assert world._embedded_joint_paths[embedded] == tuple(((f"{root}/{JOINT}",)) for root in roots)


@pytest.mark.parametrize("unbound_mode", [_UNSET, "authored"])
def test_composite_authored_mode_never_traverses_or_writes_physics(
    tmp_path: Path,
    unbound_mode: object,
) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode=unbound_mode)
    roots = _native_roots(spec)
    props = tuple(_Prim(f"{root}/props/free_prop", rigid=True) for root in roots)
    sim_utils = _SimUtils(_Stage((*_binding_prims(roots), *props)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    _RecordedUsdFileCfg.instances.clear()
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    world._author_composite_scene(container)

    assert sim_utils.match_calls == 0
    assert all(prim.kinematic_writes == [] and not prim.kinematic for prim in props)
    assert all(prim.joint_enabled_writes == [] for prim in sim_utils.stage.prims.values())
    assert len(_RecordedUsdFileCfg.instances) == spec.environments.count


def test_kinematic_mode_changes_only_unbound_rigid_props(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="kinematic", include_embedded_rigid=True)
    roots = _native_roots(spec)
    embedded_rigid = tuple(_Prim(f"{root}/props/bound_prop", rigid=True) for root in roots)
    embedded = (*_binding_prims(roots), *embedded_rigid)
    props = tuple(_Prim(f"{root}/props/free_prop", rigid=True) for root in roots)
    colliders = tuple(_Prim(f"{root}/furniture/table/collision", collision=True) for root in roots)
    sim_utils = _SimUtils(_Stage((*embedded, *props, *colliders)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    _RecordedUsdFileCfg.instances.clear()
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    world._author_composite_scene(container)

    assert sim_utils.match_calls == spec.environments.count
    assert all(prim.kinematic and prim.kinematic_writes == [True] for prim in props)
    assert all(prim.kinematic_writes == [] for prim in embedded)
    assert all(prim.joint_enabled_writes == [] for prim in embedded)


def test_static_mode_removes_only_unbound_rigid_apis_and_preserves_collision_schema(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="static", include_embedded_rigid=True)
    roots = _native_roots(spec)
    embedded_rigid = tuple(_Prim(f"{root}/props/bound_prop", rigid=True) for root in roots)
    embedded = (*_binding_prims(roots), *embedded_rigid)
    props = tuple(_Prim(f"{root}/props/free_prop", rigid=True) for root in roots)
    colliders = tuple(_Prim(f"{root}/furniture/table/collision", collision=True) for root in roots)
    sim_utils = _SimUtils(_Stage((*embedded, *props, *colliders)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    world._composite_scene_modes = {}
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    world._author_composite_scene(container)

    assert all(not prim.rigid and prim.rigid_api_removals == [_Physics.RigidBodyAPI] for prim in props)
    assert all(prim.rigid_api_removals == [] for prim in embedded)
    assert all(prim.rigid for prim in embedded if not prim.joint)
    assert all(prim.collision_enabled and prim.collision_enabled_writes == [] for prim in colliders)
    assert world._composite_scene_modes[container.path] == "static"


def test_kinematic_mode_freezes_joint_connected_private_articulation_bodies(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="kinematic")
    roots = _native_roots(spec)
    private_bodies: list[_Prim] = []
    private_joints: list[_Prim] = []
    unrelated_nested_rigids: list[_Prim] = []
    props: list[_Prim] = []
    for root in roots:
        base = _Prim(f"{root}/private_mechanism/base", rigid=True, articulation_root=True)
        door = _Prim(f"{root}/private_mechanism/door", rigid=True)
        joint = _Prim(
            f"{root}/private_mechanism/hinge",
            joint=True,
            body0=base.path,
            body1=f"{door.path}/joint_anchor",
        )
        private_bodies.extend((base, door))
        private_joints.append(joint)
        unrelated_nested_rigids.append(_Prim(f"{door.path}/unrelated_nested_prop", rigid=True))
        props.append(_Prim(f"{root}/props/free_prop", rigid=True))
    sim_utils = _SimUtils(
        _Stage(
            (
                *_binding_prims(roots),
                *private_bodies,
                *private_joints,
                *unrelated_nested_rigids,
                *props,
            )
        )
    )
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    world._author_composite_scene(container)

    assert all(prim.kinematic and prim.kinematic_writes == [True] for prim in private_bodies)
    assert all(prim.kinematic and prim.kinematic_writes == [True] for prim in unrelated_nested_rigids)
    assert all(prim.kinematic and prim.kinematic_writes == [True] for prim in props)
    assert all(not prim.joint_enabled and prim.joint_enabled_writes == [False] for prim in private_joints)


def test_kinematic_mode_rejects_cross_environment_rigid_topology_change_before_writing(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="kinematic")
    roots = _native_roots(spec)
    prop = _Prim(f"{roots[0]}/props/free_prop", rigid=True)
    sim_utils = _SimUtils(_Stage((*_binding_prims(roots), prop)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    with pytest.raises(ValueError, match="selection changed across environments"):
        world._author_composite_scene(container)

    assert prop.kinematic_writes == []
    assert world._composite_scene_roots == {}


def test_kinematic_mode_rejects_cross_environment_private_joint_selection_before_writing(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="kinematic")
    roots = _native_roots(spec)
    props = tuple(_Prim(f"{root}/props/free_prop", rigid=True) for root in roots)
    joint = _Prim(
        f"{roots[1]}/private_mechanism/fixed_joint",
        joint=True,
        body0=f"{props[1].path}/joint_anchor",
        body1=f"{roots[1]}/{FRAME_BODY}",
    )
    sim_utils = _SimUtils(_Stage((*_binding_prims(roots), *props, joint)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    with pytest.raises(ValueError, match="private-joint selection changed across environments"):
        world._author_composite_scene(container)

    assert all(prim.kinematic_writes == [] for prim in props)
    assert joint.joint_enabled_writes == []
    assert world._composite_scene_roots == {}


def test_kinematic_mode_rejects_joint_authoring_failure_before_rigid_writes(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="kinematic")
    roots = _native_roots(spec)
    props = tuple(_Prim(f"{root}/props/free_prop", rigid=True) for root in roots)
    private_joints = tuple(
        _Prim(
            f"{root}/private_mechanism/fixed_joint",
            joint=True,
            joint_write_succeeds=index != 0,
        )
        for index, root in enumerate(roots)
    )
    sim_utils = _SimUtils(_Stage((*_binding_prims(roots), *props, *private_joints)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    with pytest.raises(RuntimeError, match="failed to author jointEnabled=false"):
        world._author_composite_scene(container)

    assert private_joints[0].joint_enabled_writes == [False]
    assert all(prim.kinematic_writes == [] for prim in props)
    assert world._composite_scene_roots == {}


def test_kinematic_mode_rejects_rigid_authoring_failure_on_readback_gate(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="kinematic")
    roots = _native_roots(spec)
    props = tuple(
        _Prim(
            f"{root}/props/free_prop",
            rigid=True,
            kinematic_write_succeeds=index != 0,
        )
        for index, root in enumerate(roots)
    )
    sim_utils = _SimUtils(_Stage((*_binding_prims(roots), *props)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    with pytest.raises(RuntimeError, match="failed to author kinematicEnabled"):
        world._author_composite_scene(container)

    assert props[0].kinematic_writes == [True]
    assert not props[0].kinematic
    assert world._composite_scene_roots == {}


def test_static_mode_rejects_rigid_api_removal_failure(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode="static")
    roots = _native_roots(spec)
    props = tuple(
        _Prim(
            f"{root}/props/free_prop",
            rigid=True,
            rigid_api_remove_succeeds=index != 0,
        )
        for index, root in enumerate(roots)
    )
    sim_utils = _SimUtils(_Stage((*_binding_prims(roots), *props)))
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=sim_utils, UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    with pytest.raises(RuntimeError, match="failed to author RigidBodyAPI removal"):
        world._author_composite_scene(container)

    assert props[0].rigid_api_removals == [_Physics.RigidBodyAPI]
    assert props[0].rigid
    assert world._composite_scene_roots == {}


@pytest.mark.parametrize("invalid_mode", [None, "", "KINEMATIC", 7, True])
def test_composite_rejects_invalid_unbound_rigid_mode_before_composition(
    tmp_path: Path,
    invalid_mode: object,
) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset, unbound_mode=invalid_mode)
    roots = _native_roots(spec)
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=_SimUtils(_Stage(_binding_prims(roots))), UsdPhysics=_Physics)
    world._spec = spec
    world._composite_scene_roots = {}
    _RecordedUsdFileCfg.instances.clear()
    container = next(entity for entity in spec.entities if entity.kind is EntityKind.COMPOSITE_SCENE)

    with pytest.raises(ValueError, match="must be exactly 'authored', 'kinematic', or 'static'"):
        world._author_composite_scene(container)

    assert _RecordedUsdFileCfg.instances == []
    assert world._composite_scene_roots == {}


def test_embedded_binding_rejects_missing_native_articulation_root(tmp_path: Path) -> None:
    asset = tmp_path / "mixed-room.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _composite_world(asset)
    roots = _native_roots(spec)
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(
        sim_utils=_SimUtils(_Stage(_binding_prims(roots, root_api=False))),
        UsdPhysics=_Physics,
    )
    world._spec = spec
    world._composite_scene_roots = {EntityPath("/scene"): roots}
    world._usd_articulations = {}
    world._usd_rigids = {}
    world._embedded_joint_paths = {}
    world._kinematic_rigids = {}

    with pytest.raises(ValueError, match="ArticulationRootAPI"):
        world._bind_embedded_entities()
    assert world._usd_articulations == {}


def test_embedded_joint_mapping_uses_exact_paths_not_duplicate_short_names() -> None:
    expected = (
        ("/World/env_0/scene/cabinet_a/RevoluteJoint",),
        ("/World/env_1/scene/cabinet_a/RevoluteJoint",),
    )
    native = (
        (
            "/World/env_0/scene/cabinet_b/RevoluteJoint",
            "/World/env_0/scene/cabinet_a/RevoluteJoint",
        ),
        (
            "/World/env_1/scene/cabinet_b/RevoluteJoint",
            "/World/env_1/scene/cabinet_a/RevoluteJoint",
        ),
    )
    assert _declared_joint_path_map(EntityPath("/scene/cabinet_a"), native, expected) == (1,)
    with pytest.raises(ValueError, match="not DOFs"):
        _declared_joint_path_map(
            EntityPath("/scene/cabinet_a"),
            native,
            (("/World/env_0/scene/missing",), ("/World/env_1/scene/missing",)),
        )


class _Tensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.device = "cpu"

    def clone(self) -> _Tensor:
        return _Tensor(f"{self.name}.clone")

    def __getitem__(self, key: object) -> _Tensor:
        return _Tensor(f"{self.name}[{key!r}]")


class _Torch:
    int64 = "int64"

    @staticmethod
    def tensor(values: object, *, device: object, dtype: object) -> tuple[object, object, object]:
        return values, device, dtype


class _RigidView:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_transforms(self, values: object, indices: object) -> None:
        del values, indices
        self.calls.append("transforms")

    def set_velocities(self, values: object, indices: object) -> None:
        del values, indices
        self.calls.append("velocities")


class _ArticulationView:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("set_"):

            def setter(values: object, indices: object) -> None:
                del values, indices
                self.calls.append(name)

            return setter
        raise AttributeError(name)


class _Simulation:
    def __init__(self) -> None:
        self.forward_count = 0

    def forward(self) -> None:
        self.forward_count += 1


def test_composite_partial_reset_restores_rigids_and_complete_articulation_state() -> None:
    rigid_view = _RigidView()
    articulation_view = _ArticulationView()
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(torch=_Torch())
    world._sim = _Simulation()
    world._composite_rigid_states = [
        _CompositeRigidState(
            view=rigid_view,
            initial_transforms=_Tensor("poses"),
            initial_velocities=_Tensor("velocities"),
            environment_by_index=(0, 0, 1, 1),
            kinematic=False,
        )
    ]
    tensors = [_Tensor(str(index)) for index in range(9)]
    world._composite_articulation_states = [_CompositeArticulationState(articulation_view, *tensors)]
    world._articulations = {}
    world._usd_articulation_views = {}
    world._rigids = {}
    world._contacts = {}
    world._usd_rigid_views = {}
    world._deformables = {}
    world._fluids = {}
    world._cameras = {}
    world._mounted_cameras = {}
    world._debug_lifetimes = {}

    world.reset((1,))

    assert rigid_view.calls == ["transforms", "velocities"]
    assert articulation_view.calls == [
        "set_root_transforms",
        "set_root_velocities",
        "set_dof_positions",
        "set_dof_velocities",
        "set_dof_position_targets",
        "set_dof_velocity_targets",
        "set_dof_actuation_forces",
        "set_dof_stiffnesses",
        "set_dof_dampings",
    ]
    assert world._sim.forward_count == 1

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


def _composite_world(asset: Path, *, environments: int = 2) -> WorldSpec:
    container_path = EntityPath("/scene")
    container = EntitySpec(container_path, EntityKind.COMPOSITE_SCENE, asset_uri=asset.as_uri())
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
    return WorldSpec(
        "composite-scene-test",
        (door, container),
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
    def __init__(self, value: bool = False) -> None:
        self.value = value

    def Get(self) -> bool:
        return self.value


class _RigidBodyAPI:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    def GetKinematicEnabledAttr(self) -> _Attribute:
        return _Attribute(self.prim.kinematic)


class _ArticulationRootAPI:
    pass


class _Joint:
    pass


class _Physics:
    RigidBodyAPI = _RigidBodyAPI
    ArticulationRootAPI = _ArticulationRootAPI
    Joint = _Joint


class _Path:
    def __init__(self, value: str) -> None:
        self.pathString = value

    def __str__(self) -> str:
        return self.pathString


class _Prim:
    def __init__(
        self,
        path: str,
        *,
        rigid: bool = False,
        articulation_root: bool = False,
        joint: bool = False,
        kinematic: bool = False,
    ) -> None:
        self.path = path
        self.rigid = rigid
        self.articulation_root = articulation_root
        self.joint = joint
        self.kinematic = kinematic

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
        return False

    def IsA(self, schema: object) -> bool:
        return schema is _Physics.Joint and self.joint


class _Stage:
    def __init__(self, prims: tuple[_Prim, ...]) -> None:
        self.prims = {prim.path: prim for prim in prims}

    def GetPrimAtPath(self, path: str) -> _Prim | None:
        return self.prims.get(path)


class _SimUtils:
    UsdFileCfg = _RecordedUsdFileCfg

    def __init__(self, stage: _Stage) -> None:
        self.stage = stage

    def get_current_stage(self) -> _Stage:
        return self.stage

    def get_all_matching_child_prims(self, root: str, predicate: Any) -> list[_Prim]:
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
                _Prim(f"{root}/{JOINT}", joint=True),
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

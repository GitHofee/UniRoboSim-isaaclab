from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from unirobosim import (
    PHYSICAL_WORLD_SCHEMA_VERSION,
    CameraModality,
    CameraMountSpec,
    CameraSpec,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    PlanningSceneIncompleteError,
    Pose,
    WorldBuildError,
    WorldSpec,
)

from unirobosim_isaaclab.native import (
    IsaacLabNativeWorld,
    _articulation_mount_body_suffix,
    _camera_native_data_type,
    _native_asset_path,
    _native_name,
    _scaled_dimensions,
    _usd_file_cfg,
)
from unirobosim_isaaclab.planning_scene import validate_planning_build_spec
from unirobosim_isaaclab.world import IsaacLabWorld


def _static_world(asset: Path, *, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> WorldSpec:
    return WorldSpec(
        "static-scene-test",
        (
            EntitySpec(
                EntityPath("/environment"),
                EntityKind.STATIC_SCENE,
                asset_uri=asset.as_uri(),
                scale_xyz=scale,
            ),
        ),
        environments=EnvironmentSpec(2),
        schema_version=PHYSICAL_WORLD_SCHEMA_VERSION,
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


class _StaticSimUtils:
    UsdFileCfg = _RecordedUsdFileCfg

    def __init__(self, forbidden: tuple[object, ...] = ()) -> None:
        self.forbidden = forbidden

    def get_all_matching_child_prims(self, root: str, predicate: object) -> list[object]:
        del root, predicate
        return list(self.forbidden)


class _ForbiddenPrim:
    def __init__(self, path: str) -> None:
        self.path = path

    def GetPath(self) -> str:
        return self.path


class _PhysicsTokens:
    RigidBodyAPI = object()
    ArticulationRootAPI = object()

    class Joint:
        pass


def test_static_scene_preflight_accepts_only_existing_local_usd(tmp_path: Path) -> None:
    asset = tmp_path / "room with texture.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _static_world(asset, scale=(2.0, 3.0, 4.0))
    IsaacLabWorld.validate_build_spec(spec, backend_id="nvidia.isaaclab")
    assert _native_asset_path(asset.as_uri()) == str(asset)
    requirements = {item.capability.value for item in spec.requirements}
    assert {"scene.static@1", "entity.scale.static_scene@1"} <= requirements

    missing = _static_world(tmp_path / "missing.usd")
    with pytest.raises(WorldBuildError, match="existing local USD"):
        IsaacLabWorld.validate_build_spec(missing, backend_id="nvidia.isaaclab")


def test_static_scene_authors_raw_usd_per_environment_without_rigid_wrappers(tmp_path: Path) -> None:
    asset = tmp_path / "room.usd"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    entity = _static_world(asset, scale=(1.5, 2.0, 0.5)).entities[0]
    _RecordedUsdFileCfg.instances.clear()
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(sim_utils=_StaticSimUtils(), UsdPhysics=_PhysicsTokens)
    world._spec = _static_world(asset, scale=entity.scale_xyz)
    world._static_scene_roots = {}

    world._author_static_scene(entity)

    roots = world._static_scene_roots[entity.path]
    native_name = _native_name(entity.path)
    assert roots == (
        f"/World/env_0/{native_name}",
        f"/World/env_1/{native_name}",
    )
    assert len(_RecordedUsdFileCfg.instances) == 2
    for index, cfg in enumerate(_RecordedUsdFileCfg.instances):
        assert cfg.kwargs == {
            "usd_path": str(asset),
            "scale": (1.5, 2.0, 0.5),
            "activate_contact_sensors": False,
        }
        assert cfg.calls == [(roots[index], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))]


def test_static_scene_rejects_embedded_dynamic_physics(tmp_path: Path) -> None:
    asset = tmp_path / "dynamic-room.usd"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    spec = _static_world(asset)
    world = object.__new__(IsaacLabNativeWorld)
    world._m = SimpleNamespace(
        sim_utils=_StaticSimUtils((_ForbiddenPrim("/World/env_0/room/body"),)),
        UsdPhysics=_PhysicsTokens,
    )
    world._spec = spec
    world._static_scene_roots = {}
    with pytest.raises(ValueError, match="must not contain rigid bodies"):
        world._author_static_scene(spec.entities[0])
    assert spec.entities[0].path not in world._static_scene_roots


def test_static_scene_planning_demand_fails_closed(tmp_path: Path) -> None:
    asset = tmp_path / "planning-room.usd"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    with pytest.raises(PlanningSceneIncompleteError, match="complete static-scene collider forest"):
        validate_planning_build_spec(_static_world(asset), backend_id="nvidia.isaaclab")


def test_all_usd_assets_share_scale_aware_local_path_lowering(tmp_path: Path) -> None:
    asset = tmp_path / "asset with spaces.usd"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    entity = EntitySpec(
        EntityPath("/box"),
        EntityKind.RIGID_BODY,
        asset_uri=asset.as_uri(),
        scale_xyz=(2.0, 3.0, 4.0),
    )
    _RecordedUsdFileCfg.instances.clear()
    modules = SimpleNamespace(sim_utils=SimpleNamespace(UsdFileCfg=_RecordedUsdFileCfg))
    cfg = _usd_file_cfg(modules, entity, activate_contact_sensors=True)
    assert cfg.kwargs == {
        "usd_path": str(asset),
        "scale": (2.0, 3.0, 4.0),
        "activate_contact_sensors": True,
    }
    assert _scaled_dimensions((0.2, 0.3, 0.4), entity.scale_xyz) == pytest.approx((0.4, 0.9, 1.6))


@dataclass
class _Relation:
    target: str | None

    def GetTargets(self) -> tuple[str, ...]:
        return () if self.target is None else (self.target,)


class _BodyPrim:
    def __init__(self, path: str, name: str) -> None:
        self.path = path
        self.name = name

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.name

    def HasAPI(self, api: object) -> bool:
        return api is _MountPhysics.RigidBodyAPI

    def IsA(self, schema: object) -> bool:
        del schema
        return False


class _JointPrim(_BodyPrim):
    def __init__(self, path: str, parent: str, child: str) -> None:
        super().__init__(path, path.rsplit("/", 1)[-1])
        self.parent = parent
        self.child = child

    def HasAPI(self, api: object) -> bool:
        del api
        return False

    def IsA(self, schema: object) -> bool:
        return schema is _MountPhysics.Joint

    def GetBody0Rel(self) -> _Relation:
        return _Relation(self.parent)

    def GetBody1Rel(self) -> _Relation:
        return _Relation(self.child)


class _MountPhysics:
    RigidBodyAPI = object()

    class Joint:
        def __new__(cls, prim: _JointPrim) -> _JointPrim:
            return prim


class _MountSimUtils:
    def __init__(self, prims: tuple[_BodyPrim, ...]) -> None:
        self.prims = prims

    def get_all_matching_child_prims(self, root: str, predicate: Any) -> list[_BodyPrim]:
        return [prim for prim in self.prims if prim.path == root or prim.path.startswith(f"{root}/") if predicate(prim)]


def _mount_modules(root: str) -> SimpleNamespace:
    base = _BodyPrim(f"{root}/base", "base")
    wrist = _BodyPrim(f"{root}/arm/wrist", "wrist")
    joint = _JointPrim(f"{root}/joints/wrist_joint", base.path, wrist.path)
    return SimpleNamespace(sim_utils=_MountSimUtils((base, wrist, joint)), UsdPhysics=_MountPhysics)


def test_camera_mount_resolves_root_or_named_link_and_modalities() -> None:
    root = "/World/env_0/robot"
    modules = _mount_modules(root)
    assert _articulation_mount_body_suffix(modules, root, None) == "/base"
    assert _articulation_mount_body_suffix(modules, root, "wrist") == "/arm/wrist"
    with pytest.raises(ValueError, match="exactly one articulation body"):
        _articulation_mount_body_suffix(modules, root, "missing")
    assert [_camera_native_data_type(item) for item in CameraModality] == [
        "rgb",
        "distance_to_camera",
        "normals",
    ]


class _RecordedCfg:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _CameraCfg(_RecordedCfg):
    OffsetCfg = _RecordedCfg


def test_camera_authoring_places_mounted_sensor_below_native_link(tmp_path: Path) -> None:
    asset = tmp_path / "robot.usd"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    robot = EntitySpec(
        EntityPath("/robot"),
        EntityKind.ARTICULATION,
        joint_names=("joint",),
        asset_uri=str(asset),
    )
    camera = EntitySpec(
        EntityPath("/camera"),
        EntityKind.CAMERA_SENSOR,
        pose=Pose((0.1, 0.0, 0.2)),
        camera=CameraSpec(modalities=(CameraModality.RGB, CameraModality.NORMALS)),
        mount=CameraMountSpec(robot.path, "wrist"),
    )
    spec = WorldSpec(
        "mounted-camera",
        (camera, robot),
        schema_version=PHYSICAL_WORLD_SCHEMA_VERSION,
        build_resource_manifest_sha256="b" * 64,
    )
    root = f"/World/env_0/{_native_name(robot.path)}"
    modules = _mount_modules(root)
    modules.CameraCfg = _CameraCfg
    modules.Camera = lambda cfg: cfg
    modules.sim_utils.PinholeCameraCfg = _RecordedCfg
    world = object.__new__(IsaacLabNativeWorld)
    world._m = modules
    world._spec = spec
    world._config = SimpleNamespace(enable_cameras=True, render=True)
    world._cameras = {}
    world._mounted_cameras = {}

    world._author_camera(camera)

    cfg = world._cameras[camera.path]
    assert cfg.kwargs["prim_path"] == f"{root}/arm/wrist/{_native_name(camera.path)}".replace("env_0", "env_.*")
    assert cfg.kwargs["data_types"] == ["rgb", "normals"]
    assert cfg.kwargs["update_latest_camera_pose"] is True
    assert cfg.kwargs["offset"].kwargs["pos"] == (0.1, 0.0, 0.2)
    binding = world._mounted_cameras[camera.path]
    assert binding.parent_path == robot.path
    assert binding.body_name == "wrist"
    assert binding.body_suffix == "/arm/wrist"

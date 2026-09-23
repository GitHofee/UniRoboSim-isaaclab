from types import SimpleNamespace

import pytest
from unirobosim import EntityKind, EntityPath

from unirobosim_isaaclab.native import IsaacLabNativeWorld, _native_name


@pytest.mark.parametrize('environment', [0, 3])
@pytest.mark.parametrize('body_suffix', ['', '/Geometry/body/arm/wrist'])
def test_mounted_camera_entity_reads_use_authored_body_path(environment, body_suffix):
    camera, parent = EntityPath('/sensors/head'), EntityPath('/robots/g2')
    world = object.__new__(IsaacLabNativeWorld)
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=4))
    world._entity_specs = {camera: SimpleNamespace(kind=EntityKind.CAMERA_SENSOR, embedded_binding=None)}
    world._entity_prim_path_cache = {}
    world._mounted_cameras = {camera: SimpleNamespace(parent_path=parent, body_suffix=body_suffix)}
    expected = f'/World/env_{environment}/{_native_name(parent)}{body_suffix}/{_native_name(camera)}'
    captured = []
    world._read_usd_prim_pose = lambda path, env: captured.append((path, env))
    world._read_usd_entity_prim_pose(camera, environment)
    assert captured == [(expected, environment)]
    assert world._entity_prim_path(camera, environment) == expected
    with pytest.raises(IndexError):
        world._entity_prim_path(camera, 4)


@pytest.mark.parametrize('kind', [EntityKind.CAMERA_SENSOR, EntityKind.ARTICULATION])
@pytest.mark.parametrize('environment', [0, 3])
def test_world_camera_and_non_camera_paths_remain_at_environment_root(kind, environment):
    path = EntityPath('/camera')
    world = object.__new__(IsaacLabNativeWorld)
    world._spec = SimpleNamespace(environments=SimpleNamespace(count=4))
    world._entity_specs = {path: SimpleNamespace(kind=kind, embedded_binding=None)}
    world._entity_prim_path_cache = {}
    world._mounted_cameras = {}
    assert world._entity_prim_path(path, environment) == f'/World/env_{environment}/{_native_name(path)}'

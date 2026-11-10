import sys
from types import ModuleType, SimpleNamespace

import pytest
from unirobosim import (
    ArrayValue,
    CapabilityId,
    CapabilityRequirement,
    EntityKind,
    EntityPath,
    EntitySpec,
    ParticleFluidSpec,
    WorldSpec,
)

from unirobosim_isaaclab.native import IsaacLabNativeWorld
from unirobosim_isaaclab.planning_scene import validate_planning_build_spec


def test_particle_fluid_can_coexist_with_a_planning_articulation() -> None:
    robot = EntitySpec(EntityPath('/robot'), EntityKind.ARTICULATION, joint_names=('wrist',))
    fluid = EntitySpec(EntityPath('/coffee'), EntityKind.PARTICLE_FLUID,
        particle_fluid=ParticleFluidSpec(ArrayValue.from_nested(((0.0, 0.0, 1.0),))))
    world = WorldSpec('coffee', (robot, fluid),
        requirements=(CapabilityRequirement(CapabilityId('planning.scene@2')),))
    validate_planning_build_spec(world, backend_id='nvidia.isaaclab')


def test_nonfluid_world_preserves_device_and_factory_without_sdk_import(monkeypatch: pytest.MonkeyPatch) -> None:
    world = object.__new__(IsaacLabNativeWorld)
    world._has_fluid = False
    world._sim = SimpleNamespace(device='cuda:3')
    factory = object()
    monkeypatch.setitem(sys.modules, 'isaaclab_physx.assets.articulation', None)
    assert world._physical_tensor_device() == 'cuda:3'
    assert world._physical_asset_type(factory) is factory


def test_fluid_assets_use_local_cpu_frontends_without_mutating_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    class ManagedAsset:
        def __init__(self, configured_device: str) -> None:
            self._device = configured_device

    factory_articulation, factory_rigid, factory_contact = object(), object(), object()
    for name, class_name in (
        ('isaaclab_physx.assets.articulation', 'Articulation'),
        ('isaaclab_physx.assets.rigid_object', 'RigidObject'),
        ('isaaclab_physx.sensors.contact_sensor', 'ContactSensor'),
    ):
        module = ModuleType(name)
        setattr(module, class_name, ManagedAsset)
        monkeypatch.setitem(sys.modules, name, module)
    world = object.__new__(IsaacLabNativeWorld)
    world._has_fluid = True
    world._sim = SimpleNamespace(device='cuda:3')
    world._m = SimpleNamespace(Articulation=factory_articulation,
        RigidObject=factory_rigid, ContactSensor=factory_contact)
    assert world._physical_tensor_device() == 'cpu'
    for factory in (factory_articulation, factory_rigid, factory_contact):
        asset_type = world._physical_asset_type(factory)
        asset = asset_type('cuda:3')
        assert isinstance(asset, ManagedAsset)
        assert asset._device == 'cpu'
        asset._device = 'cuda:1'
        assert asset._device == 'cpu'
    assert world._sim.device == 'cuda:3'
    assert world._m.Articulation is factory_articulation
    assert world._m.RigidObject is factory_rigid
    assert world._m.ContactSensor is factory_contact

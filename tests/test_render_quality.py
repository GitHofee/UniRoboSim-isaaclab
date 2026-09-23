import sys
from types import SimpleNamespace

import pytest
from unirobosim import RENDER_QUALITY_CAPABILITY_ID, RenderQualityWorld

from unirobosim_isaaclab.native import IsaacLabNativeWorld

from .helpers import FakeNativeRuntime, make_articulation_asset, make_world
from .test_lifecycle_world import open_test_session

KEYS = ("/rtx/indirectDiffuse/enabled", "/rtx/ambientOcclusion/enabled")


class Settings:
    def __init__(self):
        self.values = dict(zip(KEYS, (False, True), strict=True))
        self.fail = False

    def get(self, key):
        return self.values[key]

    def set(self, key, value):
        self.values[key] = value

    def set_bool(self, key, value):
        if not (self.fail and key == KEYS[1]):
            self.set(key, value)


@pytest.mark.parametrize("pair", [(False, False), (True, False), (False, True), (True, True)])
def test_native_readback_restore_and_reused_process_isolation(monkeypatch, pair):
    settings = Settings()
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    for flags in (pair, (False, False)):
        world = object.__new__(IsaacLabNativeWorld)
        world._closed = False
        world._sim = object()
        world._invalidate_render = lambda: None
        result = world.configure_render_quality(enable_global_illumination=flags[0], enable_ambient_occlusion=flags[1])
        assert result == flags
        world._restore_render_quality()
        assert tuple(settings.values.values()) == (False, True)


def test_native_failed_readback_rolls_back(monkeypatch):
    settings = Settings()
    settings.fail = True
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    world = object.__new__(IsaacLabNativeWorld)
    world._closed = False
    world._sim = object()
    with pytest.raises(RuntimeError, match="readback"):
        world.configure_render_quality(enable_global_illumination=True, enable_ambient_occlusion=False)
    assert tuple(settings.values.values()) == (False, True)
    assert not hasattr(world, "_render_quality_baseline")


def test_public_port_validates_and_roundtrips_without_tick(tmp_path):
    runtime = FakeNativeRuntime()
    _, session = open_test_session(runtime)
    try:
        world = session.build(make_world(make_articulation_asset(tmp_path / "robot.usda")))
        assert isinstance(world, RenderQualityWorld)
        assert session.descriptor.capabilities.get(RENDER_QUALITY_CAPABILITY_ID) is not None
        calls = []
        world._native.configure_render_quality = lambda **kw: calls.append(kw) or tuple(kw.values())
        before = world.tick
        assert world.configure_render_quality(enable_global_illumination=False, enable_ambient_occlusion=True) == (
            False,
            True,
        )
        assert world.tick == before and len(calls) == 1
        with pytest.raises(Exception, match="booleans"):
            world.configure_render_quality(enable_global_illumination=1, enable_ambient_occlusion=True)
        assert len(calls) == 1
    finally:
        session.close()

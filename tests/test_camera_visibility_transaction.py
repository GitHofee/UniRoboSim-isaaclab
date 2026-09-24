from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from unirobosim_isaaclab.camera_visibility_transaction import render_camera_groups


class Attribute:
    def __init__(self, value=None):
        self.value = value

    def HasAuthoredValueOpinion(self):
        return self.value is not None

    def Get(self):
        return self.value or "inherited"

    def Set(self, value):
        self.value = value

    def Clear(self):
        self.value = None


class Product:
    def __init__(self):
        self.enabled = True
        self.frames = []

    def get_updates_enabled(self):
        return self.enabled

    def set_updates_enabled(self, value):
        self.enabled = value


@pytest.mark.parametrize("fail", [False, True])
def test_each_view_sees_only_its_exclusion_and_all_state_is_restored(fail):
    attrs = {"shell_l": Attribute(), "shell_r": Attribute("inherited")}
    products = {name: [Product()] for name in ("head", "left", "right")}
    viewport = SimpleNamespace(updates_enabled=True)
    viewport_frames = []
    renders = []

    def render():
        hidden = tuple(name for name, attr in attrs.items() if attr.Get() == "invisible")
        renders.append(hidden)
        for bindings in products.values():
            for product in bindings:
                if product.enabled:
                    product.frames.append(hidden)
        if viewport.updates_enabled:
            viewport_frames.append(hidden)
        if fail and hidden:
            raise RuntimeError("render failed")

    @contextmanager
    def scope(hidden):
        before = {name: attr.value for name, attr in attrs.items()}
        try:
            for name in hidden:
                attrs[name].Set("invisible")
            yield
        finally:
            for name, value in before.items():
                attrs[name].value = value

    def invoke():
        render_camera_groups(
            {"head": (), "left": ("shell_l",), "right": ("shell_r",)},
            products,
            [viewport],
            render,
            scope,
        )

    if fail:
        with pytest.raises(RuntimeError, match="render failed"):
            invoke()
    else:
        invoke()
        assert products["head"][0].frames == [()]
        assert products["left"][0].frames == [("shell_l",)]
        assert products["right"][0].frames == [("shell_r",)]
        assert len(renders) == 3  # Three product frames total, never nine.
    assert viewport_frames == ([] if fail else [()])
    assert attrs["shell_l"].value is None
    assert attrs["shell_r"].value == "inherited"
    assert all(p.enabled for bindings in products.values() for p in bindings)
    assert viewport.updates_enabled


def test_visibility_scope_leaves_reference_and_session_opinions_unchanged():
    import subprocess
    import sys

    code = """
from pxr import Usd, UsdGeom, Sdf
from unirobosim_isaaclab.camera_visibility_transaction import temporary_mesh_visibility
source = Sdf.Layer.CreateAnonymous()
s = Usd.Stage.Open(source)
mesh = UsdGeom.Mesh.Define(s, '/Mesh')
mesh.GetVisibilityAttr().Set('inherited')
mesh.GetVisibilityAttr().Set('inherited', 2)
stage = Usd.Stage.CreateInMemory()
stage.DefinePrim('/Shell').GetReferences().AddReference(source.identifier, '/Mesh')
layer = Sdf.Layer.CreateAnonymous()
stage.GetSessionLayer().subLayerPaths = [layer.identifier]
before = (source.ExportToString(), stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
for fail in (False, True):
    try:
        with temporary_mesh_visibility(stage, ('/Shell',), usd=Usd, layer=layer, selected=('/Shell',)):
            assert stage.GetPrimAtPath('/Shell').GetAttribute('visibility').Get() == 'invisible'
            if fail:
                raise RuntimeError('expected')
    except RuntimeError as error:
        assert str(error) == 'expected'
    after = (source.ExportToString(), stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
    assert before == after
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

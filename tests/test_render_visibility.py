import pytest
from unirobosim import EntityPath, RenderVisibilityOverride
from unirobosim_isaaclab.render_visibility import RenderVisibilityScope
import importlib.util
import subprocess
import sys
import os


def fixture():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/environment")
    roof = UsdGeom.Xform.Define(stage, "/World/environment/ceilings")
    roof.GetVisibilityAttr().Set("invisible")
    mesh = UsdGeom.Mesh.Define(stage, "/World/environment/ceilings/mesh")
    mesh.GetPointsAttr().Set([(0, 0, 3), (1, 0, 3), (0, 1, 3)])
    roots = {("/environment", 0): "/World/environment"}
    return stage, roots, mesh


def request(path="ceilings"):
    return RenderVisibilityOverride(EntityPath("/environment"), path)


def case_only_visibility_and_cleanup():
    stage, roots, mesh = fixture()
    before = stage.GetRootLayer().ExportToString()
    scope = RenderVisibilityScope(stage, (request(),), roots)
    assert UsdGeom.Imageable(mesh).ComputeVisibility() == "inherited"
    assert len(scope.receipts) == 1
    assert stage.GetRootLayer().ExportToString() == before
    scope.close()
    scope.close()
    assert UsdGeom.Imageable(mesh).ComputeVisibility() == "invisible"
    assert list(stage.GetSessionLayer().subLayerPaths) == []


def case_batch_validation_before_mutation():
    stage, roots, mesh = fixture()
    with pytest.raises(ValueError):
        RenderVisibilityScope(stage, (request(), request("missing")), roots)
    assert UsdGeom.Imageable(mesh).ComputeVisibility() == "invisible"
    assert list(stage.GetSessionLayer().subLayerPaths) == []


def case_preconditions(case):
    stage, roots, mesh = fixture()
    requests = (request(),)
    if case == "duplicate":
        requests = (request(), request())
    if case == "wrong_expected":
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/environment/ceilings")).GetVisibilityAttr().Set("inherited")
    if case == "nonimageable":
        stage.DefinePrim("/World/environment/shader", "Shader")
        requests = (request("shader"),)
    if case == "hidden_ancestor":
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/environment")).GetVisibilityAttr().Set("invisible")
    with pytest.raises(ValueError):
        RenderVisibilityScope(stage, requests, roots)
    assert list(stage.GetSessionLayer().subLayerPaths) == []


def case_stronger_session_opinion_readback_rolls_back():
    stage, roots, mesh = fixture()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/environment/ceilings")).GetVisibilityAttr().Set("invisible")
    unrelated = Sdf.Layer.CreateAnonymous("unrelated.usda")
    stage.GetSessionLayer().subLayerPaths = [unrelated.identifier]
    edit_target = stage.GetEditTarget().GetLayer().identifier
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):
        RenderVisibilityScope(stage, (request(),), roots)
    assert stage.GetSessionLayer().ExportToString() == before
    assert stage.GetEditTarget().GetLayer().identifier == edit_target
    assert UsdGeom.Imageable(mesh).ComputeVisibility() == "invisible"


def case_apply_once_reset_scope():
    from types import SimpleNamespace
    from unirobosim import EntityKind
    from unirobosim_isaaclab.render_visibility import apply, clear
    from unirobosim_isaaclab.native import _native_name

    stage, roots, mesh = fixture()
    # Native naming follows the same fixed world namespace as production.
    path = "/World/env_0/" + _native_name(EntityPath("/environment"))
    UsdGeom.Xform.Define(stage, path)
    roof = UsdGeom.Xform.Define(stage, path + "/ceilings")
    roof.GetVisibilityAttr().Set("invisible")
    world = SimpleNamespace(
        _spec=SimpleNamespace(
            entities=(SimpleNamespace(path=EntityPath("/environment"), kind=EntityKind.COMPOSITE_SCENE),),
            environments=SimpleNamespace(count=1),
        ),
        _m=SimpleNamespace(sim_utils=SimpleNamespace(get_current_stage=lambda: stage)),
        _invalidate_render=lambda: None,
    )
    apply(world, (request(),))
    with pytest.raises(ValueError, match="already applied"):
        apply(world, (request(),))
    clear(world)
    assert roof.GetVisibilityAttr().Get() == "invisible"
    apply(world, (request(),))
    clear(world)
    assert roof.GetVisibilityAttr().Get() == "invisible"


@pytest.mark.parametrize(
    "name,arg",
    [
        ("apply_once_reset_scope", None),
        ("only_visibility_and_cleanup", None),
        ("batch_validation_before_mutation", None),
        ("stronger_session_opinion_readback_rolls_back", None),
        *[("preconditions", case) for case in ["duplicate", "wrong_expected", "nonimageable", "hidden_ancestor"]],
    ],
)
def test_usd_visibility_isolated(name, arg):
    if importlib.util.find_spec("pxr") is None:
        pytest.skip("USD unavailable")
    command = [sys.executable, __file__, name]
    if arg is not None:
        command.append(arg)
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    from pxr import Usd, UsdGeom, Sdf

    globals()["case_" + sys.argv[1]](*sys.argv[2:])

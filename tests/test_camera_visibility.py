import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from unirobosim_isaaclab.camera_visibility import configure_camera_exclusions, partition_groups


def test_identical_exclusion_sets_share_partition_and_default_is_always_preserved():
    groups, cameras = partition_groups({"left": ("/left",), "left2": ("/left", "/left"), "head": ()})
    assert len(groups) == 2
    assert cameras["left"] == cameras["left2"]
    assert cameras["head"] == groups[()]


def _assert_render_references_preserve_collision_and_unconfigured_camera_visibility():
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for name in ("left", "right"):
        mesh = UsdGeom.Mesh.Define(stage, "/" + name).GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh)
    for name in ("head", "left_camera", "right_camera", "unconfigured"):
        UsdGeom.Camera.Define(stage, "/" + name)
    settings = {}
    clones = configure_camera_exclusions(
        stage,
        {"/left_camera": ("/left",), "/right_camera": ("/right",), "/head": ()},
        usd=Usd,
        usd_geom=UsdGeom,
        sdf=Sdf,
        settings=SimpleNamespace(set=lambda k, v: settings.__setitem__(k, v)),
    )
    assert len(clones) == 2
    assert sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()) == 2
    assert all(not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI) for path in clones)
    assert stage.GetPrimAtPath("/head").GetAttribute("omni:scenePartition").Get() == (
        stage.GetPrimAtPath("/unconfigured").GetAttribute("omni:scenePartition").Get()
    )
    assert all(
        stage.GetPrimAtPath(path).GetAttribute("visibility").Get() == "inherited" for path in ("/left", "/right")
    )


def _assert_invalid_selection_does_not_partially_change_cameras():
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Camera.Define(stage, "/camera")
    UsdGeom.Xform.Define(stage, "/body")
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="exact Mesh"):
        configure_camera_exclusions(
            stage,
            {"/camera": ("/body",)},
            usd=Usd,
            usd_geom=UsdGeom,
            sdf=Sdf,
            settings=None,
        )
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize(
    "assertion",
    [
        "_assert_render_references_preserve_collision_and_unconfigured_camera_visibility",
        "_assert_invalid_selection_does_not_partially_change_cameras",
    ],
)
def test_usd_behavior_in_isolated_process(assertion):
    result = subprocess.run(
        [sys.executable, "-c", f"from tests.test_camera_visibility import {assertion}; {assertion}()"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

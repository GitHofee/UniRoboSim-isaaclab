"""Exercise the real path resolver method without importing the optional SDK."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

source = Path(__file__).parents[1] / "src/unirobosim_isaaclab/native_planning.py"
method = next(node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.FunctionDef) and node.name == "_entity_root")
class NativePlanningError(Exception):
    pass
class EntityKind:
    CAMERA_SENSOR = "camera"
namespace = {"Any": object, "EntityKind": EntityKind, "NativePlanningError": NativePlanningError, "_native_name": lambda x: x}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

@pytest.mark.parametrize("env", [0, 2])
@pytest.mark.parametrize("suffix", ["head", "robot/link/head", "robot/head"])
def test_camera_uses_authored_path(env, suffix):
    expected = f"/World/env_{env}/{suffix}"
    calls = []
    world = SimpleNamespace(_entity_prim_path=lambda path, index: expected)
    stage = SimpleNamespace(GetPrimAtPath=lambda path: calls.append(path) or SimpleNamespace(IsValid=lambda: path == expected))
    self = SimpleNamespace(_world=world, _stage=stage)
    assert namespace["_entity_root"](self, SimpleNamespace(path="head", kind="camera"), env).IsValid()
    assert calls == [expected]

@pytest.mark.parametrize("kind", ["camera", "articulation"])
def test_missing_prim_still_rejected(kind):
    self = SimpleNamespace(_world=SimpleNamespace(_entity_prim_path=lambda *args: "/missing"), _stage=SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: False)))
    with pytest.raises(NativePlanningError, match="catalog_invalid"):
        namespace["_entity_root"](self, SimpleNamespace(path="head", kind=kind), 0)

def test_non_camera_original_root_unchanged():
    calls = []
    self = SimpleNamespace(_world=None, _stage=SimpleNamespace(GetPrimAtPath=lambda path: calls.append(path) or SimpleNamespace(IsValid=lambda: True)))
    namespace["_entity_root"](self, SimpleNamespace(path="robot", kind="articulation"), 3)
    assert calls == ["/World/env_3/robot"]
